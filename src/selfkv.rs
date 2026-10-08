// SPDX-License-Identifier: MIT OR Apache-2.0
//! The self-attention products and the f16 self KV cache (0.1.3): what `whisper_build_graph_decoder` computes in every
//! decoder layer before self-attention itself, the cache `whisper_decode_internal` keeps between calls and the mask it
//! builds for each (testing/selfkv/NOTES.md). Read from the pinned source and the shipped `libggml-cpu.so`, then checked
//! by the oracle on every such node of every decoder call `whisper_full` makes for the 8 recorded inputs:
//!
//! - **attn_ln**: 0.0.8's three nodes (`norm`, `· w`, `+ b`) at the batch's row count — one row for a step.
//! - **Q = `scale(mul_mat(attn.query.weight, cur) + attn.query.bias, KQscale)`** — the bias before the scale;
//!   **K = `scale(mul_mat(attn.key.weight, cur), KQscale)`** — no bias, scaled here (the encoder's K is not; the cross
//!   graph scales only K); **V = `mul_mat(attn.value.weight, cur) + attn.value.bias`**. `KQscale` is 0.1.1's
//!   `(float) pow(64.0, −0.25)` = `0x3EB504F3`, one f32 multiply (`ggml_vec_scale_f32`, `b` = 0). The products are
//!   0.0.9's (`from_float` split by thread, then `ggml_vec_dot_f16` per output); for a one-row step the reference runs
//!   the same code as a matrix-vector product.
//! - **the CPYs** (0.0.9's scalar bit trick) into `kv_self.k` / `kv_self.v` at cell `head`: the cache holds
//!   `GGML_PAD(n_text_ctx, 256)` = 512 cells per layer of `n_state` f16 each, layer `il`'s cell `c` at row `il · 512 + c`,
//!   the buffer cleared when the state is made and at the start of every window.
//! - **the cells** ([`KvSelf`]): `whisper_kv_cache_find_slot` (the head walks forward to the first `n_tokens` free cells
//!   in a row and stays there), then `n = min(size, max(1, cell_max))` — the padding is 1 on the CPU, so there are no
//!   padded columns.
//! - **KQ_mask**: f32 `[n_tokens][n]`, 0 or −∞ where a cell lacks the row's sequence or holds a later position, then
//!   cast to f16 (a CPY: 0 → `0x0000`, −∞ → `0xFC00`).
//!
//! The fast path keeps every rounding: attn_ln is computed row by row into the products' conversion (its output never
//! written), one conversion serves Q, K and V; a prompt (four rows or more) takes 0.0.9's 4 × 3 register block over
//! panels of frames, threads by frames; a step (fewer than four rows) is a matrix-vector product read straight from the
//! f16 weights — eight output channels at a time, `vcvtph2ps` on each weight block as `ggml_vec_dot_f16` widens it,
//! the same chains and reduction — on one thread (a scoped spawn costs more than a step's products; the split by
//! output channels is there for larger matrix-vector work and tested); the scale, the bias and the f16 conversion are the
//! epilogue, K and V written straight into the cache's cells; the mask is built once per call, in f16 directly.

use crate::cross::kscale;
use crate::decoder::Batch;
use crate::f16::{fp16_to_fp32, fp16_to_fp32_row, fp32_to_fp16, fp32_to_fp16_row};
use crate::matmul::{self, add_bias, cpy_f16, cpy_f16_model, from_float_row, load_panel, panel_product, par_frames, Kernel, Linear, Shared, PANEL};
use crate::model::Model;
use crate::norm::{self, LayerNorm, Node};

/// `GGML_PAD(n_text_ctx, 256)` (whisper.cpp:3390): the cache's cells per layer.
pub fn kv_self_size(n_text_ctx: usize) -> usize {
    n_text_ctx.div_ceil(256) * 256
}

/// Up to this many rows a call is a matrix-vector product (a step); past it, panels of frames (a prompt).
pub const GEMV_MAX_ROWS: usize = 3;

/// Multiply-adds a matrix-vector call must have per thread before [`SelfAttn::layer_into`] starts another: measured on
/// the 2-core gate laptop, a scoped spawn costs more than a whole step's three products (~0.44 M multiply-adds in
/// ~85 µs at one thread; 2 threads took longer), so a step runs on one thread.
pub const GEMV_MIN_MACS_PER_THREAD: usize = 4 << 20;

/// One cache cell (`whisper_kv_cell`): its position (−1 = free) and the set of sequences it belongs to (bit s = seq s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub pos: i32,
    pub seq: u32,
}

impl Default for Cell {
    fn default() -> Cell {
        Cell { pos: -1, seq: 0 }
    }
}

impl Cell {
    pub fn has_seq(&self, s: i32) -> bool {
        (0..32).contains(&s) && self.seq & (1 << s) != 0
    }
}

/// The self-attention cache, as `whisper_state::kv_self` holds it: `k` and `v` each `[n_layer][size][n_state]` f16.
#[derive(Clone)]
pub struct KvSelf {
    pub n_layer: usize,
    pub n_state: usize,
    /// cells per layer (`GGML_PAD(n_text_ctx, 256)`)
    pub size: usize,
    /// the first cell of the last slot found (`kv_self.head`)
    pub head: usize,
    /// cells the attention reads (`kv_self.n`)
    pub n: usize,
    pub cells: Vec<Cell>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}

/// How the mask is built: `MaskVariant::default()` is the reference's; every other setting is a discriminator.
#[derive(Clone, Copy, Default, Debug)]
pub struct MaskVariant {
    /// a cell at the row's own position masked too (`pos >= p` for `pos > p`)
    pub off_by_one: bool,
    /// the sequence ignored (only the positions) — a free cell (pos −1) is then attended; with one sequence and no free
    /// cell below `n` this cannot be told apart
    pub no_seq: bool,
    /// `n` padded to 32 cells (the Metal padding), the extra columns −∞
    pub pad32: bool,
    /// −∞ written as f32's most negative finite value (the cast then saturates it to f16's −∞ … or not: see the test)
    pub neg_max_f32: bool,
    /// −∞ written as −65504 (f16's most negative finite), in f32 and so in f16
    pub neg_max_f16: bool,
    /// the cast by the row converter (`vcvtps2ph`): it agrees with the bit trick on 0 and −∞, so this cannot be told apart
    pub row_converter: bool,
}

impl KvSelf {
    /// `whisper_kv_cache_init(F16, n_state, n_layer, GGML_PAD(n_text_ctx, 256))`: every cell free, the buffer +0.
    pub fn new(n_layer: usize, n_state: usize, n_text_ctx: usize) -> KvSelf {
        let size = kv_self_size(n_text_ctx);
        KvSelf { n_layer, n_state, size, head: 0, n: 0, cells: vec![Cell::default(); size], k: vec![0; n_layer * size * n_state], v: vec![0; n_layer * size * n_state] }
    }

    pub fn for_model(m: &Model) -> KvSelf {
        let h = &m.hparams;
        KvSelf::new(h.n_text_layer as usize, h.n_text_state as usize, h.n_text_ctx as usize)
    }

    /// `whisper_kv_cache_clear`: every cell free, the head at 0, the buffer +0 (whisper does this at every window).
    pub fn clear(&mut self) {
        self.cells.fill(Cell::default());
        self.head = 0;
        self.k.fill(0);
        self.v.fill(0);
    }

    /// `whisper_kv_cache_find_slot` (whisper.cpp:1019), as written: from the head, the first `n_tokens` free cells in a
    /// row (wrapping to 0 when they would run past the end), the batch's positions and sequences put in them.
    pub fn find_slot(&mut self, b: &Batch) -> Result<(), String> {
        let (n_ctx, n_tokens) = (self.size, b.n_tokens());
        if n_tokens > n_ctx {
            return Err(format!("kv_self: {n_tokens} tokens > {n_ctx} cells"));
        }
        let mut n_tested = 0;
        loop {
            if self.head + n_tokens > n_ctx {
                n_tested += n_ctx - self.head;
                self.head = 0;
                continue;
            }
            let mut found = true;
            for i in 0..n_tokens {
                if self.cells[self.head + i].pos >= 0 {
                    found = false;
                    self.head += i + 1;
                    n_tested += i + 1;
                    break;
                }
            }
            if found {
                break;
            }
            if n_tested >= n_ctx {
                return Err(format!("kv_self: no slot for {n_tokens} tokens"));
            }
        }
        for i in 0..n_tokens {
            let c = &mut self.cells[self.head + i];
            c.pos = b.pos[i];
            for _ in 0..b.n_seq_id[i] {
                let s = b.seq_id[i];
                if !(0..32).contains(&s) {
                    return Err(format!("kv_self: sequence {s} (voaice keeps 32)"));
                }
                c.seq |= 1 << s;
            }
        }
        Ok(())
    }

    /// `whisper_kv_cache_cell_max`: one past the last used cell, scanning down to cell 1 (so at least 1).
    pub fn cell_max(&self) -> usize {
        (1..self.size).rev().find(|&i| self.cells[i].pos >= 0 && self.cells[i].seq != 0).map_or(1, |i| i + 1)
    }

    /// What `whisper_decode_internal` does before it builds the graph: find the slot, then `n = min(size, max(pad,
    /// GGML_PAD(cell_max, pad)))` with the CPU's pad of 1.
    pub fn prepare(&mut self, b: &Batch) -> Result<(), String> {
        self.find_slot(b)?;
        self.n = self.size.min(self.cell_max().max(1));
        Ok(())
    }

    /// Row of layer `il`'s cell `c` in `k` / `v` (in elements: × n_state).
    pub fn row(&self, il: usize, c: usize) -> usize {
        il * self.size + c
    }

    /// The mask by the model: f32 `[n_tokens][n]` as whisper.cpp:2925 fills it, then the cast to f16.
    pub fn mask_model(&self, b: &Batch, v: MaskVariant) -> (Vec<f32>, Vec<u16>) {
        let n_kv = if v.pad32 { self.n.div_ceil(32) * 32 } else { self.n };
        let ninf = if v.neg_max_f32 {
            f32::MIN
        } else if v.neg_max_f16 {
            -65504.0
        } else {
            f32::NEG_INFINITY
        };
        let mut m = vec![0.0f32; b.n_tokens() * n_kv];
        for j in 0..b.n_tokens() {
            let (pos, seq) = (b.pos[j], b.seq_id[j]);
            for i in 0..n_kv {
                let c = self.cells.get(i).copied().unwrap_or_default();
                let later = if v.off_by_one { c.pos >= pos } else { c.pos > pos };
                if i >= self.n || (!v.no_seq && !c.has_seq(seq)) || later {
                    m[j * n_kv + i] = ninf;
                }
            }
        }
        let mut h = vec![0u16; m.len()];
        if v.row_converter {
            fp32_to_fp16_row(&m, &mut h);
        } else {
            h.iter_mut().zip(&m).for_each(|(o, &x)| *o = fp32_to_fp16(x));
        }
        (m, h)
    }

    /// The fast path: the f16 mask straight into `out` (resized to `n_tokens · n`, its capacity kept) — 0 → `0x0000`,
    /// −∞ → `0xFC00`, the bits the cast gives (the f32 values are only ever these two).
    pub fn mask_into(&self, b: &Batch, out: &mut Vec<u16>) {
        let n = self.n;
        out.clear();
        out.resize(b.n_tokens() * n, 0xFC00);
        for (j, row) in out.chunks_exact_mut(n.max(1)).enumerate().take(b.n_tokens()) {
            let (pos, seq) = (b.pos[j], b.seq_id[j]);
            for (o, c) in row.iter_mut().zip(&self.cells[..n]) {
                if c.has_seq(seq) && c.pos <= pos {
                    *o = 0;
                }
            }
        }
    }

    /// Bytes held: the two f16 buffers and the cells.
    pub fn bytes(&self) -> usize {
        2 * (self.k.capacity() + self.v.capacity()) + self.cells.capacity() * std::mem::size_of::<Cell>()
    }
}

/// One decoder layer's attn_ln and self-attention products.
pub struct SelfLayer {
    pub attn_ln: LayerNorm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
}

pub struct SelfAttn {
    pub layers: Vec<SelfLayer>,
    pub n_state: usize,
    pub kqscale: f32,
}

/// One layer's nodes by the model, each `[rows][n_state]`.
#[derive(Default, Debug)]
pub struct SelfNodes {
    pub norm: Vec<f32>,
    pub ln_mul: Vec<f32>,
    pub ln_add: Vec<f32>,
    pub q_mm: Vec<f32>,
    pub q_add: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub k_mm: Vec<f32>,
    pub k_scale: Vec<f32>,
    pub v_mm: Vec<f32>,
    pub v_add: Vec<f32>,
    pub k_cpy: Vec<u16>,
    pub v_cpy: Vec<u16>,
}

/// How the model computes a layer: `SelfVariant::default()` is the reference's; every other setting is a
/// discriminator (a reading the oracle must reject — or, where marked, cannot tell apart).
#[derive(Clone, Copy, Default, Debug)]
pub struct SelfVariant {
    /// attn_ln's changes (0.0.8's discriminators)
    pub ln: norm::Variant,
    /// the products' changes (0.0.9's discriminators)
    pub mm: matmul::Variant,
    /// the activations scaled before Q's and K's products (the scale moved inside the dot)
    pub scale_first: bool,
    /// Q scaled before its bias: `mm · s + b`
    pub q_scale_before_bias: bool,
    /// Q not scaled here (left for the attention, as the encoder's flash attention scales by 1/8)
    pub q_unscaled: bool,
    /// K not scaled (as the encoder's K)
    pub k_unscaled: bool,
    /// K given a bias (the query's: the model has no key bias)
    pub k_biased: bool,
    /// V scaled too
    pub v_scaled: bool,
    /// the scale in double: `(float)((double) x · pow(64, −0.25))`
    pub scale_double: bool,
    /// the CPYs by the row converter — it agrees with the bit trick on every finite value, so this cannot be told apart
    pub cpy_row_converter: bool,
}

/// Optional copies of a layer's f32 nodes from the fast path (each `[rows][n_state]`).
#[derive(Default)]
pub struct SelfTaps<'a> {
    pub q_mm: Option<&'a mut [f32]>,
    pub q_add: Option<&'a mut [f32]>,
    pub k_mm: Option<&'a mut [f32]>,
    pub k_scale: Option<&'a mut [f32]>,
    pub v_mm: Option<&'a mut [f32]>,
    pub v_add: Option<&'a mut [f32]>,
}

fn tap(t: &Shared<f32>, a: usize, b: usize, w: usize, src: &[f32]) {
    // SAFETY: the caller's rows (or channels) are its own, inside the output (asserted by rows())
    if let Some(t) = unsafe { t.rows(a, b, w) } {
        t.copy_from_slice(src);
    }
}

impl SelfAttn {
    pub fn new(m: &Model) -> Result<SelfAttn, String> {
        let h = &m.hparams;
        let n_state = h.n_text_state as usize;
        if h.n_text_head as usize * 64 != n_state {
            return Err(format!("self-attention: n_text_state {n_state} with {} heads: not whisper's head size 64", h.n_text_head));
        }
        let layers = (0..h.n_text_layer as usize)
            .map(|il| {
                let p = format!("decoder.blocks.{il}");
                let lin = |w: &str, b: Option<&str>| Linear::new(m, &format!("{p}.{w}"), b.map(|b| format!("{p}.{b}")).as_deref());
                Ok(SelfLayer {
                    attn_ln: LayerNorm::new(m, &format!("{p}.attn_ln"))?,
                    q: lin("attn.query.weight", Some("attn.query.bias"))?,
                    k: lin("attn.key.weight", None)?,
                    v: lin("attn.value.weight", Some("attn.value.bias"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(SelfAttn { layers, n_state, kqscale: kscale(n_state / h.n_text_head as usize) })
    }

    /// The reference thread count whose `from_float` split the products reproduce (only a NaN can tell).
    pub fn set_split(&mut self, nth: usize) {
        for l in &mut self.layers {
            for x in [&mut l.q, &mut l.k, &mut l.v] {
                x.split = nth.max(1);
            }
        }
    }

    /// Layer `il`'s nodes by the model for the layer's input `x` (`[rows][n_state]`), converted as `nth` threads convert.
    pub fn model(&self, il: usize, x: &[f32], nth: usize, v: SelfVariant) -> SelfNodes {
        let l = &self.layers[il];
        let s = self.kqscale;
        let scale = |a: f32| if v.scale_double { (a as f64 * (64f64).powf(-0.25)) as f32 } else { a * s };
        let norm = l.attn_ln.run_model(x, v.ln, Node::Norm);
        let ln_mul = l.attn_ln.run_model(x, v.ln, Node::Mul);
        let ln_add = l.attn_ln.run_model(x, v.ln, Node::Add);
        let scaled: Vec<f32> = ln_add.iter().map(|&a| a * s).collect();
        let qk_in = if v.scale_first { &scaled } else { &ln_add };
        let q_mm = l.q.model(qk_in, nth, v.mm, false);
        let q_b = l.q.b.as_deref().unwrap();
        let q_add = if v.q_scale_before_bias { add_rows(&q_mm.iter().map(|&a| scale(a)).collect::<Vec<_>>(), q_b) } else { add_rows(&q_mm, q_b) };
        let q_scale: Vec<f32> = if v.scale_first || v.q_unscaled || v.q_scale_before_bias { q_add.clone() } else { q_add.iter().map(|&a| scale(a)).collect() };
        let k_mm = l.k.model(qk_in, nth, v.mm, false);
        let k_pre = if v.k_biased { add_rows(&k_mm, q_b) } else { k_mm.clone() };
        let k_scale: Vec<f32> = if v.scale_first || v.k_unscaled { k_pre } else { k_pre.iter().map(|&a| scale(a)).collect() };
        let v_mm = l.v.model(&ln_add, nth, v.mm, false);
        let mut v_add = add_rows(&v_mm, l.v.b.as_deref().unwrap());
        if v.v_scaled {
            v_add.iter_mut().for_each(|a| *a = scale(*a));
        }
        let cpy = |x: &[f32]| {
            if v.cpy_row_converter {
                let mut y = vec![0u16; x.len()];
                fp32_to_fp16_row(x, &mut y);
                y
            } else {
                cpy_f16_model(x)
            }
        };
        let (k_cpy, v_cpy) = (cpy(&k_scale), cpy(&v_add));
        SelfNodes { norm, ln_mul, ln_add, q_mm, q_add, q_scale, k_mm, k_scale, v_mm, v_add, k_cpy, v_cpy }
    }

    /// The fast path for layer `il`: its input `x` (`[rows][n_state]`) through attn_ln into Q (scaled, into `q`), K
    /// (scaled) and V (+ bias) written as f16 into `kv`'s cells `head .. head + rows` of layer `il`. Up to three rows (a
    /// step) are a matrix-vector product, threads splitting the output channels when the work pays for them (see
    /// [`GEMV_MIN_MACS_PER_THREAD`]: a step takes one); more (a prompt), panels of frames with threads splitting the
    /// frames. Every row's every value is computed whole in one thread, so the count changes no bit.
    pub fn layer_into(&self, il: usize, x: &[f32], threads: usize, kv: &mut KvSelf, q: &mut [f32], taps: SelfTaps) {
        let rows = x.len() / self.n_state.max(1);
        let threads = if rows <= GEMV_MAX_ROWS { threads.min((3 * rows * self.n_state * self.n_state / GEMV_MIN_MACS_PER_THREAD).max(1)) } else { threads };
        self.layer_into_split(il, x, threads, kv, q, taps);
    }

    /// [`SelfAttn::layer_into`] with exactly `threads` threads (clamped to the work): the split the tests check.
    pub fn layer_into_split(&self, il: usize, x: &[f32], threads: usize, kv: &mut KvSelf, q: &mut [f32], taps: SelfTaps) {
        let ns = self.n_state;
        let rows = x.len() / ns;
        assert!(x.len() == rows * ns && q.len() == rows * ns && rows > 0, "self-attention: x and q are not [rows][n_state]");
        assert!(kv.n_state == ns && il < kv.n_layer && kv.head + rows <= kv.size, "self-attention: the cache's slot");
        let at = kv.row(il, kv.head) * ns;
        let (kc, vc) = (&mut kv.k[at..at + rows * ns], &mut kv.v[at..at + rows * ns]);
        if rows <= GEMV_MAX_ROWS {
            self.gemv_into(il, x, threads, kc, vc, q, taps);
        } else {
            self.panels_into(il, x, threads, kc, vc, q, taps);
        }
    }

    /// A prompt: 0.0.9's panels, attn_ln in the conversion, one conversion for the three products, the epilogues.
    #[allow(clippy::too_many_arguments)]
    fn panels_into(&self, il: usize, x: &[f32], threads: usize, kc: &mut [u16], vc: &mut [u16], q: &mut [f32], taps: SelfTaps) {
        let l = &self.layers[il];
        let (ns, s) = (self.n_state, self.kqscale);
        let rows = x.len() / ns;
        let kern = Kernel::detect();
        let (sq, sk, sv) = (Shared::new(Some(q)), Shared::new(Some(kc)), Shared::new(Some(vc)));
        let t = [taps.q_mm, taps.q_add, taps.k_mm, taps.k_scale, taps.v_mm, taps.v_add].map(Shared::new);
        par_frames(rows, threads, &|a, b| {
            let mut px = vec![0.0f32; PANEL * ns];
            let mut tmp = vec![0.0f32; ns];
            let mut raw = vec![0.0f32; PANEL * ns];
            for p in (a..b).step_by(PANEL) {
                let r = PANEL.min(b - p);
                let rn = r * ns;
                load_panel(kern, &x[p * ns..(p + r) * ns], Some(&l.attn_ln), l.k.split, &mut tmp, &mut px);
                // SAFETY (every rows() below): frames [p, p + r) lie in this thread's range [a, b)
                unsafe {
                    // K × KQscale, the CPY into the cache
                    panel_product(kern, &px, r, &l.k, &mut raw);
                    tap(&t[2], p, p + r, ns, &raw[..rn]);
                    raw[..rn].iter_mut().for_each(|y| *y *= s);
                    tap(&t[3], p, p + r, ns, &raw[..rn]);
                    cpy_f16(&raw[..rn], sk.rows(p, p + r, ns).unwrap());
                    // V + bias, the CPY
                    panel_product(kern, &px, r, &l.v, &mut raw);
                    tap(&t[4], p, p + r, ns, &raw[..rn]);
                    let vb = l.v.b.as_deref().unwrap();
                    raw[..rn].chunks_exact_mut(ns).for_each(|row| row.iter_mut().zip(vb).for_each(|(y, &b)| *y += b));
                    tap(&t[5], p, p + r, ns, &raw[..rn]);
                    cpy_f16(&raw[..rn], sv.rows(p, p + r, ns).unwrap());
                    // Q + bias, × KQscale
                    panel_product(kern, &px, r, &l.q, &mut raw);
                    tap(&t[0], p, p + r, ns, &raw[..rn]);
                    let qo = sq.rows(p, p + r, ns).unwrap();
                    add_bias(&raw[..rn], l.q.b.as_deref(), qo);
                    tap(&t[1], p, p + r, ns, qo);
                    qo.iter_mut().for_each(|y| *y *= s);
                }
            }
        });
    }

    /// A step: each row through attn_ln and `from_float` once, then the three products as matrix-vector products from
    /// the f16 weights, threads by blocks of eight output channels.
    #[allow(clippy::too_many_arguments)]
    fn gemv_into(&self, il: usize, x: &[f32], threads: usize, kc: &mut [u16], vc: &mut [u16], q: &mut [f32], taps: SelfTaps) {
        let l = &self.layers[il];
        let (ns, s) = (self.n_state, self.kqscale);
        let rows = x.len() / ns;
        // attn_ln and from_float once per row (the file's order: the weights are read as stored)
        let kern = Kernel::detect();
        let mut xr = vec![0.0f32; rows * ns];
        {
            let (mut y, mut h) = (vec![0.0f32; ns], vec![0u16; ns]);
            for (xi, o) in x.chunks_exact(ns).zip(xr.chunks_exact_mut(ns)) {
                l.attn_ln.row_into(xi, &mut y);
                from_float_row(&y, &mut h, l.k.split);
                fp16_to_fp32_row(&h, o);
            }
        }
        let groups = ns.div_ceil(8);
        let threads = threads.clamp(1, groups);
        let (sq, sk, sv) = (Shared::new(Some(q)), Shared::new(Some(kc)), Shared::new(Some(vc)));
        let t = [taps.q_mm, taps.q_add, taps.k_mm, taps.k_scale, taps.v_mm, taps.v_add].map(Shared::new);
        let xr = &xr;
        let job = |ti: usize| {
            let (c0, c1) = ((groups * ti / threads * 8).min(ns), (groups * (ti + 1) / threads * 8).min(ns));
            let w = c1 - c0;
            let mut out = vec![0.0f32; w];
            for r in 0..rows {
                let xi = &xr[r * ns..(r + 1) * ns];
                let at = r * ns + c0;
                // SAFETY (every rows() below, as rows of width 1): this thread's channels [c0, c1) of row r are its own
                unsafe {
                    gemv(kern, xi, &l.k, c0, c1, &mut out);
                    tap(&t[2], at, at + w, 1, &out);
                    out.iter_mut().for_each(|y| *y *= s);
                    tap(&t[3], at, at + w, 1, &out);
                    cpy_f16(&out, sk.rows(at, at + w, 1).unwrap());
                    gemv(kern, xi, &l.v, c0, c1, &mut out);
                    tap(&t[4], at, at + w, 1, &out);
                    out.iter_mut().zip(&l.v.b.as_deref().unwrap()[c0..c1]).for_each(|(y, &b)| *y += b);
                    tap(&t[5], at, at + w, 1, &out);
                    cpy_f16(&out, sv.rows(at, at + w, 1).unwrap());
                    gemv(kern, xi, &l.q, c0, c1, &mut out);
                    tap(&t[0], at, at + w, 1, &out);
                    let qo = sq.rows(at, at + w, 1).unwrap();
                    qo.iter_mut().zip(out.iter().zip(&l.q.b.as_deref().unwrap()[c0..c1])).for_each(|(y, (&a, &b))| *y = a + b);
                    tap(&t[1], at, at + w, 1, qo);
                    qo.iter_mut().for_each(|y| *y *= s);
                }
            }
        };
        if threads == 1 {
            job(0);
        } else {
            std::thread::scope(|sc| {
                for ti in 1..threads {
                    let job = &job;
                    sc.spawn(move || job(ti));
                }
                job(0);
            });
        }
    }
}

/// `x + b` per row of `b.len()`.
fn add_rows(x: &[f32], b: &[f32]) -> Vec<f32> {
    x.chunks_exact(b.len()).flat_map(|r| r.iter().zip(b).map(|(a, c)| a + c)).collect()
}

/// Output channels `c0..c1` of `lin` for one converted row `x` (f16 values widened, the file's order): each
/// `ggml_vec_dot_f16(k, W row c, x)` read from the f16 weights. A step reads every weight once, and a decoder step
/// streams more weights than this laptop's 4 MB L3 holds: measured, the held f32 copy (2× the bytes) made one layer
/// hot in cache ~2× faster but four layers in a row ~1.7× slower than these f16 reads, whose `vcvtph2ps` is the cost.
fn gemv(kern: Kernel, x: &[f32], lin: &Linear, c0: usize, c1: usize, out: &mut [f32]) {
    let k = lin.k;
    let mut c = c0;
    #[cfg(target_arch = "x86_64")]
    if matches!(kern, Kernel::Avx2) && k.is_multiple_of(32) && std::is_x86_feature_detected!("f16c") {
        while c + 8 <= c1 {
            // SAFETY: AVX2 + FMA + F16C (detected); x holds k values, w rows c..c + 8 of k each, k a multiple of 32
            unsafe { x86::gemv8(x, &lin.w[c * k..(c + 8) * k], k, &mut out[c - c0..c - c0 + 8]) };
            c += 8;
        }
    }
    let _ = kern;
    let mut wf = vec![0.0f32; if c < c1 { k } else { 0 }];
    while c < c1 {
        wf.iter_mut().zip(&lin.w[c * k..(c + 1) * k]).for_each(|(o, &h)| *o = fp16_to_fp32(h));
        out[c - c0] = crate::conv::dot_f16_model(&wf, x);
        c += 1;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// The lane pairing of `ggml_vec_dot_f16`'s reduction for eight sums at once (as matmul.rs's).
    #[inline(always)]
    unsafe fn reduce8(s: &[__m256; 8]) -> __m256 {
        let mut tt = [_mm256_setzero_ps(); 4];
        for (r, t) in tt.iter_mut().enumerate() {
            let lo = _mm256_permute2f128_ps::<0x20>(s[r], s[r + 4]);
            let hi = _mm256_permute2f128_ps::<0x31>(s[r], s[r + 4]);
            *t = _mm256_add_ps(lo, hi);
        }
        _mm256_hadd_ps(_mm256_hadd_ps(tt[0], tt[1]), _mm256_hadd_ps(tt[2], tt[3]))
    }

    /// Eight dots `ggml_vec_dot_f16(k, w row r, x)` (r < 8): for each accumulator j, the chains over the blocks
    /// `32b + 8j` of all eight rows (x loaded once for eight rows, each weight block widened by `vcvtph2ps` as the
    /// reference widens it), then `(a0 + a2) + (a1 + a3)` and the reference's lane pairing.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn gemv8(x: &[f32], w: &[u16], k: usize, out: &mut [f32]) {
        assert!(x.len() >= k && w.len() >= 8 * k && out.len() >= 8 && k.is_multiple_of(32));
        // SAFETY (all loads and the store): x[32b + 8j ..][..8] with 32b + 8j + 8 <= k; w row r the same within its k
        unsafe {
            let (xp, wp) = (x.as_ptr(), w.as_ptr());
            let mut part = [[_mm256_setzero_ps(); 8]; 4];
            for (j, pj) in part.iter_mut().enumerate() {
                let mut a = [_mm256_setzero_ps(); 8];
                let mut i = 8 * j;
                while i < k {
                    let xv = _mm256_loadu_ps(xp.add(i));
                    for (r, ar) in a.iter_mut().enumerate() {
                        let wv = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(r * k + i) as *const __m128i));
                        *ar = _mm256_fmadd_ps(xv, wv, *ar);
                    }
                    i += 32;
                }
                *pj = a;
            }
            let mut s = [_mm256_setzero_ps(); 8];
            for (d, sd) in s.iter_mut().enumerate() {
                *sd = _mm256_add_ps(_mm256_add_ps(part[0][d], part[2][d]), _mm256_add_ps(part[1][d], part[3][d]));
            }
            _mm256_storeu_ps(out.as_mut_ptr(), reduce8(&s));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn layer(ns: usize, seed: &mut u64) -> SelfLayer {
        let lin = |bias: bool, seed: &mut u64| {
            let w: Vec<u16> = (0..ns * ns).map(|_| fp32_to_fp16(((rnd(seed) % 2001) as f32 - 1000.0) / 4000.0)).collect();
            let b = bias.then(|| (0..ns).map(|_| ((rnd(seed) % 2001) as f32 - 1000.0) / 1000.0).collect());
            Linear::from_parts(ns, ns, w, b).unwrap()
        };
        let w: Vec<f32> = (0..ns).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let b: Vec<f32> = (0..ns).map(|i| (i % 5) as f32 / 16.0 - 0.1).collect();
        SelfLayer { attn_ln: LayerNorm::from_parts(w, b, 1e-5).unwrap(), q: lin(true, seed), k: lin(false, seed), v: lin(true, seed) }
    }

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|a| a.to_bits()).collect()
    }

    /// The fast path (both shapes, several thread counts, with taps, into a cache at a head) = the model.
    #[test]
    fn fast_equals_model() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let ns = 96;
        let sa = SelfAttn { layers: vec![layer(ns, &mut seed), layer(ns, &mut seed)], n_state: ns, kqscale: kscale(64) };
        for rows in [1usize, 2, 3, 4, 7, 70] {
            let x: Vec<f32> = (0..rows * ns).map(|_| ((rnd(&mut seed) % 20001) as f32 - 10000.0) / 997.0).collect();
            for il in 0..2 {
                let want = sa.model(il, &x, 1, SelfVariant::default());
                for threads in [1, 2, 3, 5] {
                    let mut kv = KvSelf::new(2, ns, 100);
                    kv.head = 11;
                    kv.k.fill(0xFFFF);
                    let mut q = vec![0.0f32; rows * ns];
                    let mut tp: Vec<Vec<f32>> = (0..6).map(|_| vec![0.0f32; rows * ns]).collect();
                    let [a, b, c, d, e, f] = &mut tp[..] else { unreachable!() };
                    let taps = SelfTaps { q_mm: Some(a), q_add: Some(b), k_mm: Some(c), k_scale: Some(d), v_mm: Some(e), v_add: Some(f) };
                    sa.layer_into_split(il, &x, threads, &mut kv, &mut q, taps);
                    let at = kv.row(il, 11) * ns;
                    let tag = format!("rows {rows} layer {il} threads {threads}");
                    assert_eq!(bits(&q), bits(&want.q_scale), "{tag}: q");
                    for (got, want, name) in [(&tp[0], &want.q_mm, "q_mm"), (&tp[1], &want.q_add, "q_add"), (&tp[2], &want.k_mm, "k_mm"), (&tp[3], &want.k_scale, "k_scale"), (&tp[4], &want.v_mm, "v_mm"), (&tp[5], &want.v_add, "v_add")] {
                        assert_eq!(bits(got), bits(want), "{tag}: {name}");
                    }
                    assert_eq!(&kv.k[at..at + rows * ns], &want.k_cpy[..], "{tag}: k cells");
                    assert_eq!(&kv.v[at..at + rows * ns], &want.v_cpy[..], "{tag}: v cells");
                    assert!(kv.k[..at].iter().chain(&kv.k[at + rows * ns..]).all(|&h| h == 0xFFFF), "{tag}: only the slot written");
                }
            }
        }
    }

    /// find_slot, cell_max and n as whisper's: a prompt from cell 0, each step at the next cell; the mask causal, its
    /// fast f16 = the model's cast.
    #[test]
    fn cells_and_mask() {
        let mut kv = KvSelf::new(1, 32, 448);
        assert_eq!(kv.size, 512);
        let mut b = Batch::with_capacity(8);
        b.prep_legacy(&[1, 2, 3], 0, 0);
        kv.prepare(&b).unwrap();
        assert_eq!((kv.head, kv.n), (0, 3));
        let mut m16 = Vec::new();
        kv.mask_into(&b, &mut m16);
        let (m32, want16) = kv.mask_model(&b, MaskVariant::default());
        assert_eq!(m16, want16);
        assert_eq!(m32.iter().map(|&x| if x == 0.0 { 0 } else { 1 }).collect::<Vec<_>>(), [0, 1, 1, 0, 0, 1, 0, 0, 0]);
        assert_eq!(want16[1], 0xFC00);
        for i in 0..4 {
            b.prep_step(9, 3, i, 0);
            kv.prepare(&b).unwrap();
            assert_eq!((kv.head, kv.n), (3 + i, 4 + i));
            kv.mask_into(&b, &mut m16);
            assert_eq!(m16, kv.mask_model(&b, MaskVariant::default()).1);
            assert!(m16.iter().all(|&h| h == 0));
        }
        // a one-row prompt: cell_max never looks at cell 0, so n is 1 either way
        kv.clear();
        b.prep_legacy(&[1], 0, 0);
        kv.prepare(&b).unwrap();
        assert_eq!((kv.head, kv.n), (0, 1));
    }
}
