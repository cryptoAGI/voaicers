// SPDX-License-Identifier: MIT OR Apache-2.0
//! The encoder's matrix products on activations as the shipped ggml-cpu computes them (0.0.9,
//! testing/matmul/NOTES.md): per block, Q (+ bias), K (no bias), V (+ bias) on attn_ln's output, K and V copied to
//! f16 for flash attention, the attention's out projection (+ bias, + the block's input), and the MLP: mlp_ln, fc1
//! (+ bias), GELU, fc2 (+ bias, + the residual). Read from the pinned source and the shipped `libggml-cpu.so`, then
//! checked by the oracle on every one of those nodes of whisper's encoder graph:
//!
//! - **`mul_mat(W, x)`** with W an f16 [K, N] weight and x the f32 activations [K, frames]: `vec_dot_type` of F16 is
//!   F16, so x is first converted by `from_float` = `ggml_cpu_fp32_to_fp16` (0.0.3's row converter: `vcvtps2ph` on
//!   blocks of 8, then of 4, the portable bit trick on the last `len % 4`) — **each thread converting its element range
//!   `[ith·K/nth, (ith+1)·K/nth)` of every row**, so where a NaN lands relative to those ranges decides which converter
//!   sees it; every finite value converts the same either way. Then every output is one `ggml_vec_dot_f16(K, W row,
//!   converted x row)` (0.0.6's kernel, whole in one thread; the product of two halves is exact, so the operand order
//!   changes no bit), written to the f32 node. No tinyBLAS (`LLAMAFILE` off), no repack (none for f16).
//! - The bias, the residual and GELU are separate nodes: one f32 add each, 0.0.3's GELU. K has no bias.
//! - **The CPY of K and V into flash attention's f16 cache** converts element by element with the *scalar*
//!   `GGML_CPU_FP32_TO_FP16` (the bit trick), not the row converter.
//!
//! The fast path keeps every one of those roundings and changes only what is free: the activations are converted once
//! per frame (Q, K and V share one conversion, which the reference makes three times), straight into a panel in the
//! kernel's layout, rounded through F16C and widened back to f32 (exact); attn_ln and mlp_ln are computed row by row
//! into that conversion, so their f32 output is never written; a 4-frame × 3-row register block shares each weight
//! load across four frames and each frame load across three rows, every dot chained in `ggml_vec_dot_f16`'s order;
//! bias, residual, GELU and the f16 copies are the epilogue of a panel of frames, and the MLP runs a panel at a time
//! from the out projection to fc2, so its 1536-wide hidden layer never leaves the panel. Threads split frames.

use crate::conv::dot_f16_model;
use crate::f16::{fp16_to_fp32, fp32_to_fp16, fp32_to_fp16_f16c, fp32_to_fp16_row};
use crate::gelu::{gelu_f32, Gelu};
use crate::model::{Dtype, Model};
use crate::norm::LayerNorm;

/// Frames converted and multiplied together (a panel): the weights are read once per panel.
pub const PANEL: usize = 64;

/// The element ranges `[ith·k/nth, (ith+1)·k/nth)` of one row that `ggml_compute_forward_mul_mat`'s `nth` threads
/// each convert with one `from_float` call.
pub fn split_ranges(k: usize, nth: usize) -> impl Iterator<Item = (usize, usize)> {
    let nth = nth.max(1);
    (0..nth).map(move |i| (i * k / nth, (i + 1) * k / nth))
}

/// `from_float` on one row as the reference's `nth` threads convert it: each range one `ggml_cpu_fp32_to_fp16` call.
pub fn from_float_row(x: &[f32], y: &mut [u16], nth: usize) {
    assert_eq!(x.len(), y.len(), "from_float: lengths differ");
    for (a, b) in split_ranges(x.len(), nth) {
        fp32_to_fp16_row(&x[a..b], &mut y[a..b]);
    }
}

/// How the model computes the products: the reference's way is `Variant::default()`; every other setting is a
/// discriminator (a reading of the source the oracle must reject).
#[derive(Clone, Copy, Default, Debug)]
pub struct Variant {
    /// every activation through the scalar bit trick (the CPY's converter) instead of the row converter
    pub convert_scalar: bool,
    /// the row converter over each whole row, ignoring how the threads split it
    pub convert_unsplit: bool,
    /// the activations not rounded to f16 (an f16 weight times an f32 activation, the same order)
    pub no_f16: bool,
    /// one f32 accumulator in index order instead of 4 × 8 lanes
    pub single_accumulator: bool,
    /// the four accumulators reduced `((a0 + a1) + a2) + a3` instead of `(a0 + a2) + (a1 + a3)`
    pub sequential_reduce: bool,
    /// the bias as the first accumulator's starting value (added before the products, not after the node)
    pub bias_in_accumulator: bool,
    /// the residual added before the bias: `(mm + x) + b`
    pub residual_before_bias: bool,
    /// GELU computed from x itself (`gelu_f32`), not through ggml's f16 table
    pub gelu_unrounded: bool,
}

/// One weight (f16 [n rows][k]) and its bias, as whisper's graph multiplies frames of k activations by it.
pub struct Linear {
    pub k: usize,
    pub n: usize,
    /// the f16 weights, row c = output channel c (the file's order)
    pub w: Vec<u16>,
    /// the weights widened (exact), in the kernel's layout: per row, accumulator j's blocks contiguous
    wp: Vec<f32>,
    pub b: Option<Vec<f32>>,
    /// the reference thread count whose `from_float` split the fast path reproduces (see [`split_ranges`]); only a NaN
    /// can tell one split from another
    pub split: usize,
}

/// Position of element `i` of a row of `k` (a multiple of 32) in the kernel's layout: element `32b + 8j + l` (block
/// b, accumulator j, lane l of `ggml_vec_dot_f16`) at `j·k/4 + 8b + l`.
#[inline]
pub fn perm(i: usize, k: usize) -> usize {
    let (b, j, l) = (i / 32, (i / 8) % 4, i % 8);
    j * (k / 4) + 8 * b + l
}

impl Linear {
    /// `<weight>` (f16 [k, n]) and, if named, `<bias>` (f32 [n]) of the model.
    pub fn new(m: &Model, weight: &str, bias: Option<&str>) -> Result<Linear, String> {
        let t = m.tensor(weight).ok_or(format!("no {weight}"))?;
        if t.dtype != Dtype::F16 || t.ne[2] != 1 || t.ne[3] != 1 {
            return Err(format!("{weight}: expected a 2-d f16 tensor"));
        }
        let w: Vec<u16> = m.tensor_bytes(t).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let b = match bias {
            None => None,
            Some(name) => {
                let tb = m.tensor(name).ok_or(format!("no {name}"))?;
                if tb.dtype != Dtype::F32 {
                    return Err(format!("{name}: expected f32"));
                }
                Some(m.tensor_bytes(tb).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
            }
        };
        Linear::from_parts(t.ne[0] as usize, t.ne[1] as usize, w, b)
    }

    pub fn from_parts(k: usize, n: usize, w: Vec<u16>, b: Option<Vec<f32>>) -> Result<Linear, String> {
        if k == 0 || n == 0 || w.len() != k * n || b.as_ref().is_some_and(|b| b.len() != n) {
            return Err("linear: shapes".into());
        }
        let mut wp = vec![0.0f32; k * n];
        if k.is_multiple_of(32) {
            for c in 0..n {
                for i in 0..k {
                    wp[c * k + perm(i, k)] = fp16_to_fp32(w[c * k + i]);
                }
            }
        } else {
            for (o, &h) in wp.iter_mut().zip(&w) {
                *o = fp16_to_fp32(h);
            }
        }
        Ok(Linear { k, n, w, wp, b, split: 1 })
    }

    /// The activations of one frame as the dot reads them (f16, widened), by the model: `from_float` split over `nth`.
    pub fn convert_model(x: &[f32], nth: usize, v: Variant) -> Vec<f32> {
        if v.no_f16 {
            return x.to_vec();
        }
        let mut h = vec![0u16; x.len()];
        if v.convert_scalar {
            for (o, &a) in h.iter_mut().zip(x) {
                *o = fp32_to_fp16(a);
            }
        } else {
            from_float_row(x, &mut h, if v.convert_unsplit { 1 } else { nth });
        }
        h.iter().map(|&b| fp16_to_fp32(b)).collect()
    }

    /// The MUL_MAT node (or with `bias`, the ADD after it) by the model, for frames `x` ([rows][k]) converted as `nth`
    /// threads convert them; out [rows][n].
    pub fn model(&self, x: &[f32], nth: usize, v: Variant, bias: bool) -> Vec<f32> {
        let (k, n) = (self.k, self.n);
        assert!(x.len().is_multiple_of(k), "linear: the input is not whole rows");
        let wf: Vec<f32> = self.w.iter().map(|&h| fp16_to_fp32(h)).collect();
        let mut out = vec![0.0f32; x.len() / k * n];
        for (xr, o) in x.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
            let y = Linear::convert_model(xr, nth, v);
            for (c, oc) in o.iter_mut().enumerate() {
                let w = &wf[c * k..(c + 1) * k];
                let b = if bias { self.b.as_ref().map_or(0.0, |b| b[c]) } else { 0.0 };
                *oc = if v.bias_in_accumulator && bias {
                    dot_variant(w, &y, v, Some(b))
                } else {
                    let d = dot_variant(w, &y, v, None);
                    match (bias, &self.b) {
                        (true, Some(b)) => d + b[c],
                        _ => d,
                    }
                };
            }
        }
        out
    }

    /// The reference thread count to reproduce in the conversion (only a NaN can tell; see [`split_ranges`]).
    pub fn with_split(mut self, nth: usize) -> Linear {
        self.split = nth.max(1);
        self
    }
}

/// `ggml_vec_dot_f16(w, y)` (both widened from f16) with a variant's changes; `bias0`: the first accumulator's start.
fn dot_variant(w: &[f32], y: &[f32], v: Variant, bias0: Option<f32>) -> f32 {
    if v.single_accumulator {
        return w.iter().zip(y).fold(bias0.unwrap_or(0.0), |s, (a, b)| s + a * b);
    }
    if !v.sequential_reduce && bias0.is_none() {
        return dot_f16_model(w, y);
    }
    let n = w.len();
    let np = n & !31;
    let mut acc = [[0.0f32; 8]; 4];
    if let Some(b) = bias0 {
        acc[0][0] = b;
    }
    for i in (0..np).step_by(32) {
        for (j, a) in acc.iter_mut().enumerate() {
            for (l, al) in a.iter_mut().enumerate() {
                *al += w[i + 8 * j + l] * y[i + 8 * j + l];
            }
        }
    }
    let s: Vec<f32> = (0..8)
        .map(|l| if v.sequential_reduce { ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l] } else { (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]) })
        .collect();
    let t = [s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]];
    let mut sum = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
    for k in np..n {
        sum += (w[k] * y[k]) as f64;
    }
    sum as f32
}

/// The residual ADD node: `a + r`, element by element (or with `residual_before_bias`, `(mm + r) + b` from the
/// MUL_MAT node `a` and the bias `b`).
pub fn residual_model(a: &[f32], r: &[f32]) -> Vec<f32> {
    a.iter().zip(r).map(|(x, y)| x + y).collect()
}

/// The GELU node by the model (0.0.3's op), or with `gelu_unrounded` the formula on x itself.
pub fn gelu_model(g: &Gelu, x: &[f32], v: Variant) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    if v.gelu_unrounded {
        for (o, &a) in y.iter_mut().zip(x) {
            *o = if a <= -10.0 { 0.0 } else if a >= 10.0 { a } else { gelu_f32(a) };
        }
    } else {
        g.row_scalar(x, &mut y);
    }
    y
}

/// The f32 → f16 CPY node (K and V into flash attention's cache): the scalar bit trick, element by element.
pub fn cpy_f16_model(x: &[f32]) -> Vec<u16> {
    x.iter().map(|&a| fp32_to_fp16(a)).collect()
}

// ---- the fast path -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) enum Kernel {
    Model,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Kernel {
    pub(crate) fn detect() -> Kernel {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
            return Kernel::Avx2;
        }
        Kernel::Model
    }
}

/// One frame's activations rounded to f16 and widened, into the kernel's layout (k a multiple of 32), as `from_float`
/// split over `split` threads converts them: `vcvtps2ph` everywhere except a NaN in the last `len % 4` of a range,
/// which takes the bit trick (only there do the two differ).
fn convert_perm(kern: Kernel, x: &[f32], split: usize, out: &mut [f32]) {
    let k = x.len();
    match kern {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX and F16C (detect); x and out have length k, a multiple of 32 (asserted by the caller)
        Kernel::Avx2 => unsafe { x86::convert_perm(x, out) },
        Kernel::Model => {
            for (i, &a) in x.iter().enumerate() {
                out[perm(i, k)] = fp16_to_fp32(fp32_to_fp16_f16c(a));
            }
        }
    }
    for (a, b) in split_ranges(k, split) {
        for i in b - (b - a) % 4..b {
            if x[i].is_nan() {
                out[perm(i, k)] = fp16_to_fp32(fp32_to_fp16(x[i]));
            }
        }
    }
}

/// The dots of `rows` frames of a panel (`px`: [rows4][k] in the kernel's layout, rows4 = rows rounded up to 4, the
/// padding rows zero) with every weight row: `raw[f·n + c]`.
pub(crate) fn panel_product(kern: Kernel, px: &[f32], rows: usize, lin: &Linear, raw: &mut [f32]) {
    let (k, n) = (lin.k, lin.n);
    let rows4 = rows.div_ceil(4) * 4;
    let n3 = n - n % 3;
    match kern {
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => {
            let mut res = [[0.0f32; 4]; 3];
            for c in (0..n3).step_by(3) {
                for f in (0..rows4).step_by(4) {
                    // SAFETY: AVX2 + FMA (detect); px holds rows4 rows of k, wp n rows of k, k a multiple of 32
                    unsafe { x86::block43(&px[f * k..(f + 4) * k], &lin.wp[c * k..(c + 3) * k], k, &mut res) };
                    for (cc, rc) in res.iter().enumerate() {
                        for (ff, &v) in rc.iter().enumerate() {
                            if f + ff < rows {
                                raw[(f + ff) * n + c + cc] = v;
                            }
                        }
                    }
                }
            }
            for c in n3..n {
                for f in 0..rows {
                    raw[f * n + c] = dot_perm(&px[f * k..(f + 1) * k], &lin.wp[c * k..(c + 1) * k]);
                }
            }
        }
        Kernel::Model => {
            for c in 0..n {
                for f in 0..rows {
                    raw[f * n + c] = dot_perm(&px[f * k..(f + 1) * k], &lin.wp[c * k..(c + 1) * k]);
                }
            }
        }
    }
}

/// `ggml_vec_dot_f16`'s order on two rows in the kernel's layout (k a multiple of 32: no tail).
fn dot_perm(x: &[f32], w: &[f32]) -> f32 {
    let k = x.len();
    let q = k / 4;
    let mut acc = [[0.0f32; 8]; 4];
    for (j, a) in acc.iter_mut().enumerate() {
        for b in 0..q / 8 {
            for (l, al) in a.iter_mut().enumerate() {
                *al += x[j * q + 8 * b + l] * w[j * q + 8 * b + l];
            }
        }
    }
    let s: Vec<f32> = (0..8).map(|l| (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l])).collect();
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))
}

/// Eight (or fewer) f32 to f16 as the CPY node converts them (the scalar bit trick): F16C unless a NaN is among them.
pub(crate) fn cpy_f16(x: &[f32], y: &mut [u16]) {
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        while i + 8 <= x.len() {
            // SAFETY: F16C and AVX (checked above); x[i..i + 8] and y[i..i + 8] are in bounds
            if !unsafe { x86::cpy8(&x[i..i + 8], &mut y[i..i + 8]) } {
                for j in i..i + 8 {
                    y[j] = fp32_to_fp16(x[j]);
                }
            }
            i += 8;
        }
    }
    for j in i..x.len() {
        y[j] = fp32_to_fp16(x[j]);
    }
}

pub(crate) struct Shared<T>(*mut T, usize);
// SAFETY: threads write disjoint frame ranges of the outputs, and the scope joins them before they are read
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    pub(crate) fn new(s: Option<&mut [T]>) -> Shared<T> {
        match s {
            Some(s) => Shared(s.as_mut_ptr(), s.len()),
            None => Shared(std::ptr::null_mut(), 0),
        }
    }
    /// rows [a, b) of width w, or None when this output was not asked for
    /// SAFETY: the caller's frame range is its own (no other thread writes it) and lies inside the output
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn rows(&self, a: usize, b: usize, w: usize) -> Option<&mut [T]> {
        if self.0.is_null() {
            return None;
        }
        assert!(b * w <= self.1, "an output is too short");
        // SAFETY: as the function's contract
        Some(unsafe { std::slice::from_raw_parts_mut(self.0.add(a * w), (b - a) * w) })
    }
}

/// Run `job(a, b)` on frame ranges [a, b) of `rows` frames (multiples of 4 except the last), one per thread.
pub(crate) fn par_frames(rows: usize, threads: usize, job: &(dyn Fn(usize, usize) + Sync)) {
    let quads = rows.div_ceil(4);
    let threads = threads.clamp(1, quads.max(1));
    std::thread::scope(|s| {
        for ti in 0..threads {
            let (a, b) = ((quads * ti / threads * 4).min(rows), (quads * (ti + 1) / threads * 4).min(rows));
            if ti + 1 == threads {
                job(a, b);
            } else {
                s.spawn(move || job(a, b));
            }
        }
    });
}

/// What [`Linear::run_into`] does after the product, row by row.
#[derive(Clone, Copy)]
pub enum Epilogue<'a> {
    /// the MUL_MAT node
    None,
    /// + bias (the ADD node)
    Bias,
    /// + bias, then GELU (fc1's three nodes)
    BiasGelu(&'a Gelu),
}

impl Linear {
    /// The product (and its epilogue) of every frame of `x` ([rows][k]) into `out` ([rows][n], the caller's), or with
    /// `ln`, of `ln`'s output computed row by row into the conversion (never written). Threads split frames.
    pub fn run_into(&self, x: &[f32], ln: Option<&LayerNorm>, threads: usize, epi: Epilogue, out: &mut [f32]) {
        let (k, n) = (self.k, self.n);
        assert!(k.is_multiple_of(32), "linear: k must be a multiple of 32 (every whisper product is)");
        assert!(x.len().is_multiple_of(k) && out.len() == x.len() / k * n, "linear: shapes");
        let rows = x.len() / k;
        let kern = Kernel::detect();
        let o = Shared::new(Some(out));
        par_frames(rows, threads, &|a, b| {
            let mut px = vec![0.0f32; PANEL * k];
            let mut tmp = vec![0.0f32; k];
            let mut raw = vec![0.0f32; PANEL * n];
            // SAFETY: frames [a, b) are this thread's
            let out = unsafe { o.rows(a, b, n) }.unwrap();
            for p in (a..b).step_by(PANEL) {
                let r = PANEL.min(b - p);
                load_panel(kern, &x[p * k..(p + r) * k], ln, self.split, &mut tmp, &mut px);
                panel_product(kern, &px, r, self, &mut raw);
                let dst = &mut out[(p - a) * n..(p - a + r) * n];
                match epi {
                    Epilogue::None => dst.copy_from_slice(&raw[..r * n]),
                    Epilogue::Bias | Epilogue::BiasGelu(_) => {
                        add_bias(&raw[..r * n], self.b.as_deref(), dst);
                        if let Epilogue::BiasGelu(g) = epi {
                            gelu_in_place(g, dst, &mut raw[..r * n]);
                        }
                    }
                }
            }
        });
    }
}

/// `dst = src + b` per row (b broadcast; no bias: a copy).
pub(crate) fn add_bias(src: &[f32], b: Option<&[f32]>, dst: &mut [f32]) {
    match b {
        None => dst.copy_from_slice(src),
        Some(b) => {
            for (s, d) in src.chunks_exact(b.len()).zip(dst.chunks_exact_mut(b.len())) {
                for ((o, &x), &y) in d.iter_mut().zip(s).zip(b) {
                    *o = x + y;
                }
            }
        }
    }
}

/// GELU of `x` in place, through `scratch` (the same length).
fn gelu_in_place(g: &Gelu, x: &mut [f32], scratch: &mut [f32]) {
    scratch.copy_from_slice(x);
    g.row(scratch, x);
}

/// `r` frames (`x`: [r][k], or `ln`'s output of them) converted into the panel's first `r` rows, padding rows to 4 zero.
pub(crate) fn load_panel(kern: Kernel, x: &[f32], ln: Option<&LayerNorm>, split: usize, tmp: &mut [f32], px: &mut [f32]) {
    let k = tmp.len();
    let r = x.len() / k;
    for (f, xr) in x.chunks_exact(k).enumerate() {
        let src = match ln {
            Some(ln) => {
                ln.row_into(xr, tmp);
                &*tmp
            }
            None => xr,
        };
        convert_perm(kern, src, split, &mut px[f * k..(f + 1) * k]);
    }
    let r4 = r.div_ceil(4) * 4;
    px[r * k..r4 * k].fill(0.0);
}

/// The products of one block of the encoder, with its two layer norms and GELU (attention itself is v0.1.0's).
pub struct Block {
    pub attn_ln: LayerNorm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub mlp_ln: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub gelu: Gelu,
}

/// Optional copies of the attention half's intermediate nodes (the oracle reads them; the encoder does not need them).
#[derive(Default)]
pub struct QkvTaps<'a> {
    pub k_mm: Option<&'a mut [f32]>,
    pub v_mm: Option<&'a mut [f32]>,
    pub v_add: Option<&'a mut [f32]>,
    pub q_mm: Option<&'a mut [f32]>,
}

/// Optional copies of the MLP half's intermediate nodes.
#[derive(Default)]
pub struct MlpTaps<'a> {
    pub o_mm: Option<&'a mut [f32]>,
    pub o_add: Option<&'a mut [f32]>,
    pub o_res: Option<&'a mut [f32]>,
    pub fc1_mm: Option<&'a mut [f32]>,
    pub fc1_add: Option<&'a mut [f32]>,
    pub gelu: Option<&'a mut [f32]>,
    pub fc2_mm: Option<&'a mut [f32]>,
    pub fc2_add: Option<&'a mut [f32]>,
}

fn tap(t: Option<&mut [f32]>, src: &[f32]) {
    if let Some(t) = t {
        t.copy_from_slice(src);
    }
}

impl Block {
    /// Block `il` of the model's encoder.
    pub fn new(m: &Model, il: usize) -> Result<Block, String> {
        let p = format!("encoder.blocks.{il}");
        let lin = |w: &str, b: Option<&str>| Linear::new(m, &format!("{p}.{w}"), b.map(|b| format!("{p}.{b}")).as_deref());
        Ok(Block {
            attn_ln: LayerNorm::new(m, &format!("{p}.attn_ln"))?,
            q: lin("attn.query.weight", Some("attn.query.bias"))?,
            k: lin("attn.key.weight", None)?,
            v: lin("attn.value.weight", Some("attn.value.bias"))?,
            o: lin("attn.out.weight", Some("attn.out.bias"))?,
            mlp_ln: LayerNorm::new(m, &format!("{p}.mlp_ln"))?,
            fc1: lin("mlp.0.weight", Some("mlp.0.bias"))?,
            fc2: lin("mlp.2.weight", Some("mlp.2.bias"))?,
            gelu: Gelu::new(),
        })
    }

    /// The reference thread count whose `from_float` split the products reproduce (only a NaN can tell).
    pub fn set_split(&mut self, nth: usize) {
        for l in [&mut self.q, &mut self.k, &mut self.v, &mut self.o, &mut self.fc1, &mut self.fc2] {
            l.split = nth.max(1);
        }
    }

    /// The attention half's inputs from the block's input `x` ([rows][n_state]): attn_ln, then Q + bias (`q`, f32),
    /// K and V + bias copied to f16 (`k16`, `v16`: the CPY nodes flash attention reads). attn_ln is computed into the
    /// conversion, once for the three products.
    pub fn qkv_into(&self, x: &[f32], threads: usize, q: &mut [f32], k16: &mut [u16], v16: &mut [u16], taps: QkvTaps) {
        let (k, n) = (self.q.k, self.q.n);
        assert!(self.k.k == k && self.v.k == k && self.k.n == n && self.v.n == n && k.is_multiple_of(32));
        let rows = x.len() / k;
        assert!(x.len() == rows * k && q.len() == rows * n && k16.len() == rows * n && v16.len() == rows * n);
        let kern = Kernel::detect();
        let (sq, sk, sv) = (Shared::new(Some(q)), Shared::new(Some(k16)), Shared::new(Some(v16)));
        let (tk, tv, tva, tq) = (Shared::new(taps.k_mm), Shared::new(taps.v_mm), Shared::new(taps.v_add), Shared::new(taps.q_mm));
        par_frames(rows, threads, &|a, b| {
            let mut px = vec![0.0f32; PANEL * k];
            let mut tmp = vec![0.0f32; k];
            let mut raw = vec![0.0f32; PANEL * n];
            let mut add = vec![0.0f32; PANEL * n];
            for p in (a..b).step_by(PANEL) {
                let r = PANEL.min(b - p);
                let rn = r * n;
                load_panel(kern, &x[p * k..(p + r) * k], Some(&self.attn_ln), self.k.split, &mut tmp, &mut px);
                // SAFETY (every rows() below): frames [p, p + r) lie in this thread's range [a, b)
                unsafe {
                    // K: no bias; the CPY reads the MUL_MAT
                    panel_product(kern, &px, r, &self.k, &mut raw);
                    tap(tk.rows(p, p + r, n), &raw[..rn]);
                    cpy_f16(&raw[..rn], sk.rows(p, p + r, n).unwrap());
                    // V + bias; the CPY reads the ADD
                    panel_product(kern, &px, r, &self.v, &mut raw);
                    tap(tv.rows(p, p + r, n), &raw[..rn]);
                    add_bias(&raw[..rn], self.v.b.as_deref(), &mut add[..rn]);
                    tap(tva.rows(p, p + r, n), &add[..rn]);
                    cpy_f16(&add[..rn], sv.rows(p, p + r, n).unwrap());
                    // Q + bias
                    panel_product(kern, &px, r, &self.q, &mut raw);
                    tap(tq.rows(p, p + r, n), &raw[..rn]);
                    add_bias(&raw[..rn], self.q.b.as_deref(), sq.rows(p, p + r, n).unwrap());
                }
            }
        });
    }

    /// The rest of the block after attention: `att` (the attention's output, [rows][n_state]) through the out
    /// projection + bias + the block's input `x`, then mlp_ln, fc1 + bias, GELU, fc2 + bias + that residual, into
    /// `out` (the next block's input). One panel of frames at a time from the first product to the last: mlp_ln's
    /// output, the 1536-wide hidden layer and the residual stay in the panel.
    pub fn mlp_into(&self, att: &[f32], x: &[f32], threads: usize, out: &mut [f32], taps: MlpTaps) {
        let (ns, nh) = (self.o.n, self.fc1.n);
        assert!(self.o.k == ns && self.fc1.k == ns && self.fc2.k == nh && self.fc2.n == ns && ns.is_multiple_of(32) && nh.is_multiple_of(32));
        let rows = x.len() / ns;
        assert!(x.len() == rows * ns && att.len() == x.len() && out.len() == x.len());
        let kern = Kernel::detect();
        let so = Shared::new(Some(out));
        let t = [taps.o_mm, taps.o_add, taps.o_res, taps.fc1_mm, taps.fc1_add, taps.gelu, taps.fc2_mm, taps.fc2_add].map(Shared::new);
        par_frames(rows, threads, &|a, b| {
            let mut px = vec![0.0f32; PANEL * ns];
            let mut ph = vec![0.0f32; PANEL * nh];
            let mut tmp = vec![0.0f32; nh];
            let mut raw = vec![0.0f32; PANEL * nh];
            let mut hid = vec![0.0f32; PANEL * nh];
            let mut res = vec![0.0f32; PANEL * ns];
            for p in (a..b).step_by(PANEL) {
                let r = PANEL.min(b - p);
                let (rs, rh) = (r * ns, r * nh);
                // SAFETY (every rows() below): frames [p, p + r) lie in this thread's range [a, b)
                unsafe {
                    // out projection + bias, + the block's input
                    load_panel(kern, &att[p * ns..(p + r) * ns], None, self.o.split, &mut tmp[..ns], &mut px);
                    panel_product(kern, &px, r, &self.o, &mut raw);
                    tap(t[0].rows(p, p + r, ns), &raw[..rs]);
                    add_bias(&raw[..rs], self.o.b.as_deref(), &mut res[..rs]);
                    tap(t[1].rows(p, p + r, ns), &res[..rs]);
                    for (o, &y) in res[..rs].iter_mut().zip(&x[p * ns..(p + r) * ns]) {
                        *o += y;
                    }
                    tap(t[2].rows(p, p + r, ns), &res[..rs]);
                    // mlp_ln into fc1's conversion; fc1 + bias, GELU
                    load_panel(kern, &res[..rs], Some(&self.mlp_ln), self.fc1.split, &mut tmp[..ns], &mut px);
                    panel_product(kern, &px, r, &self.fc1, &mut raw);
                    tap(t[3].rows(p, p + r, nh), &raw[..rh]);
                    add_bias(&raw[..rh], self.fc1.b.as_deref(), &mut hid[..rh]);
                    tap(t[4].rows(p, p + r, nh), &hid[..rh]);
                    self.gelu.row(&hid[..rh], &mut raw[..rh]);
                    tap(t[5].rows(p, p + r, nh), &raw[..rh]);
                    // fc2 + bias, + the residual
                    load_panel(kern, &raw[..rh], None, self.fc2.split, &mut tmp, &mut ph);
                    panel_product(kern, &ph, r, &self.fc2, &mut hid);
                    tap(t[6].rows(p, p + r, ns), &hid[..rs]);
                    let dst = so.rows(p, p + r, ns).unwrap();
                    add_bias(&hid[..rs], self.fc2.b.as_deref(), dst);
                    tap(t[7].rows(p, p + r, ns), dst);
                    for (o, &y) in dst.iter_mut().zip(&res[..rs]) {
                        *o += y;
                    }
                }
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// One frame (k a multiple of 32) rounded through F16C and widened, into the kernel's layout: the 8-block
    /// `32b + 8j` to `j·k/4 + 8b`.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn convert_perm(x: &[f32], out: &mut [f32]) {
        let k = x.len();
        assert!(out.len() >= k && k.is_multiple_of(32));
        let q = k / 4;
        // SAFETY: every 8-block read is inside x, every write inside out[..k] (asserted)
        unsafe {
            for b in 0..k / 32 {
                for j in 0..4 {
                    let v = _mm256_loadu_ps(x.as_ptr().add(32 * b + 8 * j));
                    let r = _mm256_cvtph_ps(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v));
                    _mm256_storeu_ps(out.as_mut_ptr().add(j * q + 8 * b), r);
                }
            }
        }
    }

    /// Eight f32 to f16 through F16C; false (y untouched) if any is NaN.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn cpy8(x: &[f32], y: &mut [u16]) -> bool {
        assert!(x.len() >= 8 && y.len() >= 8);
        // SAFETY: 8 f32 read, 8 u16 written (asserted)
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr());
            if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(v, v)) != 0 {
                return false;
            }
            _mm_storeu_si128(y.as_mut_ptr().cast(), _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v));
        }
        true
    }

    /// The lane pairing of `ggml_vec_dot_f16`'s reduction for eight sums at once (see conv.rs).
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

    /// 4 frames × 3 weight rows in the kernel's layout, k a multiple of 32 (no tail): for each accumulator j, the 12
    /// chains over the blocks i + 8j (contiguous here) in registers, each weight vector loaded once for 4 frames and
    /// each frame vector once for 3 rows; then `(a0 + a2) + (a1 + a3)` and the reference's lane pairing.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn block43(x: &[f32], w: &[f32], k: usize, res: &mut [[f32; 4]; 3]) {
        assert!(x.len() >= 4 * k && w.len() >= 3 * k && k.is_multiple_of(32));
        // SAFETY (all loads): frame f spans x[f·k, f·k + k), row c w[c·k, c·k + k) — asserted above
        unsafe {
            let (xp, wp) = (x.as_ptr(), w.as_ptr());
            let q = k / 4;
            let mut part = [[_mm256_setzero_ps(); 12]; 4];
            for (j, pj) in part.iter_mut().enumerate() {
                let mut a = [_mm256_setzero_ps(); 12];
                let mut i = j * q;
                while i < (j + 1) * q {
                    let w0 = _mm256_loadu_ps(wp.add(i));
                    let w1 = _mm256_loadu_ps(wp.add(k + i));
                    let w2 = _mm256_loadu_ps(wp.add(2 * k + i));
                    for f in 0..4 {
                        let xv = _mm256_loadu_ps(xp.add(f * k + i));
                        a[f] = _mm256_fmadd_ps(xv, w0, a[f]);
                        a[4 + f] = _mm256_fmadd_ps(xv, w1, a[4 + f]);
                        a[8 + f] = _mm256_fmadd_ps(xv, w2, a[8 + f]);
                    }
                    i += 8;
                }
                *pj = a;
            }
            let mut s = [_mm256_setzero_ps(); 16];
            for (d, sd) in s.iter_mut().enumerate().take(12) {
                *sd = _mm256_add_ps(_mm256_add_ps(part[0][d], part[2][d]), _mm256_add_ps(part[1][d], part[3][d]));
            }
            let r0 = reduce8(&[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
            let r1 = reduce8(&[s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]]);
            let mut o = [0.0f32; 16];
            _mm256_storeu_ps(o.as_mut_ptr(), r0);
            _mm256_storeu_ps(o.as_mut_ptr().add(8), r1);
            for (c, rc) in res.iter_mut().enumerate() {
                rc.copy_from_slice(&o[4 * c..4 * c + 4]);
            }
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
    fn lin(k: usize, n: usize, bias: bool, seed: &mut u64) -> Linear {
        let w: Vec<u16> = (0..k * n).map(|_| fp32_to_fp16(((rnd(seed) % 2001) as f32 - 1000.0) / 4096.0)).collect();
        let b = bias.then(|| (0..n).map(|_| ((rnd(seed) % 2001) as f32 - 1000.0) / 512.0).collect());
        Linear::from_parts(k, n, w, b).unwrap()
    }
    fn acts(rows: usize, k: usize, seed: &mut u64) -> Vec<f32> {
        (0..rows * k).map(|_| ((rnd(seed) % 1_000_001) as f32 - 500_000.0) / 37_000.0).collect()
    }
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn perm_is_a_permutation() {
        for k in [32, 64, 384, 1536] {
            let mut seen = vec![false; k];
            for i in 0..k {
                seen[perm(i, k)] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    /// The fast product (any thread count, any frame count, panels and padding) = the model, with and without bias,
    /// with GELU, and with a layer norm fused into the conversion.
    #[test]
    fn fast_equals_model() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let g = Gelu::new();
        for (k, n, rows) in [(32, 3, 1), (64, 9, 5), (384, 12, 37), (96, 7, 70), (1536, 6, 9)] {
            let l = lin(k, n, true, &mut seed);
            let x = acts(rows, k, &mut seed);
            let want_mm = l.model(&x, 1, Variant::default(), false);
            let want_add = l.model(&x, 1, Variant::default(), true);
            let want_gelu = gelu_model(&g, &want_add, Variant::default());
            for threads in [1, 3] {
                let mut out = vec![0.0f32; rows * n];
                l.run_into(&x, None, threads, Epilogue::None, &mut out);
                assert!(same(&out, &want_mm), "mm k {k} n {n} rows {rows} t {threads}");
                l.run_into(&x, None, threads, Epilogue::Bias, &mut out);
                assert!(same(&out, &want_add), "add k {k} n {n} rows {rows}");
                l.run_into(&x, None, threads, Epilogue::BiasGelu(&g), &mut out);
                assert!(same(&out, &want_gelu), "gelu k {k} n {n} rows {rows}");
            }
            let w: Vec<f32> = (0..k).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
            let b: Vec<f32> = (0..k).map(|i| (i % 5) as f32 / 16.0 - 0.1).collect();
            let ln = LayerNorm::from_parts(w, b, 1e-5).unwrap();
            let normed = ln.run_model(&x, crate::norm::Variant::default(), crate::norm::Node::Add);
            let want = l.model(&normed, 1, Variant::default(), true);
            let mut out = vec![0.0f32; rows * n];
            l.run_into(&x, Some(&ln), 2, Epilogue::Bias, &mut out);
            assert!(same(&out, &want), "fused norm k {k}");
        }
    }

    /// A NaN takes the bit trick only in the last `len % 4` of a thread's range: the fast path with `split` = the
    /// model at that thread count, and the two converters give different bits there.
    #[test]
    fn nan_follows_the_split() {
        let mut seed = 7u64;
        let l = lin(384, 6, true, &mut seed);
        let mut x = acts(2, 384, &mut seed);
        x[152] = f32::from_bits(0x7FC1_2345); // the scalar tail of thread 1's range at 5 threads
        x[384 + 8] = f32::from_bits(0xFFD0_F000);
        let at1 = l.model(&x, 1, Variant::default(), true);
        let at5 = l.model(&x, 5, Variant::default(), true);
        assert_ne!(at1[0].to_bits(), at5[0].to_bits(), "the split must matter for this NaN");
        assert_eq!(at1[6].to_bits(), at5[6].to_bits(), "an 8-block NaN converts the same at any split");
        for nth in [1, 5, 7] {
            let l = Linear::from_parts(l.k, l.n, l.w.clone(), l.b.clone()).unwrap().with_split(nth);
            let mut out = vec![0.0f32; 12];
            l.run_into(&x, None, 1, Epilogue::Bias, &mut out);
            assert!(same(&out, &l.model(&x, nth, Variant::default(), true)), "split {nth}");
        }
    }

    #[test]
    fn cpy_is_the_scalar_trick() {
        let x = vec![1.0, -0.0, 65520.0, 1e-8, f32::from_bits(0x7FC1_2345), 3.3, -7.25, 0.1, 2.5, f32::NAN];
        let mut y = vec![0u16; x.len()];
        cpy_f16(&x, &mut y);
        assert_eq!(y, cpy_f16_model(&x));
    }
}
