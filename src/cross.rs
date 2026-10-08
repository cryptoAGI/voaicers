// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-attention K and V (0.1.1): what `whisper_build_graph_cross` computes once per audio window from `embd_enc`
//! and leaves in `whisper_state::kv_cross` for every decoder step (testing/cross/NOTES.md). Read from the pinned
//! source and the shipped `libggml-cpu.so`, then checked by the oracle on every node of `sched_cross` and on the
//! `kv_cross` buffer itself, padding rows included:
//!
//! - per decoder layer, **K = `mul_mat(cross_attn.key.weight, embd_enc)`** (no bias: the model has none) **then a
//!   SCALE node** — one f32 multiply by `Kscale = (float) pow(64.0, -0.25)` = `0x3EB504F3` (`ggml_vec_scale_f32`,
//!   `vmulps`; its `b` is 0, so the FMA path is not taken) — **V = `mul_mat(cross_attn.value.weight, embd_enc)` + the
//!   value bias** (its own ADD node). The products are 0.0.9's: `from_float` (the row converter, split by thread) then
//!   `ggml_vec_dot_f16` per output.
//! - both CPY'd to f16 with the **scalar** bit trick (0.0.9's CPY) into `kv_cross` at row `il · n_pad`: per layer
//!   `n_pad` = `GGML_PAD(n_ctx, 256)` = 1,536 rows of `n_state` f16, frame-major (flash attention's layout). The CPYs
//!   write rows 0..n_ctx; the cache is cleared when the state is made, so **rows n_ctx..n_pad of every layer are +0**.
//!
//! The fast path keeps every rounding: `embd_enc` is converted **once** per panel of frames for all eight products
//! (the reference converts it in each of its eight MUL_MATs), each product is 0.0.9's 4 × 3 register block, the scale,
//! the bias and the f16 conversion are the panel's epilogue, written straight into the caller's cache. Threads split
//! frames.

use crate::attention::n_kv_pad;
use crate::f16::{fp16_to_fp32, fp32_to_fp16, fp32_to_fp16_row};
use crate::matmul::{self, add_bias, cpy_f16, cpy_f16_model, load_panel, panel_product, par_frames, Kernel, Linear, Shared};
use crate::model::Model;

/// Frames converted together for the eight products (each weight is read once per panel).
pub const CROSS_PANEL: usize = 128;

/// `Kscale` as whisper.cpp:2298 computes it: `pow(float(n_state_head), -0.25)` — `std::pow(float, double)` in double,
/// then the float initialiser. For whisper's head size 64 it is `0x3EB504F3`.
pub fn kscale(n_state_head: usize) -> f32 {
    (n_state_head as f32 as f64).powf(-0.25) as f32
}

/// One decoder layer's cross K and V weights.
pub struct CrossLayer {
    pub k: Linear,
    pub v: Linear,
}

pub struct Cross {
    pub layers: Vec<CrossLayer>,
    pub n_ctx: usize,
    pub n_state: usize,
    /// rows per layer in the cache (`GGML_PAD(n_ctx, 256)`)
    pub n_pad: usize,
    pub kscale: f32,
}

/// The cross-attention cache, as `whisper_state::kv_cross` holds it: `k` and `v` each `[n_layer][n_pad][n_state]`
/// f16, the rows past `n_ctx` of every layer +0.
#[derive(Default, Clone)]
pub struct KvCross {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}

impl KvCross {
    pub fn bytes(&self) -> usize {
        2 * (self.k.capacity() + self.v.capacity())
    }
}

/// How the model computes the cross graph: `CrossVariant::default()` is the reference's; every other setting is a
/// discriminator (a reading of the source the oracle must reject — or, where marked, cannot tell apart).
#[derive(Clone, Copy, Default, Debug)]
pub struct CrossVariant {
    /// the activations scaled by Kscale before K's product (the scale moved inside the dot)
    pub scale_first: bool,
    /// K's weights scaled by Kscale and rounded to f16 (the scale folded into the weights)
    pub scale_in_weights: bool,
    /// the scale applied in double: `(float)((double)k · pow(64, −0.25))`
    pub scale_double: bool,
    /// the scale applied after the f16 copy (widened, multiplied, rounded again)
    pub scale_after_f16: bool,
    /// V scaled too
    pub v_scaled: bool,
    /// V's bias left out
    pub v_unbiased: bool,
    /// V's bias added to K as well (before the scale)
    pub k_biased: bool,
    /// the CPYs through the row converter (`vcvtps2ph`) instead of the scalar bit trick — the two agree on every
    /// finite value (0.0.3), so on finite activations this cannot be told apart
    pub cpy_row_converter: bool,
    /// the activations not rounded to f16 before the products (0.0.9's discriminator, here on the cross products)
    pub no_f16: bool,
}

/// The model's nodes for some frames of one layer (each `[frames][n_state]`).
#[derive(Default)]
pub struct CrossNodes {
    pub k_mm: Vec<f32>,
    pub k_scale: Vec<f32>,
    pub k_cpy: Vec<u16>,
    pub v_mm: Vec<f32>,
    pub v_add: Vec<f32>,
    pub v_cpy: Vec<u16>,
}

impl Cross {
    pub fn new(m: &Model) -> Result<Cross, String> {
        let h = &m.hparams;
        let (n_ctx, n_state) = (h.n_audio_ctx as usize, h.n_text_state as usize);
        if h.n_audio_state as usize != n_state || h.n_text_head as usize * 64 != n_state {
            return Err(format!("cross: n_text_state {n_state}, n_audio_state {}, {} heads: not whisper's shapes", h.n_audio_state, h.n_text_head));
        }
        let layers = (0..h.n_text_layer as usize)
            .map(|il| {
                let p = format!("decoder.blocks.{il}.cross_attn");
                Ok(CrossLayer {
                    k: Linear::new(m, &format!("{p}.key.weight"), None)?,
                    v: Linear::new(m, &format!("{p}.value.weight"), Some(&format!("{p}.value.bias")))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Cross { layers, n_ctx, n_state, n_pad: n_kv_pad(n_ctx), kscale: kscale(n_state / h.n_text_head as usize) })
    }

    /// Elements of `k` (and of `v`) in the cache.
    pub fn cache_len(&self) -> usize {
        self.layers.len() * self.n_pad * self.n_state
    }

    /// A zeroed cache of the right size.
    pub fn new_cache(&self) -> KvCross {
        KvCross { k: vec![0; self.cache_len()], v: vec![0; self.cache_len()] }
    }

    /// The reference thread count whose `from_float` split the products reproduce (only a NaN can tell).
    pub fn set_split(&mut self, nth: usize) {
        for l in &mut self.layers {
            l.k.split = nth.max(1);
            l.v.split = nth.max(1);
        }
    }

    /// Layer `il`'s nodes by the model for frames `x` (`[frames][n_state]`, any subset of `embd_enc`'s rows) converted
    /// as `nth` threads convert them, with a variant's changes.
    pub fn model(&self, il: usize, x: &[f32], nth: usize, v: CrossVariant) -> CrossNodes {
        let l = &self.layers[il];
        let s = self.kscale;
        let mv = matmul::Variant { no_f16: v.no_f16, ..Default::default() };
        let k_mm = if v.scale_first {
            let xs: Vec<f32> = x.iter().map(|&a| a * s).collect();
            l.k.model(&xs, nth, mv, false)
        } else if v.scale_in_weights {
            let w: Vec<u16> = l.k.w.iter().map(|&h| fp32_to_fp16(fp16_to_fp32(h) * s)).collect();
            Linear::from_parts(l.k.k, l.k.n, w, None).unwrap().model(x, nth, mv, false)
        } else {
            l.k.model(x, nth, mv, false)
        };
        let k_pre: Vec<f32> = if v.k_biased { add_rows(&k_mm, l.v.b.as_deref().unwrap()) } else { k_mm.clone() };
        let k_scale: Vec<f32> = if v.scale_first || v.scale_in_weights || v.scale_after_f16 {
            k_pre.clone()
        } else if v.scale_double {
            k_pre.iter().map(|&a| (a as f64 * (64f64).powf(-0.25)) as f32).collect()
        } else {
            k_pre.iter().map(|&a| a * s).collect()
        };
        let mut k_cpy = cpy(&k_scale, v);
        if v.scale_after_f16 {
            k_cpy = k_cpy.iter().map(|&h| fp32_to_fp16(fp16_to_fp32(h) * s)).collect();
        }
        let v_mm = l.v.model(x, nth, mv, false);
        let mut v_add = if v.v_unbiased { v_mm.clone() } else { add_rows(&v_mm, l.v.b.as_deref().unwrap()) };
        if v.v_scaled {
            v_add.iter_mut().for_each(|a| *a *= s);
        }
        let v_cpy = cpy(&v_add, v);
        CrossNodes { k_mm, k_scale, k_cpy, v_mm, v_add, v_cpy }
    }

    /// The whole cache by the model (`embd_enc`: `[n_ctx][n_state]`), the padding rows +0.
    pub fn model_cache(&self, embd: &[f32], nth: usize, v: CrossVariant) -> (KvCross, Vec<CrossNodes>) {
        let mut kv = self.new_cache();
        let nodes: Vec<CrossNodes> = (0..self.layers.len()).map(|il| self.model(il, embd, nth, v)).collect();
        let n = self.n_ctx * self.n_state;
        for (il, nd) in nodes.iter().enumerate() {
            let at = il * self.n_pad * self.n_state;
            kv.k[at..at + n].copy_from_slice(&nd.k_cpy);
            kv.v[at..at + n].copy_from_slice(&nd.v_cpy);
        }
        (kv, nodes)
    }

    /// The fast path: every layer's K and V of `embd` (`[n_ctx][n_state]`) into `kv` (resized to the cache's size on
    /// the first call and reused; the padding rows set to +0). With `taps` (`[n_layer][4][n_ctx][n_state]`: k_mm,
    /// k_scale, v_mm, v_add), also the f32 nodes.
    pub fn run_into(&self, embd: &[f32], threads: usize, kv: &mut KvCross, taps: Option<&mut [f32]>) {
        let (ns, nc, np, nl) = (self.n_state, self.n_ctx, self.n_pad, self.layers.len());
        assert_eq!(embd.len(), nc * ns, "cross: embd_enc is not [n_ctx][n_state]");
        assert!(ns.is_multiple_of(32), "cross: n_state must be a multiple of 32");
        kv.k.resize(self.cache_len(), 0);
        kv.v.resize(self.cache_len(), 0);
        for il in 0..nl {
            kv.k[(il * np + nc) * ns..(il + 1) * np * ns].fill(0);
            kv.v[(il * np + nc) * ns..(il + 1) * np * ns].fill(0);
        }
        assert!(taps.as_ref().is_none_or(|t| t.len() == nl * 4 * nc * ns), "cross: taps are not [n_layer][4][n_ctx][n_state]");
        let kern = Kernel::detect();
        let s = self.kscale;
        let split = self.layers[0].k.split;
        let (sk, sv, st) = (Shared::new(Some(&mut kv.k)), Shared::new(Some(&mut kv.v)), Shared::new(taps));
        par_frames(nc, threads, &|a, b| {
            let mut px = vec![0.0f32; CROSS_PANEL * ns];
            let mut tmp = vec![0.0f32; ns];
            let mut raw = vec![0.0f32; CROSS_PANEL * ns];
            let mut add = vec![0.0f32; CROSS_PANEL * ns];
            for p in (a..b).step_by(CROSS_PANEL) {
                let r = CROSS_PANEL.min(b - p);
                let rn = r * ns;
                // one conversion of these frames for all 2 · n_layer products
                load_panel(kern, &embd[p * ns..(p + r) * ns], None, split, &mut tmp, &mut px);
                for (il, l) in self.layers.iter().enumerate() {
                    let row0 = il * np + p;
                    let tap = |node: usize| (il * 4 + node) * nc + p;
                    // SAFETY (every rows() below): frames [p, p + r) lie in this thread's range [a, b); the cache rows
                    // il·n_pad + [p, p + r) and the tap rows are this thread's alone
                    unsafe {
                        // K: the product, × Kscale (the SCALE node), the CPY
                        panel_product(kern, &px, r, &l.k, &mut raw);
                        if let Some(t) = st.rows(tap(0), tap(0) + r, ns) {
                            t.copy_from_slice(&raw[..rn]);
                        }
                        raw[..rn].iter_mut().for_each(|x| *x *= s);
                        if let Some(t) = st.rows(tap(1), tap(1) + r, ns) {
                            t.copy_from_slice(&raw[..rn]);
                        }
                        cpy_f16(&raw[..rn], sk.rows(row0, row0 + r, ns).unwrap());
                        // V: the product, + bias (the ADD node), the CPY
                        panel_product(kern, &px, r, &l.v, &mut raw);
                        if let Some(t) = st.rows(tap(2), tap(2) + r, ns) {
                            t.copy_from_slice(&raw[..rn]);
                        }
                        add_bias(&raw[..rn], l.v.b.as_deref(), &mut add[..rn]);
                        if let Some(t) = st.rows(tap(3), tap(3) + r, ns) {
                            t.copy_from_slice(&add[..rn]);
                        }
                        cpy_f16(&add[..rn], sv.rows(row0, row0 + r, ns).unwrap());
                    }
                }
            }
        });
    }

    /// The same into a new cache.
    pub fn run(&self, embd: &[f32], threads: usize) -> KvCross {
        let mut kv = KvCross::default();
        self.run_into(embd, threads, &mut kv, None);
        kv
    }
}

/// `x + b` per row of `b.len()`.
fn add_rows(x: &[f32], b: &[f32]) -> Vec<f32> {
    x.chunks_exact(b.len()).flat_map(|r| r.iter().zip(b).map(|(a, c)| a + c)).collect()
}

/// The CPY node by the model: the scalar bit trick, or with `cpy_row_converter` the row converter.
fn cpy(x: &[f32], v: CrossVariant) -> Vec<u16> {
    if v.cpy_row_converter {
        let mut y = vec![0u16; x.len()];
        fp32_to_fp16_row(x, &mut y);
        y
    } else {
        cpy_f16_model(x)
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

    #[test]
    fn kscale_is_whispers() {
        assert_eq!(kscale(64).to_bits(), 0x3EB5_04F3);
        // every other way of writing 64^-0.25 in f32 gives the same bits (so the oracle's op_params pin it, not a reading)
        assert_eq!((1.0f32 / 8.0f32.sqrt()).to_bits(), 0x3EB5_04F3);
        assert_eq!(0.125f32.sqrt().to_bits(), 0x3EB5_04F3);
        assert_eq!(64f32.powf(-0.25).to_bits(), 0x3EB5_04F3);
    }

    /// Small shapes: the fast path (padding rows included, at several thread counts, with taps) equals the model.
    #[test]
    fn fast_equals_model() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let (ns, nc, nl) = (96usize, 300usize, 2usize);
        let lin = |bias: bool, seed: &mut u64| {
            let w: Vec<u16> = (0..ns * ns).map(|_| fp32_to_fp16(((rnd(seed) % 2001) as f32 - 1000.0) / 4000.0)).collect();
            let b = bias.then(|| (0..ns).map(|_| ((rnd(seed) % 2001) as f32 - 1000.0) / 1000.0).collect());
            Linear::from_parts(ns, ns, w, b).unwrap()
        };
        let layers = (0..nl).map(|_| CrossLayer { k: lin(false, &mut seed), v: lin(true, &mut seed) }).collect();
        let c = Cross { layers, n_ctx: nc, n_state: ns, n_pad: n_kv_pad(nc), kscale: kscale(64) };
        let x: Vec<f32> = (0..nc * ns).map(|_| ((rnd(&mut seed) % 20001) as f32 - 10000.0) / 997.0).collect();
        let (want, nodes) = c.model_cache(&x, 1, CrossVariant::default());
        for threads in [1, 2, 3, 4] {
            let mut kv = KvCross { k: vec![0xFFFF; c.cache_len()], v: vec![0xFFFF; c.cache_len()] };
            let mut taps = vec![0.0f32; nl * 4 * nc * ns];
            c.run_into(&x, threads, &mut kv, Some(&mut taps));
            assert!(kv.k == want.k && kv.v == want.v, "{threads} threads: the cache differs (padding included)");
            for (il, nd) in nodes.iter().enumerate() {
                for (i, want) in [&nd.k_mm, &nd.k_scale, &nd.v_mm, &nd.v_add].into_iter().enumerate() {
                    let got = &taps[(il * 4 + i) * nc * ns..(il * 4 + i + 1) * nc * ns];
                    assert!(got.iter().zip(want.iter()).all(|(a, b)| a.to_bits() == b.to_bits()), "layer {il} node {i} differs");
                }
            }
        }
        assert!(want.k[nc * ns..c.n_pad * ns].iter().all(|&h| h == 0), "padding is +0");
    }
}
