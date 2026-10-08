// SPDX-License-Identifier: MIT OR Apache-2.0
//! The encoder's self-attention as the shipped ggml-cpu computes it (v0.1.0, testing/attention/NOTES.md):
//! `ggml_flash_attn_ext(Q, K, V, no mask, scale 1/√64)` over whisper's `kv_pad` cache, read from the pinned source and
//! the shipped `libggml-cpu.so`, then checked by the oracle on every FLASH_ATTN_EXT node of the encoder graph.
//!
//! whisper's graph sends the encoder down ggml's **tiled** path (`ggml_compute_forward_flash_attn_ext_tiled`: Q f32,
//! K and V f16, 1,500 query rows ≥ the 64-row tile, head size a multiple of 8), not the one-chunk path:
//!
//! - **Q stays f32.** K and V are the f16 copies 0.0.9's CPY nodes wrote into `kv_pad`, widened exactly; the cache
//!   has 1,536 rows (`GGML_PAD(1500, 256)`) and the last 36 are the +0 the buffer was cleared to, attended like any
//!   other key (no mask): each scores exactly +0, adds `exp(0 − M)` to the denominator and nothing to the numerator.
//! - Keys go in **tiles of 64**. A score is one f32 FMA chain over the 64 dimensions from +0 (`simd_gemm`: every
//!   element its own chain, so the register blocking and the thread split change no bit), then `· 0.125`.
//! - Per tile and row: the tile's max (a `max > x ? max : x` chain from −inf), `Mnew = fmaxf(M, max)`; if it grew,
//!   the output and the sum are rescaled by **glibc's `expf(M − Mnew)`**; the probabilities are **ggml's own
//!   `ggml_v_expf(x − Mnew)`** (ARM's optimized-routines expf in 8 lanes, `ggml_vec_soft_max_f32`), summed 8 at a time
//!   in f32 as `((y0 + y4) + (y2 + y6)) + ((y1 + y5) + (y3 + y7))` and those sums in double, then
//!   `S = (float)((double)S + sum)`.
//! - The output accumulates **in f32**: per tile `o[d] = fma(p, v[d], o[d])` over the tile's 64 keys in order (again
//!   `simd_gemm`'s chains). At the end `o · (1/S)`.
//!
//! The model ([`attention_model`]) follows that order literally; its [`Variant`]s are the readings the oracle must
//! reject, among them the one-chunk path itself (Q to f16, `ggml_vec_dot_f16` scores, V accumulated in f16), which the
//! reference also records (`cplan.use_ref`). The fast path ([`Attention::run_into`]) keeps every rounding and changes
//! only the work around it: K and V are widened (K transposed per tile) once per call instead of once per query tile,
//! the scale, max, exponentials and sums are done on the tile in registers, the rescale is folded into the output
//! product's load, and threads take (head, 64-query tile) units.

use crate::conv::vec_dot_f16;
use crate::f16::{fp16_to_fp32, fp16_to_fp32_row, fp32_to_fp16_f16c, fp32_to_fp16_row};

/// whisper's head size (`n_state / n_head` for every model) and ggml's tile sizes (`GGML_FA_TILE_Q`, `_KV`).
pub const HEAD_DIM: usize = 64;
pub const KV_TILE: usize = 64;
/// voaice's query tile (the reference's is 64): rows are independent, so the tile changes no bit; 60 = 10 register
/// blocks of 6 rows, and whisper's 1,500 frames are 25 whole tiles.
pub const Q_TILE: usize = 60;
/// `KQscale = 1.0f/sqrtf(float(n_state_head))` = 1/8, exactly.
pub const SCALE: f32 = 0.125;
/// `GGML_PAD(n, 256)`: the rows of whisper's `kv_pad` cache.
pub fn n_kv_pad(n_ctx: usize) -> usize {
    n_ctx.div_ceil(256) * 256
}

// glibc libm, the symbols libggml-cpu.so imports for the tiled kernel (`objdump -T`: expf@GLIBC_2.27, fmaxf@GLIBC_2.2.5)
extern "C" {
    fn expf(x: f32) -> f32;
    fn fmaxf(a: f32, b: f32) -> f32;
}
/// glibc's `expf` (the running-max rescale).
#[inline]
pub fn libm_expf(x: f32) -> f32 {
    // SAFETY: expf is a pure libm function
    unsafe { expf(x) }
}
#[inline]
fn libm_fmaxf(a: f32, b: f32) -> f32 {
    // SAFETY: fmaxf is a pure libm function
    unsafe { fmaxf(a, b) }
}

// ggml_v_expf's constants (vec.h, the AVX2 + FMA branch), as f32 bit patterns
const R: u32 = 0x4B40_0000; // 0x1.8p23
const LOG2E: u32 = 0x3FB8_AA3B; // 0x1.715476p+0
const LN2_HI: u32 = 0x3F31_7200; // 0x1.62e4p-1
const LN2_LO: u32 = 0x35BF_BE8E; // 0x1.7f7d1cp-20
const P0: u32 = 0x3C07_2010; // 0x1.0e4020p-7
const P1: u32 = 0x3D2B_9F17; // 0x1.573e2ep-5
const P2: u32 = 0x3E2A_AF33; // 0x1.555e66p-3
const P3: u32 = 0x3EFF_FEDB; // 0x1.fffdb6p-2
const P4: u32 = 0x3F7F_FFF6; // 0x1.ffffecp-1

/// One lane of `ggml_v_expf` (vec.h, AVX2 + FMA), operation for operation. The vector form computes the fast result
/// `fma(j, k, k)` for all eight lanes unless one lane has |n| > 126, then the scaled form for all eight, which gives
/// the same `k·j + k` on the other lanes: so every lane's value depends only on its own input.
#[inline]
pub fn v_expf(x: f32) -> f32 {
    let f = f32::from_bits;
    let r = f(R);
    let z = x.mul_add(f(LOG2E), r);
    let n = z - r;
    let b = (-n).mul_add(f(LN2_LO), (-n).mul_add(f(LN2_HI), x));
    let e = z.to_bits() << 23;
    let k = f(e.wrapping_add(1.0f32.to_bits()));
    let u = b * b;
    let j = f(P0).mul_add(b, f(P1)).mul_add(u, f(P2).mul_add(b, f(P3))).mul_add(u, f(P4) * b);
    if n.abs() <= 126.0 || n.is_nan() {
        return j.mul_add(k, k);
    }
    let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = f(g.wrapping_add(0x7F00_0000));
    let s2 = f(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// `ggml_vec_soft_max_f32(n, y, x, max)` as the AVX2 + FMA build computes it: `y = exp(x − max)` eight at a time,
/// each eight summed in f32 as `(hi4 + lo4)`, `(h0 + h2, h1 + h3)`, `(h0 + h2) + (h1 + h3)`, those sums added in
/// double in order; the `n % 8` tail through libm `expf`, each added in double. `exp` replaces the 8-lane exponential
/// (a discriminator); `sum_f32` adds the eights in f32 (another).
pub fn soft_max_model(x: &mut [f32], max: f32, exp: fn(f32) -> f32, sum_f32: bool) -> f64 {
    let n8 = x.len() / 8 * 8;
    let (mut sum, mut sum32) = (0.0f64, 0.0f32);
    for c in x[..n8].as_chunks_mut::<8>().0 {
        for v in c.iter_mut() {
            *v = exp(*v - max);
        }
        let h = [c[4] + c[0], c[5] + c[1], c[6] + c[2], c[7] + c[3]];
        let s = (h[0] + h[2]) + (h[1] + h[3]);
        sum += s as f64;
        sum32 += s;
    }
    for v in x[n8..].iter_mut() {
        *v = libm_expf(*v - max);
        sum += *v as f64;
        sum32 += *v;
    }
    if sum_f32 {
        sum32 as f64
    } else {
        sum
    }
}

/// How the model computes attention: the reference's way is `Variant::default()`; every other setting is a
/// discriminator (a reading of the source the oracle must reject — or, for `scale_on_q`, one it cannot tell apart).
#[derive(Clone, Copy, Default, Debug)]
pub struct Variant {
    /// the probabilities through glibc `expf` instead of `ggml_v_expf`
    pub libm_probs: bool,
    /// the running-max rescale through `ggml_v_expf` instead of glibc `expf`
    pub v_expf_rescale: bool,
    /// no running max: the row's global max first, then one pass with no rescale
    pub no_running_max: bool,
    /// only the n_ctx written keys: kv_pad's 36 zero rows left out
    pub pad_excluded: bool,
    /// Q · scale before the dot instead of the score after it (×1/8 is exact: expected indistinguishable)
    pub scale_on_q: bool,
    /// the scores as a multiply then an add (no FMA)
    pub scores_unfused: bool,
    /// the output's accumulation as a multiply then an add (no FMA)
    pub output_unfused: bool,
    /// the softmax's eight-sums added in f32 (no double)
    pub sum_f32: bool,
    /// out / S instead of out · (1/S)
    pub divide: bool,
    /// the one-chunk path (`use_ref`): Q to f16, `ggml_vec_dot_f16` scores, libm `expf` per key, V accumulated in f16
    pub one_chunk: bool,
    /// in the one-chunk model, `S = fma(S, ms, vs)` (the build does not contract it: GCC splits the update by branch,
    /// `vmulss` then `vaddss`, and the reference's own use_ref output agrees)
    pub one_chunk_s_fused: bool,
}

/// One query row of one head through the tiled kernel's order. `k`, `v`: the head's keys and values widened to f32,
/// [n_kv][64] (kv_pad's zero rows included); `n_keys` ≤ n_kv keys are attended (all of them in the reference).
pub fn row_model(q: &[f32], k: &[f32], v: &[f32], n_keys: usize, var: Variant) -> [f32; HEAD_DIM] {
    let d = HEAD_DIM;
    let n_kv = k.len() / d;
    let qs: Vec<f32> = q.iter().map(|&x| if var.scale_on_q { x * SCALE } else { x }).collect();
    let score = |j: usize| -> f32 {
        if j >= n_keys {
            return f32::NEG_INFINITY; // a partial last tile: ggml sets the padded scores to −inf
        }
        let mut acc = 0.0f32;
        for (&a, &b) in qs.iter().zip(&k[j * d..(j + 1) * d]) {
            acc = if var.scores_unfused { acc + a * b } else { a.mul_add(b, acc) };
        }
        if var.scale_on_q { acc } else { acc * SCALE }
    };
    let prob: fn(f32) -> f32 = if var.libm_probs { libm_expf } else { v_expf };
    let rescale: fn(f32) -> f32 = if var.v_expf_rescale { v_expf } else { libm_expf };
    let global = if var.no_running_max {
        (0..n_kv).map(score).fold(f32::NEG_INFINITY, |m, x| if m > x { m } else { x })
    } else {
        f32::NEG_INFINITY
    };
    let (mut s, mut m, mut o) = (0.0f32, f32::NEG_INFINITY, [0.0f32; HEAD_DIM]);
    for t in (0..n_kv).step_by(KV_TILE) {
        let tile = KV_TILE.min(n_kv - t);
        let mut kq = [0.0f32; KV_TILE];
        for (tk, x) in kq.iter_mut().enumerate().take(tile) {
            *x = score(t + tk);
        }
        let tile_max = kq[..tile].iter().fold(f32::NEG_INFINITY, |mx, &x| if mx > x { mx } else { x });
        if tile_max == f32::NEG_INFINITY {
            continue; // every probability 0: the reference zeroes the row and the product adds ±0 (not reached here)
        }
        if var.no_running_max {
            m = global;
        } else {
            let mnew = libm_fmaxf(m, tile_max);
            if mnew > m {
                let ms = rescale(m - mnew);
                for x in o.iter_mut() {
                    *x *= ms;
                }
                s *= ms;
            }
            m = mnew;
        }
        let sum = soft_max_model(&mut kq[..tile], m, prob, var.sum_f32);
        s = if var.sum_f32 { s + sum as f32 } else { (s as f64 + sum) as f32 };
        for (od, x) in o.iter_mut().enumerate() {
            for (tk, &p) in kq[..tile].iter().enumerate() {
                let vv = v[(t + tk) * d + od];
                *x = if var.output_unfused { *x + p * vv } else { p.mul_add(vv, *x) };
            }
        }
    }
    let s_inv = if s == 0.0 { 0.0 } else { 1.0 / s };
    for x in o.iter_mut() {
        *x = if var.divide { *x / s } else { *x * s_inv };
    }
    o
}

/// One query row of one head through the one-chunk path (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`, what
/// `use_ref` runs): Q converted by K's `from_float` (the row converter), each score `ggml_vec_dot_f16(64, k, q16) ·
/// scale`, then per key: a new max rescales the f16 accumulator (`ggml_vec_scale_f16`: widen, ·ms, vcvtps2ph) with
/// libm `expf(Mold − M)`, otherwise `vs = expf(s − M)`; `ggml_vec_mad_f16` (widen, `fma(v, vs, acc)`, vcvtps2ph);
/// `S = S·ms + vs` (two roundings). `k16`, `v16`: the head's [n_kv][64] f16 bits.
pub fn row_model_one_chunk(q: &[f32], k16: &[u16], v16: &[u16], var: Variant) -> [f32; HEAD_DIM] {
    let d = HEAD_DIM;
    let mut q16 = [0u16; HEAD_DIM];
    fp32_to_fp16_row(q, &mut q16);
    let (mut s, mut m, mut acc) = (0.0f32, f32::NEG_INFINITY, [0u16; HEAD_DIM]);
    for j in 0..k16.len() / d {
        let sc = vec_dot_f16(&k16[j * d..(j + 1) * d], &q16) * SCALE;
        let (mold, mut ms, mut vs) = (m, 1.0f32, 1.0f32);
        if sc > m {
            m = sc;
            ms = libm_expf(mold - m);
            for a in acc.iter_mut() {
                *a = fp32_to_fp16_f16c(fp16_to_fp32(*a) * ms);
            }
        } else {
            vs = libm_expf(sc - m);
        }
        for (a, &vv) in acc.iter_mut().zip(&v16[j * d..(j + 1) * d]) {
            *a = fp32_to_fp16_f16c(fp16_to_fp32(vv).mul_add(vs, fp16_to_fp32(*a)));
        }
        s = if var.one_chunk_s_fused { s.mul_add(ms, vs) } else { s * ms + vs };
    }
    let s_inv = if s == 0.0 { 0.0 } else { 1.0 / s };
    let mut o = [0.0f32; HEAD_DIM];
    for (x, &a) in o.iter_mut().zip(&acc) {
        *x = fp16_to_fp32(a) * s_inv;
    }
    o
}

/// The head-major f32 copies of K and V the model reads: [n_head][n_kv][64], rows past k16's are kv_pad's +0.
fn widen_heads(x16: &[u16], n_state: usize, n_kv: usize) -> Vec<f32> {
    let (nh, rows) = (n_state / HEAD_DIM, x16.len() / n_state);
    let mut out = vec![0.0f32; nh * n_kv * HEAD_DIM];
    for h in 0..nh {
        for j in 0..rows.min(n_kv) {
            for dd in 0..HEAD_DIM {
                out[(h * n_kv + j) * HEAD_DIM + dd] = fp16_to_fp32(x16[j * n_state + h * HEAD_DIM + dd]);
            }
        }
    }
    out
}

/// Flash attention as the reference computes whisper's encoder node, by the model (one thread, slow, literal).
/// `q`: [n_ctx][n_state] f32 (Q + bias, frame-major); `k16`, `v16`: [rows][n_state] f16 bits (the CPY nodes, rows ≤
/// n_kv); `n_kv`: the cache's rows (1,536), those past `rows` being +0. Returns [n_ctx][n_state] (the node after its
/// permute: frame-major, head h at columns 64h..64h + 63).
pub fn attention_model(q: &[f32], k16: &[u16], v16: &[u16], n_state: usize, n_kv: usize, var: Variant) -> Vec<f32> {
    let frames: Vec<usize> = (0..q.len() / n_state.max(1)).collect();
    attention_model_frames(q, k16, v16, n_state, n_kv, var, &frames)
}

/// [`attention_model`] for the query frames `frames` only (every head): [frames.len()][n_state].
pub fn attention_model_frames(q: &[f32], k16: &[u16], v16: &[u16], n_state: usize, n_kv: usize, var: Variant, frames: &[usize]) -> Vec<f32> {
    assert!(n_state.is_multiple_of(HEAD_DIM) && q.len().is_multiple_of(n_state) && k16.len() == v16.len());
    let (nh, n_ctx, rows) = (n_state / HEAD_DIM, q.len() / n_state, k16.len() / n_state);
    assert!(rows <= n_kv && frames.iter().all(|&f| f < n_ctx));
    let mut out = vec![0.0f32; frames.len() * n_state];
    if var.one_chunk {
        let mut kh = vec![0u16; n_kv * HEAD_DIM];
        let mut vh = vec![0u16; n_kv * HEAD_DIM];
        for h in 0..nh {
            for j in 0..rows {
                let src = j * n_state + h * HEAD_DIM;
                kh[j * HEAD_DIM..(j + 1) * HEAD_DIM].copy_from_slice(&k16[src..src + HEAD_DIM]);
                vh[j * HEAD_DIM..(j + 1) * HEAD_DIM].copy_from_slice(&v16[src..src + HEAD_DIM]);
            }
            for (o, &i) in frames.iter().enumerate() {
                let r = &q[i * n_state + h * HEAD_DIM..i * n_state + (h + 1) * HEAD_DIM];
                out[o * n_state + h * HEAD_DIM..o * n_state + (h + 1) * HEAD_DIM].copy_from_slice(&row_model_one_chunk(r, &kh, &vh, var));
            }
        }
        return out;
    }
    let (kw, vw) = (widen_heads(k16, n_state, n_kv), widen_heads(v16, n_state, n_kv));
    let n_keys = if var.pad_excluded { rows } else { n_kv };
    for h in 0..nh {
        let (k, v) = (&kw[h * n_kv * HEAD_DIM..(h + 1) * n_kv * HEAD_DIM], &vw[h * n_kv * HEAD_DIM..(h + 1) * n_kv * HEAD_DIM]);
        for (o, &i) in frames.iter().enumerate() {
            let r = &q[i * n_state + h * HEAD_DIM..i * n_state + (h + 1) * HEAD_DIM];
            out[o * n_state + h * HEAD_DIM..o * n_state + (h + 1) * HEAD_DIM].copy_from_slice(&row_model(r, k, v, n_keys, var));
        }
    }
    out
}

/// The fast path: whisper's encoder attention for `n_state` (a multiple of 64) over a cache of `n_kv` rows (a
/// multiple of 64, ≥ the frames).
pub struct Attention {
    pub n_state: usize,
    pub n_head: usize,
    pub n_kv: usize,
}

struct Shared(*mut f32, usize);
// SAFETY: threads write disjoint parts (checked by construction at each use), the scope joins before any read
unsafe impl Sync for Shared {}
impl Shared {
    /// # Safety
    /// [a, a + n) is written by this thread alone while the scope runs.
    #[allow(clippy::mut_from_ref)]
    unsafe fn part(&self, a: usize, n: usize) -> &mut [f32] {
        assert!(a + n <= self.1);
        // SAFETY: inside the buffer (asserted); exclusive by the caller's contract
        unsafe { std::slice::from_raw_parts_mut(self.0.add(a), n) }
    }
}

impl Attention {
    pub fn new(n_state: usize, n_kv: usize) -> Result<Attention, String> {
        if n_state == 0 || !n_state.is_multiple_of(HEAD_DIM) || n_kv == 0 || !n_kv.is_multiple_of(KV_TILE) {
            return Err(format!("attention: n_state {n_state} or n_kv {n_kv} is not a multiple of 64"));
        }
        Ok(Attention { n_state, n_head: n_state / HEAD_DIM, n_kv })
    }

    /// The scratch [`Attention::run_into`] needs: one head's K, transposed per tile, and V, widened (f32).
    pub fn scratch_len(&self) -> usize {
        2 * self.n_kv * HEAD_DIM
    }

    /// `q` [n_ctx][n_state] f32, `k16`, `v16` [rows][n_state] f16 bits (rows ≤ n_kv; the rest of the cache is +0) →
    /// `out` [n_ctx][n_state]. `scratch` is resized to [`Attention::scratch_len`] on first use, then kept. Head by
    /// head, every thread first widens its share of the head's key tiles, then (after a barrier) computes its share of
    /// the query tiles against all of them; one scope per call.
    pub fn run_into(&self, q: &[f32], k16: &[u16], v16: &[u16], threads: usize, scratch: &mut Vec<f32>, out: &mut [f32]) {
        let (ns, nh, n_kv) = (self.n_state, self.n_head, self.n_kv);
        assert!(q.len().is_multiple_of(ns) && out.len() == q.len() && k16.len() == v16.len() && k16.len().is_multiple_of(ns));
        let (n_ctx, rows) = (q.len() / ns, k16.len() / ns);
        assert!(rows <= n_kv, "attention: more keys than the cache holds");
        if !kernel_available() {
            out.copy_from_slice(&attention_model(q, k16, v16, ns, n_kv, Variant::default()));
            return;
        }
        scratch.resize(self.scratch_len(), 0.0);
        let (nt, nq) = (n_kv / KV_TILE, n_ctx.div_ceil(Q_TILE));
        let (blk, half) = (KV_TILE * HEAD_DIM, n_kv * HEAD_DIM);
        let threads = threads.clamp(1, nq.max(1));
        let bar = std::sync::Barrier::new(threads);
        let (sh, so) = (Shared(scratch.as_mut_ptr(), scratch.len()), Shared(out.as_mut_ptr(), out.len()));
        let job = |ti: usize| {
            let mut kq = vec![0.0f32; Q_TILE * KV_TILE];
            let mut acc = vec![0.0f32; Q_TILE * HEAD_DIM];
            let mut row = [0.0f32; HEAD_DIM];
            for h in 0..nh {
                // 1. this thread's key tiles of head h: K transposed per tile (kt[tile][d][key]), V (vf[key][d])
                for t in nt * ti / threads..nt * (ti + 1) / threads {
                    // SAFETY: tile t of both halves belongs to this thread until the barrier
                    let (kt, vf) = unsafe { (sh.part(t * blk, blk), sh.part(half + t * blk, blk)) };
                    for tk in 0..KV_TILE {
                        let j = t * KV_TILE + tk;
                        if j < rows {
                            let src = j * ns + h * HEAD_DIM;
                            fp16_to_fp32_row(&k16[src..src + HEAD_DIM], &mut row);
                            for (dd, &x) in row.iter().enumerate() {
                                kt[dd * KV_TILE + tk] = x;
                            }
                            fp16_to_fp32_row(&v16[src..src + HEAD_DIM], &mut vf[tk * HEAD_DIM..(tk + 1) * HEAD_DIM]);
                        } else {
                            for dd in 0..HEAD_DIM {
                                kt[dd * KV_TILE + tk] = 0.0;
                            }
                            vf[tk * HEAD_DIM..(tk + 1) * HEAD_DIM].fill(0.0);
                        }
                    }
                }
                bar.wait();
                // 2. this thread's query tiles of head h against every key tile
                {
                    // SAFETY: nothing writes the scratch between the two barriers
                    let scr = unsafe { std::slice::from_raw_parts(sh.0, sh.1) };
                    let (kt, vf) = (&scr[..half], &scr[half..]);
                    for qt in nq * ti / threads..nq * (ti + 1) / threads {
                        let t0 = qt * Q_TILE;
                        let r = Q_TILE.min(n_ctx - t0);
                        // SAFETY: the CPU has AVX2, FMA and F16C (kernel_available); the slices are the sizes it reads
                        unsafe { x86::tile(&q[t0 * ns + h * HEAD_DIM..], ns, r, kt, vf, nt, &mut kq, &mut acc) };
                        for i in 0..r {
                            // SAFETY: row t0 + i, columns of head h: this thread's query tile alone
                            let o = unsafe { so.part((t0 + i) * ns + h * HEAD_DIM, HEAD_DIM) };
                            o.copy_from_slice(&acc[i * HEAD_DIM..(i + 1) * HEAD_DIM]);
                        }
                    }
                }
                bar.wait();
            }
        };
        let job = &job;
        std::thread::scope(|s| {
            for ti in 0..threads - 1 {
                s.spawn(move || job(ti));
            }
            job(threads - 1);
        });
    }

    /// The same into a new vector.
    pub fn run(&self, q: &[f32], k16: &[u16], v16: &[u16], threads: usize) -> Vec<f32> {
        let mut scratch = Vec::new();
        let mut out = vec![0.0f32; q.len()];
        self.run_into(q, k16, v16, threads, &mut scratch, &mut out);
        out
    }
}

fn kernel_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod x86 {
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn tile(_: &[f32], _: usize, _: usize, _: &[f32], _: &[f32], _: usize, _: &mut [f32], _: &mut [f32]) {
        unreachable!("kernel_available() is false off x86-64")
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// `ggml_v_expf` on eight lanes, instruction for instruction (vec.h, AVX2 + FMA).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn v_expf8(x: __m256) -> __m256 {
        let r = _mm256_set1_ps(f32::from_bits(R));
        let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(LOG2E)), r);
        let n = _mm256_sub_ps(z, r);
        let b = _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(LN2_LO)), _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(LN2_HI)), x));
        let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
        let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0))));
        let c = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_andnot_ps(_mm256_set1_ps(-0.0), n), _mm256_set1_ps(126.0)));
        let u = _mm256_mul_ps(b, b);
        let j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(f32::from_bits(P0)), b, _mm256_set1_ps(f32::from_bits(P1))), u, _mm256_fmadd_ps(_mm256_set1_ps(f32::from_bits(P2)), b, _mm256_set1_ps(f32::from_bits(P3)))),
            u,
            _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(P4)), b),
        );
        if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
            return _mm256_fmadd_ps(j, k, k);
        }
        let g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())), _mm256_set1_epi32(0x8200_0000u32 as i32));
        let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7F00_0000)));
        let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        let d = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_andnot_ps(_mm256_set1_ps(-0.0), n), _mm256_set1_ps(192.0)));
        _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(
                _mm256_castsi256_ps(d),
                _mm256_or_ps(
                    _mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                    _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
                ),
            ),
        )
    }

    /// The `RM` query rows' scores for one key tile: `kq[i][key] = (fma chain over d from +0) · scale`, two blocks of
    /// eight keys per pass (simd_gemm's 6 × 16 shape), `q` rows `stride` apart, `kt` the tile [d][key].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scores<const RM: usize>(q: *const f32, stride: usize, kt: *const f32, kq: *mut f32) {
        // SAFETY (all pointer reads/writes): the caller passes RM rows of 64 q values, a [64][64] tile, RM × 64 kq
        unsafe {
            let sc = _mm256_set1_ps(SCALE);
            for jj in (0..KV_TILE).step_by(16) {
                let mut acc = [[_mm256_setzero_ps(); 2]; RM];
                for dd in 0..HEAD_DIM {
                    let b0 = _mm256_loadu_ps(kt.add(dd * KV_TILE + jj));
                    let b1 = _mm256_loadu_ps(kt.add(dd * KV_TILE + jj + 8));
                    for (i, a) in acc.iter_mut().enumerate() {
                        let p = _mm256_broadcast_ss(&*q.add(i * stride + dd));
                        a[0] = _mm256_fmadd_ps(b0, p, a[0]);
                        a[1] = _mm256_fmadd_ps(b1, p, a[1]);
                    }
                }
                for (i, a) in acc.iter().enumerate() {
                    _mm256_storeu_ps(kq.add(i * KV_TILE + jj), _mm256_mul_ps(a[0], sc));
                    _mm256_storeu_ps(kq.add(i * KV_TILE + jj + 8), _mm256_mul_ps(a[1], sc));
                }
            }
        }
    }

    /// The `RM` rows' outputs for one key tile: `o[i][d]` (first multiplied by `ms[i]` when the max grew) then
    /// `fma(p[i][key], v[key][d], o)` over the tile's keys in order.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pv<const RM: usize>(p: *const f32, vf: *const f32, o: *mut f32, ms: &[f32], grew: &[bool]) {
        // SAFETY (all pointer reads/writes): RM rows of 64 probabilities, a [64][64] V tile, RM × 64 outputs
        unsafe {
            for jj in (0..HEAD_DIM).step_by(16) {
                let mut acc = [[_mm256_setzero_ps(); 2]; RM];
                for (i, a) in acc.iter_mut().enumerate() {
                    a[0] = _mm256_loadu_ps(o.add(i * HEAD_DIM + jj));
                    a[1] = _mm256_loadu_ps(o.add(i * HEAD_DIM + jj + 8));
                    if grew[i] {
                        let m = _mm256_set1_ps(ms[i]);
                        a[0] = _mm256_mul_ps(a[0], m);
                        a[1] = _mm256_mul_ps(a[1], m);
                    }
                }
                for tk in 0..KV_TILE {
                    let b0 = _mm256_loadu_ps(vf.add(tk * HEAD_DIM + jj));
                    let b1 = _mm256_loadu_ps(vf.add(tk * HEAD_DIM + jj + 8));
                    for (i, a) in acc.iter_mut().enumerate() {
                        let x = _mm256_broadcast_ss(&*p.add(i * KV_TILE + tk));
                        a[0] = _mm256_fmadd_ps(b0, x, a[0]);
                        a[1] = _mm256_fmadd_ps(b1, x, a[1]);
                    }
                }
                for (i, a) in acc.iter().enumerate() {
                    _mm256_storeu_ps(o.add(i * HEAD_DIM + jj), a[0]);
                    _mm256_storeu_ps(o.add(i * HEAD_DIM + jj + 8), a[1]);
                }
            }
        }
    }

    /// One row's tile max (eight vectors, a max tree): for finite scores any order gives the reference's value — only
    /// ±0 or NaN could tell, and the sign of a zero max changes no `exp(x − M)` and no comparison the kernel makes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn row_max(x: *const f32) -> f32 {
        // SAFETY: x points at the row's 64 scores
        unsafe {
            let v = |b: usize| _mm256_loadu_ps(x.add(8 * b));
            let m4 = _mm256_max_ps(_mm256_max_ps(_mm256_max_ps(v(0), v(1)), _mm256_max_ps(v(2), v(3))), _mm256_max_ps(_mm256_max_ps(v(4), v(5)), _mm256_max_ps(v(6), v(7))));
            let m2 = _mm_max_ps(_mm256_castps256_ps128(m4), _mm256_extractf128_ps::<1>(m4));
            let m1 = _mm_max_ps(m2, _mm_movehl_ps(m2, m2));
            _mm_cvtss_f32(_mm_max_ss(m1, _mm_movehdup_ps(m1)))
        }
    }

    /// One row's probabilities for a key tile: `ggml_v_expf(x − m)` in place, the eight-sums in the reference's pairing
    /// added in double; returns that double sum (`ggml_vec_soft_max_f32`'s return value).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn row_exp(x: *mut f32, m: f32) -> f64 {
        // SAFETY: x points at the row's 64 scores
        unsafe {
            let mv = _mm256_set1_ps(m);
            let mut sum = 0.0f64;
            for b in 0..8 {
                let y = v_expf8(_mm256_sub_ps(_mm256_loadu_ps(x.add(8 * b)), mv));
                _mm256_storeu_ps(x.add(8 * b), y);
                let h = _mm_add_ps(_mm256_extractf128_ps::<1>(y), _mm256_castps256_ps128(y));
                let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
                let h = _mm_add_ss(h, _mm_movehdup_ps(h));
                sum += _mm_cvtss_f32(h) as f64;
            }
            sum
        }
    }

    /// Eight rows' probabilities for a key tile ([`row_exp`] for each), the horizontal sums and the double additions
    /// done for the eight rows at once: lane r of each double accumulator is row r's `ggml_vec_soft_max_f32` sum, its
    /// eight-sums added in the same order. `s[r] = (float)((double)s[r] + sum)`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rows8_exp(x: *mut f32, m: &[f32], s: &mut [f32]) {
        // SAFETY: x points at eight rows of 64 scores (KV_TILE apart); m and s hold eight values each
        unsafe {
            assert!(m.len() >= 8 && s.len() >= 8);
            // rows in lane order a c e g | b d f h = 0 2 4 6 | 1 3 5 7
            let (mut lo, mut hi) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            for b in 0..8 {
                let mut y = [_mm256_setzero_ps(); 8];
                for (r, yr) in y.iter_mut().enumerate() {
                    let p = x.add(r * KV_TILE + 8 * b);
                    *yr = v_expf8(_mm256_sub_ps(_mm256_loadu_ps(p), _mm256_set1_ps(m[r])));
                    _mm256_storeu_ps(p, *yr);
                }
                // [h of row r | h of row r + 1], h = hi4 + lo4; then (h0 + h2, h1 + h3) in lanes 0, 1 of each half
                let mut u = [_mm256_setzero_ps(); 4];
                for (k, uk) in u.iter_mut().enumerate() {
                    let (ya, yb) = (y[2 * k], y[2 * k + 1]);
                    let h = _mm256_add_ps(_mm256_permute2f128_ps::<0x31>(ya, yb), _mm256_permute2f128_ps::<0x20>(ya, yb));
                    *uk = _mm256_add_ps(h, _mm256_permute_ps::<0x4E>(h));
                }
                // (h0 + h2) + (h1 + h3): rows 0 2 4 6 | 1 3 5 7
                let t = _mm256_hadd_ps(_mm256_shuffle_ps::<0x44>(u[0], u[1]), _mm256_shuffle_ps::<0x44>(u[2], u[3]));
                lo = _mm256_add_pd(lo, _mm256_cvtps_pd(_mm256_castps256_ps128(t)));
                hi = _mm256_add_pd(hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(t)));
            }
            let mut d = [0.0f64; 8];
            _mm256_storeu_pd(d.as_mut_ptr(), lo);
            _mm256_storeu_pd(d.as_mut_ptr().add(4), hi);
            for (k, &r) in [0usize, 2, 4, 6, 1, 3, 5, 7].iter().enumerate() {
                s[r] = (s[r] as f64 + d[k]) as f32;
            }
        }
    }

    /// [`v_expf8`] on eight values (the unit test compares it with the scalar model).
    #[cfg(test)]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn v_expf8_slice(x: &[f32; 8]) -> [f32; 8] {
        let mut y = [0.0f32; 8];
        // SAFETY: eight values read, eight written
        unsafe { _mm256_storeu_ps(y.as_mut_ptr(), v_expf8(_mm256_loadu_ps(x.as_ptr()))) };
        y
    }

    /// One (head, query tile) unit: `r` ≤ 64 query rows from `q` (`stride` apart), every key tile of the head's
    /// `kt` ([nt][64 d][64 keys]) and `vf` ([nt·64 keys][64 d]), into `acc` ([r][64], the normalized outputs).
    #[allow(clippy::too_many_arguments)] // the query rows, the head's K and V, the two per-thread tiles
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn tile(q: &[f32], stride: usize, r: usize, kt: &[f32], vf: &[f32], nt: usize, kq: &mut [f32], acc: &mut [f32]) {
        assert!((1..=Q_TILE).contains(&r) && q.len() >= (r - 1) * stride + HEAD_DIM);
        assert!(kt.len() >= nt * KV_TILE * HEAD_DIM && vf.len() >= nt * KV_TILE * HEAD_DIM);
        assert!(kq.len() >= Q_TILE * KV_TILE && acc.len() >= Q_TILE * HEAD_DIM);
        let (mut m, mut s, mut ms, mut grew) = ([f32::NEG_INFINITY; Q_TILE], [0.0f32; Q_TILE], [0.0f32; Q_TILE], [false; Q_TILE]);
        acc[..r * HEAD_DIM].fill(0.0);
        // SAFETY: every pointer below stays inside the slices asserted above
        unsafe {
            for ti in 0..nt {
                let ktp = kt.as_ptr().add(ti * KV_TILE * HEAD_DIM);
                let vfp = vf.as_ptr().add(ti * KV_TILE * HEAD_DIM);
                let mut i0 = 0;
                while i0 < r {
                    let qp = q.as_ptr().add(i0 * stride);
                    let kp = kq.as_mut_ptr().add(i0 * KV_TILE);
                    match r - i0 {
                        1 => scores::<1>(qp, stride, ktp, kp),
                        2 => scores::<2>(qp, stride, ktp, kp),
                        3 => scores::<3>(qp, stride, ktp, kp),
                        4 => scores::<4>(qp, stride, ktp, kp),
                        5 => scores::<5>(qp, stride, ktp, kp),
                        _ => scores::<6>(qp, stride, ktp, kp),
                    }
                    i0 += 6;
                }
                // the softmax in two passes, so no vector is live across glibc's expf: the max and the rescale, ...
                for i in 0..r {
                    let tile_max = row_max(kq.as_ptr().add(i * KV_TILE));
                    let mnew = if tile_max > m[i] { tile_max } else { m[i] };
                    grew[i] = mnew > m[i];
                    if grew[i] {
                        ms[i] = if m[i] == f32::NEG_INFINITY { 0.0 } else { libm_expf(m[i] - mnew) };
                        s[i] *= ms[i];
                    }
                    m[i] = mnew;
                }
                // ... then the probabilities and `S = (float)((double)S + sum)`
                let r8 = r / 8 * 8;
                for i in (0..r8).step_by(8) {
                    rows8_exp(kq.as_mut_ptr().add(i * KV_TILE), &m[i..], &mut s[i..]);
                }
                for i in r8..r {
                    s[i] = (s[i] as f64 + row_exp(kq.as_mut_ptr().add(i * KV_TILE), m[i])) as f32;
                }
                let mut i0 = 0;
                while i0 < r {
                    let pp = kq.as_ptr().add(i0 * KV_TILE);
                    let op = acc.as_mut_ptr().add(i0 * HEAD_DIM);
                    let (msr, gr) = (&ms[i0..], &grew[i0..]);
                    match r - i0 {
                        1 => pv::<1>(pp, vfp, op, msr, gr),
                        2 => pv::<2>(pp, vfp, op, msr, gr),
                        3 => pv::<3>(pp, vfp, op, msr, gr),
                        4 => pv::<4>(pp, vfp, op, msr, gr),
                        5 => pv::<5>(pp, vfp, op, msr, gr),
                        _ => pv::<6>(pp, vfp, op, msr, gr),
                    }
                    i0 += 6;
                }
            }
            for (i, &si) in s.iter().enumerate().take(r) {
                let s_inv = _mm256_set1_ps(if si == 0.0 { 0.0 } else { 1.0 / si });
                for jj in (0..HEAD_DIM).step_by(8) {
                    let p = acc.as_mut_ptr().add(i * HEAD_DIM + jj);
                    _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), s_inv));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn v_expf_matches_the_eight_lane_kernel() {
        if !kernel_available() {
            return;
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut xs: Vec<f32> = vec![0.0, -0.0, -1.0, -87.0, -88.5, -103.0, -104.0, -126.0, -127.0, -150.0, -200.0, 88.0, 89.0, f32::NEG_INFINITY, f32::NAN];
        for _ in 0..200_000 {
            let r = rng(&mut seed);
            xs.push(match r % 3 {
                0 => -((r >> 8) as u32 as f32 / u32::MAX as f32) * 40.0,
                1 => f32::from_bits((r >> 16) as u32),
                _ => -((r >> 8) as u32 as f32 / u32::MAX as f32) * 300.0,
            });
        }
        while !xs.len().is_multiple_of(8) {
            xs.push(-1.5);
        }
        for c in xs.as_chunks::<8>().0 {
            // SAFETY: AVX2 + FMA present (kernel_available)
            let y = unsafe { x86::v_expf8_slice(c) };
            for (a, b) in c.iter().zip(&y) {
                let m = v_expf(*a);
                assert!(m.to_bits() == b.to_bits() || (m.is_nan() && b.is_nan()), "v_expf({a:e}) model {m:e} kernel {b:e}");
            }
        }
    }

    #[test]
    fn fast_equals_model_on_random_shapes() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        for &(ns, n_ctx, n_kv) in &[(64usize, 5usize, 64usize), (128, 70, 128), (384, 130, 256), (128, 64, 64)] {
            let mut q = vec![0.0f32; ns * n_ctx];
            for x in q.iter_mut() {
                *x = ((rng(&mut seed) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0;
            }
            let mk = |seed: &mut u64| -> Vec<u16> {
                (0..ns * n_ctx).map(|_| fp32_to_fp16_f16c(((rng(seed) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)).collect()
            };
            let (k16, v16) = (mk(&mut seed), mk(&mut seed));
            let want = attention_model(&q, &k16, &v16, ns, n_kv, Variant::default());
            let a = Attention::new(ns, n_kv).unwrap();
            for th in [1, 3] {
                let got = a.run(&q, &k16, &v16, th);
                let bad = got.iter().zip(&want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                assert_eq!(bad, 0, "ns {ns} n_ctx {n_ctx} n_kv {n_kv} threads {th}");
            }
        }
    }

    #[test]
    fn soft_max_pairing() {
        let mut x = [0.5f32, -1.0, -2.0, -3.0, -0.25, -7.0, -1.5, -0.125];
        let s = soft_max_model(&mut x, 0.5, v_expf, false);
        let h = [x[4] + x[0], x[5] + x[1], x[6] + x[2], x[7] + x[3]];
        assert_eq!(s, ((h[0] + h[2]) + (h[1] + h[3])) as f64);
        assert_eq!(x[0], 1.0);
    }
}
