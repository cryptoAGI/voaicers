// SPDX-License-Identifier: MIT OR Apache-2.0
//! The encoder's layer norms as the shipped ggml-cpu computes them (0.0.8, testing/norm/NOTES.md): whisper's
//! `ggml_add(ggml_mul(ggml_norm(x, 1e-5), w), b)`, nine times in tiny.en's encoder (each block's `attn_ln` and
//! `mlp_ln`, then `ln_post`). Read from the pinned source and the shipped `libggml-cpu.so` (`objdump -d`), then
//! checked by the oracle on the encoder graph's own NORM, MUL and ADD nodes:
//!
//! - **`ggml_compute_forward_norm_f32`**, one row (frame) at a time, each row whole in one thread (rows `ith, ith +
//!   nth, …`): `ggml_vec_sum_f32` sums the row in **double**, element by element in index order (`vcvtss2sd` +
//!   `vaddsd`: not vectorised, there is no fast-math), and rounds it to f32; `mean = sum / (float)n` in f32.
//! - **`ggml_vec_cvar_f32`** (the AVX2 + FMA branch; no FMA is emitted in it): per block of 8, `d = x − mean` (f32,
//!   stored as the node's first draft), `p = d·d` (`vmulps`), then in f32 `h = p[4..8] + p[0..4]`,
//!   `(h0 + h2) + (h1 + h3)`, widened to double and added to a double sum; a tail of `n mod 8` adds `(d·d)` widened
//!   (none for 384 or 512). It returns `sum / n` in double, which the caller rounds to f32.
//! - `scale = 1.0f / sqrtf(variance + eps)`: an f32 add, `vsqrtss`, `vdivss`; then every `d` is multiplied by it.
//! - **MUL and ADD are separate nodes** (binary-ops.cpp: `z[i] = x[i] op y[i]`, the weight and bias broadcast per row):
//!   `(n·w) + b`, two roundings, no FMA. The CPU backend fuses only RMS_NORM + MUL, never NORM + MUL + ADD.
//!
//! The fast path ([`LayerNorm::run_into`]) keeps every one of those roundings and changes only what cannot change a
//! bit: the three nodes are one pass per row with no intermediate tensors; the double sum, which the reference adds
//! one element at a time (a chain of n dependent adds), is added in four vector lanes **when the row proves that no
//! partial sum in any order can round** (every value is a multiple of the smallest one's ulp and n · max|x| fits in
//! 53 bits of that ulp — then every order gives the exact sum, which is what the sequential sum gave); a row that
//! cannot prove it takes the sequential sum. Threads split rows.

use crate::model::{Dtype, Model};

/// Rows a thread must have before [`LayerNorm::run_into`] starts another (see there).
pub const MIN_ROWS_PER_THREAD: usize = 2048;

/// whisper's `hparams.eps` (whisper.cpp:602), stored by `ggml_norm` in the node's `op_params`.
pub const EPS: f32 = 1e-5;

/// How a row is normalized: the reference's way is `Variant::default()`; every other setting is a discriminator
/// (an order or precision the oracle must reject).
#[derive(Clone, Copy, Default, Debug)]
pub struct Variant {
    /// the row's sum accumulated in f32 instead of double
    pub sum_f32: bool,
    /// the mean as `(double sum / n)` rounded once, instead of the sum rounded to f32 and then divided in f32
    pub mean_double: bool,
    /// one pass: variance = Σx²/n − mean² in double, instead of cvar's centred two-pass
    pub single_pass: bool,
    /// cvar's squares added to the double one at a time, without the 8-lane f32 reduction of each block
    pub cvar_sequential: bool,
    /// `1 / (sqrt(variance) + eps)`: eps outside the square root
    pub eps_outside: bool,
    /// the scale computed in double, rounded once
    pub scale_double: bool,
    /// `d / sqrt(variance + eps)` instead of `d · (1 / sqrt(variance + eps))`
    pub divide: bool,
    /// `n·w + b` as one fused multiply-add (the MUL and ADD fused) instead of two roundings
    pub fma: bool,
    /// the double sum in the fast path's eight lanes on every row, without the proof that the order is free
    pub sum_lanes: bool,
}

/// A lane sum as a vector unit makes it: x[8k + j] into lane j (j < 8), the lanes added pairwise, then the tail in
/// order — the fast path's kind of sum (its own lanes are paired differently). Equal to the in-order sum whenever
/// [`sum_is_order_free`] holds, which the fast path checks per row before it trusts its lanes.
pub fn sum_lanes(x: &[f32]) -> f64 {
    let n8 = x.len() & !7;
    let mut l = [0.0f64; 8];
    for c in x[..n8].as_chunks::<8>().0 {
        for (l, &v) in l.iter_mut().zip(c) {
            *l += v as f64;
        }
    }
    let mut t = ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
    for &v in &x[n8..] {
        t += v as f64;
    }
    t
}

/// `ggml_vec_cvar_f32` on this build (AVX2 + FMA branch): the double sum of each 8-block's squares reduced in f32 in
/// the reference's pairing, the tail one at a time, divided by n in double. `y` gets `x − mean` (the norm's draft).
pub fn cvar_model(x: &[f32], mean: f32) -> f64 {
    let n = x.len();
    let mut sum = 0.0f64;
    let mut i = 0;
    while i + 8 <= n {
        let mut p = [0.0f32; 8];
        for (j, p) in p.iter_mut().enumerate() {
            let d = x[i + j] - mean;
            *p = d * d;
        }
        let h = [p[4] + p[0], p[5] + p[1], p[6] + p[2], p[7] + p[3]];
        sum += ((h[0] + h[2]) + (h[1] + h[3])) as f64;
        i += 8;
    }
    for &v in &x[i..] {
        let d = v - mean;
        sum += (d * d) as f64;
    }
    sum / n as f64
}

/// (mean, variance, scale) of one row as `ggml_compute_forward_norm_f32` computes them (or as `v` says).
pub fn row_stats(x: &[f32], eps: f32, v: Variant) -> (f32, f32, f32) {
    let n = x.len();
    let mean = if v.sum_f32 {
        x.iter().fold(0.0f32, |s, &a| s + a) / n as f32
    } else {
        let s = if v.sum_lanes { sum_lanes(x) } else { x.iter().fold(0.0f64, |s, &a| s + a as f64) };
        if v.mean_double { (s / n as f64) as f32 } else { s as f32 / n as f32 }
    };
    let var = if v.single_pass {
        let q = x.iter().fold(0.0f64, |s, &a| s + a as f64 * a as f64);
        (q / n as f64 - mean as f64 * mean as f64) as f32
    } else if v.cvar_sequential {
        (x.iter().fold(0.0f64, |s, &a| {
            let d = a - mean;
            s + (d * d) as f64
        }) / n as f64) as f32
    } else {
        cvar_model(x, mean) as f32
    };
    let scale = if v.scale_double {
        (1.0 / (var as f64 + eps as f64).sqrt()) as f32
    } else if v.eps_outside {
        1.0f32 / (var.sqrt() + eps)
    } else {
        1.0f32 / (var + eps).sqrt()
    };
    (mean, var, scale)
}

/// The NORM node of one row (`y = (x − mean) · scale`), by the model.
pub fn norm_row(x: &[f32], eps: f32, v: Variant, y: &mut [f32]) {
    let (mean, var, scale) = row_stats(x, eps, v);
    let root = (var + eps).sqrt();
    for (y, &a) in y.iter_mut().zip(x) {
        *y = if v.divide { (a - mean) / root } else { (a - mean) * scale };
    }
}

/// What [`LayerNorm::run_into`] writes: one of the reference's three nodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Node {
    /// the NORM node: (x − mean) · scale
    Norm,
    /// the MUL node: NORM · w
    Mul,
    /// the ADD node: MUL + b (what the next op reads)
    Add,
}

/// One layer norm with its weight and bias (f32 [n]), as whisper's graph applies it to rows of n values.
pub struct LayerNorm {
    pub n: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub eps: f32,
}

struct SharedOut(*mut f32);
// SAFETY: threads write disjoint row ranges of the output, and the scope joins them before it is read
unsafe impl Sync for SharedOut {}

impl LayerNorm {
    /// `<prefix>.weight` and `<prefix>.bias` of the model (e.g. `encoder.blocks.0.attn_ln`, `encoder.ln_post`).
    pub fn new(m: &Model, prefix: &str) -> Result<LayerNorm, String> {
        let get = |s: &str| -> Result<Vec<f32>, String> {
            let name = format!("{prefix}.{s}");
            let t = m.tensor(&name).ok_or(format!("no {name}"))?;
            if t.dtype != Dtype::F32 {
                return Err(format!("{name}: expected f32, got {:?}", t.dtype));
            }
            Ok(m.tensor_bytes(t).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
        };
        LayerNorm::from_parts(get("weight")?, get("bias")?, EPS)
    }

    pub fn from_parts(w: Vec<f32>, b: Vec<f32>, eps: f32) -> Result<LayerNorm, String> {
        if w.len() != b.len() || w.is_empty() {
            return Err("layer norm: weight and bias lengths".into());
        }
        Ok(LayerNorm { n: w.len(), w, b, eps })
    }

    /// The encoder's norms in graph order, named as the oracle names them: attn_ln_0, mlp_ln_0, …, ln_post.
    pub fn encoder(m: &Model) -> Result<Vec<(String, LayerNorm)>, String> {
        let mut v = Vec::new();
        for l in 0..m.hparams.n_audio_layer as usize {
            for k in ["attn_ln", "mlp_ln"] {
                v.push((format!("{k}_{l}"), LayerNorm::new(m, &format!("encoder.blocks.{l}.{k}"))?));
            }
        }
        v.push(("ln_post".into(), LayerNorm::new(m, "encoder.ln_post")?));
        Ok(v)
    }

    /// The reference's three nodes for one row, by the model (with `v`'s changes): NORM, then · w, then + b.
    pub fn row_model(&self, x: &[f32], v: Variant, node: Node, out: &mut [f32]) {
        norm_row(x, self.eps, v, out);
        if node == Node::Norm {
            return;
        }
        for ((y, &w), &b) in out.iter_mut().zip(&self.w).zip(&self.b) {
            *y = match (node, v.fma) {
                (Node::Mul, _) => *y * w,
                (_, true) => y.mul_add(w, b),
                _ => *y * w + b,
            };
        }
    }

    /// Every row of `x` ([rows][n]) by the model into a new vector.
    pub fn run_model(&self, x: &[f32], v: Variant, node: Node) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for (x, y) in x.chunks_exact(self.n).zip(out.chunks_exact_mut(self.n)) {
            self.row_model(x, v, node, y);
        }
        out
    }

    /// The node `node` for every row of `x` ([rows][n], frame-major as the graph holds it) into a new vector.
    pub fn run(&self, x: &[f32], threads: usize, node: Node) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        self.run_into(x, threads, node, &mut out);
        out
    }

    /// The node `node` for every row of `x` into `out` (the same length), which the caller can keep. One pass per row:
    /// the sum (four lanes when the row proves it exact, else in order), cvar's 8-blocks, then
    /// `((x − mean) · scale) · w + b` with each rounding the reference makes. Threads split rows, at most one per
    /// [`MIN_ROWS_PER_THREAD`] rows: measured on the 2-core gate laptop, a second thread never paid at whisper's
    /// 1,500 rows (the op streams ~4.5 MB and is bandwidth-bound; a scoped spawn costs ~60 µs against ~0.4 ms), so
    /// the encoder's norms run on one. Each row is whole in one thread, so the count changes no bit.
    pub fn run_into(&self, x: &[f32], threads: usize, node: Node, out: &mut [f32]) {
        let rows = x.len() / self.n.max(1);
        self.run_into_split(x, threads.min(rows.div_ceil(MIN_ROWS_PER_THREAD)).max(1), node, out);
    }

    /// [`LayerNorm::run_into`] with exactly `threads` threads (clamped to the row count): the split the oracle checks.
    pub fn run_into_split(&self, x: &[f32], threads: usize, node: Node, out: &mut [f32]) {
        let n = self.n;
        assert!(x.len().is_multiple_of(n), "layer norm: the input is not whole rows");
        assert_eq!(out.len(), x.len(), "layer norm: the output is not the input's size");
        let rows = x.len() / n;
        let threads = threads.clamp(1, rows.max(1));
        let shared = SharedOut(out.as_mut_ptr());
        std::thread::scope(|s| {
            for ti in 0..threads {
                let (a, b) = (rows * ti / threads, rows * (ti + 1) / threads);
                let shared = &shared;
                let job = move || {
                    // SAFETY: rows [a, b) of the output belong to this thread alone; the scope joins before reading
                    let y = unsafe { std::slice::from_raw_parts_mut(shared.0.add(a * n), (b - a) * n) };
                    self.rows(&x[a * n..b * n], node, y);
                };
                if threads == 1 {
                    job();
                } else {
                    s.spawn(job);
                }
            }
        });
    }

    /// One row of the ADD node (the norm's output) by the fast path, for a consumer that reads it row by row and never
    /// needs it whole (0.0.9: fused into the matrix products' f32 → f16 conversion). Same bits as [`LayerNorm::run_into`].
    #[inline]
    pub fn row_into(&self, x: &[f32], y: &mut [f32]) {
        assert!(x.len() == self.n && y.len() == self.n, "layer norm: one row of n");
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2 (checked above); x and y are one row of n each
            unsafe { x86::row(x, &self.w, &self.b, self.eps, Node::Add, y) };
            return;
        }
        self.row_model(x, Variant::default(), Node::Add, y);
    }

    fn rows(&self, x: &[f32], node: Node, y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            for (x, y) in x.chunks_exact(self.n).zip(y.chunks_exact_mut(self.n)) {
                // SAFETY: the CPU has AVX2 (checked above); x and y are one row of n each
                unsafe { x86::row(x, &self.w, &self.b, self.eps, node, y) };
            }
            return;
        }
        for (x, y) in x.chunks_exact(self.n).zip(y.chunks_exact_mut(self.n)) {
            self.row_model(x, Variant::default(), node, y);
        }
    }
}

/// Whether the double sum of `x` is the same in every order: every value is a multiple of 2^q (q from the smallest
/// non-zero magnitude's exponent) and n · max|x| < 2^(53 + q), so no partial sum can round. `max_bits` and
/// `min_bits` are the largest |x| bit pattern and the smallest non-zero one (u32::MAX when every value is zero).
#[inline]
pub fn sum_is_order_free(n: usize, max_bits: u32, min_bits: u32) -> bool {
    if max_bits >= 0x7F80_0000 {
        return false; // an infinity or a NaN: keep the reference's order
    }
    if min_bits == u32::MAX || max_bits == 0 {
        return true; // all zeros
    }
    let emax = (max_bits >> 23).max(1) as i64; // biased; a subnormal counts as exponent 1 (ulp 2^-149)
    let emin = (min_bits >> 23).max(1) as i64;
    // |x| < 2^(emax - 126), values are multiples of 2^(emin - 150); n < 2^l
    let l = (usize::BITS - (n.max(1) - 1).leading_zeros()) as i64;
    l + emax - 126 <= 53 + emin - 150
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// One 8-block of cvar: `d = x − mean`, `p = d·d`, `h = p[4..8] + p[0..4]`, `(h0 + h2) + (h1 + h3)` — the
    /// reference's instructions (vsubps, vmulps, vextractf128, vaddps, vmovhlps, vaddps, vmovshdup, vaddss).
    #[inline(always)]
    pub(super) unsafe fn block1(p: *const f32, vm: __m256) -> f32 {
        // SAFETY: p points at 8 f32 (the caller's loop bound)
        unsafe {
            let d = _mm256_sub_ps(_mm256_loadu_ps(p), vm);
            let q = _mm256_mul_ps(d, d);
            let h = _mm_add_ps(_mm256_extractf128_ps::<1>(q), _mm256_castps256_ps128(q));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)))
        }
    }

    /// Four consecutive 8-blocks of cvar at once, lane k = block k: the same sums as [`block1`] (every add is the
    /// reference's pair of operands — `(h0 + h2) + (h1 + h3)` — only gathered across blocks; IEEE addition commutes).
    #[inline(always)]
    pub(super) unsafe fn blocks4(p: *const f32, vm: __m256) -> __m128 {
        // SAFETY: p points at 32 f32 (the caller's loop bound)
        unsafe {
            let h = |k: usize| {
                let d = _mm256_sub_ps(_mm256_loadu_ps(p.add(8 * k)), vm);
                let q = _mm256_mul_ps(d, d);
                _mm_add_ps(_mm256_extractf128_ps::<1>(q), _mm256_castps256_ps128(q))
            };
            let (a, b, c, d) = (h(0), h(1), h(2), h(3));
            // u = [a0 + a2, b0 + b2, a1 + a3, b1 + b3], v likewise for c, d
            let u = _mm_add_ps(_mm_unpacklo_ps(a, b), _mm_unpackhi_ps(a, b));
            let v = _mm_add_ps(_mm_unpacklo_ps(c, d), _mm_unpackhi_ps(c, d));
            // [a0+a2, b0+b2, c0+c2, d0+d2] + [a1+a3, b1+b3, c1+c3, d1+d3]
            _mm_add_ps(_mm_movelh_ps(u, v), _mm_movehl_ps(v, u))
        }
    }

    #[inline(always)]
    unsafe fn hsum_pd(v: __m256d) -> f64 {
        let mut l = [0.0f64; 4];
        // SAFETY: l holds 4 f64
        unsafe { _mm256_storeu_pd(l.as_mut_ptr(), v) };
        (l[0] + l[1]) + (l[2] + l[3])
    }
    #[inline(always)]
    unsafe fn hmax_epu32(v: __m256i) -> u32 {
        let mut l = [0u32; 8];
        // SAFETY: l holds 8 u32
        unsafe { _mm256_storeu_si256(l.as_mut_ptr().cast(), v) };
        l.into_iter().max().unwrap_or(0)
    }
    #[inline(always)]
    unsafe fn hmin_epu32(v: __m256i) -> u32 {
        let mut l = [0u32; 8];
        // SAFETY: l holds 8 u32
        unsafe { _mm256_storeu_si256(l.as_mut_ptr().cast(), v) };
        l.into_iter().min().unwrap_or(u32::MAX)
    }
    #[inline(always)]
    unsafe fn hmax4_epu32(v: __m128i) -> u32 {
        let mut l = [0u32; 4];
        // SAFETY: l holds 4 u32
        unsafe { _mm_storeu_si128(l.as_mut_ptr().cast(), v) };
        l.into_iter().max().unwrap_or(0)
    }
    /// The smallest non-zero |bits| from the minimum of |bits| − 1 (u32::MAX: every value was zero)
    #[inline(always)]
    fn min_from_m1(m1: u32) -> u32 {
        if m1 == u32::MAX { u32::MAX } else { m1 + 1 }
    }
    #[inline(always)]
    unsafe fn hmin4_epu32(v: __m128i) -> u32 {
        let mut l = [0u32; 4];
        // SAFETY: l holds 4 u32
        unsafe { _mm_storeu_si128(l.as_mut_ptr().cast(), v) };
        l.into_iter().min().unwrap_or(u32::MAX)
    }

    /// The three nodes of one row with every rounding of the reference (see the module's notes).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn row(x: &[f32], w: &[f32], b: &[f32], eps: f32, node: Node, y: &mut [f32]) {
        let n = x.len();
        let n8 = n & !7;
        // SAFETY: every load and store below is within x, w, b, y (n8 <= n; the tails are scalar)
        unsafe {
            let px = x.as_ptr();
            // 1. the double sum: lanes (any order) + the order-free proof in one pass, else the reference's order
            let abs_mask = _mm256_set1_epi32(0x7FFF_FFFF);
            let one = _mm256_set1_epi32(1);
            let (mut s0, mut s1, mut s2, mut s3) = (_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd());
            let (mut mx, mut mn) = (_mm256_setzero_si256(), _mm256_set1_epi32(-1));
            let mut i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(px.add(i));
                s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
                s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v)));
                let a = _mm256_and_si256(_mm256_castps_si256(v), abs_mask);
                mx = _mm256_max_epu32(mx, a);
                mn = _mm256_min_epu32(mn, _mm256_sub_epi32(a, one)); // 0 -> u32::MAX: zeros never count as smallest
                i += 8;
                if i < n8 {
                    // the next block into the other pair: four chains, not two
                    let v = _mm256_loadu_ps(px.add(i));
                    s2 = _mm256_add_pd(s2, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
                    s3 = _mm256_add_pd(s3, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v)));
                    let a = _mm256_and_si256(_mm256_castps_si256(v), abs_mask);
                    mx = _mm256_max_epu32(mx, a);
                    mn = _mm256_min_epu32(mn, _mm256_sub_epi32(a, one));
                    i += 8;
                }
            }
            let (mut max_bits, mut min_m1) = (hmax_epu32(mx), hmin_epu32(mn));
            for &v in &x[n8..] {
                let a = v.to_bits() & 0x7FFF_FFFF;
                max_bits = max_bits.max(a);
                min_m1 = min_m1.min(a.wrapping_sub(1));
            }
            let sum = if sum_is_order_free(n, max_bits, min_from_m1(min_m1)) {
                let mut t = hsum_pd(_mm256_add_pd(_mm256_add_pd(s0, s2), _mm256_add_pd(s1, s3)));
                for &v in &x[n8..] {
                    t += v as f64;
                }
                t
            } else {
                x.iter().fold(0.0f64, |s, &a| s + a as f64)
            };
            let mean = sum as f32 / n as f32;

            // 2. ggml_vec_cvar_f32's blocks of 8 and its f32 pairing, four blocks at a time; their double sum in lanes
            //    when the 8-block sums (and the tail's squares) prove the order free, else in block order
            let vm = _mm256_set1_ps(mean);
            let nb = n8 / 8;
            let nb4 = nb & !3;
            let mut acc = _mm256_setzero_pd();
            let (mut mx, mut mn) = (_mm_setzero_si128(), _mm_set1_epi32(-1));
            let mut i = 0;
            while i < nb4 * 8 {
                let r = blocks4(px.add(i), vm);
                acc = _mm256_add_pd(acc, _mm256_cvtps_pd(r));
                let a = _mm_and_si128(_mm_castps_si128(r), _mm256_castsi256_si128(abs_mask));
                mx = _mm_max_epu32(mx, a);
                mn = _mm_min_epu32(mn, _mm_sub_epi32(a, _mm256_castsi256_si128(one)));
                i += 32;
            }
            let (mut max_bits, mut min_m1) = (hmax4_epu32(mx), hmin4_epu32(mn));
            let mut rest = [0.0f32; 11]; // < 4 blocks and < 8 tail squares
            let mut nr = 0;
            while i < n8 {
                rest[nr] = block1(px.add(i), vm);
                nr += 1;
                i += 8;
            }
            for &v in &x[n8..] {
                let d = v - mean;
                rest[nr] = d * d;
                nr += 1;
            }
            for &r in &rest[..nr] {
                let a = r.to_bits() & 0x7FFF_FFFF;
                max_bits = max_bits.max(a);
                min_m1 = min_m1.min(a.wrapping_sub(1));
            }
            let var = if sum_is_order_free(nb + (n - n8), max_bits, min_from_m1(min_m1)) {
                let mut t = hsum_pd(acc);
                for &r in &rest[..nr] {
                    t += r as f64;
                }
                t
            } else {
                let mut t = 0.0f64;
                let mut i = 0;
                while i < n8 {
                    t += block1(px.add(i), vm) as f64;
                    i += 8;
                }
                for &v in &x[n8..] {
                    let d = v - mean;
                    t += (d * d) as f64;
                }
                t
            };
            let var = (var / n as f64) as f32;
            let scale = 1.0f32 / (var + eps).sqrt();

            // 3. (x − mean) · scale, then · w, then + b — each rounded as its own node
            let vs = _mm256_set1_ps(scale);
            let py = y.as_mut_ptr();
            let mut i = 0;
            while i < n8 {
                let d = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(px.add(i)), vm), vs);
                let r = match node {
                    Node::Norm => d,
                    Node::Mul => _mm256_mul_ps(d, _mm256_loadu_ps(w.as_ptr().add(i))),
                    Node::Add => _mm256_add_ps(_mm256_mul_ps(d, _mm256_loadu_ps(w.as_ptr().add(i))), _mm256_loadu_ps(b.as_ptr().add(i))),
                };
                _mm256_storeu_ps(py.add(i), r);
                i += 8;
            }
            for i in n8..n {
                let d = (x[i] - mean) * scale;
                y[i] = match node {
                    Node::Norm => d,
                    Node::Mul => d * w[i],
                    Node::Add => d * w[i] + b[i],
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    fn rows(seed: u64, rows: usize, n: usize, wide: bool) -> Vec<f32> {
        let mut s = seed;
        (0..rows * n)
            .map(|_| {
                let r = xorshift(&mut s);
                let m = ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0;
                if wide && r.is_multiple_of(7) { m * f32::powi(2.0, (r % 40) as i32 - 30) } else { m }
            })
            .collect()
    }

    #[test]
    fn order_free_bound() {
        // 384 values in [1, 2): exponent range 0, always order-free; a 2^-21 next to a 1.0 is not (21 > 20)
        assert!(sum_is_order_free(384, 1.5f32.to_bits(), 1.0f32.to_bits()));
        assert!(sum_is_order_free(384, 1.5f32.to_bits(), f32::powi(2.0, -20).to_bits()));
        assert!(!sum_is_order_free(384, 1.5f32.to_bits(), f32::powi(2.0, -21).to_bits()));
        assert!(!sum_is_order_free(384, f32::INFINITY.to_bits(), 1.0f32.to_bits()));
        assert!(sum_is_order_free(384, 0, u32::MAX));
    }

    #[test]
    fn order_free_sum_is_the_sequential_sum() {
        // whenever the bound holds, a shuffled double sum equals the in-order one
        let mut s = 99u64;
        let mut checked = 0;
        for k in 0..2000 {
            let x = rows(k + 1, 1, 384, k % 2 == 0);
            let mx = x.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).max().unwrap();
            let mn = x.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).filter(|&a| a != 0).min().unwrap_or(u32::MAX);
            if !sum_is_order_free(384, mx, mn) {
                continue;
            }
            let seq = x.iter().fold(0.0f64, |a, &v| a + v as f64);
            let mut y = x.clone();
            for i in (1..y.len()).rev() {
                y.swap(i, (xorshift(&mut s) % (i as u64 + 1)) as usize);
            }
            assert_eq!(seq.to_bits(), y.iter().fold(0.0f64, |a, &v| a + v as f64).to_bits());
            checked += 1;
        }
        assert!(checked > 900);
    }

    #[test]
    fn the_proof_is_needed() {
        // 2^30, seven 2^-25, -2^30, zeros: in order, the small values vanish into 2^30 (56 bits); in lanes they are
        // summed apart and survive. The bound refuses the row, and the fast path keeps the reference's order.
        let mut x = vec![0.0f32; 384];
        x[0] = f32::powi(2.0, 30);
        x[1..8].fill(f32::powi(2.0, -25));
        x[8] = -f32::powi(2.0, 30);
        let seq = x.iter().fold(0.0f64, |a, &v| a + v as f64);
        assert_eq!(seq, 0.0);
        assert_eq!(sum_lanes(&x), 7.0 * f64::powi(2.0, -25));
        let mx = x.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).max().unwrap();
        let mn = x.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).filter(|&a| a != 0).min().unwrap();
        assert!(!sum_is_order_free(384, mx, mn));
        // the mean differs (0 against 7·2^-25/384); on this row the NORM's outputs happen not to (2^30 swamps it)
        assert_eq!(row_stats(&x, EPS, Variant::default()).0, 0.0);
        assert!(row_stats(&x, EPS, Variant { sum_lanes: true, ..Variant::default() }).0 > 0.0);
        let ln = LayerNorm::from_parts(vec![1.5; 384], vec![0.25; 384], EPS).unwrap();
        let want = ln.run_model(&x, Variant::default(), Node::Add);
        assert!(ln.run(&x, 1, Node::Add).iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn fast_equals_model() {
        let mut s = 5u64;
        for (n, wide) in [(384, false), (384, true), (512, true), (13, false), (8, true), (1, false), (1031, true)] {
            let w: Vec<f32> = (0..n).map(|_| (xorshift(&mut s) >> 40) as f32 / (1u64 << 22) as f32 - 2.0).collect();
            let b: Vec<f32> = (0..n).map(|_| (xorshift(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
            let ln = LayerNorm::from_parts(w, b, EPS).unwrap();
            let mut x = rows(n as u64, 37, n, wide);
            x[3] = 0.0;
            x[n.min(5)] = -0.0;
            for node in [Node::Norm, Node::Mul, Node::Add] {
                let want = ln.run_model(&x, Variant::default(), node);
                for threads in [1, 3] {
                    let mut got = vec![0.0f32; x.len()];
                    ln.run_into_split(&x, threads, node, &mut got);
                    let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    assert_eq!(bad, 0, "n {n} wide {wide} {node:?} threads {threads}");
                }
            }
        }
        // a zero row (variance 0: scale 1/sqrt(eps)) and a row with an infinity (the in-order path, NaNs out)
        let ln = LayerNorm::from_parts(vec![1.0; 16], vec![0.0; 16], EPS).unwrap();
        let mut x = vec![0.0f32; 32];
        x[20] = f32::INFINITY;
        let (a, b) = (ln.run(&x, 1, Node::Add), ln.run_model(&x, Variant::default(), Node::Add));
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
