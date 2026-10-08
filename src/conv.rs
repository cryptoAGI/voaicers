// SPDX-License-Identifier: MIT OR Apache-2.0
//! The encoder's first convolution as the shipped ggml-cpu computes it, and the f16 dot product under it (0.0.6).
//! Read from the pinned source and the shipped `libggml-cpu.so` (`objdump -d`), then checked by the oracle on the
//! conv graph's own nodes (testing/conv1/NOTES.md):
//!
//! - **conv1 is `ggml_conv_1d_ph(w, mel, 1, 1)`** = `im2col` to **f16** (`GGML_CPU_FP32_TO_FP16`, the portable bit
//!   trick of [`crate::f16::fp32_to_fp16`]) then `mul_mat(im2col [240, 3000], w [240, 384])`. The weights are
//!   mul_mat's *second* operand and already f16, which is the F16 `vec_dot_type`, so nothing else is converted: each
//!   output (frame `t`, channel `c`) is one `ggml_vec_dot_f16(240, im2col row t, weight row c)`, computed whole by one
//!   thread. Row t of im2col is, for input channel `ic` and tap `kw`, element `3·ic + kw` = the mel at frame
//!   `t + kw − 1` rounded to f16, or +0 outside the window.
//! - **`ggml_vec_dot_f16` on this build** (AVX2 + FMA + F16C, no AVX-512 — `GGML_NATIVE` on the laptop's Zen+;
//!   production's Zen 3 has the same extensions): `np = n & !31`; four accumulators of eight f32 lanes, block `j` of
//!   each 32 halves (`i + 8j`) fused-multiply-added into accumulator `j`; then `(acc0 + acc2) + (acc1 + acc3)` lane
//!   by lane, the high 128 bits added to the low, two `hadd`s — `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))` —
//!   and the f32 widened to **double**; the last `n − np` products (each an f32 product of two widened halves) are
//!   added to that double in index order, and the double is rounded to f32 once at the end. `LLAMAFILE` is off in
//!   the build (no tinyBLAS) and the CPU repack buffers have no f16 case.
//! - **A product of two f16 values is exact in f32** (11 + 11 significant bits; the smallest, 2⁻⁴⁸, is normal), so
//!   the fused multiply-add and a multiply followed by an add give the same bits here. The portable model below
//!   multiplies and adds; it is exact against the FMA build, and would be against a non-FMA AVX build too.
//! - Neither the thread count nor observing every node changes a bit (the oracle checks 1 against 4 threads, and the
//!   scheduler's output with and without its eval callback).
//!
//! The fast path keeps every dot's float order and changes only what is free: the weights and the f16-rounded mel
//! are widened to f32 once (widening is exact), im2col is never materialized (a tile of 8 frames is built in 7.5 KB
//! on the stack from the rounded mel), eight dots share each weight load, the horizontal reduction runs for eight
//! frames at once with the same pairings, and the tail's eight doubles run in two vectors. Bias and GELU can be fused
//! into the tile's epilogue ([`Conv1::run_gelu`]): the add is one rounding either way, GELU is 0.0.3's op.

use crate::f16::{fp16_to_fp32, fp32_to_fp16};
use crate::gelu::Gelu;
use crate::model::{Dtype, Model};

/// `ggml_vec_dot_f16(n, x, y)` as the AVX path computes it, on halves already widened to f32 (exact): the model the
/// fast kernels must equal, and the one used where AVX2 + FMA are absent.
pub fn dot_f16_model(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "dot: lengths differ");
    let n = x.len();
    let np = n & !31;
    let mut acc = [[0.0f32; 8]; 4];
    for i in (0..np).step_by(32) {
        for (j, a) in acc.iter_mut().enumerate() {
            for (l, al) in a.iter_mut().enumerate() {
                let k = i + 8 * j + l;
                *al += x[k] * y[k]; // exact product: the same bits as the build's vfmadd231ps
            }
        }
    }
    let mut s = [0.0f32; 8];
    for l in 0..8 {
        s[l] = (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]);
    }
    let t = [s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]];
    let mut sum = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
    for k in np..n {
        sum += (x[k] * y[k]) as f64;
    }
    sum as f32
}

/// `ggml_vec_dot_f16` on f16 bit patterns (what the kernel oracle feeds the shipped one).
pub fn vec_dot_f16(x: &[u16], y: &[u16]) -> f32 {
    let xw: Vec<f32> = x.iter().map(|&h| fp16_to_fp32(h)).collect();
    let yw: Vec<f32> = y.iter().map(|&h| fp16_to_fp32(h)).collect();
    dot_f16_model(&xw, &yw)
}

/// NOT the reference — the orders the oracle must reject (`oracle_vec_dot_f16_discriminators`).
pub mod wrong {
    use crate::f16::fp16_to_fp32;
    fn widen(x: &[u16]) -> Vec<f32> {
        x.iter().map(|&h| fp16_to_fp32(h)).collect()
    }
    /// One f32 accumulator, index order (the naive loop).
    pub fn single_accumulator(x: &[u16], y: &[u16]) -> f32 {
        widen(x).iter().zip(widen(y)).fold(0.0f32, |s, (a, b)| s + a * b)
    }
    /// The right lanes, but the tail added in f32 instead of double.
    pub fn tail_in_f32(x: &[u16], y: &[u16]) -> f32 {
        let (x, y) = (widen(x), widen(y));
        let np = x.len() & !31;
        let mut s = super::dot_f16_model(&x[..np], &y[..np]);
        for k in np..x.len() {
            s += x[k] * y[k];
        }
        s
    }
    /// The right lanes, the accumulators reduced in sequence `((acc0 + acc1) + acc2) + acc3`.
    pub fn sequential_reduce(x: &[u16], y: &[u16]) -> f32 {
        let (x, y) = (widen(x), widen(y));
        let n = x.len();
        let np = n & !31;
        let mut acc = [[0.0f32; 8]; 4];
        for i in (0..np).step_by(32) {
            for (j, a) in acc.iter_mut().enumerate() {
                for (l, al) in a.iter_mut().enumerate() {
                    *al += x[i + 8 * j + l] * y[i + 8 * j + l];
                }
            }
        }
        let s: Vec<f32> = (0..8).map(|l| ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l]).collect();
        let t = [s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]];
        let mut sum = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
        for k in np..n {
            sum += (x[k] * y[k]) as f64;
        }
        sum as f32
    }
}

/// conv1's im2col node, as ggml-cpu writes it: f16 [n_frames][3·n_mel] from the mel window at `offset` (whisper's
/// slice: frames past `n_len` are 0). `mel` is [n_mel][n_len] (`whisper_mel`). The node the oracle compares.
pub fn im2col_f16(mel: &[f32], n_mel: usize, n_len: usize, offset: usize, n_frames: usize) -> Vec<u16> {
    let k = 3 * n_mel;
    let mut out = vec![0u16; n_frames * k];
    for t in 0..n_frames {
        for ic in 0..n_mel {
            for kw in 0..3 {
                let src = t + kw;
                // the window index t + kw − 1, in [0, n_frames); the frame behind it may lie past n_len (then 0.0)
                if src >= 1 && src - 1 < n_frames {
                    let f = offset + src - 1;
                    let v = if f < n_len { mel[ic * n_len + f] } else { 0.0 };
                    out[t * k + 3 * ic + kw] = fp32_to_fp16(v);
                }
            }
        }
    }
    out
}

/// conv1 (and its bias and GELU) for one model, the weights widened once.
pub struct Conv1 {
    pub n_mel: usize,
    pub n_out: usize,
    /// K = 3·n_mel inputs per output
    k: usize,
    /// [n_out][k], the f16 weights widened to f32 (exact)
    w: Vec<f32>,
    /// [n_out]
    pub bias: Vec<f32>,
    gelu: Gelu,
}

impl Conv1 {
    pub fn new(m: &Model) -> Result<Conv1, String> {
        let wt = m.tensor("encoder.conv1.weight").ok_or("no encoder.conv1.weight")?;
        let bt = m.tensor("encoder.conv1.bias").ok_or("no encoder.conv1.bias")?;
        if wt.dtype != Dtype::F16 || bt.dtype != Dtype::F32 || wt.ne[0] != 3 {
            return Err(format!("conv1: expected f16 [3, n_mel, n_state] weights and f32 bias, got {:?} {:?}", wt.dtype, wt.ne));
        }
        let w: Vec<u16> = m.tensor_bytes(wt).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let bias: Vec<f32> = m.tensor_bytes(bt).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Conv1::from_parts(wt.ne[1] as usize, wt.ne[2] as usize, &w, bias)
    }

    /// From the f16 weights ([n_out][n_mel][3], the file's order) and the bias.
    pub fn from_parts(n_mel: usize, n_out: usize, w: &[u16], bias: Vec<f32>) -> Result<Conv1, String> {
        let k = 3 * n_mel;
        if w.len() != n_out * k || bias.len() != n_out {
            return Err("conv1: weight or bias length".into());
        }
        Ok(Conv1 { n_mel, n_out, k, w: w.iter().map(|&h| fp16_to_fp32(h)).collect(), bias, gelu: Gelu::new() })
    }

    /// conv1 without its bias: [n_out][n_frames] (`mul_mat`'s output: ne0 = frames), as the shipped build computes it.
    pub fn run(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_out * n_frames];
        self.run_into(mel, n_len, offset, n_frames, threads, false, &mut out);
        out
    }

    /// conv1 + bias, then GELU: the conv graph's node 6, which conv2's im2col reads.
    pub fn run_gelu(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_out * n_frames];
        self.run_into(mel, n_len, offset, n_frames, threads, true, &mut out);
        out
    }

    /// conv1 + bias without GELU (node 5), computed by the fast path then the add (for the oracle).
    pub fn run_bias(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize) -> Vec<f32> {
        let mut y = self.run(mel, n_len, offset, n_frames, threads);
        for (row, &b) in y.chunks_exact_mut(n_frames).zip(&self.bias) {
            for v in row {
                *v += b;
            }
        }
        y
    }

    /// conv1 (`bias_gelu`: + bias, GELU) of the mel window at `offset` into `out` ([n_out][n_frames]), which a caller
    /// can keep across calls. `mel` is [n_mel][n_len] (`whisper_mel`); frames past `n_len` are 0, as whisper's slice.
    /// Nothing the size of the input is allocated: no im2col, no rounded copy of the mel — each thread builds tiles
    /// of 8 frames (7.5 KB) from the mel, rounding to f16 as it goes.
    #[allow(clippy::too_many_arguments)] // the window (mel, n_len, offset, n_frames), the threads, the epilogue, the output
    pub fn run_into(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize, bias_gelu: bool, out: &mut [f32]) {
        assert!(mel.len() >= self.n_mel * n_len, "conv1: the mel is shorter than n_mel x n_len");
        assert_eq!(out.len(), self.n_out * n_frames, "conv1: the output is not n_out x n_frames");
        let win = Window { mel, n_len, offset, n_frames };
        // threads split the frames, in whole tiles: each builds its own tiles once and computes every channel for
        // them, so each thread writes the columns [t_begin, t_end) of every output row; every dot is whole in one thread
        let tiles = n_frames.div_ceil(TILE);
        let threads = threads.clamp(1, tiles.max(1));
        let shared = SharedOut(out.as_mut_ptr());
        std::thread::scope(|s| {
            for ti in 0..threads {
                let (a, b) = (tiles * ti / threads * TILE, (tiles * (ti + 1) / threads * TILE).min(n_frames));
                let (win, shared) = (&win, &shared);
                let job = move || self.frames(win, a, b, shared, bias_gelu);
                if threads == 1 {
                    job();
                } else {
                    s.spawn(job);
                }
            }
        });
    }

    /// Every output channel for frames [t_begin, t_end) (a whole number of tiles, the last one possibly short).
    fn frames(&self, win: &Window, t_begin: usize, t_end: usize, out: &SharedOut, bias_gelu: bool) {
        const R: usize = TILE;
        let (k, n_frames) = (self.k, win.n_frames);
        // the window frames whose mel frame exists: [0, valid)
        let valid = n_frames.min(win.n_len.saturating_sub(win.offset));
        let np = k & !31;
        let tail = k - np;
        // a tile of R frames: main part [R][np] row-major, tail transposed [tail][R] (the epilogue loads R at once)
        let mut xm = vec![0.0f32; R * np];
        let mut xt = vec![0.0f32; tail * R];
        let mut res = [0.0f32; R];
        let fast = Kernel::detect();
        for t0 in (t_begin..t_end).step_by(R) {
            let nr = R.min(t_end - t0);
            // im2col element 3·ic + kw of frame t0 + r is window index w = t0 + r + kw − 1, rounded to f16; outside
            // [0, n_frames) or past n_len, +0. Rows r >= nr (a short last tile) are computed and never stored.
            for ic in 0..self.n_mel {
                let row = &win.mel[ic * win.n_len..(ic + 1) * win.n_len];
                for kw in 0..3 {
                    let kk = 3 * ic + kw;
                    let mut v = [0.0f32; R];
                    if t0 + kw >= 1 && t0 + kw - 1 + R <= valid {
                        let w0 = win.offset + t0 + kw - 1;
                        v.copy_from_slice(&row[w0..w0 + R]);
                    } else {
                        for (r, vr) in v.iter_mut().enumerate() {
                            let w = t0 + r + kw;
                            if w >= 1 && w - 1 < valid {
                                *vr = row[win.offset + w - 1];
                            }
                        }
                    }
                    round_f16(&mut v);
                    for (r, &v) in v.iter().enumerate() {
                        if kk < np {
                            xm[r * np + kk] = v;
                        } else {
                            xt[(kk - np) * R + r] = v;
                        }
                    }
                }
            }
            for c in 0..self.n_out {
                let w = &self.w[c * k..(c + 1) * k];
                fast.tile(&xm, &xt, w, np, &mut res);
                // SAFETY: row c, columns [t0, t0 + nr) lie inside `out` ([n_out][n_frames]) and inside this thread's
                // frames [t_begin, t_end), which no other thread writes; `out` outlives the scope that runs this
                let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(c * n_frames + t0), nr) };
                if bias_gelu {
                    let b = self.bias[c];
                    let mut z = [0.0f32; R];
                    for (zz, &v) in z.iter_mut().zip(&res) {
                        *zz = v + b;
                    }
                    self.gelu.row(&z[..nr], o);
                } else {
                    o.copy_from_slice(&res[..nr]);
                }
            }
        }
    }
}

const TILE: usize = 8;

/// The mel window conv1 reads: `n_frames` frames from `offset` of a [n_mel][n_len] mel.
struct Window<'a> {
    mel: &'a [f32],
    n_len: usize,
    offset: usize,
    n_frames: usize,
}

/// Eight values rounded to f16 and widened back, as im2col's `GGML_CPU_FP32_TO_FP16` rounds them (then exact). With
/// F16C the eight go through `vcvtps2ph` / `vcvtph2ps`, which 0.0.3's oracle shows equal the bit trick on every
/// f32 except NaN; a block holding a NaN takes the bit trick itself.
#[inline]
fn round_f16(v: &mut [f32; 8]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        // SAFETY: the CPU has F16C and AVX (checked above)
        if unsafe { x86::round8_f16c(v) } {
            return;
        }
    }
    for x in v {
        *x = fp16_to_fp32(fp32_to_fp16(*x));
    }
}
struct SharedOut(*mut f32);
// SAFETY: threads write disjoint column ranges of the output, and the scope joins them before it is read
unsafe impl Sync for SharedOut {}

#[derive(Clone, Copy)]
enum Kernel {
    Model,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Kernel {
    fn detect() -> Kernel {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Kernel::Avx2;
        }
        Kernel::Model
    }

    /// Eight dots: frames r = 0..8 of the tile against one weight row, each in `ggml_vec_dot_f16`'s order.
    #[inline]
    fn tile(self, xm: &[f32], xt: &[f32], w: &[f32], np: usize, res: &mut [f32; 8]) {
        match self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the CPU has AVX2 and FMA (detect); the slices have the lengths the kernel indexes (asserted there)
            Kernel::Avx2 => unsafe { x86::tile8(xm, xt, w, np, res) },
            Kernel::Model => {
                let k = w.len();
                let mut x = vec![0.0f32; k];
                for (r, out) in res.iter_mut().enumerate() {
                    x[..np].copy_from_slice(&xm[r * np..(r + 1) * np]);
                    for kk in np..k {
                        x[kk] = xt[(kk - np) * 8 + r];
                    }
                    *out = dot_f16_model(&x, w);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Round eight f32 to f16 and back through F16C; false (and `v` untouched) if any is NaN.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn round8_f16c(v: &mut [f32; 8]) -> bool {
        // SAFETY: v is 8 f32
        unsafe {
            let x = _mm256_loadu_ps(v.as_ptr());
            if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(x, x)) != 0 {
                return false;
            }
            _mm256_storeu_ps(v.as_mut_ptr(), _mm256_cvtph_ps(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x)));
        }
        true
    }

    /// Eight frames x one weight row. Accumulator j of frame r is chained over the blocks i + 8j exactly as the
    /// reference chains it (one chain per (j, r), so j can be the outer loop: eight chains live at a time, the weight
    /// block loaded once for the eight). The reduction pairs lanes as the reference's vaddps / vextractf128 / vhaddps
    /// do, for eight frames at once; the tail runs in two vectors of four doubles, in index order.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tile8(xm: &[f32], xt: &[f32], w: &[f32], np: usize, res: &mut [f32; 8]) {
        let k = w.len();
        assert!(xm.len() >= 8 * np && xt.len() >= 8 * (k - np) && np <= k && np.is_multiple_of(32));
        // SAFETY (all loads): xm row r spans [r·np, r·np + np), w [0, k), xt [0, 8·(k − np)) — asserted above
        unsafe {
            let x = xm.as_ptr();
            let wp = w.as_ptr();
            let mut part = [[_mm256_setzero_ps(); 8]; 4];
            for (j, pj) in part.iter_mut().enumerate() {
                let mut acc = [_mm256_setzero_ps(); 8];
                let mut i = 0;
                while i < np {
                    let wv = _mm256_loadu_ps(wp.add(i + 8 * j));
                    for (r, a) in acc.iter_mut().enumerate() {
                        *a = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(r * np + i + 8 * j)), wv, *a);
                    }
                    i += 32;
                }
                *pj = acc;
            }
            // (acc0 + acc2) + (acc1 + acc3), then each frame's high half onto its low half: frames r and r + 4 share
            // a register, [t_r | t_r+4], t_l = s_l + s_l+4
            let mut s = [_mm256_setzero_ps(); 8];
            for (r, sr) in s.iter_mut().enumerate() {
                *sr = _mm256_add_ps(_mm256_add_ps(part[0][r], part[2][r]), _mm256_add_ps(part[1][r], part[3][r]));
            }
            let mut tt = [_mm256_setzero_ps(); 4];
            for (r, t) in tt.iter_mut().enumerate() {
                let lo = _mm256_permute2f128_ps::<0x20>(s[r], s[r + 4]);
                let hi = _mm256_permute2f128_ps::<0x31>(s[r], s[r + 4]);
                *t = _mm256_add_ps(lo, hi);
            }
            // hadd(t_a, t_b) = [a0+a1, a2+a3, b0+b1, b2+b3] per 128-bit half; once more gives (t0+t1)+(t2+t3) per frame
            let h01 = _mm256_hadd_ps(tt[0], tt[1]);
            let h23 = _mm256_hadd_ps(tt[2], tt[3]);
            let red = _mm256_hadd_ps(h01, h23); // frames [0, 1, 2, 3 | 4, 5, 6, 7]
            let mut lo = _mm256_cvtps_pd(_mm256_castps256_ps128(red));
            let mut hi = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(red));
            let xtp = xt.as_ptr();
            for kk in 0..k - np {
                let p = _mm256_mul_ps(_mm256_set1_ps(*wp.add(np + kk)), _mm256_loadu_ps(xtp.add(kk * 8)));
                lo = _mm256_add_pd(lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
                hi = _mm256_add_pd(hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(p)));
            }
            let out = _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
            _mm256_storeu_ps(res.as_mut_ptr(), out);
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

    /// The AVX2 tile and the model agree on random finite halves, at every length class (tails 0..31, n < 32).
    #[test]
    fn tile_equals_model() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let finite = |s: &mut u64| loop {
            let h = xorshift(s) as u16;
            if h & 0x7C00 != 0x7C00 {
                return fp16_to_fp32(h);
            }
        };
        for k in [3usize, 31, 32, 33, 64, 95, 240, 1152] {
            let np = k & !31;
            for _ in 0..20 {
                let w: Vec<f32> = (0..k).map(|_| finite(&mut s)).collect();
                let rows: Vec<Vec<f32>> = (0..8).map(|_| (0..k).map(|_| finite(&mut s)).collect()).collect();
                let mut xm = vec![0.0; 8 * np];
                let mut xt = vec![0.0; 8 * (k - np)];
                for (r, row) in rows.iter().enumerate() {
                    xm[r * np..(r + 1) * np].copy_from_slice(&row[..np]);
                    for kk in np..k {
                        xt[(kk - np) * 8 + r] = row[kk];
                    }
                }
                let want: Vec<u32> = rows.iter().map(|x| dot_f16_model(x, &w).to_bits()).collect();
                let mut got = [0.0f32; 8];
                Kernel::detect().tile(&xm, &xt, &w, np, &mut got);
                assert_eq!(got.map(f32::to_bits).to_vec(), want, "k = {k}");
                let mut got = [0.0f32; 8];
                Kernel::Model.tile(&xm, &xt, &w, np, &mut got);
                assert_eq!(got.map(f32::to_bits).to_vec(), want, "k = {k} (model)");
            }
        }
    }

    /// The whole fast path against im2col + the model dot, on windows with every edge: an offset, a mel shorter than
    /// the window, a short last tile, 1 and 3 threads.
    #[test]
    fn conv1_equals_im2col_and_model() {
        let mut s = 0x0DDB_A11C_0FFE_E123u64;
        let (n_mel, n_out) = (80usize, 5usize);
        let k = 3 * n_mel;
        let w: Vec<u16> = (0..n_out * k).map(|_| (xorshift(&mut s) as u16) & 0xBBFF).collect(); // finite, |w| < 2^15
        let c = Conv1::from_parts(n_mel, n_out, &w, (0..n_out).map(|i| i as f32 * 0.25 - 0.5).collect()).unwrap();
        for (n_len, offset, n_frames) in [(50usize, 0usize, 37usize), (50, 20, 37), (30, 0, 40), (100, 99, 9), (64, 64, 8), (200, 3, 1)] {
            let mel: Vec<f32> = (0..n_mel * n_len).map(|_| (xorshift(&mut s) % 4001) as f32 / 1000.0 - 2.0).collect();
            let im = im2col_f16(&mel, n_mel, n_len, offset, n_frames);
            for threads in [1, 3] {
                let got = c.run(&mel, n_len, offset, n_frames, threads);
                for ch in 0..n_out {
                    for t in 0..n_frames {
                        let want = vec_dot_f16(&im[t * k..(t + 1) * k], &w[ch * k..(ch + 1) * k]);
                        assert_eq!(got[ch * n_frames + t].to_bits(), want.to_bits(), "n_len {n_len} offset {offset} frame {t} ch {ch}");
                    }
                }
            }
        }
    }

    /// A block holding a NaN is rounded by the bit trick itself (vcvtps2ph would keep payload bits).
    #[test]
    fn round_f16_nan_block() {
        let mut v = [1.0f32, f32::from_bits(0x7FA0_1234), 0.1, -3.3, 65520.0, 1e-8, -0.0, 7.0];
        let want: Vec<u32> = v.iter().map(|&x| fp16_to_fp32(fp32_to_fp16(x)).to_bits()).collect();
        round_f16(&mut v);
        assert_eq!(v.map(f32::to_bits).to_vec(), want);
        let mut v = [1.0f32, 0.1, 0.2, 0.3, 65519.0, 6e-8, -0.0, 2049.0];
        let want: Vec<u32> = v.iter().map(|&x| fp16_to_fp32(fp32_to_fp16(x)).to_bits()).collect();
        round_f16(&mut v);
        assert_eq!(v.map(f32::to_bits).to_vec(), want);
    }

    #[test]
    fn small_dots() {
        // n < 32: the reduction is +0, then the tail in double
        assert_eq!(vec_dot_f16(&[0x3C00, 0x4000], &[0x4000, 0x4200]), 8.0); // 1·2 + 2·3
        // 32 ones: one lane each, the reduce adds them
        assert_eq!(vec_dot_f16(&[0x3C00; 32], &[0x3C00; 32]), 32.0);
        // the tail in double keeps what an f32 tail would lose: 2^24 then 1 then -2^24 ... (via 32 + 3)
        let mut x = vec![0u16; 35];
        let mut y = vec![0u16; 35];
        x[0] = 0x7BFF; // 65504
        y[0] = 0x7BFF; // 65504² = 4,290,774,016: an f32 with ulp 512
        x[32] = 0x3C00;
        y[32] = 0x3C00; // + 1 (lost in f32, kept in double)
        x[33] = 0xFBFF;
        y[33] = 0x7BFF; // − 65504²
        assert_eq!(vec_dot_f16(&x, &y), 1.0);
        assert_eq!(wrong::tail_in_f32(&x, &y), 0.0);
    }
}
