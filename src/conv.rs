// SPDX-License-Identifier: MIT OR Apache-2.0
//! The encoder's conv stage as the shipped ggml-cpu computes it: conv1 and the f16 dot product under it (0.0.6), conv2
//! and the positional embedding (0.0.7, testing/conv2/NOTES.md; [`Conv2`], [`ConvStage`]).
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
        assert_eq!(out.len(), self.n_out * n_frames, "conv1: the output is not n_out x n_frames");
        self.run_dst(mel, n_len, offset, n_frames, threads, Dst::F32(SharedOut(out.as_mut_ptr()), bias_gelu));
    }

    /// conv1 + bias, GELU, stored as the f16 bits conv2's im2col makes of it (`GGML_CPU_FP32_TO_FP16`, [n_out][n_frames]):
    /// conv2 reads nothing else of conv1's output, so this half-size buffer gives conv2 the same bits
    /// ([`Conv2::run_into_f16`]).
    pub fn run_into_f16(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize, out: &mut [u16]) {
        assert_eq!(out.len(), self.n_out * n_frames, "conv1: the output is not n_out x n_frames");
        self.run_dst(mel, n_len, offset, n_frames, threads, Dst::F16(SharedOut16(out.as_mut_ptr())));
    }

    fn run_dst(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize, shared: Dst) {
        assert!(mel.len() >= self.n_mel * n_len, "conv1: the mel is shorter than n_mel x n_len");
        let win = Window { mel, n_len, offset, n_frames };
        // threads split the frames, in whole tiles: each builds its own tiles once and computes every channel for
        // them, so each thread writes the columns [t_begin, t_end) of every output row; every dot is whole in one thread
        let tiles = n_frames.div_ceil(TILE);
        let threads = threads.clamp(1, tiles.max(1));
        std::thread::scope(|s| {
            for ti in 0..threads {
                let (a, b) = (tiles * ti / threads * TILE, (tiles * (ti + 1) / threads * TILE).min(n_frames));
                let (win, shared) = (&win, &shared);
                let job = move || self.frames(win, a, b, shared);
                if threads == 1 {
                    job();
                } else {
                    s.spawn(job);
                }
            }
        });
    }

    /// Every output channel for frames [t_begin, t_end) (a whole number of tiles, the last one possibly short).
    fn frames(&self, win: &Window, t_begin: usize, t_end: usize, out: &Dst) {
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
                // SAFETY (both arms): row c, columns [t0, t0 + nr) lie inside `out` ([n_out][n_frames]) and inside this
                // thread's frames [t_begin, t_end), which no other thread writes; `out` outlives the scope that runs this
                match out {
                    Dst::F32(o, bias_gelu) => {
                        let o = unsafe { std::slice::from_raw_parts_mut(o.0.add(c * n_frames + t0), nr) };
                        if *bias_gelu {
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
                    Dst::F16(o) => {
                        let b = self.bias[c];
                        let mut z = [0.0f32; R];
                        for (zz, &v) in z.iter_mut().zip(&res) {
                            *zz = v + b;
                        }
                        let mut g = [0.0f32; R];
                        self.gelu.row(&z, &mut g);
                        let h = to_f16_8(&g);
                        let o = unsafe { std::slice::from_raw_parts_mut(o.0.add(c * n_frames + t0), nr) };
                        o.copy_from_slice(&h[..nr]);
                    }
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
struct SharedOut16(*mut u16);
// SAFETY: as SharedOut
unsafe impl Sync for SharedOut16 {}
/// conv1's output: f32 (+ bias and GELU or not), or f16 bits after bias and GELU (conv2's input)
enum Dst {
    F32(SharedOut, bool),
    F16(SharedOut16),
}

/// Eight f32 to f16 bits as `GGML_CPU_FP32_TO_FP16` (the bit trick): through F16C unless the block holds a NaN.
#[inline]
fn to_f16_8(v: &[f32; 8]) -> [u16; 8] {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        // SAFETY: the CPU has F16C and AVX (checked above)
        if let Some(h) = unsafe { x86::to_f16_8_f16c(v) } {
            return h;
        }
    }
    v.map(fp32_to_fp16)
}

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

    /// F × C dots in the file's column order (k not a multiple of 32: no whisper model), each by the model.
    fn block(self, x: &[f32], w: &[f32], k: usize, nc: usize, res: &mut [[f32; F]; C]) {
        for (c, rc) in res.iter_mut().enumerate().take(nc) {
            for (f, v) in rc.iter_mut().enumerate() {
                *v = dot_f16_model(&x[f * k..(f + 1) * k], &w[c * k..(c + 1) * k]);
            }
        }
    }

    /// F × C dots on rows in the kernel's layout ([`Conv2::col`], k a multiple of 32), each in `ggml_vec_dot_f16`'s order.
    #[inline]
    fn block_perm(self, x: &[f32], w: &[f32], k: usize, res: &mut [[f32; F]; C]) {
        match self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 and FMA (detect); the slices' lengths are asserted in the kernel
            Kernel::Avx2 => unsafe { x86::block43(x, w, k, res) },
            Kernel::Model => {
                // accumulator j, lane l, block b is column j·(k/4) + 8b + l: the model's sums, read in this layout
                let q = k / 4;
                for (c, rc) in res.iter_mut().enumerate() {
                    for (f, v) in rc.iter_mut().enumerate() {
                        let (xr, wr) = (&x[f * k..(f + 1) * k], &w[c * k..(c + 1) * k]);
                        let mut acc = [[0.0f32; 8]; 4];
                        for (j, a) in acc.iter_mut().enumerate() {
                            for b in 0..q / 8 {
                                for (l, al) in a.iter_mut().enumerate() {
                                    *al += xr[j * q + 8 * b + l] * wr[j * q + 8 * b + l];
                                }
                            }
                        }
                        let s: Vec<f32> = (0..8).map(|l| (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l])).collect();
                        *v = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
                    }
                }
            }
        }
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

    /// Eight f32 to f16 bits through F16C; None if any is NaN.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn to_f16_8_f16c(v: &[f32; 8]) -> Option<[u16; 8]> {
        // SAFETY: v is 8 f32, h 8 u16
        unsafe {
            let x = _mm256_loadu_ps(v.as_ptr());
            if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(x, x)) != 0 {
                return None;
            }
            let mut h = [0u16; 8];
            _mm_storeu_si128(h.as_mut_ptr().cast(), _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x));
            Some(h)
        }
    }

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

    /// The lane pairing of the reference's reduction for eight sums at once: each `s[d]` (already `(acc0 + acc2) +
    /// (acc1 + acc3)`) → `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))`, returned as lanes [0..8] = d.
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

    /// 4 frames × 3 weight rows, k a multiple of 32 (no tail), both in [`super::Conv2::col`]'s layout. Accumulator j of
    /// each of the 12 dots is chained over the blocks i + 8j (contiguous here) as the reference chains it; per j, the 12 chains live in registers, each weight vector is
    /// loaded once for 4 frames and each frame vector once for 3 rows (7 loads per 12 FMAs, against 9 per 8 in tile8).
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
                // the blocks i + 8j of the file's order, contiguous in this layout: [j·q, j·q + q)
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
            // dot d = c·4 + f; the f32 sum widened to double and back is the same f32 (no tail)
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

// ---- 0.0.7: conv2 and the positional embedding -------------------------------------------------------------------

/// conv2's im2col node (or any strided one), as ggml-cpu writes it: f16 [n_frames][3·n_in], element `3·ic + kw` of
/// row `t` = `x[ic][stride·t + kw − 1]` rounded to f16, +0 outside [0, n_len). `x` is [n_in][n_len] (ne0 = frames).
pub fn im2col_strided_f16(x: &[f32], n_in: usize, n_len: usize, stride: usize, n_frames: usize) -> Vec<u16> {
    let k = 3 * n_in;
    let mut out = vec![0u16; n_frames * k];
    for t in 0..n_frames {
        for ic in 0..n_in {
            for kw in 0..3 {
                let src = stride * t + kw; // the input frame src − 1
                if src >= 1 && src - 1 < n_len {
                    out[t * k + 3 * ic + kw] = fp32_to_fp16(x[ic * n_len + src - 1]);
                }
            }
        }
    }
    out
}

/// conv2's input: f32 values, or the f16 bits its im2col makes of them.
enum Input<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
}
impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            Input::F32(x) => x.len(),
            Input::F16(x) => x.len(),
        }
    }
}

/// What conv2 writes ([`Conv2::run_into`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    /// the MUL_MAT node: [n_out][n_frames], no bias
    Raw,
    /// + bias, GELU: `embd_conv`, [n_out][n_frames]
    BiasGelu,
    /// + bias, GELU, then the encoder's first op: `e_pe + cont(transpose(·))`, [n_frames][n_out] (frame-major)
    Positions,
}

/// conv2 (`ggml_conv_1d_ph(w2, x, 2, 1)`), its bias and GELU, and the positional embedding the encoder adds next,
/// for one model, the weights widened once. Each output is one `ggml_vec_dot_f16(3·n_in, im2col row t, weight row
/// c)` in the AVX build's order ([`dot_f16_model`]); with n_in a multiple of 32 (every whisper model) there is no tail.
pub struct Conv2 {
    pub n_in: usize,
    pub n_out: usize,
    pub stride: usize,
    /// 3·n_in
    k: usize,
    /// [n_out padded to a multiple of C][k], the f16 weights widened to f32 (exact); with k a multiple of 32, each row
    /// is stored in the kernel's order ([`Conv2::col`]) and the padding rows are zeros (computed, never stored)
    w: Vec<f32>,
    /// k a multiple of 32: the AVX2 kernel's layout (otherwise the file's order and the model dot)
    perm: bool,
    /// [k]: where im2col element kk goes in a row ([`Conv2::col`], or kk)
    cols: Vec<usize>,
    /// [n_out]
    pub bias: Vec<f32>,
    /// [n_ctx][n_out]: encoder.positional_embedding (f32, ne0 = n_out)
    pub pe: Vec<f32>,
    gelu: Gelu,
}

/// Frames per block: each thread builds the im2col rows of FB frames (FB·3·n_in f32: 144 KiB for tiny) and runs
/// every channel over them, so a weight row is loaded from memory once per block, not once per kernel call.
const FB: usize = 32;
/// The register block: F frames × C channels per kernel call (12 accumulators, 3 weight vectors, 1 frame vector).
const F: usize = 4;
const C: usize = 3;

impl Conv2 {
    pub fn new(m: &Model) -> Result<Conv2, String> {
        let wt = m.tensor("encoder.conv2.weight").ok_or("no encoder.conv2.weight")?;
        let bt = m.tensor("encoder.conv2.bias").ok_or("no encoder.conv2.bias")?;
        let pt = m.tensor("encoder.positional_embedding").ok_or("no encoder.positional_embedding")?;
        if wt.dtype != Dtype::F16 || bt.dtype != Dtype::F32 || pt.dtype != Dtype::F32 || wt.ne[0] != 3 || pt.ne[0] != wt.ne[2] {
            return Err(format!("conv2: expected f16 [3, n, n] weights, f32 bias and f32 [n, n_ctx] positions, got {:?} {:?} {:?}", wt.dtype, wt.ne, pt.ne));
        }
        let w: Vec<u16> = m.tensor_bytes(wt).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let f32s = |b: &[u8]| -> Vec<f32> { b.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() };
        Conv2::from_parts(wt.ne[1] as usize, wt.ne[2] as usize, 2, &w, f32s(m.tensor_bytes(bt)), f32s(m.tensor_bytes(pt)))
    }

    /// From the f16 weights ([n_out][n_in][3], the file's order), the bias and the positions ([n_ctx][n_out]).
    pub fn from_parts(n_in: usize, n_out: usize, stride: usize, w: &[u16], bias: Vec<f32>, pe: Vec<f32>) -> Result<Conv2, String> {
        let k = 3 * n_in;
        if w.len() != n_out * k || bias.len() != n_out || !pe.len().is_multiple_of(n_out.max(1)) || stride == 0 {
            return Err("conv2: weight, bias or positions length".into());
        }
        let perm = k.is_multiple_of(32);
        let rows = if perm { n_out.div_ceil(C) * C } else { n_out };
        let mut ww = vec![0.0f32; rows * k];
        for (c, row) in w.chunks_exact(k).enumerate() {
            for (kk, &h) in row.iter().enumerate() {
                ww[c * k + if perm { Conv2::col(k, kk) } else { kk }] = fp16_to_fp32(h);
            }
        }
        let cols = (0..k).map(|kk| if perm { Conv2::col(k, kk) } else { kk }).collect();
        Ok(Conv2 { n_in, n_out, stride, k, w: ww, perm, cols, bias, pe, gelu: Gelu::new() })
    }

    /// Where element `kk` of a row lives in the kernel's layout: the blocks accumulator j reads (`i + 8j`, i = 0, 32,
    /// ...) made contiguous, so each of the four passes streams k/4 floats per row instead of striding over all k.
    /// A permutation of both operands' columns changes no product and no chain: the bits are the reference's.
    #[inline]
    fn col(k: usize, kk: usize) -> usize {
        (kk % 32 / 8) * (k / 4) + (kk / 32) * 8 + kk % 8
    }

    /// Output frames for an input of `n_len` frames (pad 1, kernel 3): (n_len − 1) / stride + 1.
    pub fn frames_out(&self, n_len: usize) -> usize {
        if n_len == 0 { 0 } else { (n_len - 1) / self.stride + 1 }
    }

    /// conv2 of `x` given as the f16 bits its im2col would make ([`Conv1::run_into_f16`]): the same results as
    /// [`Conv2::run_into`] on the f32 values, from half the bytes.
    pub fn run_into_f16(&self, x: &[u16], n_len: usize, threads: usize, epi: Epilogue, out: &mut [f32]) {
        self.run_input(Input::F16(x), n_len, threads, epi, out);
    }

    /// conv2 of `x` ([n_in][n_len], conv1's GELU output) with `epi`, into a new vector.
    pub fn run(&self, x: &[f32], n_len: usize, threads: usize, epi: Epilogue) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_out * self.frames_out(n_len)];
        self.run_into(x, n_len, threads, epi, &mut out);
        out
    }

    /// conv2 of `x` ([n_in][n_len]) into `out` (n_out × frames_out(n_len); its layout is the epilogue's), which a
    /// caller can keep. No im2col is held: each thread rounds its blocks of FB frames to f16 as it builds them.
    /// Threads split the output frames in whole blocks; every dot is whole in one thread.
    pub fn run_into(&self, x: &[f32], n_len: usize, threads: usize, epi: Epilogue, out: &mut [f32]) {
        self.run_input(Input::F32(x), n_len, threads, epi, out);
    }

    fn run_input(&self, x: Input, n_len: usize, threads: usize, epi: Epilogue, out: &mut [f32]) {
        let n_frames = self.frames_out(n_len);
        assert!(x.len() >= self.n_in * n_len, "conv2: the input is shorter than n_in x n_len");
        assert_eq!(out.len(), self.n_out * n_frames, "conv2: the output is not n_out x frames");
        if epi == Epilogue::Positions {
            assert!(self.pe.len() >= n_frames * self.n_out, "conv2: more frames than positions");
        }
        let blocks = n_frames.div_ceil(FB);
        let threads = threads.clamp(1, blocks.max(1));
        let shared = SharedOut(out.as_mut_ptr());
        std::thread::scope(|s| {
            for ti in 0..threads {
                let (a, b) = (blocks * ti / threads * FB, (blocks * (ti + 1) / threads * FB).min(n_frames));
                let (shared, x) = (&shared, &x);
                let job = move || self.frames(x, n_len, a, b, n_frames, shared, epi);
                if threads == 1 {
                    job();
                } else {
                    s.spawn(job);
                }
            }
        });
    }

    /// Every output channel for frames [t_begin, t_end).
    #[allow(clippy::too_many_arguments)] // the input (x, n_len), the frames (begin, end, total), the output, the epilogue
    fn frames(&self, x: &Input, n_len: usize, t_begin: usize, t_end: usize, n_frames: usize, out: &SharedOut, epi: Epilogue) {
        let (k, n_out, st) = (self.k, self.n_out, self.stride);
        let fast = Kernel::detect();
        // the block's im2col rows [FB][k] (rows past t_end stay 0 and are never stored), its dots [n_out][FB]
        let mut xb = vec![0.0f32; FB * k];
        let mut seg = vec![0.0f32; (st * (FB - 1) + 3).div_ceil(8) * 8];
        let mut dots = vec![0.0f32; n_out * FB];
        let mut z = [0.0f32; FB];
        let mut g = [0.0f32; FB];
        for t0 in (t_begin..t_end).step_by(FB) {
            let nb = FB.min(t_end - t0);
            // row r, element 3·ic + kw = x[ic][st·(t0 + r) + kw − 1], 0 outside [0, n_len), rounded to f16: per input
            // channel, the span of input frames the block reads is copied and rounded once (eight at a time), then
            // spread into the rows. Rows past nb keep what an earlier block left: computed, never stored.
            let span = st * (nb - 1) + 3;
            for ic in 0..self.n_in {
                let f0 = st * t0; // input frame f0 − 1 is seg[0]
                for (i8, chunk) in seg[..span.div_ceil(8) * 8].as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let i0 = 8 * i8;
                    let whole = i0 + f0 >= 1 && i0 + f0 + 7 <= n_len;
                    let at = ic * n_len + i0 + f0; // + q − 1
                    match x {
                        Input::F32(x) => {
                            if whole {
                                chunk.copy_from_slice(&x[at - 1..at + 7]);
                            } else {
                                for (q, v) in chunk.iter_mut().enumerate() {
                                    let f = i0 + q + f0; // input frame f − 1
                                    *v = if f >= 1 && f - 1 < n_len { x[at + q - 1] } else { 0.0 };
                                }
                            }
                            round_f16(chunk);
                        }
                        // already the im2col's f16: widening is exact (vcvtph2ps = the table on every pattern, 0.0.3)
                        Input::F16(x) => {
                            if whole {
                                crate::f16::fp16_to_fp32_row(&x[at - 1..at + 7], &mut chunk[..]);
                            } else {
                                for (q, v) in chunk.iter_mut().enumerate() {
                                    let f = i0 + q + f0;
                                    *v = if f >= 1 && f - 1 < n_len { fp16_to_fp32(x[at + q - 1]) } else { 0.0 };
                                }
                            }
                        }
                    }
                }
                let cols = &self.cols[3 * ic..3 * ic + 3];
                for r in 0..nb {
                    let xr = &mut xb[r * k..(r + 1) * k];
                    let sg = &seg[st * r..st * r + 3];
                    xr[cols[0]] = sg[0];
                    xr[cols[1]] = sg[1];
                    xr[cols[2]] = sg[2];
                }
            }
            let nf = nb.div_ceil(F) * F; // whole register blocks; the rows past nb are zeros, computed and dropped
            let mut c0 = 0;
            while c0 < n_out {
                let nc = C.min(n_out - c0);
                for f0 in (0..nf).step_by(F) {
                    let mut res = [[0.0f32; F]; C];
                    if self.perm {
                        // whole register blocks (the zero padding rows past n_out are computed and dropped)
                        fast.block_perm(&xb[f0 * k..(f0 + F) * k], &self.w[c0 * k..(c0 + C) * k], k, &mut res);
                    } else {
                        fast.block(&xb[f0 * k..(f0 + F) * k], &self.w[c0 * k..(c0 + nc) * k], k, nc, &mut res);
                    }
                    for (c, rc) in res.iter().enumerate().take(nc) {
                        dots[(c0 + c) * FB + f0..(c0 + c) * FB + f0 + F].copy_from_slice(rc);
                    }
                }
                c0 += nc;
            }
            for c in 0..n_out {
                let d = &dots[c * FB..c * FB + nb];
                match epi {
                    Epilogue::Raw => {
                        // SAFETY: row c, columns [t0, t0 + nb) are inside `out` and this thread's frames; see Conv1
                        let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(c * n_frames + t0), nb) };
                        o.copy_from_slice(d);
                    }
                    Epilogue::BiasGelu => {
                        for (zz, &v) in z.iter_mut().zip(d) {
                            *zz = v + self.bias[c];
                        }
                        // SAFETY: as above
                        let o = unsafe { std::slice::from_raw_parts_mut(out.0.add(c * n_frames + t0), nb) };
                        self.gelu.row(&z[..nb], o);
                    }
                    Epilogue::Positions => {
                        for (zz, &v) in z.iter_mut().zip(d) {
                            *zz = v + self.bias[c];
                        }
                        self.gelu.row(&z[..nb], &mut g[..nb]);
                        for (r, &gv) in g[..nb].iter().enumerate() {
                            let t = t0 + r;
                            // SAFETY: element (t, c) of the [n_frames][n_out] output; t is this thread's frame
                            unsafe { *out.0.add(t * n_out + c) = self.pe[t * n_out + c] + gv };
                        }
                    }
                }
            }
        }
    }
}

/// The conv stage: the mel window → conv1 + bias, GELU → conv2 + bias, GELU → + positions, i.e. the encoder's input
/// (`inpL`, [n_ctx][n_state], frame-major), with conv1's output kept in a buffer the caller owns.
pub struct ConvStage {
    pub conv1: Conv1,
    pub conv2: Conv2,
}

impl ConvStage {
    pub fn new(m: &Model) -> Result<ConvStage, String> {
        let s = ConvStage { conv1: Conv1::new(m)?, conv2: Conv2::new(m)? };
        if s.conv1.n_out != s.conv2.n_in {
            return Err("conv1's outputs are not conv2's inputs".into());
        }
        Ok(s)
    }

    /// `n_frames` mel frames from `offset` (2·n_ctx in whisper) → `out` ([frames_out][n_out]); `scratch` holds conv1's
    /// output as the f16 bits conv2's im2col makes of it (resized to n_state × n_frames on first use, then kept).
    #[allow(clippy::too_many_arguments)] // the window (mel, n_len, offset, n_frames), the threads, the two buffers
    pub fn run_into(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize, scratch: &mut Vec<u16>, out: &mut [f32]) {
        scratch.resize(self.conv1.n_out * n_frames, 0);
        self.conv1.run_into_f16(mel, n_len, offset, n_frames, threads, scratch);
        self.conv2.run_into_f16(scratch, n_frames, threads, Epilogue::Positions, out);
    }

    /// The same into a new vector.
    pub fn run(&self, mel: &[f32], n_len: usize, offset: usize, n_frames: usize, threads: usize) -> Vec<f32> {
        let mut scratch = Vec::new();
        let mut out = vec![0.0f32; self.conv2.n_out * self.conv2.frames_out(n_frames)];
        self.run_into(mel, n_len, offset, n_frames, threads, &mut scratch, &mut out);
        out
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

    /// conv1's f16 output = its f32 output through the im2col's conversion, at every edge (and values >= 10, which
    /// GELU passes unrounded).
    #[test]
    fn conv1_f16_out_is_the_im2col_conversion() {
        let mut s = 0xFEED_FACE_1234_0001u64;
        let (n_mel, n_out) = (80usize, 3usize);
        let w: Vec<u16> = (0..n_out * 3 * n_mel).map(|_| (xorshift(&mut s) as u16) & 0xBBFF).collect();
        let c = Conv1::from_parts(n_mel, n_out, &w, vec![0.5, -1.0, 11.0]).unwrap();
        for (n_len, n_frames) in [(50usize, 37usize), (30, 40)] {
            let mel: Vec<f32> = (0..n_mel * n_len).map(|_| (xorshift(&mut s) % 4001) as f32 / 1000.0 - 2.0).collect();
            let want: Vec<u16> = c.run_gelu(&mel, n_len, 0, n_frames, 1).iter().map(|&v| fp32_to_fp16(v)).collect();
            for threads in [1, 3] {
                let mut got = vec![0u16; n_out * n_frames];
                c.run_into_f16(&mel, n_len, 0, n_frames, threads, &mut got);
                assert_eq!(got, want, "n_len {n_len} threads {threads}");
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

    /// conv2's fast path (every epilogue) against im2col + the model dot, on random inputs with every edge: odd and
    /// even lengths, a short last block, more channels than one register block holds and a remainder, 1 and 3 threads;
    /// and the AVX2 4 × 3 block against the model.
    #[test]
    fn conv2_equals_im2col_and_model() {
        let mut s = 0x5EED_C0DE_2222_7777u64;
        for (n_in, n_out) in [(32usize, 7usize), (64, 6), (5, 4)] {
            let k = 3 * n_in;
            let w: Vec<u16> = (0..n_out * k).map(|_| (xorshift(&mut s) as u16) & 0xBBFF).collect();
            let bias: Vec<f32> = (0..n_out).map(|i| i as f32 * 0.25 - 0.5).collect();
            let pe: Vec<f32> = (0..n_out * 64).map(|_| (xorshift(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
            let c = Conv2::from_parts(n_in, n_out, 2, &w, bias.clone(), pe.clone()).unwrap();
            let g = Gelu::new();
            for n_len in [1usize, 2, 7, 64, 77, 128] {
                let x: Vec<f32> = (0..n_in * n_len).map(|_| (xorshift(&mut s) % 4001) as f32 / 1000.0 - 2.0).collect();
                let nf = c.frames_out(n_len);
                let im = im2col_strided_f16(&x, n_in, n_len, 2, nf);
                let mut raw = vec![0f32; n_out * nf];
                for ch in 0..n_out {
                    for t in 0..nf {
                        raw[ch * nf + t] = vec_dot_f16(&im[t * k..(t + 1) * k], &w[ch * k..(ch + 1) * k]);
                    }
                }
                let mut bg = vec![0f32; n_out * nf];
                for ch in 0..n_out {
                    let z: Vec<f32> = raw[ch * nf..(ch + 1) * nf].iter().map(|v| v + bias[ch]).collect();
                    g.row_scalar(&z, &mut bg[ch * nf..(ch + 1) * nf]);
                }
                let mut pos = vec![0f32; n_out * nf];
                for ch in 0..n_out {
                    for t in 0..nf {
                        pos[t * n_out + ch] = pe[t * n_out + ch] + bg[ch * nf + t];
                    }
                }
                let xh: Vec<u16> = x.iter().map(|&v| fp32_to_fp16(v)).collect();
                for threads in [1, 3] {
                    for (epi, want) in [(Epilogue::Raw, &raw), (Epilogue::BiasGelu, &bg), (Epilogue::Positions, &pos)] {
                        let got = c.run(&x, n_len, threads, epi);
                        let bad = got.iter().zip(want.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                        assert_eq!(bad, 0, "n_in {n_in} n_out {n_out} n_len {n_len} threads {threads} {epi:?}");
                        // the same from the f16 bits the im2col makes
                        let mut got = vec![0f32; want.len()];
                        c.run_into_f16(&xh, n_len, threads, epi, &mut got);
                        let bad = got.iter().zip(want.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                        assert_eq!(bad, 0, "f16 input: n_in {n_in} n_out {n_out} n_len {n_len} threads {threads} {epi:?}");
                    }
                }
            }
        }
        // the AVX2 block and the layout's model against the model dot in the file's order, k = 96 and 1152
        for k in [96usize, 1152] {
            let fin = |s: &mut u64| loop {
                let h = xorshift(s) as u16;
                if h & 0x7C00 != 0x7C00 {
                    return fp16_to_fp32(h);
                }
            };
            let x: Vec<f32> = (0..F * k).map(|_| fin(&mut s)).collect();
            let w: Vec<f32> = (0..C * k).map(|_| fin(&mut s)).collect();
            let permute = |v: &[f32]| {
                let mut p = vec![0f32; v.len()];
                for (r, row) in v.chunks_exact(k).enumerate() {
                    for (kk, &e) in row.iter().enumerate() {
                        p[r * k + Conv2::col(k, kk)] = e;
                    }
                }
                p
            };
            let (xp, wp) = (permute(&x), permute(&w));
            for kernel in [Kernel::detect(), Kernel::Model] {
                let mut got = [[0f32; F]; C];
                kernel.block_perm(&xp, &wp, k, &mut got);
                for c in 0..C {
                    for f in 0..F {
                        assert_eq!(got[c][f].to_bits(), dot_f16_model(&x[f * k..(f + 1) * k], &w[c * k..(c + 1) * k]).to_bits(), "k {k}");
                    }
                }
            }
        }
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
