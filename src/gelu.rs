// SPDX-License-Identifier: MIT OR Apache-2.0
//! GELU as whisper's encoder computes it: `ggml_gelu` on an f32 tensor, which ggml-cpu evaluates through a lookup
//! table of 65,536 f16 results (`GGML_GELU_FP16` is defined unconditionally in ggml-cpu/vec.h). Established from the
//! pinned source and the shipped `libggml-cpu.so` (`objdump -d`), and checked by the oracle on every table entry and
//! on the op's output:
//!
//! - **The table** (`ggml_table_gelu_f16`, an exported symbol) is filled once in `ggml_cpu_init`: for every 16-bit
//!   pattern i, `f = ggml_compute_fp16_to_fp32(i)`, then `GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f))` — the portable
//!   conversions of [`crate::f16`].
//! - **`ggml_gelu_f32(x) = 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))`**, compiled with
//!   `-march=native` and GCC's default `-ffp-contract=fast`, so the shipped code is not the source's rounding order:
//!   `GELU_COEF_A*x*x + 1.0f` is **one FMA** (`vmulss` A·x, then `vfmadd213ss` x·(A·x) + 1), and the rest is
//!   `(SQRT_2_OVER_PI*x) * that`, glibc's `tanhf` (the symbol libggml-cpu imports), `+ 1.0f`, `(0.5f*x) * that`.
//!   [`gelu_f32`] does exactly that (`mul_add` is the single rounding, in hardware or in glibc's `fmaf`); the
//!   unfused order of the source is kept as [`gelu_f32_unfused`], which the oracle must reject.
//! - **The op** (`ggml_vec_gelu_f32`): `x <= -10` gives `+0.0`, `x >= 10` gives `x`, anything else (NaN included,
//!   both comparisons being false) gives `table_f32_f16[table_gelu_f16[fp32_to_fp16(x)]]`, the index from the
//!   portable scalar conversion (NaN → `sign | 0x7E00`). Rows are split between threads whole, so the thread count
//!   cannot change a value.
//!
//! Faster than the reference, same bits: the table is held already widened to f32 (one lookup instead of two),
//! and on x86-64 with AVX2 + F16C eight lanes go at once — the index from `vcvtps2ph`, which equals the portable
//! conversion on every non-NaN f32 (checked on all 2³²), with NaN lanes given the portable index; then one gather.

use crate::f16::{fp16_to_fp32, fp32_to_fp16};

// glibc libm, the symbol libggml-cpu.so imports for the table (`objdump -T`: tanhf@GLIBC_2.2.5)
extern "C" {
    fn tanhf(x: f32) -> f32;
}

/// `GELU_COEF_A = 0.044715f` (vec.h), as the shipped binary holds it (`.rodata` 0x3d372713).
pub const GELU_COEF_A: f32 = 0.044715;
/// `SQRT_2_OVER_PI = 0.79788456080286535587989211986876f`, rounded to f32 as the shipped binary holds it (0x3f4c422a).
pub const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3F4C_422A);

/// `ggml_gelu_f32` as libggml-cpu.so computes it (FMA-contracted; see the module notes).
#[inline]
pub fn gelu_f32(x: f32) -> f32 {
    let inner = (GELU_COEF_A * x).mul_add(x, 1.0);
    // SAFETY: tanhf is a pure libm function
    let t = unsafe { tanhf((SQRT_2_OVER_PI * x) * inner) };
    (0.5 * x) * (t + 1.0)
}

/// The source's order with every operation rounded (no FMA): what a build without contraction would compute. The
/// discriminator `oracle_gelu_table_discriminates_unfused` requires the oracle to tell it from the shipped table.
pub fn gelu_f32_unfused(x: f32) -> f32 {
    let inner = 1.0 + GELU_COEF_A * x * x;
    // SAFETY: as above
    let t = unsafe { tanhf(SQRT_2_OVER_PI * x * inner) };
    0.5 * x * (1.0 + t)
}

/// The table of `ggml_cpu_init`, entry for entry, from a given scalar GELU.
pub fn table_with(gelu: fn(f32) -> f32) -> Vec<u16> {
    (0..=u16::MAX).map(|i| fp32_to_fp16(gelu(fp16_to_fp32(i)))).collect()
}

/// The GELU op: `ggml_table_gelu_f16` and the same results widened, for the op's single lookup.
pub struct Gelu {
    /// `ggml_table_gelu_f16`, all 65,536 entries
    pub f16: Vec<u16>,
    /// `ggml_table_f32_f16[ggml_table_gelu_f16[i]]`
    wide: Vec<f32>,
}

impl Default for Gelu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gelu {
    pub fn new() -> Self {
        let f16 = table_with(gelu_f32);
        let wide = f16.iter().map(|&h| fp16_to_fp32(h)).collect();
        Gelu { f16, wide }
    }

    /// `ggml_vec_gelu_f32` with `GGML_GELU_FP16`, for one row (or any run of values: each is independent).
    pub fn row(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), y.len(), "gelu row: lengths differ");
        let mut done = 0;
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: the CPU has AVX2 and F16C (checked above)
            done = unsafe { self.row_avx2(x, y) };
        }
        self.row_scalar(&x[done..], &mut y[done..]);
    }

    /// The reference's loop, one value at a time.
    pub fn row_scalar(&self, x: &[f32], y: &mut [f32]) {
        for (o, &v) in y.iter_mut().zip(x) {
            *o = if v <= -10.0 {
                0.0
            } else if v >= 10.0 {
                v
            } else {
                self.wide[fp32_to_fp16(v) as usize]
            };
        }
    }

    /// Eight at a time; returns how many values it did (a multiple of 8). Same bits as [`Gelu::row_scalar`].
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn row_avx2(&self, x: &[f32], y: &mut [f32]) -> usize {
        use std::arch::x86_64::*;
        let n = x.len() & !7;
        let lo = _mm256_set1_ps(-10.0);
        let hi = _mm256_set1_ps(10.0);
        let sign = _mm256_set1_epi32(0x8000);
        let qnan = _mm256_set1_epi32(0x7E00);
        let t = self.wide.as_ptr();
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= x.len() == y.len(); every gather index is < 65,536 == self.wide.len()
            unsafe {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let mut idx = _mm256_cvtepu16_epi32(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v));
                // NaN lanes: the portable conversion's index, sign | 0x7E00
                let nan = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
                let nan_idx = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32::<16>(_mm256_castps_si256(v)), sign), qnan);
                idx = _mm256_blendv_epi8(idx, nan_idx, nan);
                let g = _mm256_i32gather_ps::<4>(t, idx);
                let r = _mm256_blendv_ps(g, _mm256_setzero_ps(), _mm256_cmp_ps::<_CMP_LE_OQ>(v, lo));
                let r = _mm256_blendv_ps(r, v, _mm256_cmp_ps::<_CMP_GE_OQ>(v, hi));
                _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
            }
            i += 8;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_the_binarys() {
        assert_eq!(GELU_COEF_A.to_bits(), 0x3D37_2713);
        assert_eq!(SQRT_2_OVER_PI, 0.797_884_560_802_865_4_f64 as f32); // the source's literal, rounded
    }

    #[test]
    fn the_op_clamps_and_looks_up() {
        let g = Gelu::new();
        let x = [-10.0f32, -9.99, -1e30, 10.0, 9.99, 1e30, 0.0, -0.0, 1.0, f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let mut y = [0f32; 13];
        g.row_scalar(&x, &mut y);
        assert_eq!(y[0].to_bits(), 0);
        assert_eq!(y[2].to_bits(), 0);
        assert_eq!(y[3], 10.0);
        assert_eq!(y[5], 1e30);
        assert_eq!(y[11], f32::INFINITY);
        assert_eq!(y[12].to_bits(), 0);
        assert_eq!(y[1].to_bits(), fp16_to_fp32(g.f16[fp32_to_fp16(-9.99) as usize]).to_bits());
        assert!((y[8] - 0.8412).abs() < 1e-3, "gelu(1) = {}", y[8]);
    }

    /// The vector path against the scalar loop on every f16 value, their neighbours, NaNs and the clamps, at every
    /// alignment of the 8-lane blocks.
    #[test]
    fn vector_and_scalar_give_the_same_bits() {
        let g = Gelu::new();
        let mut x: Vec<f32> = Vec::new();
        for h in 0..=u16::MAX {
            let f = fp16_to_fp32(h);
            x.push(f);
            x.push(f32::from_bits(f.to_bits().wrapping_add(1)));
            x.push(f32::from_bits(f.to_bits().wrapping_sub(1)));
        }
        x.extend([f32::from_bits(0x7F80_0001), f32::from_bits(0xFFC1_2345), -10.0, 10.0, f32::from_bits(0xC11F_FFFF)]);
        for skip in 0..8 {
            let xs = &x[skip..];
            let (mut a, mut b) = (vec![0f32; xs.len()], vec![0f32; xs.len()]);
            g.row(xs, &mut a);
            g.row_scalar(xs, &mut b);
            for i in 0..xs.len() {
                assert_eq!(a[i].to_bits(), b[i].to_bits(), "skip {skip} i {i} x {:#010x}", xs[i].to_bits());
            }
        }
    }
}
