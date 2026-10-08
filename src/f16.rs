// SPDX-License-Identifier: MIT OR Apache-2.0
//! IEEE half precision as ggml-cpu converts it — the f32 ↔ f16 steps every encoder kernel stands on (im2col writes
//! f16, `mul_mat` converts each activation row to f16 for `ggml_vec_dot_f16`, flash attention converts Q, the GELU
//! table is indexed by an f16). Established from the pinned source and the shipped `libggml-cpu.so`
//! (`objdump -d`, testing/oracle), then checked by the oracle on every f16 and every f32 bit pattern:
//!
//! - **Two f32 → f16 conversions coexist in one build.** On an x86-64 host with F16C (this build is `GGML_NATIVE`;
//!   the laptop is Zen+, production Zen 3), `ggml-cpu/simd-mappings.h` defines only `GGML_CPU_COMPUTE_FP32_TO_FP16`
//!   as `_cvtss_sh`; the macro the kernels call, `GGML_CPU_FP32_TO_FP16`, falls through to ggml-impl.h's portable
//!   bit trick `ggml_compute_fp32_to_fp16` (Maratyszcza/FP16). That scalar copy is what im2col, GELU's index and
//!   the GELU table use ([`fp32_to_fp16`]). The row converter `ggml_cpu_fp32_to_fp16` — the type traits'
//!   `from_float` for F16, used by `mul_mat` and flash attention — runs blocks of 8, then 4, through the hardware
//!   `vcvtps2ph` (round to nearest even, imm 0) and only the last `n % 4` through the scalar copy
//!   ([`fp32_to_fp16_row`]).
//! - **They differ only on NaN.** The bit trick returns the canonical `sign | 0x7E00` for every NaN; `vcvtps2ph`
//!   quiets the NaN and keeps the top ten bits of its payload ([`fp32_to_fp16_f16c`]). Every other input — ties,
//!   subnormals, overflow to infinity — rounds to nearest even in both (checked on all 2³² inputs).
//! - **GCC contracted the bit trick in libggml-cpu**: `base = (|f| * 2¹¹²) * 2⁻¹¹⁰` then `bias + base` became
//!   `vmulss` + `vfmadd231ss`. The product by 2⁻¹¹⁰ is exact (a power of two, never subnormal at that point), so the
//!   fused and unfused forms give the same bits; the port keeps the source's two roundings, and the oracle compares
//!   it with both the contracted copy (libggml-cpu) and the uncontracted one (libggml-base's `ggml_fp32_to_fp16`).
//! - **f16 → f32** is exact by construction (every f16 is an f32). ggml-cpu reads it from `ggml_table_f32_f16`,
//!   filled at `ggml_cpu_init` by the portable `ggml_compute_fp16_to_fp32` ([`fp16_to_fp32`]); rows use the
//!   hardware `vcvtph2ps` for blocks of 8 and 4 and the table for the tail. The question is only NaN: the portable
//!   form multiplies by 2⁻¹¹², which quiets a signalling NaN, and the hardware quiets it too — the oracle compares
//!   all 65,536 patterns against the table, the row (hardware) and libggml-base.
//!
//! The scalar functions are pure integer and f32 operations (no intrinsics), so they hold on any target; the rows
//! take the F16C instructions when the CPU has them (`is_x86_feature_detected!`), which the oracle shows are
//! bit-identical to the software model of the same instruction, and otherwise run that model.

/// `ggml_compute_fp16_to_fp32` (ggml-impl.h), operation for operation: what `ggml_table_f32_f16[h]` holds.
#[inline]
pub fn fp16_to_fp32(h: u16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let exp_offset = 0xE0u32 << 23;
    let exp_scale = f32::from_bits(0x0780_0000); // 0x1.0p-112f
    let normalized = f32::from_bits((two_w >> 4) + exp_offset) * exp_scale;
    let magic_mask = 126u32 << 23;
    let magic_bias = 0.5f32;
    let denormalized = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;
    let denormalized_cutoff = 1u32 << 27;
    f32::from_bits(sign | if two_w < denormalized_cutoff { denormalized.to_bits() } else { normalized.to_bits() })
}

/// `ggml_compute_fp32_to_fp16` (ggml-impl.h) — `GGML_CPU_FP32_TO_FP16` on x86-64, the scalar conversion of im2col,
/// GELU and the row tail. Round to nearest even; overflow to ±inf; every NaN to `sign | 0x7E00`.
#[inline]
#[allow(clippy::assign_op_pattern)] // `base = bias + base` is the source's operand order, kept as written
pub fn fp32_to_fp16(f: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x7780_0000); // 0x1.0p+112f
    let scale_to_zero = f32::from_bits(0x0880_0000); // 0x1.0p-110f
    let mut base = (f.abs() * scale_to_inf) * scale_to_zero;
    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let mut bias = shl1_w & 0xFF00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    base = f32::from_bits((bias >> 1) + 0x0780_0000) + base;
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x0000_7C00;
    let mantissa_bits = bits & 0x0000_0FFF;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | if shl1_w > 0xFF00_0000 { 0x7E00 } else { nonsign }) as u16
}

/// What `vcvtps2ph` with imm 0 computes, in integers: round to nearest even, overflow to ±inf, f32 subnormals
/// rounded (not flushed: MXCSR's DAZ is off), and a NaN quieted with the top ten bits of its payload kept.
#[inline]
pub fn fp32_to_fp16_f16c(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let a = x & 0x7FFF_FFFF;
    if a > 0x7F80_0000 {
        return sign | 0x7E00 | ((a >> 13) & 0x3FF) as u16;
    }
    if a >= 0x477F_F000 {
        return sign | 0x7C00; // |f| >= 65520 (the midpoint of 65504 and 2^16) rounds to infinity
    }
    if a >= 0x3880_0000 {
        // a normal f16: rebias the exponent (127 → 15) and round the 13 dropped bits to nearest even; a carry out of
        // the mantissa lands in the exponent, which is the correct next binade
        let r = a - 0x3800_0000;
        return sign | ((r + 0xFFF + ((r >> 13) & 1)) >> 13) as u16;
    }
    if a <= 0x3300_0000 {
        return sign; // |f| <= 2^-25: half the smallest subnormal or less rounds to (even) zero
    }
    // an f16 subnormal: the value in units of 2^-24, rounded to nearest even (a carry to 0x400 is the smallest normal)
    let e = a >> 23;
    let m = (a & 0x7F_FFFF) | 0x80_0000;
    let shift = 126 - e; // 14..=24
    let q = m >> shift;
    let rem = m & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    sign | (q + u32::from(rem > half || (rem == half && q & 1 == 1))) as u16
}

/// The deliberately wrong variant the oracle must reject (`oracle_f16_discriminates_round_half_away`): the same as
/// [`fp32_to_fp16_f16c`] for every input except exact ties, which round away from zero instead of to even.
pub fn fp32_to_fp16_round_half_away(f: f32) -> u16 {
    let x = f.to_bits();
    let a = x & 0x7FFF_FFFF;
    let r = fp32_to_fp16_f16c(f);
    if a > 0x7F80_0000 || a >= 0x477F_F000 {
        return r;
    }
    // (is it exactly halfway, the magnitude truncated to f16 units)
    let (tie, q) = if a >= 0x3880_0000 {
        (a & 0x1FFF == 0x1000, (a - 0x3800_0000) >> 13)
    } else if a > 0x3300_0000 {
        let shift = 126 - (a >> 23);
        let m = (a & 0x7F_FFFF) | 0x80_0000;
        (m & ((1 << shift) - 1) == 1 << (shift - 1), m >> shift)
    } else {
        (a == 0x3300_0000, 0)
    };
    // to even went down exactly when the truncated value was even; away goes up (a carry walks into the exponent)
    if tie && q & 1 == 0 { ((x >> 16) & 0x8000) as u16 | (q + 1) as u16 } else { r }
}

/// `ggml_cpu_fp32_to_fp16(x, y, n)` as this build compiles it (x86-64, F16C, no AVX-512): blocks of 8, then of 4,
/// through `vcvtps2ph` (round to nearest even), and the last `n % 4` values through the scalar [`fp32_to_fp16`].
/// The two differ only on NaN, so a NaN's f16 bits depend on its position in the row — as in the reference.
pub fn fp32_to_fp16_row(x: &[f32], y: &mut [u16]) {
    assert_eq!(x.len(), y.len(), "fp32_to_fp16_row: lengths differ");
    let blocks = x.len() & !3;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        // SAFETY: the CPU has F16C and AVX (checked above)
        unsafe { x86::fp32_to_fp16_blocks(&x[..blocks], &mut y[..blocks]) };
        for (o, &v) in y[blocks..].iter_mut().zip(&x[blocks..]) {
            *o = fp32_to_fp16(v);
        }
        return;
    }
    for (o, &v) in y[..blocks].iter_mut().zip(&x[..blocks]) {
        *o = fp32_to_fp16_f16c(v);
    }
    for (o, &v) in y[blocks..].iter_mut().zip(&x[blocks..]) {
        *o = fp32_to_fp16(v);
    }
}

/// `ggml_cpu_fp16_to_fp32(x, y, n)`: blocks of 8 and 4 through `vcvtph2ps`, the tail through the table, which is
/// [`fp16_to_fp32`]. Widening is exact, so both give the same f32 (the oracle checks all 65,536 patterns).
pub fn fp16_to_fp32_row(x: &[u16], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "fp16_to_fp32_row: lengths differ");
    let blocks = x.len() & !3;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        // SAFETY: the CPU has F16C and AVX (checked above)
        unsafe { x86::fp16_to_fp32_blocks(&x[..blocks], &mut y[..blocks]) };
        done = blocks;
    }
    for (o, &h) in y[done..].iter_mut().zip(&x[done..]) {
        *o = fp16_to_fp32(h);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// `x.len()` a multiple of 4: 8 at a time, then one block of 4 if left, as `ggml_cpu_fp32_to_fp16`'s loops.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn fp32_to_fp16_blocks(x: &[f32], y: &mut [u16]) {
        let n = x.len();
        let mut i = 0;
        // SAFETY (all loads and stores): i + 8 <= n (or i + 4 <= n) and x, y have length n
        unsafe {
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                _mm_storeu_si128(y.as_mut_ptr().add(i).cast(), _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v));
                i += 8;
            }
            if i + 4 <= n {
                let v = _mm_loadu_ps(x.as_ptr().add(i));
                _mm_storel_epi64(y.as_mut_ptr().add(i).cast(), _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v));
            }
        }
    }

    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn fp16_to_fp32_blocks(x: &[u16], y: &mut [f32]) {
        let n = x.len();
        let mut i = 0;
        // SAFETY: as above
        unsafe {
            while i + 8 <= n {
                let h = _mm_loadu_si128(x.as_ptr().add(i).cast());
                _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
                i += 8;
            }
            if i + 4 <= n {
                let h = _mm_loadl_epi64(x.as_ptr().add(i).cast());
                _mm_storeu_ps(y.as_mut_ptr().add(i), _mm_cvtph_ps(h));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact value of an f16, from its fields (independent of both conversions under test).
    fn exact(h: u16) -> f64 {
        let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = ((h >> 10) & 31) as i32;
        let m = (h & 0x3FF) as f64;
        match e {
            0 => s * m * 2f64.powi(-24),
            31 => if m == 0.0 { s * f64::INFINITY } else { f64::NAN },
            _ => s * (1024.0 + m) * 2f64.powi(e - 25),
        }
    }

    #[test]
    fn every_f16_widens_to_its_exact_value_and_back() {
        for h in 0..=u16::MAX {
            let f = fp16_to_fp32(h);
            if (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0 {
                assert!(f.is_nan(), "{h:#06x}");
                // quieted, payload kept in the top bits
                assert_eq!(f.to_bits(), ((h as u32 & 0x8000) << 16) | 0x7FC0_0000 | ((h as u32 & 0x3FF) << 13), "{h:#06x}");
                continue;
            }
            assert_eq!(f as f64, exact(h), "{h:#06x}");
            assert_eq!(fp32_to_fp16(f), h, "{h:#06x} round trip (portable)");
            assert_eq!(fp32_to_fp16_f16c(f), h, "{h:#06x} round trip (F16C model)");
        }
    }

    /// The portable bit trick and the F16C model agree on every non-NaN f32 (sampled here; the oracle runs all 2^32
    /// against the shipped library), and on NaN differ exactly as documented.
    #[test]
    fn portable_and_f16c_agree_except_on_nan() {
        let mut u = 0u32;
        loop {
            let f = f32::from_bits(u);
            let (p, h) = (fp32_to_fp16(f), fp32_to_fp16_f16c(f));
            if f.is_nan() {
                assert_eq!(p, ((u >> 16) & 0x8000) as u16 | 0x7E00);
                assert_eq!(h & 0x7E00, 0x7E00);
            } else {
                assert_eq!(p, h, "{u:#010x}");
            }
            match u.checked_add(251) {
                Some(n) => u = n,
                None => break,
            }
        }
    }

    #[test]
    fn ties_go_to_even_and_the_discriminator_does_not() {
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: even is 1.0 (0x3C00); away is 0x3C01
        let t = f32::from_bits(0x3F80_1000);
        assert_eq!(fp32_to_fp16(t), 0x3C00);
        assert_eq!(fp32_to_fp16_round_half_away(t), 0x3C01);
        // 65520 ties 65504 and 2^16: even is 2^16, which is infinity
        assert_eq!(fp32_to_fp16(65520.0), 0x7C00);
        assert_eq!(fp32_to_fp16(f32::from_bits(0x477F_EFFF)), 0x7BFF);
        // 2^-25 ties 0 and the smallest subnormal: even is 0; away is 0x0001
        assert_eq!(fp32_to_fp16(2f32.powi(-25)), 0);
        assert_eq!(fp32_to_fp16_round_half_away(2f32.powi(-25)), 1);
        assert_eq!(fp32_to_fp16(-3.0 * 2f32.powi(-25)), 0x8002); // 1.5 units: tie, even is 2
        assert_eq!(fp32_to_fp16_round_half_away(-3.0 * 2f32.powi(-25)), 0x8002); // away from zero is 2 as well
        assert_eq!(fp32_to_fp16(5.0 * 2f32.powi(-25)), 0x0002); // 2.5 units: even is 2
        assert_eq!(fp32_to_fp16_round_half_away(5.0 * 2f32.powi(-25)), 0x0003);
    }

    #[test]
    fn rows_match_the_scalars_at_every_length_and_offset() {
        let mut vals: Vec<f32> = (0..1000u32).map(|i| f32::from_bits(i.wrapping_mul(2_654_435_761))).collect();
        vals[3] = f32::from_bits(0x7F80_0001); // a signalling NaN in a block and in the tail
        vals[998] = f32::from_bits(0xFFA0_0000);
        for n in [0, 1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 999, 1000] {
            let x = &vals[1000 - n..];
            let mut y = vec![0u16; n];
            fp32_to_fp16_row(x, &mut y);
            let blocks = n & !3;
            for i in 0..n {
                let want = if i < blocks { fp32_to_fp16_f16c(x[i]) } else { fp32_to_fp16(x[i]) };
                assert_eq!(y[i], want, "n {n} i {i}");
            }
            let mut back = vec![0f32; n];
            fp16_to_fp32_row(&y, &mut back);
            for i in 0..n {
                assert_eq!(back[i].to_bits(), fp16_to_fp32(y[i]).to_bits(), "n {n} i {i}");
            }
        }
    }
}
