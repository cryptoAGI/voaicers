// SPDX-License-Identifier: MIT OR Apache-2.0
//! voaice.rs — speech-to-text in zero-dependency Rust, built the way bankml was built against llama.cpp:
//! exact first, fast second. Every stage reproduces the compiled output of the pinned whisper.cpp (upstream/PIN)
//! bit for bit, checked by an oracle that runs the shipped library on the same input (testing/oracle).
//!
//! 0.0.2 holds the first stages: the model loader with its sha256 guard ([`model`]), the WAV input ([`wav`]) and the
//! log-mel front end ([`mel`], allocation-free and threaded since 0.0.2, still 0 ULP), plus [`measure`] (CPU time and
//! peak RSS from `/proc`, for the gate). The encoder and decoder are not here yet (TODO.md).

pub mod measure;
pub mod mel;
pub mod model;
pub mod sha256;
pub mod wav;

/// Distance in units in the last place between two f32 values (0 = the same bits, or +0 against -0).
pub fn ulp_distance(a: f32, b: f32) -> u64 {
    fn ordered(x: f32) -> i64 {
        let u = x.to_bits() as i32 as i64;
        if u < 0 {
            i64::from(i32::MIN) - u
        } else {
            u
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

#[cfg(test)]
mod tests {
    #[test]
    fn ulp() {
        assert_eq!(super::ulp_distance(1.0, 1.0), 0);
        assert_eq!(super::ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 3)), 3);
        assert_eq!(super::ulp_distance(f32::from_bits(1), -f32::from_bits(1)), 2);
        assert_eq!(super::ulp_distance(0.0, -0.0), 0);
    }
}
