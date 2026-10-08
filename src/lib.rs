// SPDX-License-Identifier: MIT OR Apache-2.0
//! voaice.rs — speech-to-text in zero-dependency Rust, built the way bankml was built against llama.cpp:
//! exact first, fast second. Every stage reproduces the compiled output of the pinned whisper.cpp (upstream/PIN)
//! bit for bit, checked by an oracle that runs the shipped library on the same input (testing/oracle).
//!
//! 0.0.8 holds the first stages: the model loader with its sha256 guard ([`model`]), the WAV input ([`wav`]) and the
//! log-mel front end ([`mel`], allocation-free and threaded since 0.0.2, still 0 ULP); the first encoder kernels
//! (0.0.3): f32 ↔ f16 as ggml-cpu converts ([`f16`]) and GELU with ggml's f16 table ([`gelu`]), bit-exact on every
//! input; the streaming Ogg/Opus reader (0.0.4, [`ogg`]): pages, CRC, packets, headers and the exact duration, one page
//! in memory, checked against opus-tools 0.2 on production; any WAV to whisper's 16 kHz mono f32 (0.0.5, [`resample`]):
//! dr_wav's conversions, miniaudio's mixdown and its linear resampler with the order-4 low-pass, bit for bit as
//! whisper-cli's `read_audio_data` reads the file, streamed; the encoder's first convolution (0.0.6, [`conv`]): im2col to
//! f16, `ggml_vec_dot_f16` in the AVX build's float order, + bias and GELU, bit-exact against the conv graph's own
//! nodes and faster; conv2, its bias and GELU (`embd_conv`) and the positional embedding (0.0.7, [`conv`]) — the
//! encoder's input, bit-exact against both schedulers' nodes; the encoder's layer norms (0.0.8, [`norm`]): `ggml_norm`'s
//! in-order double sum, cvar's 8-lane pairing and `1/sqrtf(var + eps)`, then `· w + b` as two nodes — all nine
//! (each block's attn_ln and mlp_ln, ln_post) bit-exact against the encoder graph's own nodes; plus [`measure`] (CPU time and peak RSS from `/proc`,
//! for the gate). [`vclone`] holds voaice's voice identities: the vprint, byte-identical to cryptoAGI/voaice's
//! vprint.py, and the hash-chained forge log (docs/VCLONE.md). The rest of the encoder (the matrix products,
//! attention and the MLP), and the decoder, are not here yet (TODO.md).

pub mod conv;
pub mod f16;
pub mod gelu;
pub mod json;
pub mod measure;
pub mod mel;
pub mod norm;
pub mod model;
pub mod ogg;
pub mod resample;
pub mod sha256;
pub mod sha512;
pub mod vclone;
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
