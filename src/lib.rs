// SPDX-License-Identifier: MIT OR Apache-2.0
//! voaice.rs — speech-to-text in zero-dependency Rust, built the way bankml was built against llama.cpp:
//! exact first, fast second. Every stage reproduces the compiled output of the pinned whisper.cpp (upstream/PIN)
//! bit for bit, checked by an oracle that runs the shipped library on the same input (testing/oracle).
//!
//! v0.1.0 holds the whole encoder, bit for bit what the pinned whisper.cpp's `whisper_encode_with_state` computes, and
//! the stages it is built from: the model loader with its sha256 guard ([`model`]), the WAV input ([`wav`]) and the
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
//! (each block's attn_ln and mlp_ln, ln_post) bit-exact against the encoder graph's own nodes; the matrix products on
//! activations (0.0.9, [`matmul`]): `mul_mat`'s f32 → f16 `from_float` split by thread, then the f16 dot — every
//! block's Q, K, V and their f16 copies, the out projection, the MLP, their biases, GELU and residuals, bit-exact
//! against every such node and faster; flash attention (v0.1.0, [`attention`]): ggml's tiled kernel — Q in f32, the f16
//! `kv_pad` cache with its 36 zero rows attended, f32 FMA-chain scores, the online softmax with ggml's own 8-lane
//! `ggml_v_expf` for the probabilities and glibc's `expf` for the rescale, the output accumulated in f32 — and the
//! encoder whole ([`encoder`]): the mel → conv stage → four blocks → ln_post = `embd_enc`, bit-exact against every node
//! of whisper's encoder graph and against `embd_enc` itself, about four times the reference's speed at one thread; plus
//! [`measure`] (CPU time and peak RSS from `/proc`, for the gate). [`vclone`] holds voaice's voice identities: the
//! vprint, byte-identical to cryptoAGI/voaice's vprint.py, and the hash-chained forge log (docs/VCLONE.md). The decoder
//! (v0.2.0) is not here yet (TODO.md, docs/ROADMAP.md).

pub mod attention;
pub mod conv;
pub mod encoder;
pub mod f16;
pub mod gelu;
pub mod json;
pub mod matmul;
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
