# Roadmap — the listening half, then the speaking half

voaice.rs grows in stages, each bit-exact against a pinned reference before it is fast. The detailed engineering
plan for the next stage is in [TODO.md](../TODO.md); this page is the shape of the whole.

## The listening half (whisper.cpp → voaice.rs)

| version | stage | oracle |
|---|---|---|
| **0.0.1** ✓ | the model loader, the sha256 pin, the log-mel front end | 167/167 tensors, 51,864/51,864 tokens, mel 0 ULP |
| 0.1 | the encoder: two convolutions (f16 im2col), GELU (ggml's f16 table, an exported symbol), layer norm, f16 matrix products in AVX2 lane order, flash attention | every intermediate through ggml's scheduler callback; the final output via `whisper_encode_with_state` |
| 0.2 | the decoder and greedy sampling | `whisper_get_logits_from_state`, logits bit for bit |
| 0.3 | the full transcript: tokens, timestamps, probabilities, at a stated thread count | `whisper_full`'s token ids, `t0`/`t1` and `p` bits |
| 0.4 | `voaice transcribe --json` in whisper-cli's JSON shape, and a library entry point — the seam the call sites in [ARCHITECTURE.md](ARCHITECTURE.md) switch on | the same as 0.3, through the CLI |
| 0.5 | speed: SIMD kernels, threads, the KV cache — always with the oracle green | unchanged bits, then timing against whisper.cpp |
| 1.0 | production parity: whisper.cpp needed only as the oracle; `base.en` pinned; matched against production's own native ggml-cpu build | all of the above, on production's library |

## The speaking half — inspired by Kitten TTS

voaice already speaks through piper and audio.cpp. The question for voaice.rs is what a **zero-dependency Rust
speaker** should look like, and [KittenML](https://github.com/KittenML/KittenTTS)'s two generations sketch the two
ends of it.

**Kitten TTS v1** — [kitten-tts-nano-0.1](https://huggingface.co/KittenML/kitten-tts-nano-0.1) and
[0.2](https://huggingface.co/KittenML/kitten-tts-nano-0.2), **Apache-2.0**: about **15 million parameters in under
25 MB**, one ONNX file plus a small voice table, CPU-only, 24 kHz, "works literally everywhere". It is the shape the
first Rust speaker should take: a model small enough to read whole, a runtime with nothing to install, and a voice
table instead of a voice zoo. Because the licence is open, an open model of this size is a candidate for the
speaking half's first oracle — the reference's own runtime (onnxruntime on the CPU) run on the same text, compared
sample by sample, exactly as the listening half is compared with whisper.cpp.

**Kitten TTS 2** — [kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2): a **1.7B speech language model**
that writes S3 codec tokens for a 24 kHz vocoder, clones a voice **in context from 5–30 seconds** of one speaker,
carries **47 voices** (nine of them named for a language, because the voice carries the accent), and ships a **C++
runtime with a ternary GGUF** build (`model-tq2_1.gguf`). Two ideas carry over: ternary weights, which are bankml's
own ground — [bankml](https://github.com/cryptoAGI/bankml) runs ternary GGUF bit-exact against llama.cpp and faster —
and cloning from a short reference, which is voaice's vCLONE lane. **Its licence is not open**: the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md) allows research,
non-commercial and limited commercial use with registration. voaice.rs takes inspiration from its design and
copies no code and no weights; any Kitten TTS 2 work would be measured against it, never shipped with it.

KittenML also publishes ASR models ([kitten-asr-tiny](https://huggingface.co/KittenML/kitten-asr-tiny),
[kitten-asr-small-enhanced](https://huggingface.co/KittenML/kitten-asr-small-enhanced)); their cards state no
licence, so they are a point of comparison for the listening half, not a dependency.

The voices voaice.rs could speak in are already gathered: [PYTHAI/voaice](https://huggingface.co/PYTHAI/voaice)
holds the 70 open-licensed Piper voices, each with its licence and card.
