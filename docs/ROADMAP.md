# Roadmap — the listening half, then the speaking half

voaice.rs grows in stages, each bit-exact against a pinned reference before it is fast. The detailed engineering
plan for the next stage is in [TODO.md](../TODO.md); this page is the shape of the whole.

## The listening half (whisper.cpp → voaice.rs) — counting by ten

voaice.rs moves **one 0.0.1 increment at a time**, and every tenth increment is a milestone: 0.0.1 … 0.0.9 build
v0.1.0, 0.1.1 … 0.1.9 build v0.2.0, and so on to v1.0.0. Each increment is exact against its oracle before it is
measured, and the gate measures **from processing to output**: wall time, CPU seconds, heap and RSS against the
reference in the same run (testing/results/<version>.txt). The operator's focus is efficiency end to end — the least
CPU and memory from an `.opus` file in to text out, and later from text in to the smallest `.opus` out.

### The first decade: 0.0.1 → v0.1.0 (the encoder, bit-exact)

| version | increment | oracle (all bit patterns) |
|---|---|---|
| **0.0.1** ✓ | the oracle harness, the model loader + sha256 pin, the log-mel front end | 167/167 tensors, 51,864/51,864 tokens, mel 2,316,640/2,316,640 at 0 ULP |
| **0.0.2** ✓ | **the mel, optimized, still 0 ULP**: allocation-free per frame (a `MelPlan` of the invariants, the FFT recursion unrolled in place, no 30-s padded copy), SIMD across independent lanes only, threads by frames; efficiency measured (wall, CPU, heap, RSS) | mel 0 ULP; 2/3/4/8 threads = 1 thread on all 8 inputs; fused FFT still rejected; **6.1× faster than 0.0.1, 6.8× the reference at 1 thread, one third of its heap** |
| **0.0.3** ✓ | **f32 ↔ f16 and the GELU table, as ggml-cpu computes them** (`src/f16.rs`, `src/gelu.rs`): the portable bit trick (`GGML_CPU_FP32_TO_FP16` on an F16C build — im2col, GELU), the F16C row `ggml_cpu_fp32_to_fp16` (blocks of 8 and 4 through `vcvtps2ph`, the tail through the bit trick — mul_mat and flash attention), f16 → f32 (the table and `vcvtph2ps`); `ggml_table_gelu_f16` built as `ggml_cpu_init` builds it (GCC fused `A·x·x + 1` into one FMA), and the op with its ±10 clamps — then faster: one f32 lookup instead of two, eight lanes with a gather | **65,536/65,536** f16 → f32 four ways; **all 2³² f32 → f16** three ways (the boundary set of 1,429,656 value by value, the rest by per-chunk digest); **65,536/65,536** GELU table entries; the GELU op on 1,495,192 values and on all 2³²; discriminators: round-half-away rejected, the unfused GELU rejected (1 entry); then measured: the GELU op **4.31× the reference** (1,536 × 1,500, one thread; 2.96× in an earlier run of the same gate — the reference's side moved most), the table build 1.27× (the reference also builds tables voaice does not need), the f16 rows at parity (the same instruction) |
| **0.0.4** ✓ | **a streaming Ogg/Opus container reader** (`src/ogg.rs`): pages, the Ogg CRC-32 (sliced by 8), lacing and continuation, `OpusHead` / `OpusTags`, granule positions and pre-skip → the exact duration in samples, without holding the file (one page buffer, packets as slices); the page format checked against [streamair](../streamair/)'s writer by round trips | opus-tools 0.2 (`opusinfo`, `opusdec`), libopus 1.4 and libogg 1.3.5 **on production** (not on the dev laptop), recorded with the files: **35 / 35** files on 18 checks each — the duration equals opusdec's sample count on every one, every page's granule and every packet's sample count equal libogg's and libopus's; **21 / 21** adversarial files refused by name; discriminators: pre-skip added (34 / 35 wrong), no end trim (34 / 35), the code-3 count ignored (3 / 35), zlib's CRC (0 / 427 pages verify) — all caught |
| **0.0.5** ✓ | **the audio reader whisper-cli really runs** (`src/resample.rs`; the row as first written guessed at it — the source says): `read_audio_data` → miniaudio 0.11.24's decoder asking its WAV backend for f32, so **dr_wav's** conversions (u8, s16, s24, s32, f32), not `ma_pcm_*`; the mixdown is miniaudio's **`mono_out` average** `(0 + L + R) / 2` (no channel weights); then the **linear resampler** on the gcd-reduced rates with its **order-4 low-pass** (two Butterworth biquads, transposed direct form II, coefficients from glibc `sin` in double, rounded to f32) — on the input when downsampling, on the output when upsampling; the length from `ma_calculate_frame_count_after_resampling`, which can promise one frame more than is made (the tail stays 0.0); 16 kHz mono untouched. Compiled `-O3` with no `-march`: no FMA. Streamed: any chunking gives the same bits | whisper-cli's own `libcommon.a` (`resample_oracle`): **55 / 55** files, **1,955,875** samples bit-identical — 8 / 16 / 22.05 / 24 / 32 / 44.1 / 48 kHz, mono and stereo, s16 and f32, plus u8 / s24 / s32, 6-channel EXTENSIBLE, f32 beyond full scale and subnormal, 1–7-frame inputs, a LIST chunk, a truncated file, JFK at 48 kHz; random chunking 55 / 55; discriminators: low-pass order 2 or 6 (45 / 50 resampled files), mixdown L + R or L alone (8 / 8), the length without the promised frame (3 / 55) — all caught |
| **0.0.6** ✓ | **`ggml_vec_dot_f16` brought forward from 0.0.9, then encoder conv1** — conv1 is `im2col` to f16 + `mul_mat` against the f16 weights, and every one of its 1,152,000 outputs *is* one `ggml_vec_dot_f16(240)`, so it could not be exact without the kernel: 4 accumulators × 8 lanes with FMA (exact here: a product of two halves is exact in f32), the pairwise `GGML_F32x8_REDUCE`, the f32 widened to double and the 16-element tail added in double (`src/conv.rs`); conv1's bias and GELU nodes came with it (one add; 0.0.3's GELU). Then faster with the bits unchanged: no im2col (tiles of 8 frames built on the stack, rounded through F16C), eight dots per weight load, the reduction and the double tail for eight frames at once, threads by frames | the conv graph's own nodes, read through **ggml's scheduler eval callback** on `whisper_state::sched_conv`: IM2COL **5,760,000 / 5,760,000** f16 values, MUL_MAT, + bias and GELU **0 differ** on all 8 inputs at 1 and 4 threads (55,296,000 values); the kernel through `ggml_get_type_traits_cpu`: **1,436 / 1,436** dots on real rows of all 70 f16 tensors, lengths 1–300, random patterns; discriminators: one accumulator, the tail in f32, the accumulators reduced in sequence, im2col without its f16 rounding — all caught; then measured against the reference's own graph (CHANGELOG) |
| 0.0.7 | **conv2** (stride 2 → 1500 frames; its dots are n = 1,152 = 36 × 32, so no tail) with its bias and GELU nodes, + the positional embedding add | the conv graph's remaining nodes and `embd_conv` |
| 0.0.8 | **layer norm**: `ggml_compute_forward_norm_f32`'s mean and variance, in its summation order and precision, eps 1e-5, then `* w + b` | each block's norm node |
| 0.0.9 | **the matrix products on activations**: Q, K, V, the out projection and the MLP, where `mul_mat`'s f32 activation rows are first converted to f16 by `from_float` (`ggml_cpu_fp32_to_fp16`, 0.0.3 — the threads split each row's conversion by element ranges, which can move a NaN between the F16C blocks and the scalar tail) and then dotted by 0.0.6's kernel | every `mul_mat` node of block 0, through the encoder scheduler's eval callback |
| **v0.1.0** | **MILESTONE — the whole encoder bit-exact**: 4 blocks of norm → Q, K, V → **flash attention** (`ggml_flash_attn_ext`: the f16 KV cache, the online softmax and its exp, V accumulation) → out proj → MLP, then `ln_post` | `embd_enc` via `whisper_encode_with_state`, all 8 inputs, at a stated thread count |

### The next milestones (sketch; each built by its own ten increments)

| milestone | what it delivers | oracle |
|---|---|---|
| v0.2.0 | the decoder: cross-attention K/V, the f16 self-attention cache, token + positional embeddings, logits | `whisper_get_logits_from_state`: all 51,864 logits per step, bit for bit |
| v0.3.0 | the full transcript: greedy loop and logit filters, timestamp rules, 30-s windowing and seek, token timestamps, `p` — **at a stated thread count** (the reference's own transcript depends on it) | `whisper_full`'s token ids, `t0`/`t1` and the bits of `p` |
| v0.4.0 | `voaice transcribe --json` in whisper-cli's shape, a library entry point, and **`.opus` in, end to end**: 0.0.4's reader, a decoder, 0.0.5's resampler, the mel, encoder, decoder — streamed, without a WAV on disk | the same as v0.3.0, through the CLI, from `.opus` files |
| v0.5.0 | speed: SIMD kernels chosen at run time, threads where they pay, the KV cache without per-step copies, fused conv + GELU — bits unchanged | the oracles green, then CPU-seconds per audio-second against whisper-cli |
| v0.6.0 – v0.9.0 | `base.en` pinned and proven; the ggml quantized formats production may switch to (q5_0, q8_0); production's own native Zen 3 ggml-cpu build as the oracle; portability (glibc's `sincosf` / `cosf` / `log10` ported in-crate, proven over every input the mel can give them) | each on production's library |
| **v1.0.0** | **production parity**: voaice's speech-to-text runs on voaice.rs; whisper.cpp is needed only as the oracle | all of the above, on production |

### The Opus efficiency thread

Efficiency is measured, never assumed, at both ends of the voice:

- **In (listening):** an `.opus` file should reach the mel **streamed**, page by page, with memory bounded by a page
  and a mel window, not by the file — 0.0.4's reader is built that way from the start, and v0.4.0 measures CPU and
  peak memory per audio-second from `.opus` in to text out. miniaudio has no Opus decoder, so today an `.opus` input
  reaches whisper-cli only after a decode step; that step is what voaice.rs folds in.
- **Out (speaking):** rage stores **24 kb/s mono Opus**. The speaking half's output goes through
  [streamair](../streamair/) (libopus 1.4 is its oracle), and the encoder's settings — frame size (2.5–60 ms),
  complexity (0–10), VBR or CBR, the application mode — are chosen by **measured CPU-seconds per audio-second and
  bytes per second at a stated quality**, per setting, on the production CPU, not by defaults or folklore.

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
