# Architecture — modules, the oracle contract, and how voaice.rs grows

voaice.rs is built to be **extended one module at a time without weakening what is already proven**. Every module
has one job, a public surface small enough to read, and an oracle that compares it with the reference's compiled
library. A new stage is added beside the old ones, never inside them, and it ships only when its own oracle passes
in the release gate.

## The modules (0.0.3)

| module | job | public surface | proven by |
|---|---|---|---|
| `src/sha256.rs` | FIPS 180-4 SHA-256, no crates | `Sha256`, `digest()`, `hex()` | FIPS test vectors |
| `src/model.rs` | the ggml whisper model file: hparams, mel filterbank, vocab and special tokens, every tensor's name/type/shape/offset; the sha256 **pin** that refuses an unknown file | `Model`, `Hparams`, `Pin`, `Tensor`, `Dtype`, `Specials`, `expected_tensors()` | `oracle_model_hparams_tensors_vocab_filters`, `guard_refuses_a_modified_model` |
| `src/wav.rs` | 16-bit PCM mono 16 kHz WAV → f32, refusing anything else (no silent resampling) | `read()`, `parse()` | `oracle_pcm_input_identical` |
| `src/mel.rs` | whisper's log-mel front end, in whisper.cpp's own arithmetic order (its radix-2 FFT, mixed f32/f64 band sum, libm symbols); since 0.0.2 allocation-free per frame (a `MelPlan` holds the invariants), SIMD across independent lanes only, threaded by frames | `Tables`, `MelPlan`, `Mel`, `log_mel_spectrogram()`, `log_mel_spectrogram_threads()` | `oracle_mel_bit_exact` (0 ULP), `oracle_mel_threads_bit_identical`, `oracle_mel_discriminates_fused_fft` |
| `src/f16.rs` | (0.0.3) f32 ↔ f16 exactly as ggml-cpu converts on an F16C build: the portable bit trick (`GGML_CPU_FP32_TO_FP16` — im2col, GELU, row tails), the `vcvtps2ph` row of `ggml_cpu_fp32_to_fp16` (mul_mat, flash attention) and a software model of it; f16 → f32 (the table, `vcvtph2ps`). They differ only on NaN, and the rows keep the reference's NaN bits by position | `fp16_to_fp32()`, `fp32_to_fp16()`, `fp32_to_fp16_f16c()`, `fp32_to_fp16_row()`, `fp16_to_fp32_row()` | `oracle_f16_to_f32_all_65536`, `oracle_f32_to_f16_boundary_set`, `oracle_f32_to_f16_every_pattern` (all 2³²), `oracle_f16_discriminates_round_half_away` |
| `src/gelu.rs` | (0.0.3) `ggml_gelu` as the encoder runs it: `ggml_table_gelu_f16` built as `ggml_cpu_init` builds it (glibc `tanhf`, GCC's FMA in `A·x·x + 1`), the op's ±10 clamps and f16 index; held widened to f32 for one lookup, eight lanes at once with AVX2 + F16C | `Gelu` (`new`, `row`, `row_scalar`, `f16`), `gelu_f32()`, `table_with()` | `oracle_gelu_table_all_65536`, `oracle_gelu_op` (all 2³²), `oracle_gelu_table_discriminates_unfused` |
| `src/measure.rs` | CPU seconds, RSS and its peak from `/proc` (no libc binding), and the `bench` loop the gate uses | `cpu_seconds()`, `rss_kb()`, `peak_rss_kb()`, `reset_peak_rss()`, `bench()` | unit test; its numbers are only read after the oracles pass |
| `src/lib.rs` | the crate root, and `ulp_distance()` every oracle reports in | `ulp_distance()` | — |
| `src/main.rs` | the CLI: `voaice info · mel · bench-mel · bench-f16 · version`; a counting allocator for `bench-mel`'s heap peak | — | the gate runs it |

The pattern each module follows is the one bankml uses: **a pure function of its inputs, the same float operations
in the same order as the reference, and no hidden state.** That is what makes a module testable alone, and what
makes it safe to replace a module's internals later for speed: the oracle still has to say 0 ULP.

## Adding a stage — the recipe

The next stages are the encoder, the decoder, the tokenizer and timestamps (see [TODO.md](../TODO.md)). Each one
goes in the same way:

1. **Find the reference's arithmetic, not the paper's.** Read the pinned source *and* the shipped binary
   (`objdump`): which ops, which precision, which order, fused or not. Write it down in the module's header.
2. **Give the oracle a way to see it.** If whisper.cpp's public API exposes the value, use it; if not, extend the
   layout probe (`testing/oracle/layout_probe.cpp`) or hook ggml's scheduler
   (`ggml_backend_sched_set_eval_callback`) — and make the oracle *self-check* every probed location against a
   public getter before it trusts it.
3. **Write the module** as a new file (`src/encoder.rs`, `src/decoder.rs`, …) with a pure public function.
4. **Add its oracle** to `tests/oracle.rs` (bit patterns, ULP distance, a count of matches over all inputs) **and a
   discriminator**: a deliberately wrong variant (fused, reordered, wider accumulator) that the oracle must reject.
   An oracle that cannot fail proves nothing.
5. **Wire it into `testing/release_gate.sh`.** The gate records the result in `testing/results/<version>.txt` and
   measures speed only after every oracle passes.
6. **Only then optimise**, and keep the oracle green: integer sums may be reordered (exact in any order); float
   order may not. What 0.0.2's mel showed is allowed, each checked by the oracle afterwards:
   - memory and index work, freely: preallocated scratch, gathered tables, strides instead of copies;
   - SIMD **across independent accumulators only** — one lane per output, each lane's own sum in the reference's
     order — never a horizontal reduction that reassociates one sum; mul and add, not FMA (Rust never contracts);
   - skipping an operation that is an exact identity (adding +0 to a sum), with a guard for the values where it is
     not (a non-finite operand makes `x * 0` NaN, so that case takes the full path);
   - threads that split *outputs*, each output computed whole by one thread.
   And one trap: a C++ `std::max(x, c)` is `x < c ? c : x`, which keeps a NaN `x`; Rust's `f64::max` drops it.

## Where voaice uses speech-to-text today — the integration points

voaice.rs exists to replace the whisper.cpp binary these call, one call site at a time, with output that is
bit-identical (at a stated thread count) and a library API instead of a subprocess.

| where | what it asks whisper.cpp for | source |
|---|---|---|
| the STT engine table | transcripts from `whisper-cli` / `whisper-cpp` / `main`, the torch-free ASR lane | [`src/stt/STT.js`](https://github.com/Professor-Codephreak/voaice/blob/main/src/stt/STT.js) in Professor-Codephreak/voaice |
| the Python speech bridge | the same lane from Python | [`src/python_speech.js`](https://github.com/Professor-Codephreak/voaice/blob/main/src/python_speech.js), [`python/voaice_speech.py`](https://github.com/Professor-Codephreak/voaice/blob/main/python/voaice_speech.py) |
| **visemes** — lip-sync from what is said | word timings from `whisper-cli -ojf` (token offsets in ms), mapped through espeak-ng phonemes to mouth shapes | `src/visemes.js` (voaice v3.5, mindX production; not yet on GitHub) |
| the word-cloud **re-hearing gate** | a clip is kept only if whisper, hearing it alone, hears its words | `src/wordcloud/verify.js` (v3.5, production) |
| the audio.cpp engine | whisper.cpp at `/opt/whisper.cpp` as the host ASR fallback | `src/engines/audiocpp.js` (v3.5, production) |
| the render store's **listening test** | `opusdec` → `whisper-cli` on every rendered article, to prove the audio is the words (and catch a name said wrong) | mindX operations; see [cryptoAGI/voaice PRONUNCIATION.md](https://github.com/cryptoAGI/voaice/blob/main/PRONUNCIATION.md#the-listening-test) |

The planned seam is the same for all of them: `voaice transcribe <model> <wav> [--json]` emitting whisper-cli's JSON
shape (tokens, `t0`/`t1`, `p`), so a call site switches by changing one binary path, and a Rust/FFI library entry
point for the services that would rather not spawn a process.

## The voaice family

| repository | what it is |
|---|---|
| [cryptoAGI/voaicers](https://github.com/cryptoAGI/voaicers) | this — voaice.rs, the listening half in Rust |
| [cryptoAGI/voaice](https://github.com/cryptoAGI/voaice) | what a voice *is*: `.voaice` identity cards, the 18-decimal vprint (Python and the browser agree), the pronunciation table every engine speaks through |
| [Professor-Codephreak/voaice](https://github.com/Professor-Codephreak/voaice) | the voice *stack*: in-house DSP, forensic voiceprints, the editor, torch-free TTS and cloning, the STT lane voaice.rs will serve |
| [PYTHAI/voaice](https://huggingface.co/PYTHAI/voaice) | the voice library on Hugging Face: 70 open-licensed Piper voices, for anyone to use |
| [cryptoAGI/bankml](https://github.com/cryptoAGI/bankml) | the method: bit-exact against the compiled reference, then faster |
