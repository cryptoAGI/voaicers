# Architecture — modules, the oracle contract, and how voaice.rs grows

voaice.rs is built to be **extended one module at a time without weakening what is already proven**. Every module
has one job, a public surface small enough to read, and an oracle that compares it with the reference's compiled
library. A new stage is added beside the old ones, never inside them, and it ships only when its own oracle passes
in the release gate.

## The modules (0.0.8)

| module | job | public surface | proven by |
|---|---|---|---|
| `src/sha256.rs` | FIPS 180-4 SHA-256, no crates | `Sha256`, `digest()`, `hex()` | FIPS test vectors |
| `src/model.rs` | the ggml whisper model file: hparams, mel filterbank, vocab and special tokens, every tensor's name/type/shape/offset; the sha256 **pin** that refuses an unknown file | `Model`, `Hparams`, `Pin`, `Tensor`, `Dtype`, `Specials`, `expected_tensors()` | `oracle_model_hparams_tensors_vocab_filters`, `guard_refuses_a_modified_model` |
| `src/wav.rs` | 16-bit PCM mono 16 kHz WAV → f32, refusing anything else (no silent resampling) | `read()`, `parse()` | `oracle_pcm_input_identical` |
| `src/mel.rs` | whisper's log-mel front end, in whisper.cpp's own arithmetic order (its radix-2 FFT, mixed f32/f64 band sum, libm symbols); since 0.0.2 allocation-free per frame (a `MelPlan` holds the invariants), SIMD across independent lanes only, threaded by frames | `Tables`, `MelPlan`, `Mel`, `log_mel_spectrogram()`, `log_mel_spectrogram_threads()` | `oracle_mel_bit_exact` (0 ULP), `oracle_mel_threads_bit_identical`, `oracle_mel_discriminates_fused_fft` |
| `src/f16.rs` | (0.0.3) f32 ↔ f16 exactly as ggml-cpu converts on an F16C build: the portable bit trick (`GGML_CPU_FP32_TO_FP16` — im2col, GELU, row tails), the `vcvtps2ph` row of `ggml_cpu_fp32_to_fp16` (mul_mat, flash attention) and a software model of it; f16 → f32 (the table, `vcvtph2ps`). They differ only on NaN, and the rows keep the reference's NaN bits by position | `fp16_to_fp32()`, `fp32_to_fp16()`, `fp32_to_fp16_f16c()`, `fp32_to_fp16_row()`, `fp16_to_fp32_row()` | `oracle_f16_to_f32_all_65536`, `oracle_f32_to_f16_boundary_set`, `oracle_f32_to_f16_every_pattern` (all 2³²), `oracle_f16_discriminates_round_half_away` |
| `src/gelu.rs` | (0.0.3) `ggml_gelu` as the encoder runs it: `ggml_table_gelu_f16` built as `ggml_cpu_init` builds it (glibc `tanhf`, GCC's FMA in `A·x·x + 1`), the op's ±10 clamps and f16 index; held widened to f32 for one lookup, eight lanes at once with AVX2 + F16C | `Gelu` (`new`, `row`, `row_scalar`, `f16`), `gelu_f32()`, `table_with()` | `oracle_gelu_table_all_65536`, `oracle_gelu_op` (all 2³²), `oracle_gelu_table_discriminates_unfused` |
| `src/conv.rs` | (0.0.6) `ggml_vec_dot_f16` in the AVX build's float order (4 × 8 lanes, exact products, pairwise reduce, the f32 widened to double and the tail added in double) and encoder conv1 (`ggml_conv_1d_ph`: im2col to f16 + that dot), its bias and GELU; fast path: tiles of 8 frames built from the mel and rounded through F16C (no im2col held), eight dots per weight load, the reduce and the double tail across eight frames, threads by frames | `Conv1` (`new`, `from_parts`, `run`, `run_bias`, `run_gelu`, `run_into`), `vec_dot_f16()`, `dot_f16_model()`, `im2col_f16()`, `wrong::*` (discriminators) | `oracle_vec_dot_f16_kernel`, `oracle_vec_dot_f16_discriminators`, `oracle_conv1_im2col_bit_exact`, `oracle_conv1_bit_exact`, `oracle_conv1_discriminators` (tests/conv1.rs) |
| `src/conv.rs` (0.0.7) | encoder conv2 (`ggml_conv_1d_ph(w2, ·, 2, 1)`: stride-2 im2col to f16 + 1,152-long f16 dots), its bias and GELU (`embd_conv`), and the positional embedding the encoder graph adds first (`e_pe + cont(transpose(·))`); fast path: per thread, blocks of 32 frames built from conv1's output (no im2col held), a 4-frame × 3-channel AVX2 register block over rows whose columns are permuted so each accumulator's blocks are contiguous (no product or chain changes), the transpose and the add written by the epilogue; `ConvStage` joins conv1 and conv2 through an f16 buffer (conv2 reads only conv1's f16 conversion) | `Conv2` (`new`, `from_parts`, `frames_out`, `run`, `run_into`, `run_into_f16`), `Epilogue` (`Raw`, `BiasGelu`, `Positions`), `ConvStage` (`new`, `run`, `run_into`), `Conv1::run_into_f16`, `im2col_strided_f16()` | `oracle_conv2_im2col_bit_exact`, `oracle_conv2_bit_exact`, `oracle_positions_bit_exact`, `oracle_conv2_discriminators` (tests/conv2.rs) |
| `src/norm.rs` (0.0.8) | the encoder's layer norms, `ggml_add(ggml_mul(ggml_norm(x, 1e-5), w), b)`, nine in tiny.en (each block's attn_ln and mlp_ln, ln_post): `ggml_compute_forward_norm_f32`'s in-order double sum rounded to f32, `mean = sum / n` in f32, `ggml_vec_cvar_f32`'s 8-blocks (f32 `x − mean`, `d·d`, the `(h0 + h2) + (h1 + h3)` pairing, a double sum), `1.0f / sqrtf(var + eps)`, then `· w` and `+ b` as two roundings (two nodes, no FMA); fast path: the three nodes in one pass per row, the double sums in vector lanes **only on rows that prove every order exact** (else in order), no intermediate tensors, threads by rows | `LayerNorm` (`new`, `from_parts`, `encoder`, `run`, `run_into`, `run_into_split`, `run_model`, `row_model`), `MIN_ROWS_PER_THREAD`, `Node` (`Norm`, `Mul`, `Add`), `Variant` (the reference = default; each flag a discriminator), `row_stats()`, `cvar_model()`, `norm_row()`, `sum_lanes()`, `sum_is_order_free()` | `oracle_norm_nodes_bit_exact`, `oracle_attn_ln_0_from_mel`, `oracle_norm_discriminators` (tests/norm.rs) |
| `src/measure.rs` | CPU seconds, RSS and its peak from `/proc` (no libc binding), and the `bench` loop the gate uses | `cpu_seconds()`, `rss_kb()`, `peak_rss_kb()`, `reset_peak_rss()`, `bench()` | unit test; its numbers are only read after the oracles pass |
| `src/lib.rs` | the crate root, and `ulp_distance()` every oracle reports in | `ulp_distance()` | — |
| `src/main.rs` | the CLI: `voaice info · mel · bench-mel · bench-f16 · conv1 · bench-conv1 · conv · bench-conv · norm · bench-norm · version` (and the opus, resample and vclone commands); a counting allocator for `bench-mel`'s heap peak | — | the gate runs it |

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
   - threads that split *outputs*, each output computed whole by one thread;
   - (0.0.8) reassociating one float sum **only where the input proves every order exact**: when every term is a
     multiple of 2^q (q from the smallest non-zero term's exponent) and n · max|term| < 2^(53 + q), no partial sum
     can round in any order, so a lane sum equals the in-order one; a row that cannot prove it takes the in-order
     loop. The proof is checked per row at run time, never assumed for a model or an input set.
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
