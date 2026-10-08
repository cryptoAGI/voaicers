# Architecture — modules, the oracle contract, and how voaice.rs grows

voaice.rs is built to be **extended one module at a time without weakening what is already proven**. Every module
has one job, a public surface small enough to read, and an oracle that compares it with the reference's compiled
library. A new stage is added beside the old ones, never inside them, and it ships only when its own oracle passes
in the release gate.

## The modules (v0.1.0)

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
| `src/matmul.rs` (0.0.9) | the encoder's matrix products on activations as `ggml_compute_forward_mul_mat` runs them for an f16 weight: the f32 activations converted by `from_float` (`ggml_cpu_fp32_to_fp16`, each of the reference's threads converting its element range of every row — which decides how a NaN converts), then 0.0.6's `ggml_vec_dot_f16` per output; Q, K (no bias), V and their f16 CPYs (the scalar bit trick), the out projection + bias + residual, fc1 + bias, GELU, fc2 + bias + residual; fast path: attn_ln / mlp_ln computed row by row into the conversion (never written), one conversion shared by Q, K and V, panels of 64 frames in a permuted layout rounded through F16C, a 4-frame × 3-row AVX2 register block, every epilogue in the panel, the MLP a panel at a time from the out projection to fc2, threads by frames | `Linear` (`new`, `from_parts`, `model`, `convert_model`, `run_into`, `with_split`), `Epilogue` (`None`, `Bias`, `BiasGelu`), `Block` (`new`, `set_split`, `qkv_into`, `mlp_into`), `QkvTaps`, `MlpTaps`, `Variant` (the reference = default; each flag a discriminator), `split_ranges()`, `from_float_row()`, `perm()`, `residual_model()`, `gelu_model()`, `cpy_f16_model()`, `PANEL` | `oracle_matmul_nodes_bit_exact`, `oracle_block0_from_mel`, `oracle_mm_nan_split`, `oracle_matmul_discriminators` (tests/matmul.rs) |
| `src/attention.rs` (v0.1.0) | the encoder's self-attention as ggml's **tiled** flash-attention kernel computes whisper's node (`ggml_compute_forward_flash_attn_ext_tiled`): Q in f32; K and V the f16 `kv_pad` cache of 1,536 rows, its 36 +0 rows attended (no mask); per tile of 64 keys, f32 FMA-chain scores × 0.125, the tile max, a rescale of the output and the sum by glibc `expf` when the max grows, the probabilities by ggml's own 8-lane `ggml_v_expf` (summed in its pairing, in double), the output accumulated in f32 by FMA chains, × 1/S at the end; the one-chunk path (`use_ref`: Q to f16, f16 dots, V in f16) modelled as a discriminator and checked against the reference's own output; fast path: each head's K transposed per tile and V widened once per call (not once per query tile), the tile's scale, max, exponentials and sums in registers, the rescale folded into the output product's load, threads by query tiles behind a per-head barrier | `Attention` (`new`, `run`, `run_into`, `scratch_len`), `attention_model()`, `attention_model_frames()`, `row_model()`, `row_model_one_chunk()`, `soft_max_model()`, `v_expf()`, `libm_expf()`, `Variant` (the reference = default; each flag a discriminator), `n_kv_pad()`, `HEAD_DIM`, `KV_TILE`, `Q_TILE`, `SCALE` | `oracle_attention_nodes_bit_exact`, `oracle_attention_discriminators` (tests/attention.rs) |
| `src/encoder.rs` (v0.1.0) | **the whole encoder**: the mel window → `ConvStage` → per block `Block::qkv_into` → `Attention::run_into` → `Block::mlp_into` → `ln_post` = `embd_enc`, bit for bit what `whisper_encode_with_state` leaves in `whisper_state::embd_enc`; no arithmetic of its own; every buffer between stages caller-owned and reused | `Encoder` (`new`, `encode`, `encode_into`), `EncoderBuffers` (`bytes`) | `oracle_encoder_end_to_end` (tests/attention.rs) |
| `src/measure.rs` | CPU seconds, RSS and its peak from `/proc` (no libc binding), and the `bench` loop the gate uses | `cpu_seconds()`, `rss_kb()`, `peak_rss_kb()`, `reset_peak_rss()`, `bench()` | unit test; its numbers are only read after the oracles pass |
| `src/lib.rs` | the crate root, and `ulp_distance()` every oracle reports in | `ulp_distance()` | — |
| `src/main.rs` | the CLI: `voaice info · mel · bench-mel · bench-f16 · conv1 · bench-conv1 · conv · bench-conv · norm · bench-norm · qkv · bench-mm · encode · bench-attn · bench-encode · version` (and the opus, resample and vclone commands); a counting allocator for `bench-mel`'s heap peak | — | the gate runs it |

The pattern each module follows is the one bankml uses: **a pure function of its inputs, the same float operations
in the same order as the reference, and no hidden state.** That is what makes a module testable alone, and what
makes it safe to replace a module's internals later for speed: the oracle still has to say 0 ULP.

## Adding a stage — the recipe

The next stages are the decoder, the tokenizer and timestamps (see [TODO.md](../TODO.md)). Each one
goes in the same way:

1. **Find the reference's arithmetic, not the paper's.** Read the pinned source *and* the shipped binary
   (`objdump`): which ops, which precision, which order, fused or not. Write it down in the module's header.
2. **Give the oracle a way to see it.** If whisper.cpp's public API exposes the value, use it; if not, extend the
   layout probe (`testing/oracle/layout_probe.cpp`) or hook ggml's scheduler
   (`ggml_backend_sched_set_eval_callback`) — and make the oracle *self-check* every probed location against a
   public getter before it trusts it. Identify nodes by what they read (a weight's name, the node they add to), not
   by their index; and when the tensors are large, record a digest per row (0.0.9: 80 MB instead of 1.8 GB) and take
   the full inputs from an earlier record, checking each against the digest of what the node read.
3. **Write the module** as a new file (`src/decoder.rs`, …) with a pure public function.
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
   - (0.0.9) a different layout and a different split of the *same* chains: the dot's four accumulators may live in
     any registers and its operands in any permuted layout, as long as each accumulator chains the same products in
     the same order and the reduction pairs as the reference's does; a conversion the reference makes three times
     (Q, K and V each convert attn_ln's output) may be made once, because it is the same function of the same row.
   - (v0.1.0) a different tiling of *independent rows*: flash attention's softmax state is per query row, and each
     score and each output element is its own FMA chain, so the query tile (voaice's 60 against ggml's 64), the
     register blocking and the thread split are free — while the **key** tiling (64) is not: it decides when the max
     is updated and the output rescaled, and where the double sums break. Work done once per call that the
     reference repeats per query tile (widening and transposing K) is free too.
   - read **which path** the reference takes before reading the path: ggml has two flash-attention kernels and a shape
     test picks one; TODO.md's first reading was the other one. When the reference can be asked to run its other path
     (`cplan.use_ref`), record that as well: a model of the road not taken, checked against it, is a discriminator
     the reference itself confirms (and it settled a contraction the source left open).
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
