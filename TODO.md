# voaice.rs — roadmap

The order is the method: **exact first, fast second.** A stage counts when its oracle passes against the shipped
whisper.cpp (upstream/PIN) in the same run; only then is its speed measured.

## Done in 0.0.1
- [x] Stage 0 — oracle harness (`testing/oracle/`): pinned reference built, layout probe, model / vocab / filterbank /
      mel / transcript recorded from the shipped `libwhisper.so`.
- [x] Stage 1 — model loader (`src/model.rs`): ggml whisper format, sha256 guard, 167/167 tensors identical.
- [x] Stage 2 — log-mel front end (`src/mel.rs`): 2,316,640 / 2,316,640 values bit-exact, 8 inputs.

## Next, in order

### Stage 0b — close the reference gaps
- [ ] Verify the pin against production itself: `strings` on the VPS's `libggml-base.so*` for `0.16.0`, the build's
      git sha if kept, and the sha256 of `/opt/whisper.cpp/models/ggml-{tiny,base}.en.bin`. (Read-only; not done in
      0.0.1, which does not touch production.)
- [ ] Pin `ggml-base.en.bin` (147,964,211 bytes, production's default) by sha256 and run the same oracles on it.
- [ ] The production build is `GGML_NATIVE=ON` on the VPS's CPU. ggml-cpu's kernels (not the mel, which is in
      libwhisper and compiled for baseline x86-64) depend on the host's ISA: record the VPS's `/proc/cpuinfo` flags and
      make the oracle build reproduce them (`-march=` of the VPS), or build both and keep one oracle per ISA.

### Stage 3 — encoder (plan; no code yet)
Graph (`whisper_build_graph_conv` + `whisper_build_graph_encoder`, src/whisper.cpp:1976–2270), ggml CPU backend:

| step | ggml op | what must be matched |
|---|---|---|
| mel window | slice of `state->mel`, 2·n_ctx = 3000 frames from `offset` | exact copy |
| conv1 | `ggml_conv_1d_ph(w1 f16 [3,80,384], mel, s=1, p=1)` = `im2col` (to **f16**) + `mul_mat` | the f32→f16 rounding of im2col; the f16·f16 dot's accumulation order |
| + bias, GELU | `ggml_add`, `ggml_gelu` | GELU is a **lookup table** (`GGML_GELU_FP16`): x→f16, `ggml_table_gelu_f16[bits]`; x ≤ −10 → 0, x ≥ 10 → x |
| conv2 | same, stride 2 → [384, 1500] | as conv1 |
| + positional | `ggml_add(e_pe view, cont(transpose(cur)))` | exact (one add) |
| ×4 blocks | `ggml_norm` (eps 1e-5) → `*w + b` → Q,K,V `mul_mat` (f16 weights) + biases | norm's mean/variance summation order (ggml_vec_* in f64 or f32? read `ggml_compute_forward_norm_f32`) |
| attention | **flash_attn = true** (whisper-cli default): K,V copied to the f16 `kv_pad` cache, `ggml_flash_attn_ext(Q, K, V, scale 1/√64)` | online softmax order, Q→f16 conversion, `expf` vs ggml's own exp, V accumulation in f16 or f32 — read `ggml_compute_forward_flash_attn_ext_f16` |
| out proj, residual, MLP | `mul_mat` + bias, add, norm, `mul_mat` (384→1536), GELU, `mul_mat`, add | as above |
| ln_post | norm, `*w + b` | as above |

The matrix products are where the float order lives. For an f16 weight and f32 activation, ggml-cpu converts the
activation row to f16 (`vec_dot_type` of F16 is F16) and calls `ggml_vec_dot_f16`: on AVX2 that is 4 accumulators ×
8 lanes with FMA, then `GGML_F32x8_REDUCE`'s pairwise order, then a scalar tail. Threads split rows (each dot whole
in one thread), so the count should not change bits — to be checked, as the mel's was. `GGML_LLAMAFILE` is OFF in
this build (whisper.cpp's CMake does not set the default), so tinyBLAS is not in the path; the oracle must confirm it.

**How the oracle checks each step.** The public API gives the encoder's output: `whisper_encode_with_state()` after
`whisper_set_mel_with_state()` (or `pcm_to_mel`), then `state->embd_enc` — a `ggml_tensor *` at an offset the layout
probe already computes (`VOAICE_OFF_STATE_EMBD_ENC`) — read with `ggml_backend_tensor_get`. `embd_conv` likewise
(add it to the probe). For every intermediate op, whisper exposes no eval callback, so the instrumented route is:
probe the offset of `whisper_state::sched_encode` / `sched_conv` and call `ggml_backend_sched_set_eval_callback` on
the shipped scheduler; the callback copies each node's output (name, op, shape, f32 bits) — the shipped library
still does all the arithmetic, the oracle only observes. Kernel-level oracles, as bankml did for its dot products:
`ggml_table_gelu_f16` is an **exported symbol** of `libggml-cpu.so` (dump all 65,536 entries and compare to the port's
table), `ggml_cpu_fp32_to_fp16` / `ggml_vec_dot_f16` are reachable through `ggml_get_type_traits_cpu`.

Order of work: GELU table → f32↔f16 conversions → `vec_dot_f16` on sampled real rows → conv1 → conv2 → one block
(norm, attention, MLP) → all four → `embd_enc` bit-exact on the 8 test inputs.

### Stage 4 — decoder (plan)
Cross-attention K/V (`whisper_build_graph_cross`: `mul_mat` of `embd_enc` by each layer's cross K/V, scaled by
`n_state_head^-0.25` on both K and Q in the non-flash path — check which path flash_attn takes there), self-attention
with the f16 KV cache, token + positional embeddings, 4 blocks, `ln`, logits = `mul_mat(d_te, cur)`.
Oracle: `whisper_decode_with_state()` + `whisper_get_logits_from_state()` (public): all 51,864 logits per step,
bit-exact, for the prompt `[SOT, (lang), task, NOT/BEG]` and for each greedy step of the recorded transcripts.

### Stage 5 — the transcript
`whisper_full`'s greedy loop and its logit filters (suppress blank, suppress non-speech tokens, timestamp rules: the
`ts` probability-sum test, `max_initial_ts`), the 30-s windowing and seek, segment splitting, `t0/t1` and token
timestamps (`token_timestamps = true`), `p` per token. Oracle: `transcript.tsv` already records ids, t0, t1 and the
f32 bits of `p` for every token: the whole file identical is the stage's gate. Also: whisper-cli's miniaudio WAV path
(resampling, stereo) as its own oracle, and the temperature fallback (sampling with whisper's `std::mt19937`).

### Stage 6 — fast
Only after stage 5 is bit-exact. Measured, never quoted: same input, same cores (bankml's `testing/pinned.sh`), the
oracle passing in the same run. Candidates: the mel (0.0.1's port allocates per FFT level and per frame and runs one
thread — see testing/results for its time against the reference's; a different FFT is not allowed, the float order
is the reference's, but the allocations can go and frames can be threaded, each frame whole in one thread as
whisper does); `vec_dot_f16`
with AVX2/F16C in the reference's lane order; fused conv+GELU; a KV layout without per-step copies. Then
`base.en`, then the ggml quantized formats (q5_0, q8_0) production may switch to.

### Portability
- [ ] The tables and `log10` come from glibc's libm (`sincosf`, `cosf`, `log10`, the symbols libwhisper imports),
      so a host with another libm could differ in the last bit. Port the exact glibc 2.35 algorithms in-crate (as
      bankml carries its own f16), with an oracle over every f32 input the mel can produce for `sincosf`/`cosf`
      (400 + 400 arguments) and a sampled-plus-boundary oracle for `log10` on f64.
- [ ] Threading of the mel (frame i to worker i % n), checked bit-identical to one thread.

## Production parity (checked 2026-10-07)
- The pin is production's commit (see upstream/PIN). libwhisper there has no FMA, so stages 1–2 are exact on production too.
- libggml-cpu on production is a native Zen 3 build (652 vfmadd, AVX2). Before the encoder oracle claims production
  parity, record production's GGML_* build flags (or rebuild there) and compare kernel outputs against that library,
  not only the laptop's native build.
