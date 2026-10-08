# voaice.rs — roadmap

The order is the method: **exact first, fast second.** A stage counts when its oracle passes against the shipped
whisper.cpp (upstream/PIN) in the same run; only then is its speed measured.

## Done in 0.0.1
- [x] Stage 0 — oracle harness (`testing/oracle/`): pinned reference built, layout probe, model / vocab / filterbank /
      mel / transcript recorded from the shipped `libwhisper.so`.
- [x] Stage 1 — model loader (`src/model.rs`): ggml whisper format, sha256 guard, 167/167 tensors identical.
- [x] Stage 2 — log-mel front end (`src/mel.rs`): 2,316,640 / 2,316,640 values bit-exact, 8 inputs.

## Done in 0.0.2
- [x] The mel, optimized with its bits unchanged: allocation-free per frame, SIMD across independent lanes, threads
      by frames (`MelPlan::run(samples, threads)`); 0 ULP on all 8 inputs, 2/3/4/8 threads = 1 thread, the fused-FFT
      discriminator still rejected. 6.1× faster than 0.0.1, 6.8× the reference at one thread, a third of its heap.
- [x] The gate measures efficiency after the oracles: wall, CPU (`/proc/self/stat`), heap (counting allocators on
      both sides), RSS (`VmHWM`), at 1 and nproc threads, and 0.0.1 rebuilt from its tag in the same run.

## Done in 0.0.3 — f32 ↔ f16 and the GELU table (pulled ahead: the encoder's order of work starts here)
- [x] `src/f16.rs`: the conversions this F16C build of ggml-cpu actually runs, read from the source and the binary.
      `GGML_CPU_FP32_TO_FP16` is **not** `_cvtss_sh` on x86 — simd-mappings.h defines only the `COMPUTE_` macro for
      F16C, so the kernels' macro falls through to ggml-impl.h's portable bit trick (NaN → `sign | 0x7E00`; GCC
      contracted it into an FMA in libggml-cpu, which is exact there). `ggml_cpu_fp32_to_fp16` (mul_mat's and flash
      attention's `from_float`, confirmed equal to the type traits' pointer) runs `vcvtps2ph` on blocks of 8 and 4
      (NaN quieted, top payload bits kept) and the bit trick on the last `n % 4`. f16 → f32: the table (portable
      `ggml_compute_fp16_to_fp32`) and `vcvtph2ps` agree on all 65,536.
- [x] `src/gelu.rs`: `ggml_table_gelu_f16` as `ggml_cpu_init` fills it — `(0.5·x)·(tanhf((S·x)·fma(A·x, x, 1)) + 1)`,
      glibc `tanhf`, then the portable f16 — and `ggml_vec_gelu_f32` (`x <= -10` → +0, `x >= 10` → x, else the
      table at the portable f16 index; NaN takes the table path). Faster than the reference: the table held widened
      (one lookup), AVX2 + F16C eight lanes with the NaN lanes re-indexed and one gather.
- [x] Oracle (`whisper_oracle --f16`, tests `oracle_f16_*`, `oracle_f32_to_f16_*`, `oracle_gelu_*`): all 65,536 f16
      patterns four ways; all 2³² f32 patterns three ways (the 1,429,656-value boundary set compared value by value, the
      rest by per-chunk digest); the GELU table; the op on 1,495,192 values and on all 2³². Discriminators:
      round-half-away (rejected) and the unfused GELU (rejected: 1 table entry, `0xBFFF`).
- [ ] Not observed yet: the inlined scalar copies inside im2col and flash attention themselves (the same macro; the
      oracle sees the row tail's copy, the table-init's copy and the GELU op's copy). They are checked when their
      nodes are, through the scheduler callback (0.0.6 onward).

## Done in 0.0.4 — the streaming Ogg/Opus reader (`src/ogg.rs`)
- [x] Pages from any `std::io::Read`, one page buffer (65,307 bytes) allocated once and reused; a packet inside a page
      handed out as a slice of it, a packet across pages assembled in a second buffer sized exactly to the longest
      such packet. Checked on every page: `OggS`, version 0, the flags (BOS first only, EOS last only, continued
      exactly when a packet is carried), one serial, consecutive sequence numbers, Ogg's CRC-32 (0x04C11DB7,
      unreflected, init 0 — sliced by 8, 4.5× the byte-at-a-time table).
- [x] `OpusHead` (version, channels, pre-skip, input rate, gain, mapping family with the family-1 table validated) and
      `OpusTags` (vendor and comments, kept up to a bound; a longer packet is marked truncated, not refused), each
      on its pages as RFC 7845 §3 places them.
- [x] Granules (RFC 7845 §4): the start of a stream that began past zero, every mid-stream granule equal to the last
      plus the samples completed, end trimming on the last page only and never past its samples or before the
      pre-skip; per packet its TOC samples and how many to skip and keep. Duration = last granule − start − pre-skip.
- [x] Every refusal a named `Kind` with the byte offset and the page (23 kinds).
- [x] Oracle (`tests/opus.rs`, `testing/opus/`): opus-tools 0.2 + libopus 1.4 + libogg 1.3.5 on production, recorded:
      35 / 35 files on 18 checks each (duration = opusdec's samples, every page and packet by libogg digest),
      21 / 21 adversarial files refused by name, 1 valid variant accepted; discriminators caught (pre-skip added,
      no end trim, code-3 count ignored, zlib's CRC verifies 0 / 427 pages). `voaice opus info`, `voaice bench-opus`.
- [x] streamair → voaice round trips (streamair/tests/roundtrip.rs): 400 random streams back exactly.
- [ ] **Found, not fixed (streamair's next step):** streamair 0.0.1's `mux` accepts an end trim larger than the
      samples on the last page (it bounds the trim by 5,760), which makes the EOS granule go backwards; opusinfo calls
      such a file an ERROR, voaice refuses it (`known_issue_the_writer_accepts_end_trimming_past_the_last_page`).
- [ ] Not yet covered by the oracle: the family-1 mapping table's values (opusinfo does not print them), mapping
      families 2 / 3 / 255, chained and multiplexed streams (refused by name today), the reference's own speed (the
      gate does not time on production).

## Next: 0.0.5 — the resampler whisper-cli uses (see docs/ROADMAP.md)
- [ ] miniaudio's linear resampler and its low-pass filter as compiled in the pinned whisper.cpp, 48 kHz → 16 kHz and
      the other rates, mono mixdown — the samples `read_audio_data` produces, bit for bit.

## Then, in order

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

Order of work: GELU table → f32↔f16 conversions (both done in 0.0.3) → `vec_dot_f16` on sampled real rows → conv1 → conv2 → one block
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
oracle passing in the same run. Candidates (the mel's allocations and threads were done in 0.0.2; what remains of its time is glibc's
`log10`, about a fifth, and the 25-point DFTs); `vec_dot_f16`
with AVX2/F16C in the reference's lane order; fused conv+GELU; a KV layout without per-step copies. Then
`base.en`, then the ggml quantized formats (q5_0, q8_0) production may switch to.

### Portability
- [ ] The tables and `log10` come from glibc's libm (`sincosf`, `cosf`, `log10`, the symbols libwhisper imports),
      so a host with another libm could differ in the last bit. Port the exact glibc 2.35 algorithms in-crate (as
      bankml carries its own f16), with an oracle over every f32 input the mel can produce for `sincosf`/`cosf`
      (400 + 400 arguments) and a sampled-plus-boundary oracle for `log10` on f64.
- [x] Threading of the mel (0.0.2: contiguous runs of frames per thread), checked bit-identical to one thread.

## Production parity (checked 2026-10-07)
- The pin is production's commit (see upstream/PIN). libwhisper there has no FMA, so stages 1–2 are exact on production too.
- libggml-cpu on production is a native Zen 3 build (652 vfmadd, AVX2). Before the encoder oracle claims production
  parity, record production's GGML_* build flags (or rebuild there) and compare kernel outputs against that library,
  not only the laptop's native build.
