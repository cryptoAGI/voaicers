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
- [x] im2col's inlined scalar copy: observed through conv1's IM2COL node in 0.0.6 (5,760,000 / 5,760,000).
- [ ] Not observed yet: flash attention's inlined copy (checked when its node is, through the scheduler callback).

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

## Done: 0.0.5 — the audio reader whisper-cli uses (CHANGELOG.md, testing/resample/NOTES.md)
- [x] dr_wav's f32 conversions, miniaudio's `mono_out` mixdown, the linear resampler with its order-4 low-pass, the
      length rule and its zero tail — 55 / 55 files, 1,955,875 samples bit-identical to whisper-cli's `libcommon.a`;
      streamed in any chunking; discriminators caught.
- [ ] Not covered: what miniaudio decodes besides WAV (FLAC, MP3, Vorbis — whisper-cli reads them all), WAV's A-law,
      µ-law, ADPCM, f64 and odd bit depths, RF64 / Wave64 (each refused by name); the `--diarize` stereo path
      (`L + R`, channels kept); input from stdin (`-`). Rates other than the seven in the corpus are computed by the same
      code but were not each checked; a rate whose reduced pair puts a different `sin` argument in play is a different
      coefficient set (computed through the platform libm's `sin`, glibc's here).
- [ ] The low-pass is a serial IIR chain, so the inner loop is latency-bound (≈ 12 cycles a stage a sample); the
      coefficients and float order are fixed by the oracle. What remains to win is around it: `.opus` decoded straight
      into this converter (v0.4.0), and the mel fed from it without the whole vector.

## Done: 0.0.6 — `ggml_vec_dot_f16` (brought forward from 0.0.9) and encoder conv1 (CHANGELOG.md, testing/conv1/NOTES.md)
- [x] The order decided from the source and the binary before any code: conv1's mul_mat has the f16 weights as its
      *second* operand (already the F16 `vec_dot_type`, nothing converted), every output one
      `ggml_vec_dot_f16(240)` whole in one thread; the AVX path (4 × 8 lanes, FMA, pairwise reduce, double tail of 16)
      read in `objdump`; LLAMAFILE off, repack has no f16. The kernel was the roadmap's 0.0.9 — moved here, rows renumbered.
- [x] Oracle through ggml's scheduler eval callback on `whisper_state::sched_conv` (layout probe): IM2COL, MUL_MAT,
      ADD, GELU nodes bit-exact on 8 inputs at 1 and 4 threads; the kernel on 1,436 real/random dots; four
      discriminators caught; `embd_conv` unchanged by observing; the standalone graph the bench times = the node.
- [x] Faster, bits unchanged: no im2col, tiles of 8 frames rounded through F16C, 8 dots per weight load, the reduce and
      double tail vectorized across frames, threads by frames, the output kept by the caller (`Conv1::run_into`).
- [ ] Not covered: production's own libggml-cpu (Zen 3) was not run; an AVX-512 host's 16-lane path; offsets other
      than 0 against the reference (voaice's windowing at other offsets is checked against its own im2col + model
      dot in unit tests, not against whisper); `base.en` (n_state 512).
- [ ] Efficiency left on the table: the kernel issues 9 loads per 8 FMAs (one weight block, eight frame rows); a
      4-frame × 2-channel tile would issue 6, at the cost of a second epilogue shape. At 4 threads on this 2-core /
      4-thread laptop voaice gains nothing over 2 (the FMA pipes are shared by SMT siblings).

## Done: 0.0.7 — conv2 and the positional embedding (CHANGELOG.md, testing/conv2/NOTES.md)
- [x] Read from the pin: `embd_conv` is conv2's GELU (the conv graph's last node); the positions are the encoder
      graph's first op, `add(view_2d(e_pe, …, offset 0: static iter = 0), cont(transpose(embd_conv)))`; the mel input is
      always 2·n_ctx frames zero-padded, so short inputs take no other path.
- [x] Oracle through both schedulers' eval callbacks (`sched_conv` every node; `sched_encode` up to the first ADD,
      then unobserved): conv2's IM2COL 13,824,000 / 13,824,000, MUL_MAT / ADD / GELU 0 differ at 1 and 4 threads, CONT
      0 differ, the positional ADD 0 differ from voaice's mel at 1, 2, 4 threads, 8 inputs; `embd_conv` = the state's;
      `embd_enc` unchanged by observing; standalone graphs = the nodes. Discriminators: stride 1, one accumulator,
      positions before the transpose, positions a frame late, GELU before the bias — all caught.
- [x] Found: conv2's f16 rounding of its input changes only conv1-GELU outputs ≥ 10 (GELU's table already returns
      f16 values below that), so "im2col in f32" is caught on 4 inputs (min_len, noise_loud, odd_len, silence) and
      cannot be told apart on the other 4. The same fact makes an f16 buffer between conv1 and conv2 exact.
- [x] Faster, bits unchanged: blocks of 32 frames per thread (no im2col), a 4 × 3 register block (7 loads per 12 FMAs)
      over a column layout where each accumulator's blocks are contiguous, the transpose + positions in the epilogue,
      conv1 → conv2 through f16, caller-owned buffers (`Conv2::run_into`, `ConvStage::run_into`).
- [ ] Not covered: `audio_ctx` ≠ 0 (n_ctx < 1500: a shorter e_pe view and conv window) — voaice takes n_frames, but it
      was not compared; production's own libggml-cpu (Zen 3) was not run; an AVX-512 host; `base.en` (n_state 512,
      k = 1,536; 512 channels = 170 × 3 + 2, which the padded weight rows handle, unit-tested on small shapes only).
- [ ] Efficiency left: conv1 still uses 0.0.6's 8 × 1 tile (9 loads per 8 FMAs) on a strided layout; the 4 × 3 block
      with the permuted layout would need its 16-wide double tail handled (k = 240 = 7 × 32 + 16). 4 threads = 2 here.

## Done: 0.0.8 — the encoder's layer norms (CHANGELOG.md, testing/norm/NOTES.md)
- [x] Read from the pin and the binary: nine norms (attn_ln, mlp_ln per block, ln_post), each `ggml_norm(x, 1e-5f)`
      then separate MUL and ADD nodes (the CPU backend fuses only RMS_NORM + MUL). `ggml_vec_sum_f32` in double, in
      order (`vcvtss2sd` + `vaddsd`); `ggml_vec_cvar_f32`'s AVX2 + FMA branch with no FMA in it; `vsqrtss`, `vdivss`.
- [x] Oracle through `sched_encode`'s eval callback, every node observed, each NORM's input read before it ran:
      NORM, MUL, ADD of all nine chains 0 differ (373,248,000 values, 8 inputs, model + 1 and 4 threads); block 0
      from voaice's mel 0 differ at 1, 2, 4 threads; eight discriminators each caught on every input.
- [x] Faster, bits unchanged: one pass per row for the three nodes, the double sums in lanes when the row proves the
      order free (107,845 of 108,000 rows on the 8 inputs), caller-owned output (`LayerNorm::run_into`), threads by rows.
- [ ] Not covered: the inputs of blocks 1–3 and ln_post come from the reference (attention and the MLP are not
      ported); an AVX-512 build (cvar's 16-lane branch pairs differently); production's library; `base.en` (n = 512).
- [ ] Efficiency left: at 1500 × 384 the op is memory-bound on this laptop (input + output ≈ 4.5 MB > the 4 MB L3);
      the real saving is to fuse the norm into 0.0.9's `from_float` (the MUL_MAT's f32 → f16 row conversion), so the
      norm's f32 output is never written.

## Next: 0.0.9 — the matrix products on activations (see docs/ROADMAP.md)
- [ ] Q, K, V, the out projection and the MLP's two products: `mul_mat`'s f32 activation rows converted to f16 by
      `from_float` (`ggml_cpu_fp32_to_fp16`, 0.0.3; the threads split each row's conversion by element ranges), then
      0.0.6's `ggml_vec_dot_f16`; + biases. Oracle: every MUL_MAT node of block 0 through `sched_encode`'s callback
      (0.0.8's callback already observes them all), fed the recorded activations.

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
| + positional | `ggml_add(e_pe view, cont(transpose(cur)))` | exact (one add) — done in 0.0.7 |
| ×4 blocks | `ggml_norm` (eps 1e-5) → `*w + b` → Q,K,V `mul_mat` (f16 weights) + biases | norm: done in 0.0.8 (double sum in order, cvar's 8-lane f32 pairing, `1/sqrtf`, MUL and ADD unfused); the products: 0.0.9 |
| attention | **flash_attn = true** (whisper-cli default): K,V copied to the f16 `kv_pad` cache, `ggml_flash_attn_ext(Q, K, V, scale 1/√64)` | online softmax order, Q→f16 conversion, `expf` vs ggml's own exp, V accumulation in f16 or f32 — read `ggml_compute_forward_flash_attn_ext_f16` |
| out proj, residual, MLP | `mul_mat` + bias, add, norm, `mul_mat` (384→1536), GELU, `mul_mat`, add | as above |
| ln_post | norm, `*w + b` | done in 0.0.8 (fed the recorded input) |

The matrix products are where the float order lives. For an f16 weight and f32 activation, ggml-cpu converts the
activation row to f16 (`vec_dot_type` of F16 is F16) and calls `ggml_vec_dot_f16`: on AVX2 that is 4 accumulators ×
8 lanes with FMA, then `GGML_F32x8_REDUCE`'s pairwise order, then a scalar tail. Threads split rows (each dot whole
in one thread), so the count should not change bits — checked for conv1 in 0.0.6 (1 vs 4 threads identical).
`GGML_LLAMAFILE` is OFF in this build (CMakeCache.txt; the gate prints it), so tinyBLAS is not in the path, and the
traits' `vec_dot` for F16 is the exported `ggml_vec_dot_f16` (0.0.6's record checks the pointer). Note conv1 is the
other way round: there the *weights* are mul_mat's src1 and nothing is converted.

**How the oracle checks each step.** The public API gives the encoder's output: `whisper_encode_with_state()` after
`whisper_set_mel_with_state()` (or `pcm_to_mel`), then `state->embd_enc` — a `ggml_tensor *` at an offset the layout
probe already computes (`VOAICE_OFF_STATE_EMBD_ENC`) — read with `ggml_backend_tensor_get`. `embd_conv` likewise
(add it to the probe). For every intermediate op, whisper exposes no eval callback, so the instrumented route is:
probe the offset of `whisper_state::sched_encode` / `sched_conv` and call `ggml_backend_sched_set_eval_callback` on
the shipped scheduler; the callback copies each node's output (name, op, shape, f32 bits) — the shipped library
still does all the arithmetic, the oracle only observes. Kernel-level oracles, as bankml did for its dot products:
`ggml_table_gelu_f16` is an **exported symbol** of `libggml-cpu.so` (dump all 65,536 entries and compare to the port's
table), `ggml_cpu_fp32_to_fp16` / `ggml_vec_dot_f16` are reachable through `ggml_get_type_traits_cpu`.

Order of work: GELU table → f32↔f16 conversions (both done in 0.0.3) → `vec_dot_f16` on sampled real rows → conv1 (both done in 0.0.6) → conv2 + positions (0.0.7) → one block
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
