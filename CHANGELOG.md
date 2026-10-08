# Changelog

## 0.0.6 — 2026-10-08 — `ggml_vec_dot_f16` and encoder conv1, bit-exact against the conv graph's own nodes

**The encoder's first layer — im2col to f16, the f16 · f16 dot product in the AVX build's float order, + bias and
GELU — bit for bit as the shipped ggml-cpu computes it inside whisper's own scheduler, at 1 and 4 threads, then about
four times faster with less memory.** Record: `testing/results/0.0.6.txt`; how it was decided:
`testing/conv1/NOTES.md`.

### The order, decided first (and the roadmap renumbered)
- conv1 = `ggml_conv_1d_ph(w f16 [3, 80, 384], mel, 1, 1)` = `im2col` → f16 [240, 3000] (the portable
  `GGML_CPU_FP32_TO_FP16`, 0.0.3) and `mul_mat(im2col, w)`, where the **weights are the second operand** and already
  the F16 `vec_dot_type`: nothing is converted, and each of the 1,152,000 outputs is one
  `ggml_vec_dot_f16(240, im2col row, weight row)`, whole in one thread (mul_mat chunks rows, never a dot).
- So conv1 could not be exact without the kernel the roadmap had at 0.0.9: it is **brought forward into 0.0.6** and
  the rows after it renumbered (0.0.9 is now the activation-side products: f32 rows → f16 by `from_float`, then this
  dot). conv1's bias and GELU nodes came along (one f32 add; 0.0.3's GELU), ahead of 0.0.7.
- `ggml_vec_dot_f16` on this build (read in the source, then in `objdump` — 8 `vcvtph2ps`, 4 ymm `vfmadd231ps`, 2
  `vhaddps`, 1 `vaddsd`): four accumulators of eight lanes over blocks of 32; `(acc0 + acc2) + (acc1 + acc3)`; high
  half onto low; two `hadd`s; the f32 **widened to double**; the last `n mod 32` products added **in double** in index
  order; one rounding to f32. Every product of two halves is exact in f32, so the FMA equals a multiply and an add
  here — the portable model needs no FMA to be exact. `GGML_LLAMAFILE` is off (no tinyBLAS), the CPU repack has no
  f16 case, and the F16 traits' `vec_dot` is the exported `ggml_vec_dot_f16` (the record checks the pointer).

### The oracle
- `whisper_oracle --conv1`: the layout probe now finds `whisper_state::sched_conv` / `sched_encode` / `embd_conv`;
  `ggml_backend_sched_set_eval_callback` on the conv scheduler copies every node as it is computed (self-checks: only
  the CPU backend; IM2COL, MUL_MAT, ADD, GELU in order; `embd_conv` named). Observing changes nothing (`embd_conv`
  identical with and without the callback, 8 / 8); the reference's own nodes are identical at 1 and 4 threads
  (8 / 8); the standalone graph the benchmark times equals the scheduler's node (8 / 8, at 1 and 4 threads).
- `oracle_conv1_im2col_bit_exact`: **5,760,000 / 5,760,000** f16 values (8 inputs, from voaice's own mel of each WAV).
- `oracle_conv1_bit_exact`: the MUL_MAT, + bias and GELU nodes, **0 values differ** on all 8 inputs at 1 and at 4
  threads — 55,296,000 values compared.
- `oracle_vec_dot_f16_kernel`: **1,436 / 1,436** dots by the shipped kernel through `ggml_get_type_traits_cpu`: 16 row
  pairs of each of the model's 70 f16 tensors (n = 240, 384, 1,152, 1,536), every length 1–300 on real conv2 rows, 64
  random finite-f16 vectors; 303 distinct lengths.
- Discriminators, all caught: one f32 accumulator (**1,265** / 1,436 dots; on conv1 **1,068,048** / 1,152,000 values),
  the tail added in f32 (**211** / 1,436), the accumulators reduced in sequence (**680** / 1,436), im2col kept in f32
  without its f16 rounding (**1,151,932** / 1,152,000 conv1 values).
- **What this holds for:** this laptop's native libggml-cpu (Ryzen 3 3200U, Zen+: AVX2 + FMA + F16C, no AVX-512),
  which takes the AVX path. Production's Zen 3 has the same extensions, so its native build compiles the same path —
  but production's library was not run by this oracle. Offsets other than 0, and `base.en`, were not compared.

### Measured (gate step 8, only after 4d passed; 2 cores / 4 threads, load 1.7–3 from interactive use)
conv1 on the first 30-s window (3000 frames; every input costs the same). The reference: the same ops as a ggml graph
on the shipped CPU backend. Wall = best of 10. The gate's own run was noisy, so three more runs of the step follow it:

| | reference | voaice | gate run | three reruns |
|---|---|---|---|---|
| conv1, 1 thread | 86–89 ms | 21.4–24.2 ms (33.1 in the gate) | 2.63× | 3.66× · 3.97× · 4.03× |
| conv1, 2 threads | 51–54 ms (80 in the gate) | 13.5–15.9 ms | 5.79× | 3.76× · 3.21× · 4.00× |
| conv1, 4 threads | 46–59 ms | 14.7–24.6 ms | 1.87× | 3.31× · 3.94× · 3.42× |
| conv1 + bias + GELU, 1 thread | 94–119 ms | 24.7–30.7 ms | 3.78× | 3.80× · 3.88× · 3.90× |
| conv1 + bias + GELU, 4 threads | 50–53 ms | 15.9–16.3 ms | 3.19× | 3.30× · 3.09× · 3.16× |

- **About 4× the reference at one thread and 3–4× at two and four**, in **13–43 % of its CPU time** (gate: 39 against
  90 CPU-ms at one thread, 38 against 302 at two, 57 against 344 at four).
- Memory: voaice's heap at the call's peak is **4,508 KiB — the output (4,500) and two 7.5 KB tiles**; the reference's
  graph holds **5,907 KiB** (im2col 1,406 + the product 4,500), and **14,907 KiB** with the bias and GELU outputs,
  which voaice writes in place (the add and GELU are fused into each tile's epilogue).
- What changed, bits fixed: the weights widened to f32 once; no im2col and no rounded copy of the mel — each thread
  builds tiles of 8 frames from the mel, rounding 8 lanes at once through F16C (equal to the bit trick except on NaN;
  a block holding a NaN takes the bit trick); eight dots share each weight load; the pairwise reduction and the
  double tail run across eight frames in vector registers; threads split frames; the caller keeps the output.
- 4 threads gain nothing over 2 here (2 physical cores; the loop is bound by the FMA pipes the SMT siblings share).

### Added
- `src/conv.rs`: `Conv1` (`new`, `from_parts`, `run`, `run_bias`, `run_gelu`, `run_into`), `vec_dot_f16`,
  `dot_f16_model`, `im2col_f16`, `wrong::{single_accumulator, tail_in_f32, sequential_reduce}`; unit tests (the AVX2
  tile against the model at 8 lengths, the whole path against im2col + model at offsets, short windows and a short
  last tile, the NaN block). `voaice conv1`, `voaice bench-conv1`. `tests/conv1.rs` (5 oracle tests).
- `whisper_oracle --conv1`, `--bench-conv1`; layout probe offsets; gate steps 4d and 8 (all earlier checks kept).
- The production Opus re-ask (step 4b) was skipped in this run on purpose (production was not to be touched); the
  comparisons ran against the recorded answers.

## 0.0.5 — 2026-10-08 — the audio reader whisper-cli runs, bit-exact against its own libcommon.a

**Any PCM or float WAV, at any rate and channel count, to whisper's 16 kHz mono f32 — dr_wav's conversions, miniaudio
0.11.24's mono average, its linear resampler with the order-4 low-pass and its length rule — bit for bit as
whisper-cli's `read_audio_data` reads the same file, streamed in any chunking.** Record: `testing/results/0.0.5.txt`.

### The oracle
- `testing/oracle/resample_oracle` links the pinned build's `examples/libcommon.a` (the object whisper-cli links;
  `-O3 -DNDEBUG`, no `-march`, 0 FMA instructions counted) and calls `read_audio_data` as whisper-cli does without
  `--diarize`. Corpus: `testing/make_resample_audio.py`, 55 files pinned in `testing/pins/resample.sha256`.
- `oracle_resample_bit_exact`: **55 / 55** files, **1,955,875 / 1,955,875** samples — 8 / 16 / 22.05 / 24 / 32 /
  44.1 / 48 kHz, mono and stereo, s16 / f32 / u8 / s24 / s32, six channels in WAVE_FORMAT_EXTENSIBLE, f32 beyond full
  scale and subnormal, 1–7-frame inputs, a LIST chunk, a truncated data chunk, JFK at 48 kHz, 60 s of 44.1 kHz stereo.
  3 files end in the length rule's zero tail (the rule promises one frame more than the resampler makes); reproduced.
- `oracle_resample_streaming_equals_whole`: random 1–9,000-byte pushes, frames split across them: **55 / 55**.
- Discriminators: low-pass order 2 or 6 — **45 / 50** resampled files differ each (the other 5 are 1–2-frame inputs
  whose only output is the leading 0); stereo mixdown as `L + R` or `L` alone — **8 / 8**; the length without the
  promised frame — **3 / 55** (exactly the zero-tail files). All caught.
- How the source was read, line by line: `testing/resample/NOTES.md`.

### Measured (gate step 7, only after 4c passed; same laptop, 1-minute load 1.6 at the start)
One call = the whole read of a file, each side in a fresh process; wall = best of 10.

| file | audio | reference | voaice | × | heap ref / voaice |
|---|---|---|---|---|---|
| JFK held to 48 kHz mono | 11 s | 6.17 ms | 4.78 ms | 1.29× | 698 / 688 KiB |
| 44.1 kHz stereo, 60 s | 60 s | 42.2 ms | 19.7 ms | 2.14× | 3,762 / 3,750 KiB |
| 48 kHz stereo | 1.3 s | 0.91 ms | 0.46 ms | 2.00× | 91 / 82 KiB |
| 8 kHz stereo (upsampling) | 1.3 s | 0.45 ms | 0.17 ms | 2.56× | 91 / 82 KiB |
| 16 kHz stereo (no resampler) | 1.3 s | 0.11 ms | 0.03 ms | 4.04× | 91 / 82 KiB |
| 48 kHz, six channels | 1.0 s | 1.12 ms | 0.81 ms | 1.39× | 72 / 63 KiB |

Geometric mean over 8 files: **2.06×** the reference; about **0.33 CPU-ms per second of output audio** when
resampling. The low-pass is a serial IIR chain whose float order the oracle fixes, so the win is around it (no
intermediate buffers, one pass), not inside it. The heap is the output vector on both sides.

### Added
- `src/resample.rs` (`read`, `Converter` push/finish, `Linear`, `parse_header`); `voaice resample`,
  `voaice bench-resample`; `tests/resample.rs`; the gate's steps 4c and 7; `ATTRIBUTION.md` credits miniaudio / dr_wav
  (public domain or MIT-0; nothing copied).
- Not covered (refused by name): FLAC / MP3 / Vorbis, A-law, µ-law, ADPCM, f64, RF64 / Wave64, stdin, the diarize path.

## 0.0.4 — 2026-10-08 — the streaming Ogg/Opus reader, exact against opus-tools 0.2

**An `.opus` file read page by page from any `std::io::Read` — every page's CRC, sequence and flags checked, packets
reassembled across lacing and continuation, `OpusHead` and `OpusTags` parsed, granules and pre-skip turned into the
exact playable length in 48 kHz samples — with memory bounded by one page, never the file. On 35 files the length
equals the samples production's opusdec plays, and every page and packet equals what libogg and libopus see.**
Record: `testing/results/0.0.4.txt`.

### The oracle (opus-tools is not on the dev laptop; it runs where production has it)
- The reference: **opus-tools 0.2 (`opusinfo`, `opusdec`), libopus 1.4, libogg 1.3.5** on mindX production
  (AMD EPYC 7543P; packages `opus-tools 0.2-1build3`, `libopus0 1.4-1build1`, `libogg0 1.3.5-3build1`), reached
  read-only over ssh, all work in `/tmp/voaice_oracle_004`, removed after; no service touched.
  `testing/opus/reference.py` records per file: opusdec's WAV frame count at `--rate 48000`, opusinfo's printed
  fields and warnings, and libogg (through ctypes) page by page and packet by packet, each packet's samples from
  libopus's `opus_packet_get_nb_samples`. The files and the answers are pinned (`testing/opus/files.sha256`,
  `reference.jsonl`), so `cargo test` checks offline; the gate asks the reference again when the host answers (this
  run: **57 / 57 answers identical** to the record). opusenc with a fixed serial reproduced the same bytes twice.
- `oracle_opus_good_files`: **35 / 35 files on each of 18 checks** — 14 written by streamair (silence of five lengths;
  1, 7, 255 and 300 packets per page; a 70,000-byte packet and one of exactly 255 × 255 bytes across pages; every
  frame size and frame-count code; stereo; pre-skip 0 and 3,840) and 21 encoded by opusenc on production from the
  pinned WAVs (2.5 / 5 / 10 / 20 / 40 / 60 ms frames, 6–96 kb/s, hard CBR, complexity 0, stereo, downmix, **six
  channels in mapping family 1**, a 68 KB OpusTags with a picture spanning pages, UTF-8 comments, a 201-sample file
  whose only audio page is its last). The checks: **duration = opusdec's sample count**, the per-packet keep counts
  sum to it, pre-skip, channels, input rate, gain, vendor, comments (the picture as opusinfo summarises it, decoded
  from base64), playback length, packet duration max/avg/min, bytes, pages, audio packets, decoded samples, every
  page's (sequence, granule, flags) and every packet's (bytes, samples) by FNV-1a digest against libogg, the last
  granule, and that the reference itself found no fault (only advisory warnings: "high muxing delay" on pages longer
  than a second, "implausibly low preskip" on pre-skip 0).
- `oracle_opus_adversarial_files_refused`: **21 / 21** corruptions refused with the named error, its byte offset and
  page (CRC byte, body bit, capture, version, serial, BOS again, continued flag, dropped and duplicated page, three
  truncations, a cut at a page boundary, EOS removed, granule backwards / off by a packet / beyond the samples at EOS,
  OpusHead and OpusTags magic, pre-skip past the end, an appended stream), and **1 / 1** valid variant accepted (every
  granule +48,000: a stream that starts mid-broadcast; 528,000 samples = opusdec's). The test rebuilds each file from
  its rule and checks its sha256 against what the reference saw; the reference noticed all 21 as well.
- Discriminators (`oracle_opus_discriminators`): pre-skip **added** instead of subtracted is wrong on 34 / 35 files
  (right only on the pre-skip-0 file); no end trimming, 34 / 35; the code-3 frame count ignored, 3 / 35 (the files
  with multi-frame packets); **zlib's CRC-32 verifies 0 of the 427 pages**, Ogg's all 427. Each is caught.
- Round trips (`streamair/tests/roundtrip.rs`): streamair's writer → voaice's reader on 400 random streams, every
  packet's bytes, the duration and the length back exactly.

### Found
- **streamair 0.0.1's `mux` accepts an end trim larger than the samples on the last page** (it bounds the trim by
  5,760 samples), so the EOS granule goes backwards. opusinfo 0.2 on production calls such a file an ERROR ("interior
  holes or more than one page of end trimming") and opusdec plays the untrimmed length; voaice refuses it
  (`GranuleBackwards`). Pinned as `known_issue_the_writer_accepts_end_trimming_past_the_last_page`; the fix is
  streamair's next step (its source is not changed here).

### Measured (gate step 6, only after 4b passed; 4-CPU Ryzen 3 3200U, 1-minute load 1.98 at the start, falling from 6.8)

| file | bytes | audio | read from a slice | CPU per read | from the file | heap peak |
|---|---|---|---|---|---|---|
| JFK ×3, 6 kb/s, 1,651 packets | 24,617 | 33 s | 0.044 ms (534 MiB/s) | 0.045 ms (**≈ 730,000× real time**) | 0.125 ms | **66,281 B** |
| JFK, 2.5 ms frames, 4,403 packets | 50,156 | 11 s | 0.087 ms (552 MiB/s) | 0.095 ms | 0.137 ms | 66,289 B |
| JFK, 60 ms frames, 184 packets | 34,119 | 11 s | 0.032 ms (1,005 MiB/s) | 0.030 ms | 0.065 ms | 66,288 B |
| 60 s of 1-byte DTX packets | 7,782 | 60 s | 0.047 ms (157 MiB/s) | 0.063 ms | 0.256 ms | 65,499 B |
| 70,000 + 65,025-byte packets | 135,850 | 0.17 s | 0.093 ms (1,392 MiB/s) | 0.103 ms | 0.146 ms | 135,453 B |
| 68 KB OpusTags with a picture | 68,384 | 0.3 s | 0.105 ms (623 MiB/s) | 0.100 ms | 0.126 ms | 197,958 B |

- **Ogg CRC-32 sliced by 8: 1,406–1,580 MiB/s against 326–349 MiB/s one byte at a time — 4.06–4.61×** (16 MiB, best
  of 7, six runs). Both are in the crate; the tests require them equal at every length 0–299 and alignment 0–7.
- The heap is one page (65,307 bytes, allocated once) and a few hundred bytes of headers; a packet across pages adds
  exactly its length (the carry buffer reserves exactly: 135,453 = 65,307 + 70,000 + 146 — an amortised doubling
  had made it 195,503); OpusTags is held only while it is parsed. Nothing grows with the file.
- Time per read is per page and per packet, not per byte: small packets (DTX, 2.5 ms) read slower in MiB/s and faster
  in audio seconds. From the file it is three `read` calls a page and the `open`; no buffering layer is added.
- The reference's speed is **not** compared: opusinfo is not on this laptop and the gate does not time on production.
- The earlier stages, re-measured in the same run: the mel 4.78× the reference at one thread (5.52× at 4), the GELU
  op 3.24× (10.5 against 3.25 ms) — lower than 0.0.3's recorded 5.65× and 4.31× at a different load; their code is
  unchanged.

### Added
- `src/ogg.rs`: `Reader` (`new`, `with_max_packet`, `head`, `tags`, `next_packet` → `Packet { data, samples,
  decoded_before, skip, keep, page, first_page, offset }`, `summary`, `finish`), `OpusHead` and `OpusTags` (with
  `parse`), `Summary`, `Error { kind, offset, page, detail }` with 23 named `Kind`s, `crc32` (sliced by 8),
  `crc32_update`, `crc32_bytewise`, `page_crc`, `packet_samples`, `fnv1a`. Rules enforced: RFC 3533 framing; RFC
  7845 §3 header placement (OpusHead alone on the BOS page with granule 0; OpusTags ending a page, granule 0); §4
  granules (start past zero allowed, never before it unless the first audio page is also the last; mid-stream
  granules exact; end trimming only on the EOS page, never past its samples, never before the pre-skip); §5 headers
  (major version 0, family 0 ≤ 2 channels, family 1 ≤ 8 with its table validated). Refused by name, not read:
  chained streams (bytes after EOS) and multiplexed ones (a second serial). OpusTags beyond the packet bound
  (1 MiB by default) is kept up to it and marked truncated; an audio packet beyond it is refused.
- `voaice opus info <file.opus>`, `voaice bench-opus <file.opus>`.
- `tests/opus.rs` (4 tests), unit tests in `src/ogg.rs` (7), `testing/opus/` (`oracle.sh record|check`,
  `reference.py`, `make_inputs.py`, `mutate.py`, 35 pinned files, the recorded answers), gate steps 4b and 6.
- streamair: `examples/opus_corpus.rs` (its 9 corpus files) and `tests/roundtrip.rs`; `streamair/src` unchanged.

### Not checked yet
- The family-1 channel mapping table's values (opusinfo does not print it; the six-channel file's channels, pre-skip
  and duration are checked, the table only for structure). Mapping families 2, 3 and 255: no file in the corpus.
- Chained and multiplexed streams are refused, not read; granules past 2⁶³ are refused as missing.

## 0.0.3 — 2026-10-08 — f32 ↔ f16 and GELU, bit-exact on every input

**The first two encoder kernels, as the shipped ggml-cpu computes them: every f16 pattern widened, every one of the
4,294,967,296 f32 patterns narrowed, all 65,536 GELU table entries and the GELU op on every f32 input — then the op
three times faster.** Record: `testing/results/0.0.3.txt`.

### What the reference does (read from the source and `objdump -d libggml-cpu.so`, then confirmed by the oracle)
- On an F16C build `GGML_CPU_FP32_TO_FP16` is **not** the hardware instruction: simd-mappings.h defines only the
  `COMPUTE_` macro as `_cvtss_sh`, so the kernels' macro falls through to ggml-impl.h's portable bit trick (round to
  nearest even; every NaN → `sign | 0x7E00`). im2col, the GELU index and the GELU table use it. GCC contracted it in
  libggml-cpu (`vmulss` + `vfmadd231ss`); the fused product is by a power of two and exact, so the bits agree with
  libggml-base's uncontracted copy on all 2³² inputs.
- `ggml_cpu_fp32_to_fp16` — the F16 type traits' `from_float` (mul_mat, flash attention; pointer equality checked)
  — converts blocks of 8 and 4 with `vcvtps2ph` (NaN quieted, top 10 payload bits kept) and the last `n % 4` with
  the bit trick: a NaN's f16 depends on its position in the row. The two agree on every non-NaN f32.
- f16 → f32: `ggml_table_f32_f16` (portable, filled at `ggml_cpu_init`) and `vcvtph2ps` agree on all 65,536.
- `ggml_table_gelu_f16[i] = fp32_to_fp16(gelu(fp16_to_fp32(i)))` with `gelu(x) = (0.5·x)·(tanhf((S·x)·fma(A·x, x, 1)) + 1)`
  — GCC fused `A·x·x + 1` into one FMA; glibc `tanhf`. The op: `x <= -10` → +0, `x >= 10` → x, else (NaN too) the
  table at the portable index. Elementwise, so 1 and 4 threads are identical (recorded).

### Measured (testing/release_gate.sh; 4-CPU Ryzen 3 3200U, 1-minute load 7.4 at the start, falling from ≈ 30, after
an hour near 80 from other work on the machine; ±20 % is noise, and more)
- `oracle_f16_to_f32_all_65536`: **65,536 / 65,536** against libggml-base, the table, the F16C row and its tail.
- `oracle_f32_to_f16_boundary_set`: **1,429,656 / 1,429,656** each — the scalar against libggml-base and ggml-cpu's
  inlined copy, the row and the `vcvtps2ph` model against `ggml_cpu_fp32_to_fp16`. The set: every f16 value, every
  halfway point between neighbours (ties), each with its f32 neighbours, the 65,520 and 2⁻²⁵ edges, subnormals,
  infinities, 35 NaN payloads × 2 signs, ±10 ± 20 ULP, 2²⁰ random. The portable and F16C forms differ on 4,029 of
  its 4,047 NaNs, in ours as in the reference.
- `oracle_f32_to_f16_every_pattern`: **all 2³² f32 patterns**, by per-chunk digest (65,536 chunks of 65,536):
  65,536 / 65,536 chunks identical for each of the four comparisons. The reference's own scalar copies agree with
  each other on every chunk, its row and its scalar on 65,280 (the 256 chunks holding NaNs).
- `oracle_gelu_table_all_65536`: **65,536 / 65,536**. `oracle_gelu_op`: **1,495,192 / 1,495,192** (the boundary
  set and every f16 value) on the vector and the scalar path, and 65,536 / 65,536 chunks of all 2³².
- Discriminators: round-half-away differs in **31,752** boundary values and **8,448** chunks — rejected; the GELU
  without the FMA differs in **1** table entry (`0xBFFF`, −1.999: 43,474 vs 43,475) — rejected, and that one entry
  is the whole effect of the FMA.
- Efficiency, only after the above (step 5b; each side in fresh processes):

  | measure | reference | voaice 0.0.3 | ratio |
  |---|---|---|---|
  | GELU op, 1,536 × 1,500, 1 thread (reference net of its graph's two copies) | 26.04 ms | **6.04 ms** | **4.31×** |
  | GELU op, voaice's scalar path | 26.04 ms | 18.75 ms | 1.39× |
  | GELU table, first build (best of 5 processes; the reference's `ggml_cpu_init` also fills its quick-GELU and f32←f16 tables) | 3.16 ms | 2.49 ms | 1.27× |
  | f32 → f16 row, 576,000 values | 0.172 ms | 0.139 ms | 1.24× |
  | f16 → f32 row, 576,000 values | 0.184 ms | 0.172 ms | 1.07× |

  An earlier full run of the same gate on the same kernels (only doc comments changed since; higher load) measured the op at 18.43 against 6.23 ms
  (2.96×), the table 1.17×, the rows 0.90× and 1.01×: the op's gain is real, its size is not known to better than
  that range here. The rows are the same instructions on both sides; their ratios are noise. The op's gain: the table held widened
  to f32 (one lookup, not two) and eight lanes at once (`vcvtps2ph` for the index with the NaN lanes re-indexed to
  the portable `sign | 0x7E00`, compares for the clamps, one gather). The mel (step 5) is unchanged from 0.0.2:
  5.65× the reference at one thread in this run (6.21× in the earlier one).

### Added
- `src/f16.rs`: `fp16_to_fp32`, `fp32_to_fp16` (the portable ports), `fp32_to_fp16_f16c` (`vcvtps2ph` in integers),
  `fp32_to_fp16_row` / `fp16_to_fp32_row` (F16C when the CPU has it — checked bit-identical — the model otherwise),
  `fp32_to_fp16_round_half_away` (the discriminator).
- `src/gelu.rs`: `gelu_f32` (the shipped FMA order), `gelu_f32_unfused` (the discriminator), `table_with`, `Gelu`
  (`new`, `row`, `row_scalar`).
- `whisper_oracle --f16 <dir>` (records the conversions from libggml-base and libggml-cpu, the exported tables and
  the op through a ggml graph, on every f16 and every f32 pattern; about 2 minutes on one thread here) and
  `--bench-f16 init|rows`; `voaice bench-f16 init|rows`.
- Seven oracle tests (13 in all) and gate step 5b.
- Changed: the decade's order — f32↔f16 + GELU became 0.0.3 (the encoder's order of work starts there); the Ogg/Opus
  reader is now 0.0.4, the resampler 0.0.5, conv1 0.0.6 (docs/ROADMAP.md).
- `src/lib.rs` allows clippy's `chunks_exact_to_as_chunks` on the `sha512` module (new in clippy 1.99; that module
  belongs to the vclone work and is left unedited) so the gate's `clippy -D warnings` passes.

## Also in 0.0.3 — vclone (committed before it, released with it)

- `src/vclone.rs`: the vprint (`dvscope/1`) byte-identical to cryptoAGI/voaice `tools/vprint.py`. Checked on 2,000
  recorded metric sets (`testing/vclone/make_oracle.py`, which names the vprint.py it ran by sha256) and on all 10
  measured `.voaice` identities, every field (`tests/vclone.rs`).
- The forge log (`vclone-event/1`): hash-chained events for capture, measure, ref, consent, model, actor, prompt,
  skill, tool, language and forge, and `mintable()`.
- `src/json.rs` (an order-keeping JSON reader and writer) and `src/sha512.rs` (FIPS 180-4), in-crate: still zero
  dependencies.
- `voaice vclone check | print | log`. Plan and TODO: [docs/VCLONE.md](docs/VCLONE.md).

## 0.0.2 — 2026-10-08 — the mel, six times faster, still 0 ULP

**The log-mel front end without the waste: no allocation per frame, no padded copy of the audio, vectors only
across independent lanes, threads by frames — and every float operation, in its order, still the reference's.**
The oracle says so on every value of every input. Record: `testing/results/0.0.2.txt`.

### Measured (testing/release_gate.sh, the same laptop: 4 CPUs, load ≈ 7.6 during the run, so ±20 % is noise)
- `oracle_mel_bit_exact`: **2,316,640 / 2,316,640** values identical, **max 0 ULP**, 8 inputs (unchanged from 0.0.1).
- `oracle_mel_threads_bit_identical` (new): at 2, 3, 4 and 8 threads, **9,266,560** values identical to 1 thread
  and to the reference.
- `oracle_mel_discriminates_fused_fft`: the fused variant still differs in **25,242 of 328,000** values on JFK.
- The other three oracles unchanged (167/167 tensors, 51,864/51,864 tokens, PCM identical, guard refuses).
- Efficiency, only after the above; wall = best of 10, CPU per call over ≥ 1 s, heap = bytes live at the first
  call's peak; 0.0.1 **rebuilt from its tag and run in the same gate** (not its recorded numbers):

  | input | reference 1 thread | 0.0.1 | **0.0.2, 1 thread** | reference 4 threads | 0.0.2, 4 threads | CPU ms ref / 0.0.2 | heap KiB ref / 0.0.2 |
  |---|---|---|---|---|---|---|---|
  | JFK, 11 s | 60.1 ms | 62.9 ms | **9.5 ms** | 52.0 ms | 8.8 ms | 58.3 / **9.7** | 3,868 / **1,282** |
  | JFK ×3, 33 s | 198.8 ms | 167.8 ms | **24.9 ms** | 116.7 ms | 15.8 ms | 162.0 / **25.8** | 5,925 / **1,969** |
  | chirp, 2.5 s | 13.2 ms | 14.1 ms | **2.1 ms** | 8.9 ms | 1.6 ms | 13.7 / **2.2** | 3,068 / **1,016** |
  | 201 samples | 1.36 ms | 1.8 ms | **0.34 ms** | 1.52 ms | 0.34 ms | 1.6 / **0.38** | 2,836 / **938** |

  Geometric means over the 8 inputs: **6.10× faster than 0.0.1** and **6.78× faster than the reference** at one
  thread; at 4 threads 5.92× the reference at 4. CPU per call is about **one sixth** of the reference's, and the heap
  **one third**: what is left is the output itself (80 × 4,100 × 4 bytes = 1,281 KiB on JFK) — the reference also
  holds the audio padded with 30 s of zeros. Threads help less than they could here: the host was already running
  ≈ 7.6 runnable tasks on 4 CPUs, so 4 threads bought 1.1–1.6× on the long inputs; inputs under 128 frames per thread
  stay on one (their spawn costs more than it saves).

### Changed — `src/mel.rs`
- `MelPlan`: the invariants built once (window, each butterfly level's twiddles, the 25-point DFT's gathered
  `table[(k*n*16) % 400]`, each band's non-zero span of the filterbank); `MelPlan::run(samples, threads)` allocates
  only its output.
- The FFT recursion unrolled bottom-up and in place: the 16 leaf DFTs read the frame by stride, the butterflies of
  each level work on [even | odd] halves exactly where the recursion would have put them, with the spectrum split
  into re and im so butterfly k is lane k. The DFT computes its 25 outputs side by side (lane k accumulates over n in
  the reference's order). AVX2 code path chosen at run time (`is_x86_feature_detected!`), mul and add only.
  Tried and measured slower, so not kept: lanes across the 16 leaves instead of across k.
- Band sums skip the all-zero groups outside each band's span (an exact +0), except in a frame with a non-finite
  power, which takes the full loop (`inf * 0 = NaN`).
- No padded copy: a frame reads the reflected head and the audio directly.
- Threads (`log_mel_spectrogram_threads`, `MelPlan::run`): contiguous runs of frames, each frame whole in one thread;
  the clamp pass split by range. Default 1.
- Fixed: `log10(std::max(sum, 1e-10))` now keeps C++'s `max` — a NaN sum stays NaN, where Rust's `f64::max` gave
  1e-10 (only reachable with non-finite samples; unit-tested).

### Added
- `src/measure.rs`: CPU seconds (`/proc/self/stat`), RSS and peak RSS (`/proc/self/status`, peak reset through
  `/proc/self/clear_refs`), and the `bench` loop — no crate, no libc binding.
- `voaice bench-mel <model> <wav> [--threads N]` (with a counting allocator for the heap peak); `voaice mel --threads N`.
- `whisper_oracle --bench-mel <model> <wav> <threads>`: the reference measured the same way (operator new counted).
- Gate step 5 rewritten: 1 and nproc threads, both sides, plus 0.0.1 rebuilt from its tag; the record keeps only
  repo-relative paths. `testing/oracle/build.sh | head -1` replaced by a log file (under `pipefail` the head could
  SIGPIPE the build script and fail the gate).
- Unit test `same_bits_as_the_0_0_1_port_at_every_thread_count`: 0.0.1's port kept verbatim (test-only) as a second
  witness at 1, 2, 3, 4, 7 threads and in the fused variant.
- docs/ROADMAP.md: the first decade (0.0.3 the streaming Ogg/Opus reader … v0.1.0 the encoder), the milestones to
  v1.0.0, and the Opus efficiency thread.

## 0.0.1 — 2026-10-07 — the oracle, the loader, the log-mel front end (stages 0–2)

**voaice.rs begins: the reference pinned, an oracle that reads the shipped whisper.cpp library, a model loader whose
167 tensors are byte-identical to what whisper's loader holds, and a log-mel front end bit-exact on every value of
every test input.** It does not transcribe yet. Record: `testing/results/0.0.1.txt`.

### Measured (testing/release_gate.sh, this laptop: 4 CPUs, AVX2+FMA, gcc 11.4.0, glibc 2.35)
- `oracle_model_hparams_tensors_vocab_filters`: 11 hparams equal; **167 / 167 tensors** equal in name, type, shape,
  byte count and the sha256 of the bytes in the loader's memory (77,110,272 bytes); filterbank 80×201 bit-exact;
  **51,864 / 51,864** token strings equal as `whisper_token_to_str` shows them (token 188 is the byte `0x00`, which
  that C API cannot show; voaice holds it).
- `oracle_mel_bit_exact`: **2,316,640 / 2,316,640** f32 values identical, **max 0 ULP**, over 8 inputs: JFK (11 s), JFK
  ×3 (33 s, past one 30-s window), a chirp, full-scale noise with ±32767/−32768, silence, 0.3 s, a length off the hop
  (12,345), and the 201-sample minimum. `n_len` and `n_len_org` equal on all 8.
- `oracle_mel_discriminates_fused_fft`: the same pipeline with the FFT's multiply-adds fused differs in **25,242 of
  328,000** values on JFK — the oracle can tell float orders apart.
- `oracle_pcm_input_identical`: the samples voaice reads from each WAV are the f32 bits fed to whisper.
- `guard_refuses_a_modified_model`: one flipped bit in `decoder.token_embedding.weight` still parses and is refused by
  the pin, with the reason.
- Reference determinism, recorded: the mel at 1 and 4 threads is bit-identical (all 8 inputs); `whisper_full` at 1
  thread is identical run to run (all 8); **at 4 threads against 1 the token ids and text are the same but each
  token's `p` differs, and on JFK the token timestamps too** — the decoder's arithmetic depends on the thread count,
  so the transcript oracle is pinned at 1 thread (production's whisper-cli runs min(4, cores) threads).
- Speed, only after the above (mel, one thread each, best of 5, the laptop at load 9.6 so ±20 % is noise): about
  **parity** — voaice 0.82–1.22× the reference's time per input (JFK 58.7 ms against 57.7; JFK ×3 198.1 against
  175.8; chirp 14.4 against 16.1). Nothing has been optimized: this is the exact port, allocations and all.

### Added
- `upstream/PIN`: whisper.cpp `080bbbe85230f624f0b52127f1ae1218247989f9` (`v1.9.1-154-g080bbbe8`), ggml 0.16.0 — no
  release tag bundles 0.16.0 — and `ggml-tiny.en.bin` by size and sha256.
- `testing/oracle/`: `build.sh` (reference at the pin, CPU-only shared build, refuses another commit or ggml version),
  `layout_probe.cpp` (offsets of the internals the API does not expose, from the pinned source), `whisper_oracle.cpp`
  (records model, vocab, filterbank, mel at 1 and 4 threads with its time, and greedy transcripts with token ids,
  timestamps and `p` bits; self-checks every probed offset against a public getter).
- `testing/make_audio.py` + `testing/pins/audio.sha256`: 8 deterministic test WAVs (JFK from whisper.cpp's samples,
  the rest synthetic), pinned.
- `src/sha256.rs` (FIPS 180-4 vectors), `src/model.rs` (format, guard, the tensor set whisper creates for the hparams,
  extra-token names including `[_LANG_xx]`), `src/wav.rs`, `src/mel.rs`, `src/main.rs` (`voaice info | mel | version`).
- `tests/oracle.rs`, `testing/release_gate.sh`, README, TODO (stage 3–6 plan).

### Fixed (found on the way)
- `sha256::update` reset a partial block's length when its input fitted in the buffer (a hang on any input that was
  not a multiple of 64 bytes: the FIPS-vector test hung).
- The oracle's first layout self-check was wrong (`whisper_n_len_from_state` returns `n_len_org`); the check refused
  to run, which is what it is for.
