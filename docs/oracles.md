# Oracles

An oracle runs the reference's **shipped, compiled** library on the same input as voaice.rs and compares the two
results bit for bit. A count is reported as matched / total, with the largest ULP distance. Every oracle has a
discriminator where one is meaningful: a deliberately wrong variant it must reject, because an oracle that cannot
fail proves nothing.

Run them all with `testing/release_gate.sh`; the record is written to `testing/results/<version>.txt`. In
`tests/oracle.rs` they are `#[ignore]`d for plain `cargo test`, because they need the pinned model and a recorded
oracle run.

| oracle | compares | result (0.0.3) |
|---|---|---|
| `oracle_model_hparams_tensors_vocab_filters` | the 11 hparams; every tensor's name, type, shape, byte count and the sha256 of its bytes **as whisper's loader holds them in memory**; the 80×201 filterbank; every token string | 11/11 · **167/167** tensors (77,110,272 bytes) · filterbank bit-exact · **51,864/51,864** tokens |
| `oracle_pcm_input_identical` | the f32 samples voaice reads from each WAV against the samples fed to whisper | identical on all 8 inputs |
| `oracle_mel_bit_exact` | every value of the log-mel spectrogram | **2,316,640/2,316,640, max 0 ULP**, 8 inputs |
| `oracle_mel_threads_bit_identical` | the mel at 2, 3, 4 and 8 threads against 1 thread and against the reference | **9,266,560** values identical, 8 inputs |
| `oracle_mel_discriminates_fused_fft` | the discriminator: the FFT with fused multiply-adds | differs in 25,242 of 328,000 values on JFK — the oracle tells float orders apart |
| `guard_refuses_a_modified_model` | a model with one flipped bit in `decoder.token_embedding.weight` | parses, and is refused by the pin with the reason |
| `oracle_f16_to_f32_all_65536` | (0.0.3) every f16 pattern widened by the port, against libggml-base's `ggml_fp16_to_fp32`, ggml-cpu's `ggml_table_f32_f16`, its `ggml_cpu_fp16_to_fp32` row (F16C `vcvtph2ps`) and that row's scalar tail | **65,536/65,536** each of the four |
| `oracle_f32_to_f16_boundary_set` | (0.0.3) 1,429,656 f32 inputs — every f16 value and its f32 neighbours, every halfway point between adjacent f16 values and its neighbours, the overflow edge (65,520), the underflow edge (2⁻²⁵), subnormals, infinities, 35 NaN payloads × 2 signs, ±10 ± 20 ULP, 2²⁰ random — narrowed by the portable scalar (against libggml-base and ggml-cpu's inlined, FMA-contracted copy), by the row (against `ggml_cpu_fp32_to_fp16`'s F16C blocks) and by the `vcvtps2ph` model | **1,429,656/1,429,656** each; the portable and F16C conversions differ on 4,029 of the 4,047 NaN inputs, as the reference's do |
| `oracle_f32_to_f16_every_pattern` | (0.0.3) **all 2³² f32 patterns**, the same three ways, compared by a per-chunk digest (65,536 chunks of 65,536; FNV-1a over the packed outputs) | **65,536/65,536** chunks each; the reference's own scalar copies agree on all 65,536, its row and scalar on 65,280 (the 256 chunks holding NaNs) |
| `oracle_f16_discriminates_round_half_away` | the discriminator: ties rounded away from zero | differs in 31,752 boundary values and 8,448 of the 65,536 chunks — rejected |
| `oracle_gelu_table_all_65536` | (0.0.3) the exported `ggml_table_gelu_f16`, entry for entry | **65,536/65,536** |
| `oracle_gelu_table_discriminates_unfused` | the discriminator: GELU in the source's order without the FMA GCC formed | differs in 1 entry (`0xBFFF`, −1.999) — rejected; the FMA changes exactly one entry |
| `oracle_gelu_op` | (0.0.3) `ggml_gelu` through a graph on the shipped CPU backend (1 thread; 4 threads recorded identical) on the boundary set plus every f16 value (1,495,192), then on all 2³² by digest; the vector and the scalar path | **1,495,192/1,495,192** both paths; **65,536/65,536** chunks of all 2³² both paths |

Below the oracles, `cargo test` carries a second witness for the mel that needs no model:
`same_bits_as_the_0_0_1_port_at_every_thread_count` keeps 0.0.1's allocating port verbatim (test-only) and requires
the optimized path, at 1, 2, 3, 4 and 7 threads and in its fused variant, to give the same bits on synthetic audio
and a synthetic filterbank with zero runs.

## Efficiency — measured only after the oracles pass

Step 5 of the gate, in the same run, each row a fresh process for the reference (`whisper_oracle --bench-mel`) and
for voaice (`voaice bench-mel`), measured the same way on both sides:

| measure | how |
|---|---|
| wall | best of 10 calls (`ggml_time_us` / `Instant`), after one untimed call |
| CPU | `/proc/self/stat` utime + stime (all threads, joined ones included) over a loop of ≥ 1 s, per call; ticks are 1/100 s, so the loop is what makes them small |
| heap | bytes live at the first call's peak: voaice's counting `GlobalAlloc`; the reference's `operator new`/`delete` replaced in the oracle executable (libwhisper's `std::vector`s bind to it) |
| RSS | `VmHWM` after resetting it through `/proc/self/clear_refs`, minus `VmRSS` before the call — reported, but heap reuse after the model load can hide growth, so heap is the comparable number |
| 0.0.1 | rebuilt from tag `v0.0.1` in the same run (scratch under `.oracle/`, removed after); its `voaice mel` times the mel call alone |

No crate and no libc binding is used for any of it: `/proc` is read as text.

Step 5b (0.0.3) measures the f16 work the same way, each side in fresh processes (`voaice bench-f16`,
`whisper_oracle --bench-f16`): the first build of the GELU table (best of 5 processes; the reference's
`ggml_cpu_init` also fills its quick-GELU and f32←f16 tables, which voaice does not need, so that ratio flatters
voaice and says so), the f32↔f16 rows on 384 × 1,500 values, and the GELU op on 1,536 × 1,500 at one thread (the
reference's through a one-op ggml graph; its two tensor copies are measured alone and taken off).

**How the oracle sees the conversions.** Everything it compares is the shipped code: `ggml_fp16_to_fp32` /
`ggml_fp32_to_fp16` are exported by libggml-base, `ggml_cpu_fp16_to_fp32` / `ggml_cpu_fp32_to_fp16` and the data
symbols `ggml_table_f32_f16` / `ggml_table_gelu_f16` by libggml-cpu (`nm -D`). The scalar `GGML_CPU_FP32_TO_FP16` is
inlined, not exported: the oracle reaches ggml-cpu's own compiled copy by calling `ggml_cpu_fp32_to_fp16` on three
values at a time (always its tail loop), and the GELU op's copy through the op. The copies inlined in im2col and
flash attention are not observed until those nodes are (0.0.6 onward).

## The test inputs

Eight WAVs, generated by `testing/make_audio.py` and pinned by sha256 in `testing/pins/audio.sha256`: JFK (11 s,
from whisper.cpp's samples), JFK ×3 (33 s, past one 30-s window), a chirp, full-scale noise touching ±32767/−32768,
silence, 0.3 s, a length off the hop (12,345 samples), and the 201-sample minimum.

## How the oracle sees what the API hides

whisper.cpp's public C API has no getter for the mel it computes, the tensors its loader holds, or the filterbank.
`testing/oracle/layout_probe.cpp` compiles the pinned source with the same compiler and prints `offsetof()` for each;
`testing/oracle/whisper_oracle.cpp` reads the shipped library's own objects at those offsets, **after checking each
against a public getter** (the tensor map's size against the loader's count, `n_len_org` against
`whisper_n_len_from_state`, `n_mel` against `whisper_model_n_mels`), and refuses to run on any mismatch.

## Determinism of the reference

- The mel is bit-identical at 1 and 4 threads (each frame is computed whole by one thread).
- `whisper_full` at 1 thread is identical run to run.
- At 4 threads against 1, token ids and text are the same, but every token's probability differs in its bits, and
  on JFK the token timestamps move. The transcript oracle is therefore pinned at **1 thread**; a bit-exact transcript
  will always state its thread count.
