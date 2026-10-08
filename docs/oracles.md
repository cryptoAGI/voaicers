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

### The Ogg/Opus reader (0.0.4) — `tests/opus.rs`, offline against a recorded reference

The reference is not on the dev laptop (no opus-tools), so it is run where it is — mindX production: **opus-tools 0.2
(`opusinfo`, `opusdec`), libopus 1.4 and libogg 1.3.5** (Ubuntu packages `opus-tools 0.2-1build3`, `libopus0
1.4-1build1`, `libogg0 1.3.5-3build1`) — read-only over ssh, in one scratch directory that is removed after
(`testing/opus/oracle.sh`). Its answers are recorded with the files they are about, so the comparison runs offline in
plain `cargo test`, and the gate asks the reference again when the host answers.

| what | how |
|---|---|
| the files | 35 good ones pinned by sha256 (`testing/opus/files.sha256`): 14 written by streamair (silence of 0.001 / 1 / 2.5 / 7.3333 / 60 s; packets per page 1, 7, 255 and 300; a 70,000-byte packet and one of exactly 255 × 255 bytes crossing pages; every frame size and frame-count code; stereo; pre-skip 0 and 3,840) and 21 encoded by `opusenc` on production from the pinned WAVs (JFK at 2.5 / 5 / 10 / 20 / 40 / 60 ms frames, 6 to 64 kb/s, hard CBR, complexity 0, `--max-delay 0`, stereo, downmix, **six channels in mapping family 1**, a PNG that makes OpusTags 68 KB and span pages, UTF-8 and `=`-bearing comments, the 201-sample file whose only audio page is also its last). opusenc is given a fixed serial; two recordings produced the same bytes |
| the reference's answers | `testing/opus/reference.py` per file: **opusdec `--rate 48000` to WAV, frames counted from the data chunk** (the exact samples it plays); opusinfo's fields as printed (pre-skip, gain, channels, original rate, vendor, comments, packet duration max/avg/min, playback length, total data length) and its warnings; and **libogg through ctypes**: pages and packets as `ogg_sync_pageout` / `ogg_stream_packetout` give them, an FNV-1a digest of every page's (sequence, granule, flags) and of every audio packet's (bytes, samples), with each packet's samples from libopus's own `opus_packet_get_nb_samples` |
| the adversarial files | 22 made from two good files by `testing/opus/mutate.py` (a CRC byte, a body bit, the capture pattern, the version, a serial, BOS again, the continued flag, a dropped and a duplicated page, three truncations and a cut at a page boundary, EOS removed, granules backwards / off by a packet / beyond the samples at EOS, the OpusHead and OpusTags magic, pre-skip past the end, a second stream appended — and one *valid* change, every granule +48,000, a stream that starts mid-broadcast); `tests/opus.rs` makes the same bytes by the same rules and checks their sha256 against what the reference saw |

| oracle | compares | result (0.0.4) |
|---|---|---|
| `oracle_opus_good_files` | 18 checks per file: **duration = opusdec's sample count**; the per-packet keep counts sum to it; pre-skip, channels (opusinfo and opusdec), input rate, gain, vendor, comments (a picture as opusinfo summarises it, decoded from base64), playback length, packet duration max/avg/min, total bytes; pages, audio packets, decoded samples (libopus per packet), every page's sequence/granule/flags and every packet's bytes/samples (libogg digests), the last granule; and that the reference itself found no fault (advisory warnings only: "high muxing delay", "implausibly low preskip") | **35 / 35 files on each of the 18 checks** |
| `oracle_opus_adversarial_files_refused` | each corruption refused with its named error kind, the byte offset and page in the message; the valid variant accepted with opusdec's count | **21 / 21 refused** as expected, **1 / 1** valid variant accepted (start granule 48,000, duration 528,000 = opusdec's); the reference noticed all 21 too (opusinfo warned, or opusdec's output changed or failed) |
| `oracle_opus_discriminators` | readers wrong in plausible ways, run against the same answers | pre-skip **added**: wrong on 34 / 35 (right only where pre-skip is 0); no end trimming: wrong on 34 / 35; the code-3 frame count ignored: wrong on 3 / 35 (the files with multi-frame packets); **zlib's CRC-32 verifies 0 of the 427 pages**, Ogg's all 427 |
| `bounded_memory_one_byte_at_a_time` | a source that yields one byte per `read` gives the same summary | identical on the continuation, picture and 2.5 ms files |
| streamair `tests/roundtrip.rs` | streamair's writer → voaice's reader on 400 random streams (1–300 packets, 1–260 per page, payloads to 70,000 bytes, every TOC code, mono/stereo, pre-skip 0–3,999, a trim within the last packet) | every packet's bytes, the duration and the file length back exactly |

**Not checked by any oracle yet:** the channel mapping *table* of family 1 (opusinfo does not print it; the
six-channel file's channel count, pre-skip and duration are checked, the table is parsed and validated for structure
only); mapping families 2, 3 and 255 (no file in the corpus); chained and multiplexed streams (refused by name, not
read); granule positions past 2⁶³.

Below the oracles, `cargo test` carries a second witness for the mel that needs no model:
`same_bits_as_the_0_0_1_port_at_every_thread_count` keeps 0.0.1's allocating port verbatim (test-only) and requires
the optimized path, at 1, 2, 3, 4 and 7 threads and in its fused variant, to give the same bits on synthetic audio
and a synthetic filterbank with zero runs.

### The audio reader (0.0.5) — `tests/resample.rs` against whisper-cli's own `libcommon.a`

`testing/oracle/resample_oracle.cpp` links the pinned build's `examples/libcommon.a` — the object whisper-cli is
linked with: `common-whisper.cpp` and miniaudio 0.11.24 inside it, compiled `-O3 -DNDEBUG -fPIC` with no `-march`
(the gate counts its FMA instructions: 0) — and calls `read_audio_data(path, pcm, pcms, /*stereo=*/false)` exactly as
whisper-cli does without `--diarize`, writing the vector it returns. The corpus is `testing/make_resample_audio.py`
(55 files, sha256-pinned in `testing/pins/resample.sha256`). How the source was read is in
`testing/resample/NOTES.md`.

| oracle | compares | result (0.0.5) |
|---|---|---|
| `oracle_resample_bit_exact` | voaice's `resample::read` (streamed from the file) against the vector, length and every bit pattern: 8 / 16 / 22.05 / 24 / 32 / 44.1 / 48 kHz × mono, stereo × s16, f32; u8, s24, s32 at 22.05 / 44.1 / 48 kHz; six channels in WAVE_FORMAT_EXTENSIBLE; f32 at ±4, ±1, −0.0 and subnormal; 1, 2, 3, 4, 5, 7-frame inputs; a padded odd LIST chunk; a data chunk claiming more than the file holds; JFK held to 48 kHz; 60 s of 44.1 kHz stereo | **55 / 55** files, **1,955,875 / 1,955,875** samples; 3 files end in the length rule's zero tail, reproduced |
| `oracle_resample_streaming_equals_whole` | the data chunk pushed through `Converter` in pseudo-random 1–9,000-byte pieces (frames split across pushes) | **55 / 55** files equal the reference |
| `oracle_resample_discriminates` | wrong readings of miniaudio: a low-pass of order 2 or 6; the stereo mixdown as `L + R` (the diarize path's) or the left channel alone; the output length as the samples the resampler makes | order 2: **45 / 50** resampled files differ, order 6: **45 / 50** (the 5 not caught are 1–2-frame inputs whose only output is the leading 0); `L + R`: **8 / 8**, `L` alone: **8 / 8** stereo s16 files; the length: **3 / 55** (exactly the files with a zero tail) |

### Encoder conv1 and `ggml_vec_dot_f16` (0.0.6) — `tests/conv1.rs` against the conv graph's own nodes

`whisper_oracle --conv1` loads the model, computes each input's mel, and runs `whisper_encode_with_state` at offset 0
three times: twice with `ggml_backend_sched_set_eval_callback` set on the state's **conv scheduler**
(`whisper_state::sched_conv`, at an offset from the layout probe; self-checked: non-null, only the `CPU` backend, and
the graph must show IM2COL, MUL_MAT, ADD, GELU in that order) at 1 and at 4 threads, and once without it. The
callback copies every node's output as the scheduler finishes it; the shipped library does all the arithmetic. It
also calls `ggml_get_type_traits_cpu(GGML_TYPE_F16)->vec_dot` — which it checks is the exported `ggml_vec_dot_f16` —
on real rows. How the order was decided, and the disassembly it agrees with: `testing/conv1/NOTES.md`.

| oracle | compares | result (0.0.6) |
|---|---|---|
| `oracle_vec_dot_f16_kernel` | 1,436 dots by the shipped kernel: 16 random row pairs of every one of the model's 70 f16 tensors (as mul_mat reads them: n = 240, 384, 1,152, 1,536), every length 1–300 on two real conv2 rows (every tail 0–31, and n < 32 where there is no vector loop), 64 vectors of random finite f16 patterns (every exponent, both signs, subnormals) | **1,436 / 1,436** identical, 303 distinct lengths |
| `oracle_vec_dot_f16_discriminators` | one f32 accumulator in index order; the right lanes with the tail added in f32 instead of double; the four accumulators reduced in sequence instead of pairwise | **1,265**, **211** and **680** of 1,436 dots differ — each rejected |
| `oracle_conv1_im2col_bit_exact` | conv1's IM2COL node (f16 [240, 3000]: the mel window rounded by ggml-cpu's inlined `GGML_CPU_FP32_TO_FP16`) against voaice's, from voaice's own mel of the WAV | **5,760,000 / 5,760,000** f16 values, 8 inputs |
| `oracle_conv1_bit_exact` | the MUL_MAT node (conv1 without its bias, f32 [3000, 384]), the ADD (+ bias) and the GELU after it, against voaice's fast path at 1 and 4 threads; the record must also say the reference's own 1- and 4-thread nodes are identical, that `embd_conv` is the same observed and not, and that the standalone graph `--bench-conv1` times equals the scheduler's node | **0 values differ** in any of the three nodes on any of the 8 inputs at either thread count (55,296,000 compared); all three record checks yes |
| `oracle_conv1_discriminators` | conv1 with im2col kept in f32 (no f16 rounding), and with a one-accumulator dot, on JFK | **1,151,932** and **1,068,048** of 1,152,000 values differ — both rejected |

What this holds for: **this laptop's** native libggml-cpu (Zen+: AVX2, FMA, F16C, no AVX-512), which takes the AVX
path of `ggml_vec_dot_f16`. Production's native build (Zen 3) has the same extensions and so compiles the same path,
and because every product of two halves is exact in f32, the FMA and a separate multiply and add give the same bits in
this kernel — but production's own library was not run by this oracle. An AVX-512 host would take a 16-lane path
with another reduction order; voaice does not model it.

### Encoder conv2, `embd_conv` and the positional embedding (0.0.7) — `tests/conv2.rs` against both schedulers' nodes

`whisper_build_graph_conv` ends at conv2's GELU, named `embd_conv`; the positional embedding is the **encoder
graph's** first op: `ggml_add(view_2d(e_pe, 384, n_ctx, offset 384·4·n_ctx·iter), ggml_cont(ggml_transpose(view of
embd_conv)))`, with `static int iter = 0` (offset 0; the view covers e_pe whole when n_ctx = 1500). The mel input is
always 2·n_ctx frames, zero past the mel (whisper_encode_internal), so a short input takes no other path; only a
non-zero `audio_ctx` would change n_ctx, and that is not checked. `whisper_oracle --conv2` sets one eval callback on
`sched_conv` (every node) and one on `sched_encode` that observes up to the first ADD and then answers `ask` with
false, so the rest of the encoder runs unobserved. It records conv1's GELU (conv2's input), conv2's IM2COL, MUL_MAT,
ADD, GELU, the encoder graph's CONT and ADD; the 18 nodes it saw are identical on all 8 inputs (VIEW of e_pe,
TRANSPOSE, CONT, ADD — nothing else before the first norm). Self-checks, all yes on all 8: the reference's nodes at 1
and 4 threads identical; the conv graph's last node equals `whisper_state::embd_conv` read after an unobserved run;
`embd_enc` (the encoder's whole output) is the same observed and not; the standalone graphs `--bench-conv2` times
(conv2 from conv1's GELU, and the whole stage from the mel) equal the scheduler's nodes at 1 and 4 threads.

| oracle | compares | result (0.0.7) |
|---|---|---|
| `oracle_conv2_im2col_bit_exact` | conv2's IM2COL node (f16 [1152, 1500], stride 2, pad 1) against voaice's, from voaice's own conv1 (itself checked equal to the record's conv1 GELU node) | **13,824,000 / 13,824,000** f16 values, 8 inputs |
| `oracle_conv2_bit_exact` | the MUL_MAT node (f32 [1500, 384]), the ADD (+ bias) and the GELU (`embd_conv`), against voaice's fast path at 1 and 4 threads | **0 values differ** in any node, any input, either thread count (27,648,000 compared) |
| `oracle_positions_bit_exact` | the encoder graph's CONT (embd_conv transposed: frame-major) and its ADD (the encoder's input), against conv2 with the positions fused into its epilogue (from voaice's conv1) and against the whole fused stage from voaice's mel (conv1 → f16 buffer → conv2 → + positions), at 1, 2 and 4 threads | CONT **0 differ**; ADD **0 differ** on both paths at every thread count, 8 inputs (27,648,000 compared) |
| `oracle_conv2_discriminators` | on JFK: conv2 with stride 1 (its first 1,500 frames), a one-accumulator dot, positions added before the transpose (to embd_conv's channel-major layout), positions one frame late (another view offset), GELU before the bias; on every input: im2col kept in f32 | **423,936**, **544,848** of 576,000 MUL_MAT values; **575,247**, **510,942**, **575,992** of 576,000 encoder-input values — each rejected. im2col in f32: rejected on min_len (384 values differ), noise_loud (767), odd_len (384), silence (383); **not discriminable** on jfk, jfk_x3, chirp, short (below) |

**What the f16 rounding of conv2's im2col can be told from.** conv2 reads conv1's GELU output, and GELU returns the
f16 table's value (an f16, widened) for every x < 10, 0 for x ≤ −10 and x itself for x ≥ 10. So the rounding changes
only GELU outputs ≥ 10 that f16 cannot hold: 3 such values in min_len, 2 in noise_loud and odd_len, 1 in silence, none
in the other four inputs, where rounding and not rounding are the same function and no oracle could separate them.
The test counts them on every input and checks the discriminator wherever the count is not zero. The same fact lets
voaice keep conv1's output as f16 (half the bytes) with no change to any bit conv2 sees.

What this holds for: the same as 0.0.6 — this laptop's native libggml-cpu on the AVX path; production's own library
was not run. The f32 add (bias, positions) and the CONT copy are one rounding or none, so their order cannot vary.

### The encoder's layer norms (0.0.8) — `tests/norm.rs` against every NORM, MUL and ADD node of the encoder graph

whisper's encoder applies `ggml_add(ggml_mul(ggml_norm(x, 1e-5f), w), b)` nine times on tiny.en: each block's
`attn_ln` (on the block input) and `mlp_ln` (on the residual after attention), then `ln_post` (whose ADD is
`embd_enc`). `whisper_oracle --norm` sets an eval callback on `sched_encode` that observes **every** encoder node
(127), one at a time. A NORM's input is read when the scheduler *asks* about the NORM — after every earlier node ran
and before the NORM did, so an in-place norm could not have overwritten it; the NORM, the MUL that reads it and the
ADD that reads the MUL are read when computed. The record self-checks, all yes on all 8 inputs: each MUL's `src[0]` is
its NORM and each ADD's `src[0]` its MUL; their `src[1]` are the model's own tensors by pointer
(`encoder.blocks.N.{attn_ln,mlp_ln}.{weight,bias}`, `encoder.ln_post.*`); `op_params` holds eps = 1e-5f; every input is
contiguous f32; the reference's nodes are identical at 1 and 4 threads; `embd_enc` is the same observed and not; the
last ADD equals `embd_enc`; the standalone graphs `--bench-norm` times equal the nodes at 1 and 4 threads.

Blocks 1–3 and `ln_post` read the output of attention and the MLP, which voaice.rs does not compute yet (0.0.9,
v0.1.0), so those norms are fed **the reference's recorded input to that node**; block 0's `attn_ln` is also run end
to end from voaice's own mel.

| oracle | compares | result (0.0.8) |
|---|---|---|
| `oracle_norm_nodes_bit_exact` | the NORM, MUL and ADD nodes of all nine chains, from each NORM's recorded input: voaice's fast path at 1 and 4 threads and the portable model | **0 values differ** on all 8 inputs (373,248,000 compared); of the 108,000 rows, 107,845 proved their sum order-free (lanes), 155 took the in-order loop |
| `oracle_attn_ln_0_from_mel` | block 0's attn_ln from the WAV: voaice's mel → conv stage (0.0.7) → norm, · w, + b, at 1, 2 and 4 threads | the NORM's input **0 differ**, the three nodes **0 differ**, 8 inputs (41,472,000 compared) |
| `oracle_norm_discriminators` | the model with one reading changed, on every chain of every input, against the ADD node | ADD values that differ, of 5,184,000 per input (range over the 8 inputs): the sum in f32 **3.76–3.87 M**; the mean from the double sum (no f32 rounding first) **1.17–1.22 M**; a one-pass variance **0.72–1.01 M**; cvar's squares added one by one in double (no 8-lane f32 reduce) **121–196 k**; eps outside the sqrt **5.09–5.15 M**; the scale in double **0.94–1.00 M**; dividing by the root **1.00–1.02 M**; the MUL and ADD fused into an FMA **1.48–1.64 M** — each rejected on every input |

**What the lane sum's proof is for.** The reference adds a row's 384 values to a double one at a time. voaice adds
them in vector lanes only when the row proves that no partial sum can round in any order (every value a multiple of
2^q, q from the smallest non-zero |x|, and n · max|x| < 2^(53 + q)); otherwise it adds them in order. The
discriminator "lane sum on every row, no proof" changes **no** ADD value on these 8 inputs — the 155 unproven rows'
means round to the same f32 — so the inputs alone would not have caught its absence; a unit test builds a row where
it does change the mean (2^30, seven 2^−25, −2^30: in order the small values vanish) and checks that the fast path
refuses the lanes there.

What this holds for: this laptop's native libggml-cpu (Zen+, AVX2 + FMA; the AVX2 branch of `ggml_vec_cvar_f32`,
which the gate's disassembly shows has no FMA); production's own library was not run. An AVX-512 build takes cvar's
16-lane branch (`_mm512_reduce_add_ps`), another pairing — not compared. `base.en` (n = 512) not compared.

## Efficiency — measured only after the oracles pass

(0.0.4) Step 6 of the gate measures the Ogg/Opus reader after 4b passed: Ogg's CRC on 16 MiB sliced-by-8 against the
one-byte-at-a-time table (best of 7); `Reader::new` + every packet + the end checks from a slice and from the file
(best of 10, and CPU per read over ≥ 1 s); the heap live at one read's peak through the counting allocator. The
reference's speed is **not** compared: opusinfo is not on this laptop and the gate does not run timing on production.


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
flash attention are not observed until those nodes are: im2col's is, since 0.0.6 (the IM2COL node, through the
scheduler's eval callback).

Step 8 (0.0.6) times conv1 — and conv1 + bias + GELU — each side in a fresh process: the reference as a ggml graph of
the same ops on copies of the weights and the mel window, computed by the shipped CPU backend
(`whisper_oracle --bench-conv1`; the record shows this graph's output equals the scheduler's node, at 1 and 4
threads), voaice through `Conv1::run_into` with its output kept between calls (`voaice bench-conv1`), at 1, 2 and
nproc threads. The reference's memory is the bytes of the graph's own tensors (ggml_nbytes: im2col and the product,
plus the add and GELU outputs), voaice's the heap live at the first call's peak, its output included.

Step 9 (0.0.7) times conv2 (+ bias + GELU, from conv1's GELU output, which each side computes beforehand) and the whole
conv stage (the mel → the encoder's input) the same way: the reference as the standalone graphs above
(`whisper_oracle --bench-conv2`), voaice through `Conv2::run_into` and `ConvStage::run_into` with their buffers kept
between calls (`voaice bench-conv`), at 1, 2 and nproc threads. The reference's memory is every non-view node of its
graph (im2cols, products, adds, GELUs, the CONT, the positional ADD), from ggml_nbytes.

Step 10 (0.0.8) times the NORM node alone and the chain norm → · w → + b (block 0's attn_ln weights) on jfk's encoder
input ([1500, 384], computed beforehand by each side's own conv stage): the reference as a ggml graph of the same ops
(`whisper_oracle --bench-norm`; the record shows its outputs equal the scheduler's nodes), voaice through
`LayerNorm::run_into` with its output kept between calls (`voaice bench-norm`), at 1, 2 and nproc threads. The
reference's memory is its graph's non-view nodes (the norm, mul and add outputs), from ggml_nbytes.

## The test inputs

Eight WAVs, generated by `testing/make_audio.py` and pinned by sha256 in `testing/pins/audio.sha256`: JFK (11 s,
from whisper.cpp's samples), JFK ×3 (33 s, past one 30-s window), a chirp, full-scale noise touching ±32767/−32768,
silence, 0.3 s, a length off the hop (12,345 samples), and the 201-sample minimum.

## How the oracle sees what the API hides

whisper.cpp's public C API has no getter for the mel it computes, the tensors its loader holds, or the filterbank.
`testing/oracle/layout_probe.cpp` compiles the pinned source with the same compiler and prints `offsetof()` for each;
`testing/oracle/whisper_oracle.cpp` reads the shipped library's own objects at those offsets, **after checking each
against a public getter** (the tensor map's size against the loader's count, `n_len_org` against
`whisper_n_len_from_state`, `n_mel` against `whisper_model_n_mels`), and refuses to run on any mismatch. Since
0.0.6 it also finds the state's schedulers (`sched_conv`, `sched_encode`) and `embd_conv` that way, and observes
every node of the conv graph through ggml's own `ggml_backend_sched_set_eval_callback`; observing does not change the
result (`embd_conv` is bit-identical with and without the callback, on all 8 inputs).

## Determinism of the reference

- The mel is bit-identical at 1 and 4 threads (each frame is computed whole by one thread).
- conv1's four nodes (im2col, the product, + bias, GELU) are bit-identical at 1 and 4 threads: mul_mat splits rows,
  never a dot (each output is one `ggml_vec_dot_f16` call in one thread). So are conv2's four, the CONT and the
  positional ADD (0.0.7), and all nine norm → mul → add chains of the encoder (0.0.8: each row of a NORM is whole in one
  thread; MUL and ADD are elementwise).
- `whisper_full` at 1 thread is identical run to run.
- At 4 threads against 1, token ids and text are the same, but every token's probability differs in its bits, and
  on JFK the token timestamps move. The transcript oracle is therefore pinned at **1 thread**; a bit-exact transcript
  will always state its thread count.
