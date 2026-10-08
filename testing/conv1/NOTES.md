# 0.0.6 — encoder conv1: notes (written as the work goes)

## The order decision (2026-10-08, from the pinned source)
conv1 = `ggml_conv_1d_ph(e_conv_1_w f16 [3,80,384], mel f32 [3000,80], s=1, d=1)` = `ggml_conv_1d(a, b, 1, p=1, 1)`:
- `im2col(a, b, s0=1, p0=1, d0=1, is_2D=false, F16)` -> f16 [240, 3000]: row t, element ic*3 + kw =
  `GGML_CPU_FP32_TO_FP16(mel[ic][t + kw - 1])`, or 0 outside [0, 3000). (ops.cpp ggml_compute_forward_im2col_f16;
  threads split ic; each element written once — threads cannot change bits.)
- `mul_mat(reshape_2d(im2col, 240, 3000), reshape_2d(w, 240, 384))` -> f32 [3000, 384]: src0 = im2col (F16),
  src1 = the WEIGHTS (F16). vec_dot_type(F16) = F16 = src1's type, so nothing is converted; every output is one
  `type_traits_cpu[F16].vec_dot(240, x = im2col row t, y = weight row c)` call, whole in one thread
  (ggml_compute_forward_mul_mat_one_chunk; chunks split rows, never a dot). nrows = 1.
- So conv1 CANNOT be exact without ggml_vec_dot_f16's order -> vec_dot_f16 is brought forward from 0.0.9 into 0.0.6.
- ggml_vec_dot_f16, AVX path (vec.cpp + simd-mappings.h, GGML_F16_STEP 32, EPR 8, ARR 4): np = n & ~31 = 224 for
  n = 240; 4 accumulators of 8 f32 lanes, acc[j] = fma(x8, y8, acc[j]) over blocks i += 32 (j = 0..3 at i + 8j);
  REDUCE: acc0 += acc2, acc1 += acc3, acc0 += acc1 (lane-wise); then lo128 + hi128, hadd, hadd -> f32
  ((t0+t1)+(t2+t3)), widened to double (ggml_float); leftovers 224..239 summed in DOUBLE, each product an f32
  product of two widened f16; *s = (float)sumf.
- f16 x f16 products are exact in f32 (11+11 significant bits; the smallest, 2^-48, is normal in f32), so
  FMA vs mul+add gives the same bits in this kernel: Zen+ (this laptop) and Zen 3 (production) agree on THIS
  kernel by construction, as long as both take the AVX2 path. (Not true for f32 kernels later.)
- LLAMAFILE: OFF in build/CMakeCache.txt (GGML_LLAMAFILE:BOOL=OFF; ggml/CMakeLists.txt default OFF) -> no tinyBLAS.
- REPACK: ON, but repack.cpp has no GGML_TYPE_F16 case -> the f16 weights stay in the plain CPU buffer; and the
  conv weight is mul_mat's src1 here anyway (extra buffers act on src0).

## The binary agrees with the reading (objdump -d libggml-cpu.so, ggml_vec_dot_f16 at 0x6fa30, exported)
- loop: per 32 halves, 8 x vcvtph2ps (x and y) and 4 x vfmadd231ps into ymm0 (block 0), ymm1 (1), ymm3 (2), ymm2 (3);
  reduce: vaddps ymm0+ymm3 (acc0+acc2), ymm1+ymm2 (acc1+acc3), then their sum; vextractf128 hi + lo; vhaddps x2;
  vcvtss2sd; tail: table(y) * table(x) in vmulss, vcvtss2sd, vaddsd, in index order; vcvtsd2ss. No tail -> the f32
  reduction stored as is (same bits as through double). n < 32 -> the reduction is +0.0 then the tail.
- This laptop: Ryzen 3 3200U (Zen+): avx2 fma f16c, no avx512 -> the AVX path above. Production Zen 3 has the same
  ISA extensions (no AVX-512), so -march=native compiles the same path there; NOT checked on production's binary.
- The traits: ggml_get_type_traits_cpu(F16)->vec_dot == ggml_vec_dot_f16 (exported symbol), vec_dot_type f16,
  nrows 1 (recorded in vecdot.tsv by the oracle).

## The oracle (whisper_oracle --conv1), first run 2026-10-08
- Layout probe: sched_conv 42376, sched_encode 42408, embd_conv 42504, whisper_sched::sched at 0. Self-checks: each
  scheduler non-null with only the "CPU" backend; embd_conv's name "embd_conv"; the conv graph shows IM2COL,
  MUL_MAT, ADD, GELU in that order.
- conv graph, 14 nodes: IM2COL f16 [240,3000] -> RESHAPE x2 -> MUL_MAT f32 [3000,384] -> RESHAPE -> ADD -> GELU ->
  IM2COL f16 [1152,1500] -> ... (conv2). Observing every node does not change the result: embd_conv observed ==
  unobserved on all 8. 1 vs 4 threads: all four conv1 nodes identical on all 8. The standalone ggml graph
  (ggml_conv_1d_ph on copies, ggml_graph_compute_with_ctx, 1 and 4 threads) == the scheduler's nodes on all 8 —
  so --bench-conv1 times the same computation.
- vec_dot_f16 kernel record: 1,436 dots — 16 random row pairs of each of the 70 f16 tensors (n = 240, 384, 1152,
  1536), every length 1..300 on two real conv2 rows, 64 random finite-f16 vectors.
- Bug found in my first draft of the recorder: a 1,536-long random vector written into a 1,152 buffer (heap
  corruption, caught by glibc) — fixed before any record was used.

## voaice side, first oracle run (2026-10-08, load ~1)
- src/conv.rs: dot_f16_model (portable, mul+add — exact = FMA here), vec_dot_f16 on bit patterns, im2col_f16 (the
  node), Conv1 (fast path: weights widened once, rounded mel window padded, tiles of 8 frames x 1 channel, AVX2+FMA
  tile8 kernel, threads split output channels). tests/conv1.rs, 5 tests, ALL PASSED FIRST RUN:
  - vec_dot_f16 kernel: 1,436 / 1,436 dots identical (303 distinct lengths 1..1536).
  - discriminators on the kernel record: one f32 accumulator 1,265 / 1,436 differ; tail in f32 211 / 1,436; accumulators
    reduced in sequence 680 / 1,436.
  - im2col node: 8 inputs, 5,760,000 f16 values identical (voaice's own mel from the WAV).
  - MUL_MAT, ADD (+bias), GELU nodes: 0 differ on all 8 inputs at 1 and 4 threads (55,296,000 values compared).
  - conv1 discriminators on jfk: im2col kept in f32 1,151,932 / 1,152,000 differ; one f32 accumulator 1,068,048.

## Efficiency, first measurements (jfk; conv1 cost does not depend on the input: always 3000 frames)
Laptop: Ryzen 3 3200U = 2 cores / 4 SMT threads, 2.6 GHz max; 1-min load 1.4-2.9 during these runs.
- first draft (threads split channels, rounded-mel copy, fresh 4.6 MB output each call): 1t 24.0 ms vs ref 87.0
  (3.6x); 4t 20.4 vs 44.1 — 4 threads barely helped: every thread rebuilt every tile, and a fresh 4.6 MB output
  per call page-faults (malloc mmaps it); the rounded-mel copy cost 2 ms single-threaded.
- now: threads split frames (each tile built once), the mel rounded per tile 8 lanes at a time through F16C (NaN
  block -> the bit trick), no rounded copy, the output kept by the caller (run_into; the reference's graph keeps
  its tensors too):
    conv1      1t 21.0 ms vs 87.7 (4.2x) | 2t 14.3 vs 47.8 (3.3x) | 4t 15.5 vs 44.0 (2.8x)
    +bias+gelu 1t 24.2 vs 93.3 (3.9x)    | 2t 16.3 vs 54.2        | 4t 15.8 vs 48.8
    heap: voaice 4,508 KiB (the output, 4,500 KiB, + tiles) vs the reference's tensors 5,907 KiB (im2col 1,406 +
    output 4,500); with bias+GELU the reference holds 14,907 KiB (add and gelu outputs separate), voaice 4,508.
- 4 threads = 2 threads for voaice: 2 physical cores, the FMA pipes are shared by SMT siblings and voaice's loop is
  FMA/load-bound; the reference's per-dot overhead (scalar tail, horizontal reduce per dot) hides in SMT better.
- perf is not permitted on this laptop (perf_event_paranoid); timing was by instrumentation, removed.

## The gate run (testing/results/0.0.6.txt, 2026-10-08T09:47Z, load 1.7-3 from the operator's browser; no bankml gate)
- GATE PASSED. Every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample) + 5 conv1. The production opus
  re-ask was deliberately skipped (VOAICE_OPUS_HOST pointed at an unreachable name: production not touched).
- Step 8 in the gate run was noisy: conv1 1t 87.1 vs 33.1 ms (2.63x), 2t 80.2 vs 13.8 (5.79x — the reference's
  80 ms at 2 threads is an outlier), 4t 45.9 vs 24.6 (1.87x); +bias+gelu 3.78x / 3.38x / 3.19x.
- Three reruns of step 8 right after (same binaries, load 2.0-3.2), wall best of 10 each:
    conv1 1t 3.66x 3.97x 4.03x (voaice 21.4-24.2 ms, ref 86-89) | 2t 3.76x 3.21x 4.00x | 4t 3.31x 3.94x 3.42x
    gelu  1t 3.80x 3.88x 3.90x                                  | 2t 3.34x 3.53x 4.15x | 4t 3.30x 3.09x 3.16x
  So: about 3.7-4x the reference at 1 thread, 3-4x at 2 and 4; the gate's single run is within that only at 1t+gelu.
