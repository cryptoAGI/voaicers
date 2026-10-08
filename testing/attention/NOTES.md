# v0.1.0 — flash attention and the whole encoder: notes (written as the work goes)

## The op, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8; ggml-cpu in the pin's tree)
- whisper_build_graph_encoder (whisper.cpp:2128–2162), flash_attn (whisper-cli's default):
  - `Q = permute(reshape_3d(Qcur, 64, 6, 1500), 0, 2, 1, 3)` → ne [64, 1500, 6], nb1 = 384·4, nb2 = 64·4 (a view of
    q_add: frame-major f32, head h at columns 64h..64h+63). **Q stays f32.**
  - `cpy(Kcur, view_1d(kv_pad.k, 1500·384))`, likewise V (0.0.9's scalar bit-trick CPY nodes), then
    `K = view_3d(kv_pad.k, 64, n_ctx_pad = 1536, 6, nb1 = 2·384, nb2 = 2·64)`, V likewise: key j of head h is
    `kv_pad[j·384 + 64h + d]`. **1536 key rows, the last 36 never written by any CPY.**
  - kv_pad is made by `whisper_kv_cache_init(state->kv_pad, …, n_audio_state, 1, GGML_PAD(n_audio_ctx, 256))`, which
    calls `ggml_backend_buffer_clear(cache.buffer, 0)`: **the 36 padding rows are f16 +0** for the life of the state
    (every block writes rows 0..1499 only). No mask (`nullptr`), so they are attended to: each contributes a score of
    exactly +0 (see below) — `exp(0 − M)` to the denominator and nothing to the numerator.
  - `KQscale = 1.0f/sqrtf(float(64))` = 0.125 exactly (whisper.cpp:2069), max_bias 0, logit_softcap 0; then
    `reshape_2d(cur, 384, 1500)`. Precision GGML_PREC_DEFAULT → `ggml_compute_forward_flash_attn_ext_f16`.
- ggml_compute_forward_flash_attn_ext_f16 (ops.cpp:9001) chooses, for this graph:
  - not the split-KV (decode) path: that needs neq1 == 1 (here 1500);
  - **the tiled path** `ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8641): `use_tiled = !use_ref && q F32
    && K, V F32-or-F16 && k->type == v->type && neq1 >= GGML_FA_TILE_Q (64)` and `DV % GGML_F32_EPR (8) == 0` — all
    true. **TODO.md's reading (Q converted to f16, `kq_vec_dot` = ggml_vec_dot_f16, V accumulated in f16 through
    ggml_vec_mad_f16) is the one-chunk path, which whisper's encoder never takes on this build** (it is the
    `use_ref` path, and the fallback for small or odd shapes). Kept as a discriminator.
  - rows = 1500·6 = 9000 (frame-major within a head: ir → head ir / 1500, frame ir % 1500), nchunk = 4·nth (nth > 1)
    or nth, chunks taken dynamically (`ggml_threadpool_chunk_add`); each chunk walked in tiles of up to 64 query rows
    that never cross a head.
- The tiled kernel, per tile of query rows (Q_TILE 64, KV_TILE 64; nek1 = 1536 = 24 KV tiles, no partial tile):
  1. Q rows memcpy'd as f32 (rows past the tile's end zeroed); S = 0, M = −inf, VKQ32 = 0 per row.
  2. per KV tile: K rows widened f16 → f32 (GGML_CPU_FP16_TO_FP32, exact) and packed transposed K_f32[dk][kv];
     `KQ = 0; simd_gemm(KQ, Q, K_f32, 64, 64, 64)`; `ggml_vec_scale_f32(KQ, scale)`.
  3. per row: `tile_max = ggml_vec_max_f32(64)` (scalar `max = max > x ? max : x` from −inf: one vmaxss chain);
     `Mnew = fmaxf(Mold, tile_max)` (libm); if `Mnew > Mold`: `ms = expf(Mold − Mnew)` (**libm expf**), VKQ *= ms
     (vmulps), S *= ms; `M = Mnew`; `S += ggml_vec_soft_max_f32(64, kq, kq, Mnew)` — a double return added to the
     float S: `S = (float)((double)S + sum)`.
  4. V rows widened by `ggml_fp16_to_fp32_row` (exact), then `simd_gemm(VKQ32, KQ, V32, 64, 64, 64)`.
  5. after the 24 tiles: `S_inv = S == 0 ? 0 : 1.0f/S`, VKQ *= S_inv (vmulps), memcpy to dst[frame][64h..].
- simd_gemm (simd-gemm.h), AVX2: a 6 × 2·8 register kernel, then single rows; for every C element the arithmetic is
  **one sequential FMA chain over k from C's own value**: `c = fma(A[i][kk], B[kk][j], c)` for kk = 0..K−1. Each
  element is independent of every other (no reduction across lanes), so the kernel's blocking, the tile's composition
  and the thread count change no bit. So:
  - scores: `s = fma(q[63], k[63], … fma(q[0], k[0], +0))` in f32, then `s · 0.125` (exact unless subnormal).
  - a zero key: every product is ±0 and `+0 + (−0) = +0`, so the padding rows' scores are exactly +0.
  - output: `o[d] = fma(p[63], v[63][d], … fma(p[0], v[0][d], o[d]))` per KV tile, rescaled between tiles.
- ggml_vec_soft_max_f32 (vec.cpp:531), the AVX2 + FMA branch: per 8: `ggml_v_expf(x − max)` (vec.h:1215, ARM's
  optimized-routines expf in 8 lanes: `z = fma(x, 0x1.715476p+0, 0x1.8p23)`, `n = z − 0x1.8p23`,
  `b = fnma(n, 0x1.7f7d1cp-20, fnma(n, 0x1.62e4p-1, x))`, `k = (z << 23) + bits(1.0)`, `u = b·b`,
  `j = fma(fma(fma(0x1.0e4020p-7, b, 0x1.573e2ep-5), u, fma(0x1.555e66p-3, b, 0x1.fffdb6p-2)), u, 0x1.ffffecp-1·b)`,
  then `fma(j, k, k)` unless |n| > 126 for some lane, then the scaled branch for all 8 — the same value per lane
  either way), stored, then `(hi4 + lo4)`, `(h0 + h2, h1 + h3)`, `(h0 + h2) + (h1 + h3)` in f32, `sum += (double)`.
  n = 64: 8 blocks, no scalar tail (the tail would use libm expf).

## Confirmed in this laptop's libggml-cpu.so (Zen+, -march=native: avx2, fma, f16c; no avx512)
- `ggml_compute_forward_flash_attn_ext_tiled` is its own function (0x76d00), called once from
  ggml_compute_forward_flash_attn_ext. In it: 47 `vfmadd231ps` (simd_gemm's chains), one `vmaxss (%rax),%xmm1,%xmm1`
  loop from −inf (= `max > x ? max : x`), `call fmaxf@plt`, `vcomiss; ja` → `vsubss` (Mold − Mnew) → `call expf@plt`
  (expf@GLIBC_2.27, the symbol voaice calls too), the rescale by vmulps, `call ggml_vec_soft_max_f32@plt`, then
  `vcvtss2sd S; vaddsd; vcvtsd2ss` (S += the double sum), `ggml_fp16_to_fp32_row@plt`, `vdivss` for 1/S. No
  vcvtps2ph on Q (the f32 → f16 copy is only in the one-chunk path).
- `ggml_vec_soft_max_f32` (exported): the AVX2 + FMA branch as read — vsubps, vfmadd132ps (z), vaddps of −0x1.8p23
  (n: the compiler's `z + (−r)`, the same rounding as `z − r`), vfnmadd231ps ×2 (b), vfmadd213ps ×2, vmulps (u, c·b),
  vfmadd132ps ×2 (j), vmovmskps test, vfmadd132ps (k·j + k); then vextractf128, vaddps, vmovhlps, vaddps, vmovshdup,
  vaddss, vcvtss2sd, vaddsd. The scalar tail calls expf@plt (not reached at n = 64).
- So **two exponentials**: ggml's own `ggml_v_expf` for the probabilities, glibc's `expf` (2.35, the ifunc picks
  `__expf_fma` on this CPU) for the running-max rescale. voaice calls the same libm `expf` symbol; production's glibc
  and CPU would pick their own variant — not checked here.

## The oracle (whisper_oracle --encoder), first run 2026-10-08
- An eval callback on sched_encode observes every node; it runs 0.0.9's node keys (mm_cb) and adds the norm chains,
  the positional ADD and FLASH_ATTN_EXT; each computed node (views, reshapes, permutes, transposes skipped) is digested
  per row (enc.d64 + enc.tsv: 97 nodes). When the scheduler ASKS about a FLASH_ATTN_EXT node, its Q (q_add, whole), and
  kv_pad.k / .v (all 1,536 rows) are copied: the inputs' shapes, strides, op params are checked, and kv_pad's rows
  1500..1535 are all +0 in every block. The standalone graph (fa_graph: what --bench-attn times) is run on those inputs
  at 1..8 threads (= the node every time) and once with cplan.use_ref (the one-chunk path): b<il>.fa_ref.d64.
- Self-checks yes on 8/8: inputs as built, padding +0, 1 vs 4 threads identical (every digest and every attention
  output), embd_enc observed = unobserved, 1 = 2 = 4 threads unobserved, the last node = embd_enc, standalone = node at
  1..8 threads. **use_ref changes 2,303,363–2,303,984 of 2,304,000 values per input (all 6,000 rows)**: the path matters.
- 21 MB for 8 inputs (embd_enc whole + digests), ~25 s per input (encoded 5 times, standalone attention 9 times per block).

## voaice side, first oracle run (2026-10-08) — EXACT FIRST RUN
- src/attention.rs (model + variants, the one-chunk model, the AVX2 kernel), src/encoder.rs (the pipeline, caller-owned
  EncoderBuffers). Unit tests: the scalar v_expf = the 8-lane kernel on 200k inputs (incl. the |n| > 126 / 192 branches,
  −inf, NaN); fast = model on random shapes at 1 and 3 threads.
- `voaice encode` on jfk at 2 threads: embd_enc digest 50d38ec85f2778b9 = the reference's embd_enc.f32 digest.
- Full oracle run (tests/attention.rs, 248–266 s): attention 0 differ in 36,864,000 values (8 inputs × 4 blocks × 1 and 4
  threads), the model 0 of 15,600 frames; the whole encoder from voaice's mel, 96 nodes × 8 inputs × 3 thread counts
  0 rows differ, embd_enc 0 values differ (and on buffer reuse).
- Discriminators (every 25th frame and the last, 244 rows per input): all caught on every input; Q scaled first: 0
  everywhere (×1/8 exact — predicted, asserted as indistinguishable).
- **Found: my first one-chunk model contracted `S = S·ms + vs` into an FMA (scalar FMAs are all over that function in
  the binary) — the reference's own use_ref output said no: 543 of 1,952 rows differ fused, 0 unfused.** The binary
  agrees on a second look: GCC split the update by branch (new max: `vmulss` S·ms then `vaddss` 1.0; else `vaddss`).

## Efficiency (block 0 of jfk; Ryzen 3 3200U, 2 cores / 4 threads; load 1–2 from a browser)
- First fast path (K/V widened for all heads once per call, 6 × 16 register blocks, softmax per row): 94.3 ms vs the
  reference's 147.1 at one thread (1.56×), 59.0 vs 85.4 at two.
- Ablations (wrong-output builds, one thread): no softmax 74.8 ms, no output product 54.8 ms, widening alone 2.0 ms;
  no ggml_v_expf 78.7 ms. So the softmax ≈ 14.5 ms of which v_expf ≈ 11 ms (≈ 21 cycles per 8-lane call: its own op
  count on Zen+'s split 256-bit pipes), scores + output ≈ 73 ms against a 65 ms floor (221 M ymm FMAs per block at one
  per cycle).
- The softmax in two passes (max + glibc expf, then the exponentials): 94 → 89.5 ms — the 8 loaded vectors were being
  spilled around the expf call. Eight rows' sums at once in double lanes, Q tile 60: no measurable change (kept: it is
  no slower and leaves only full 6-row blocks).
- K/V widened per head behind a std::sync::Barrier (one scope per call): the scratch 4.7 → 0.8 MB (heap 6.9 → 3.0 MB,
  under the reference's 4.7), times unchanged at 1 and 2 threads, 4 threads 62 → 55 ms.
- The whole encoder (first measurement, load ~1): 1t 3,967 → 977 ms (4.06×), 2t 2,359 → 711 (3.32×), 4t 2,399 → 796
  (3.01×); the embd_enc digests equal (50d38ec85f2778b9).

## The gate run (testing/results/0.1.0.txt, 2026-10-08, load 1.5 → 3.5)
- GATE PASSED: every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample, 5 conv1, 4 conv2, 3 norm, 4 matmul)
  + 3 attention (266 s). 4b SKIPPED by VOAICE_OPUS_HOST=unreachable.invalid (the recorded answers compared). The
  encoder record took 203 s.
- Step 12: attention 1t 150.8 → 91.9 (1.64×), 2t 94.7 → 64.1 (1.48×), 4t 89.7 → 70.2 (1.28×); **the encoder 1t 3,994 →
  1,038 ms (3.85×), 2t 2,436 → 799 (3.05×), 4t 2,378 → 791 (3.01×)**; heap 17,869 KiB vs 29,398; digests equal.
- Rerun right after (.oracle/step12_rerun.txt, load 2.1 → 3.2): attention 1.61 / 1.53 / 1.46×; the encoder 3,983 → 987
  (4.04×), 2,431 → 774 (3.14×), 2,435 → 824 (2.95×).
