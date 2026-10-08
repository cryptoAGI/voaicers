# 0.0.9 — the matrix products on activations: notes (written as the work goes)

## The ops, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8; ggml-cpu in the pin's tree)
- whisper_build_graph_encoder (whisper.cpp:2038), per block il, after attn_ln (0.0.8):
  - Q = mul_mat(attn.query.weight, cur) + attn.query.bias; **K = mul_mat(attn.key.weight, cur), no bias** ("note: no
    bias for Key"); V = mul_mat(attn.value.weight, cur) + attn.value.bias.
  - flash_attn (whisper-cli's default): `ggml_cpy(Kcur, view_1d(kv_pad.k))`, likewise V — **f32 → f16 CPY nodes**,
    then FLASH_ATTN_EXT(Q permuted, K view, V view) (v0.1.0's), RESHAPE to [384, 1500].
  - out = mul_mat(attn.out.weight, attn) + attn.out.bias; inpFF = ADD(out, inpL) (the residual).
  - mlp_ln (0.0.8) → fc1 = mul_mat(mlp.0.weight, ·) + mlp.0.bias [1536, 1500] → GELU → fc2 = mul_mat(mlp.2.weight, ·)
    + mlp.2.bias → ADD(fc2, inpFF) = the next block's input (ln_post's after block 3).
- Node order as the scheduler runs it (nodes.tsv of 0.0.8's record): NORM 4, MUL 5, ADD 6, MUL_MAT 7 (K: the CPY's
  build_forward_expand pulls K first), VIEW 8 + CPY 9 (K → kv_pad.k f16), MUL_MAT 10 + ADD 11 (V), VIEW 12 + CPY 13
  (V → kv_pad.v), MUL_MAT 14 + ADD 15 (Q), RESHAPE, PERMUTE, VIEWs, FLASH_ATTN_EXT 20, RESHAPE 21, MUL_MAT 22 + ADD 23
  (out), ADD 24 (residual), NORM/MUL/ADD 25–27, MUL_MAT 28 + ADD 29 (fc1), GELU 30, MUL_MAT 31 + ADD 32 (fc2), ADD 33.
- ggml_compute_forward_mul_mat (ggml-cpu.c:1254), src0 = the f16 weight [K, N], src1 = the f32 activation [K, 1500]:
  - `vec_dot_type` of F16 is F16, `from_float` = ggml_cpu_fp32_to_fp16 (the type traits, ggml-cpu.c:221), nrows = 1.
  - src1 is converted into wdata **row by row, every thread converting its element range of every row**:
    `[(ith·K)/nth, ((ith+1)·K)/nth)` (bs = 1 for f16), each range one call of ggml_cpu_fp32_to_fp16 — vcvtps2ph
    on blocks of 8, then of 4, then the portable bit trick on the last `len % 4` (0.0.3). For K = 384 or 1536 and nth
    in {1, 2, 3, 4} every range is a multiple of 8: all through vcvtps2ph. For nth = 5 the ranges are 76/77 long and
    the last element of each 77 goes through the bit trick: **the conversion of a NaN depends on the thread count**
    (vcvtps2ph keeps the top payload bits; the trick gives sign | 0x7E00). Non-NaN values: the two agree (0.0.3).
  - barrier, then chunks of 16 × 16 (nchunk0·nchunk1 = 24·94 ≥ 4·nth: no re-chunking), each output
    `vec_dot(K, &tmp, weight row ir0, converted column ir1)` = 0.0.6's ggml_vec_dot_f16 whole in one thread; tmp
    memcpy'd to dst. Nothing accumulates across chunks: the thread count changes no bit of a finite product.
  - ggml_vec_dot_f16's x is the WEIGHT row, y the activation (conv1 had them the other way round); the product of two
    halves is exact in f32, so the operand order changes no bit (NaN payload: the only NaN is the activation's).
  - GGML_LLAMAFILE OFF (CMakeCache), so no tinyBLAS; GGML_CPU_REPACK ON but repack has no f16 tensor_traits (nm -D:
    q2_K, q4_0, q4_K, q5_K, q6_K, q8_0, mxfp4, iq4_nl only); KleidiAI, AMX off. GGML_HINT_SRC0_IS_HADAMARD not set.
  - K = 384: 12 × 32, no double tail; K = 1536 (fc2): 48 × 32, no tail.
- ADD (binary-ops.cpp): one f32 add per element, the bias broadcast per row. CPU graph fusion: only RMS_NORM + MUL
  (ggml_cpu_try_fuse_ops), so mul_mat + add + gelu are three nodes, three roundings.
- CPY f32 → f16 (ops.cpp ggml_compute_forward_dup_flt<float, ggml_fp16_t>, dst contiguous, rows split by thread):
  element by element `type_conversion_table<ggml_fp16_t>::from_f32` = f32_to_f16 = GGML_CPU_FP32_TO_FP16 — the
  **scalar** portable bit trick (0.0.3), not the row converter. Same bits except on NaN.
- GELU: ggml_vec_gelu_f32 with GGML_GELU_FP16 — 0.0.3's op (the f16 table, ±10 clamps), the same one conv1/conv2 use.

## The oracle (whisper_oracle --matmul, --mm-nan), first run 2026-10-08
- An eval callback on sched_encode observes every node (127). Nodes are identified by what they read, not by index:
  a MUL_MAT by its src[0]'s name in the model's tensor map (encoder.blocks.N.attn.{query,key,value,out}.weight,
  mlp.{0,2}.weight); an ADD by its src[0] (the MUL_MAT it adds a bias to — the bias checked by name — or o_add / fc2_add
  for the residuals); a CPY by its src[0] (k_mm, v_add); GELU by fc1_add; FLASH_ATTN_EXT in order.
- Compact record: per node one 64-bit FNV-1a digest per row (the oracle's existing digest32/digest16: every step a
  bijection, so one changed value always changes the row's digest); each MUL_MAT's src1 digested at ask time; the
  attention's output kept whole (2.3 MB × 4 blocks); the block's input kept in memory for the standalone check. 80 MB
  for 8 inputs (vs ~1.8 GB as tensors). Full values around the products come from 0.0.8's norm record (attn_ln/mlp_ln
  ADD = the products' inputs, their `in` = the residual stream).
- Self-checks yes on 8/8: every block shows its 17 nodes; the MUL_MATs read f16 weights from a plain "CPU" buffer (no
  repack); src1 contiguous f32; no ADD reads K's MUL_MAT (K unbiased); every ADD's bias is the named tensor; 1 vs 4
  threads identical (every digest and the attention output); embd_enc observed == unobserved; the standalone qkv and mlp
  graphs (what --bench-mm times) == the scheduler's nodes at 1 and 4 threads. 2 m 5 s for 8 inputs.
- --mm-nan: 12 rows of encoder.positional_embedding, 10 carrying one NaN at 152/229/306/383 (the bit-trick tail of the
  5-thread ranges), 150 (a 4-block there), 52/53 (the tail of thread 0 at 7 threads), 8 (an 8-block everywhere); payloads
  with high bits, negative, signalling. Through mul_mat(query.weight) + bias at 1..8 threads (ggml_graph_plan +
  ggml_graph_compute, cplan.n_threads checked). **The reference's own output at 5 threads differs from its 1-thread
  output in 1,920 values (5 rows), at 7 threads in 1,152 (3 rows)**; at 1, 2, 3, 4, 6, 8 threads identical.

## voaice side, first oracle run (2026-10-08) — ALL EXACT FIRST RUN
- src/matmul.rs: Linear (model with the from_float split as a parameter; Variant discriminators), Block (qkv_into with
  attn_ln fused into one conversion shared by Q, K, V; mlp_into a panel of 32 frames at a time from the out projection
  to fc2), the 4 × 3 register block (conv2's) over a permuted layout, epilogues: bias, GELU, residual, f16 CPY (bit trick).
- oracle_mm_nan_split: model and fast path (told the split) 0 differ at every thread count 1..8; discriminators: the
  scalar bit trick everywhere 3,456 values at 1/2/3/4/6/8 threads (1,536 at 5, 2,304 at 7); the row converter with the
  split ignored 1,920 at 5 threads, 1,152 at 7, 0 elsewhere (it IS the reference there).
- oracle_block0_from_mel: voaice's mel → conv stage → block 0 (attn_ln fused → Q, K, V, the CPYs; out proj → MLP from
  the recorded attention): every node 0 differ at 1, 2, 4 threads on 8 inputs.
- oracle_matmul_nodes_bit_exact: 4 blocks × 16 nodes, model + fast 1t + fast 4t, 8 inputs: 0 rows differ (2,304,000
  node rows by digest), o_res and mlp_res also 0 values differ against the norm record. oracle_matmul_discriminators
  (summed over the 4 blocks): activations not rounded to f16, one accumulator, the accumulators in sequence, the bias in
  the first accumulator: 6,000 / 6,000 q_add rows on every input; residual before the bias 281–307 k o_res values of
  2,304,000; GELU without the table 6,000 / 6,000 rows; the two from_float readings 0 (no NaN in the inputs).
  The 4 tests took 178 s (the scalar model is most of it).

## Efficiency (first measurements, jfk, block 0; load ~1.4–1.9). Laptop Ryzen 3 3200U (2 cores / 4 threads)
- 1 thread, wall best of 10 (reference → voaice): q 54.2 → 10.4 ms (5.2×), fc1+gelu 225.8 → 46.9 (4.8×), fc2 208.5 → 40.5
  (5.1×), qkv (attn_ln + 3 products + CPYs) 167.3 → 34.4 (4.9×), mlp 494.3 → 112.2 (4.4×), block 661.8 → 144.7 (4.6×).
  Q at 10.4 ms = 21 G multiply-adds/s, ~75 % of this core's 8 FMA lanes per cycle at its 3.5 GHz boost.
- 2 threads: q 31.1 → 7.2, fc2 111.3 → 22.6, qkv 89.4 → 20.0, block 385 → 92.5. 4 threads: q 30.3 → 7.4, fc2 122.5 →
  28.0, qkv 94.7 → 23.3, block 400 → 95.1 — the FMA pipes are shared by SMT siblings, so 4 = 2 for voaice (the
  reference's cvt-bound loop gains a little more from SMT).
- Heap: block 9.7 MB (q, K16, V16, the V stand-in and the output) vs the graph's 69.8 MB (every node + the work buffer).
- Tried: the weights kept f16 in the kernel's layout, widened by vcvtph2ps in the 4 × 3 block (half the weight memory):
  ~30 % slower (q 12.8 → 17.0 ms, block 171 → 206 ms, both with an env lookup in the loop). Kept f32 weights: the
  products hold 7.1 MB per block widened (2× the f16 bytes; 28 MB for the encoder), allocated with the model, outside
  the measured call — said so in the CHANGELOG.
- PANEL 32 → 64 (2026-10-08): mlp 104–108 vs 111–113 ms at one thread, 61.5–66 vs 65–67 at two (weights re-read half
  as often); q unchanged within noise. Kept 64.

## The gate run (testing/results/0.0.9.txt, 2026-10-08T12:07Z, load 1.6–1.8 → 2.9)
- A first run stopped at step 3 with exit 141: `sed '/ret *$/q'` in the disassembly reads quit before objdump finished
  writing, objdump took SIGPIPE, and pipefail + set -e ended the gate (the same lines passed in 0.0.8's runs: a race).
  The reads now use `sed -n '1,/ret/p'` (reads to the end; the same lines printed). Kept log: .oracle/gate009.sigpipe.log.
- GATE PASSED: every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample, 5 conv1, 4 conv2, 3 norm) + 4
  matmul (185 s). 4b SKIPPED by VOAICE_OPUS_HOST=unreachable.invalid (the recorded answers compared, as 0.0.6/0.0.7).
- Step 3: ggml_cpu_fp32_to_fp16 2 vcvtps2ph (the 8- and 4-blocks); ggml_compute_forward_mul_mat 0 FMA (the dot is
  called through the traits); GGML_CPU_REPACK ON, 0 f16 repack traits.
- Step 11 (1t / 2t / 4t, ratio ref/voaice): q 5.19 / 3.06 / 4.74; fc1 4.70 / 4.68 / 4.27; fc2 5.15 / 5.17 / 4.28;
  qkv 5.18 / 4.79 / 4.10; mlp 4.78 / 4.58 / 3.89; block 4.55 / 4.36 / 4.14 (666.8 → 146.6 ms at 1t). Heap block 10,351
  vs 69,751 KiB. Rerun right after (.oracle/step11_rerun.txt, load 1.7 → 2.9): q 5.07 / 3.91 / 4.16, block 4.77 /
  4.37 / 3.80 — the gate's q at 2 threads (3.06×) was the noisy one.
