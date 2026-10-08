# 0.0.8 — layer norm: notes (written as the work goes)

## The op, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8; ggml-cpu in the pin's tree)
- whisper's encoder has NINE norms, all `ggml_norm(ctx0, x, hparams.eps)` with eps = 1e-5f (whisper.cpp:602), each
  followed by `ggml_add(ggml_mul(cur, w), b)` — separate MUL and ADD nodes:
  per block il: attn_ln (norm(inpL) * attn_ln_0_w + attn_ln_0_b) and mlp_ln (norm(inpFF) * mlp_ln_w + mlp_ln_b);
  then ln_post (norm(inpL) * e_ln_w + e_ln_b) = embd_enc.
- ggml_compute_forward_norm_f32 (ops.cpp:3698), contiguous path (nb00 = nb0 = 4), rows i01 = ith, ith+nth, … (each
  row whole in one thread; rows are frames, ne00 = 384 channels):
  1. `ggml_vec_sum_f32`: ggml_float (double) sum, x[i] widened, index order, then `(float)sum`.
  2. `mean = sum / ne00` — float / (float)ne00 (vcvtsi2ss of the int64, vdivss).
  3. `variance = ggml_vec_cvar_f32(ne00, y, x, mean)` (vec.cpp:455; returns double, assigned to float): AVX2+FMA
     branch, blocks of 8: d = x - mean (f32, stored to y), p = d*d (vmulps, NO FMA), h = hi4(p) + lo4(p);
     h = h + movehl(h) → (h0+h2, h1+h3); s = (h0+h2) + (h1+h3) (add_ss with movshdup); sum += (double)s.
     Tail n mod 8 scalar: d, d*d in f32, widened, added in double. Return sum / n in double; caller rounds to f32.
  4. `scale = 1.0f / sqrtf(variance + eps)`: float add, vsqrtss, vdivss (the sqrtf@plt call is only the NaN path).
  5. `ggml_vec_scale_f32(y, scale)`: y[i] *= scale (f32 mul; vector or not, one rounding).
- objdump of the shipped libggml-cpu.so (this laptop's -march=native, Zen+): ggml_vec_cvar_f32 = vsubps, vmulps,
  vextractf128, vaddps, vmovhlps, vaddps, vmovshdup, vaddss, vcvtss2sd, vaddsd — exactly the source; no FMA. The sum
  loop in ggml_compute_forward_norm: vcvtss2sd + vaddsd one element at a time (not vectorised: no fast-math), then
  vcvtsd2ss, vdivss by vcvtsi2ss(ne00). After cvar: vcvtsd2ss, vaddss eps, vsqrtss, vdivss 1.0f. Scale: vmulps.
- MUL and ADD (binary-ops.cpp): `z[i] = x[i] op y[i]` per element, src1 broadcast per row; two nodes, two roundings.
  CPU graph fusion (ggml_cpu_try_fuse_ops) fuses only RMS_NORM + MUL, never NORM + MUL + ADD. So `(n*w) + b` with no
  FMA between them.
- n = 384 = 48 × 8: no scalar tail in cvar on tiny.en (base.en 512 likewise).

## The oracle (whisper_oracle --norm), first run 2026-10-08 (jfk: 11.5 s, load ~1)
- An eval callback on sched_encode observes EVERY encoder node (127 on tiny.en). For a NORM node, its src[0] is read
  when the scheduler ASKS about the node (all earlier nodes computed, the NORM not yet run — an in-place norm could
  not have overwritten it); the NORM, the MUL reading it (src[1] = the weight), the ADD reading that (src[1] = the
  bias) are read when computed. Checked: each MUL's src[0] is the NORM, each ADD's src[0] the MUL; the weights are
  the model's own tensors by pointer (encoder.blocks.N.{attn_ln,mlp_ln}.{weight,bias}, encoder.ln_post.*); eps in
  op_params = 1e-5f (0x1.4f8b58p-17); inputs contiguous f32.
- Graph order: VIEW e_pe, TRANSPOSE, CONT, ADD (0.0.7's), then NORM (node 4), MUL, ADD, the Q/K/V MUL_MATs …; the
  9 chains = attn_ln_0, mlp_ln_0, …, attn_ln_3, mlp_ln_3, ln_post.
- Self-checks yes on jfk: 1 vs 4 threads bit-identical (every in/norm/mul/add); embd_enc observed == unobserved;
  ln_post's ADD == embd_enc; standalone norm graph and norm→mul→add graph (what --bench-norm times) == the nodes at
  1 and 4 threads.
- Full record, 8 inputs: 1 m 34 s (load ~1-2), every self-check yes on 8/8; 635 MB in .oracle/norm (9 chains × in,
  norm, mul, add × 2.25 MB × 8).

## voaice side, first oracle run (2026-10-08) — ALL EXACT FIRST RUN
- src/norm.rs: `Variant` (the reference = default; eight discriminator flags), `cvar_model`, `row_stats`, `norm_row`,
  `LayerNorm` (new / from_parts / encoder() → the 9 in graph order, `row_model`, `run_model`, `run`, `run_into` with
  `Node::{Norm, Mul, Add}`), `sum_is_order_free`, the AVX2 row (one pass per pass-type in L1: lanes sum + proof scan,
  cvar 8-blocks with the reference's pairing, (x − mean)·scale·w + b with no FMA).
- The fast sum: the reference's sum is a chain of 384 dependent double adds (≈ 1,150+ cycles a row). If every value
  is a multiple of 2^q (q from the smallest non-zero |x|'s exponent) and n·max|x| < 2^(53+q), no partial sum in any
  order rounds, so 4-lane (×2) sums equal the sequential one. Proof scan = u32 max of |bits| and min of |bits|−1 (zero
  → u32::MAX), fused into the same pass as the lane sums. Unit test: shuffled sums equal sequential when it holds.
- tests/norm.rs, 8 inputs: NORM, MUL, ADD of all 9 chains from the recorded inputs — model, fast 1t, fast 4t: 0
  differ (373,248,000 values). Rows provably order-free 107,845, in the reference's order 155 (of 108,000).
  attn_ln_0 from voaice's own mel (conv stage → chain) at 1/2/4 threads: input 0 differ, nodes 0 differ (41,472,000).
- Discriminators (ADD values differing of 5,184,000 per input; all 8 inputs each): sum in f32 3.76–3.87 M; mean from
  the double sum 1.17–1.22 M; one-pass variance 0.72–1.01 M; cvar without the 8-lane f32 reduce 121–196 k; eps outside
  the sqrt 5.09–5.15 M; scale in double 0.94–1.00 M; divide by the root 1.00–1.02 M; mul + add fused (FMA) 1.48–1.64 M.
- The lane sum without the proof ("lane sum on every row") changes NO ADD value on the 8 inputs: the 155 unproven
  rows' means round to the same f32. A unit test (the_proof_is_needed) builds a row where it changes the mean
  (2^30, seven 2^-25, -2^30) and checks the fast path keeps the in-order sum there.

## Efficiency (jfk's encoder input [1500, 384], block 0's attn_ln). Laptop Ryzen 3 3200U (2 cores / 4 threads)
- First draft (lane sum + proof, cvar per block, fused epilogue): NORM 0.40 ms vs ref 0.80 (1t); chain 0.415 vs 1.85.
- Four double chains in pass 1 + cvar 4 blocks at a time (unpack/movelh pairing; each add the reference's operand
  pair) with its own order-free proof over the 8-block sums: ~7 % on L2-resident rows (207 vs 222 ns/row), nothing
  on 1500 rows: there the op is BANDWIDTH-bound — input 2.25 MB + output 2.25 MB (+ RFO) > the 4 MB L3; at 6,000 and
  24,000 rows 2 and 4 threads gain nothing either (~12 GB/s). A scoped spawn+join costs ~60 µs here.
- Gate run 1 (load 1.0 → 3): norm 1t 0.371 vs 0.778 (2.10x), 2t 1.50x, 4t 1.13x; chain 1t 0.430 vs 2.506 (5.83x),
  2t 4.37x, 4t 3.49x. Heap: 2,251 KiB (the output) vs the graph's 6,750 for the chain.
- Three reruns (load 3.2-3.7, .oracle/step10_reruns.txt): norm 1t 1.02/2.30/2.19x, 2t 1.22/1.31/2.08x, 4t
  0.94/0.75/0.83x (voaice SLOWER at 4: its threads cost spawns, the reference's OpenMP pool is persistent); chain
  1t 2.60/3.94/4.39x, 2t 3.94/4.27/4.82x, 4t 3.35/3.44/3.61x.
- So: `run_into` caps threads at one per MIN_ROWS_PER_THREAD = 2048 rows (whisper's 1,500 rows -> one thread;
  bits unaffected, every row whole in one thread); `run_into_split` keeps the exact split for the oracle (still
  checked at 4 threads). Then the gate was rerun for the record.

## The gate run (testing/results/0.0.8.txt, 2026-10-08T11:18Z, load 2.1-2.4) — after the thread cap
- GATE PASSED: every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample, 5 conv1, 4 conv2) + 3 norm.
  Step 4b re-asked production's opus-tools as the gate does by default (read-only): 57/57 identical.
- Step 10: norm 1t 0.382 vs 1.189 (3.11x), 2t 2.47x, 4t 0.91x (ref 0.382 over its pool, 1.89 CPU-ms; voaice one
  thread, 0.42 CPU-ms); chain 1t 0.405 vs 1.997 (4.93x), 2t 4.68x, 4t 4.11x. Heap 2,251 vs 6,750 KiB.
