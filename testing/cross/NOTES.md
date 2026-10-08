# 0.1.1 — cross-attention K and V: notes (written as the work goes)

## The op, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8)
- whisper_build_graph_cross (whisper.cpp:2272): cur = view(embd_enc) [384, 1500]; `Kscale = pow(float(64), -0.25)`
  (std::pow(float, double) -> double, then the float initialiser); per decoder layer il (tiny: 4):
  K = mul_mat(cross_attn_k_w, cur) -> ggml_scale(K, Kscale); V = mul_mat(cross_attn_v_w, cur) + cross_attn_v_b
  (**the bias is V's; K has none**); flash_attn: both `ggml_cpy` into view_1d(kv_cross.{k,v}, n_state·n_ctx,
  offset 2·n_state·il·n_ctx_pad) — layer il's rows start at il·1536, frame-major [384 per row].
- kv_cross = whisper_kv_cache_init(itype = F16, n_text_state 384, n_text_layer 4, GGML_PAD(n_audio_ctx, 256) = 1536):
  k and v each 384 × 4 × 1536 f16, the buffer cleared to 0 at init; the CPYs write rows 0..1499 of each layer, so
  rows 1500..1535 of each layer are +0 for the life of the state.
- ggml_scale (ops.cpp:4505) with b = 0: memcpy then ggml_vec_scale_f32 — one f32 multiply per element (the AVX loop
  and the scalar leftovers round the same way), threads by rows.
- **Found while reading: whisper_encode_internal runs the cross graph too** (whisper.cpp:2436: conv, encoder, then
  `// cross` on sched_cross). So `whisper_encode_with_state` — what v0.1.0's `--bench-encode` timed as "the whole
  encoder" — included the cross K/V of all four decoder layers, which voaice's encode_into did not compute. v0.1.0's
  3.85x compared voaice's encoder against whisper's encoder + cross. 0.1.1 measures both parts and the fair whole.

## The oracle (whisper_oracle --cross), first run 2026-10-08
- The eval callback on sched_cross (layout probe: whisper_state::sched_cross at 42440, kv_cross at 168, its k / v at
  +40 / +48) sees exactly 24 computed nodes, 6 per decoder layer in this order: MUL_MAT (key.weight), SCALE, CPY,
  MUL_MAT (value.weight), ADD (value.bias), CPY — no other computed node. Self-checks yes on 8 / 8: f16 weights in a
  plain CPU buffer; src1 IS embd_enc (same data pointer); Kscale op_param 0x3EB504F3 in every layer, its b = 0 (the
  multiply path); the SCALE runs in place (the allocator gave it the MUL_MAT's memory, 4 of 4: no bit consequence);
  no ADD reads K; V's ADD reads decoder.blocks.N.cross_attn.value.bias; each CPY's destination is a view of
  kv_cross.k / .v at byte offset 2·384·il·1536; kv_cross +0 at init; after the graph, all 288 padding rows (36 per
  layer, k and v) still +0; the CPY nodes' rows = the buffer's rows; 1 vs 4 threads identical; observed = unobserved;
  1 = 2 = 4 threads; embd_enc identical in every run (digest 50d38ec85f2778b9 on jfk = v0.1.0's); the standalone
  graph (what --bench-cross times) = the scheduler's kv_cross at 1 and 4 threads. 3.1 MB for 8 inputs, ~16 s each.
- objdump of ggml_compute_forward_scale: 6 vmulps (vec_scale's b == 0 branch), 6 vfmadd132ps + 2 vfmadd132ss (the
  mad1 branch for b != 0, not taken).

## voaice side, first oracle run (2026-10-08) — ALL EXACT FIRST RUN
- src/cross.rs: Cross (the eight Linear of 0.0.9), the model (`Linear::model` + scale + scalar CPY) and the fast path
  (one conversion per panel for all eight products, 0.0.9's 4 × 3 block, scale / bias / CPY as the epilogue, straight
  into the caller's cache; padding rows set to +0 every call).
- oracle_cross_nodes_bit_exact (fed the reference's embd_enc): model, fast 1t, fast 4t: 0 of 36,000 node rows and 0 of
  12,288 kv_cross rows (padding included) differ on each of 8 inputs — 864,000 node rows + 294,912 cache rows.
- oracle_cross_end_to_end (voaice's mel -> encoder -> cross at 1, 2, 4 threads): embd_enc = the record's, every node
  0 rows differ, the cache (reused across two calls, pre-filled with 0xFFFF) 0 of 12,288 — 576 node comparisons,
  1,158,912 rows.
- Discriminators, first run on k_cpy + v_cpy only: "scale in double" caught on only 6 of 8 inputs (0-3 sampled rows):
  the f16 rounding swallows most of a 1-ulp f32 difference. Compared on all six nodes instead (k_scale is f32).

## Efficiency (gate step 13, after 4i passed; jfk; laptop Ryzen 3 3200U, 2 cores / 4 threads; load ≈ 3, rerun ≈ 2.3)
- cross K/V, embd_enc -> kv_cross, wall best of 10 (gate · rerun): 1 thread 454.2 · 451.0 -> 95.4 · 92.8 ms (4.76 ·
  4.86x); 2 threads 278.8 · 263.7 -> 58.3 · 57.7 (4.78 · 4.57x); 4 threads 283.2 · 274.3 -> 67.5 · 69.1 (4.19 · 3.97x).
  1.77 G multiply-adds in 95 ms = 18.5 G/s, ~2/3 of 8 FMA lanes a cycle at 3.5 GHz. Panel 64 / 128 / 256 frames: 92.9
  / 89.0 / 88.9 ms (noise): FMA-bound, kept 128.
- the whole of whisper_encode_with_state like for like (mel -> embd_enc -> kv_cross): 1 thread 4,010.6 · 4,021.5 ->
  1,085.4 · 1,074.9 (3.70 · 3.74x); 2 threads 2,440.3 · 2,457.6 -> 835.8 · 825.0 (2.92 · 2.98x); 4 threads 2,395.9 ->
  901.6 (2.66x). Step 12 in the same gate (voaice's encoder alone against the same call): 3.79x — v0.1.0's figure.
- memory: cross 9,794 KiB (the 9,216 KiB cache + panels) vs 11,466 (sched_cross 2,250 + the same cache); whole 26,312
  vs 40,864 (v0.1.0's 29,398 + sched_cross + kv_cross). digests on both sides: kv 76f24e73e318b877, enc 50d38ec85f2778b9.
