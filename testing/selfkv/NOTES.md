# 0.1.3 — the self-attention products and the f16 self KV cache: notes (written as the work goes)

## The op, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8)
- whisper_build_graph_decoder (whisper.cpp:2458), per decoder layer il, before self-attention itself:
  `cur = add(mul(norm(inpL, eps), attn_ln.weight), attn_ln.bias)` (0.0.8's three nodes, at the batch's row count);
  `Qcur = scale(add(mul_mat(attn.query.weight, cur), attn.query.bias), KQscale)` — **the bias before the scale**;
  `Kcur = scale(mul_mat(attn.key.weight, cur), KQscale)` — **no bias ("note: no bias for Key")**, scaled here (in the
  encoder K was not scaled; in the cross graph only K was);
  `Vcur = add(mul_mat(attn.value.weight, cur), attn.value.bias)`; flash_attn: `cpy(Kcur, view_1d(kv_self.k, n_tokens·384,
  2·384·(il·n_ctx + kv_head)))`, the same for V into kv_self.v. Q goes on (permuted) to flash_attn_ext (0.1.4).
- `KQscale = pow(float(n_state_head), -0.25)` (whisper.cpp:2506) — the same expression as 0.1.1's Kscale: 0x3EB504F3.
- kv_self = whisper_kv_cache_init(F16, 384, 4, **GGML_PAD(n_text_ctx 448, 256) = 512** cells) (whisper.cpp:3387): k and v
  each [4 · 512 · 384] f16, cell-major per layer (layer il's cell c at row il·512 + c). Its buffer is cleared at init
  and by whisper_kv_cache_clear at the start of **every window** (whisper.cpp:7163), not between steps.
- whisper_decode_internal (whisper.cpp:2848): whisper_kv_cache_find_slot (the cache's head walks forward past used cells
  until n_tokens free cells are in a row; it is NOT advanced past the slot it returns), then
  `kv_self.n = min(size, max(pad, GGML_PAD(cell_max, pad)))`, where **pad = whisper_kv_cache_get_padding = 1 on the
  CPU** (flash_attn && use_gpu with Metal/CUDA only: 32 / 256), so n_kv = cell_max exactly: no padded columns.
  cell_max scans cells size−1 down to 1 (cell 0 never tested: at least 1).
- KQ_mask, f32 [n_kv, n_tokens, 1] (whisper.cpp:2508), built on the host each call (whisper.cpp:2909): 0, or −INFINITY
  where `!cells[i].has_seq_id(seq_id) || cells[i].pos > pos` (the second loop `for (i = n_tokens; i < n_tokens…)` is
  empty); then `KQ_mask_f16 = ggml_cast(KQ_mask, F16)` — a CPY node (ops.cpp dup_flt<float, ggml_fp16_t>: the scalar
  type conversion, 0.0.9's bit trick; only 0 and −inf go through it). flash_attn_ext asserts only f16 + contiguous: no
  GGML_KQ_MASK_PAD at this pin.
- mul_mat for a one-row src1 (a step) takes the same path as the encoder's (ggml-cpu.c:1254): from_float split by
  thread over the row's elements, then ggml_vec_dot_f16 per output — a matrix-vector product, nothing special-cased.

## The oracle (whisper_oracle --selfkv), first run on jfk 2026-10-08
- The callback asks for every NORM (a call begins at layer 0's, the one reading the decoder input), and for the eleven
  other nodes per layer identified by what they read (the model's tensors, structurally — no node names); the
  NORM is committed as attn_ln's when the MUL by attn_ln.weight reads it, its input (the layer's input) read whole.
  The KQ_mask cast (CPY whose src0 is named KQ_mask) with its f32 source. kv_self.head / n / size / cells (layout probe:
  kv_self at its state offset, cells as std::vector<{int32 pos; std::set<int32> seq_id}>, sizeof 56 asserted) at the
  call's first node; kv_self.k and .v whole when every node of the call is in.
- jfk: A 27 calls (1 prompt [SOT], 26 steps), B 23 (1 prompt of 226 rows, 22 steps) — n_kv up to 248 of 512 cells.
  Self-checks yes: SCALE b = 0 and 0x3EB504F3 in every layer for Q and K; every SCALE runs in place (400 / 400) and
  q's ADD too (200 / 200); the CPYs' destinations are views of kv_self.k / .v at 2·384·(il·512 + head); all three
  products read attn_ln's ADD; f16 weights; no ADD reads K; the mask f32 [n_kv, n_tokens] → f16, once per call;
  the cells' seq_id sets {0} where used, empty where not, no used cell past n_kv; **head = the batch's first position
  and n_kv = head + n_tokens on every call** (greedy, one sequence); mask values only 0 and −inf.
- **Thread behaviour (found):** at 1 and 4 threads, call by call while the batch is the same, every observed node of
  layer 0 is identical (its input is: 0.1.2), and so are the mask and the cache wherever all four layers' inputs were
  the same. **Layer 1's input already differs at 4 threads on every step (A 27 / 27 calls, B 22 / 23) — but not on B's
  226-row prompt.** So the thread dependence enters inside layer 0 after these nodes (self-attention, cross-attention
  or the MLP — 0.1.4+), and only for one-row batches. The products themselves: 0 node rows differ wherever their input
  is the same (no case of a same input giving different rows).
- Observing changes nothing: the results observed = unobserved at 1 and at 4 threads, both configs. 6.7 MB for jfk.
- All 8 inputs (gate run): A 146 calls per thread count, B 164; 1,886 rows at 1 thread; last position 307, n up to 308
  of 512 cells; 44 MB; 273 s. The same thread pattern on every input: layer 1's input differs at 4 threads on every
  one-row call and on none of the multi-row prompts (B's 226-row prompts, and jfk_x3's second-window `[SOT, NOT]`, 2
  rows) — single-row batches only.
- objdump: `ggml_compute_forward_dup` (dup_flt<float, ggml_fp16_t> inlined; the mask's cast and the K/V CPYs) holds no
  `vcvtps2ph`: the scalar conversion, 0.0.9's bit trick.

## voaice side (src/selfkv.rs), first oracle run 2026-10-08 — ALL EXACT FIRST RUN
- `KvSelf`: the cells (`pos`, a sequence bitmask), `find_slot` / `cell_max` / `prepare` written as whisper's, `clear` at
  every window, the f16 k / v buffers `[4][512][384]`; `mask_model` (f32 then the cast) and `mask_into` (f16 directly).
  `SelfAttn`: attn_ln + the three Linear of 0.0.9 per layer; `model` (one value at a time, with discriminator
  variants) and `layer_into` (the fast path, K and V straight into the cells).
- oracle_selfkv_nodes_bit_exact: 620 calls, every layer fed its recorded input: 362,112 node rows (model and fast),
  11,316 mask rows (model f32, model f16, fast f16), 5,079,040 cache rows (the model's cache and the fast one, which
  starts from 0xFFFF and must be cleared at every window) — 0 differ. head, n and the cells equal on every call.
- oracle_selfkv_layer0_from_voaice_input: layer 0 from 0.1.2's run_batch at 1, 2, 4 threads: 2,040,432 rows, 0 differ.
- Discriminators (per input, 1-thread calls): every reading caught on every input that decodes, except as designed:
  the −∞-as-a-finite readings are caught only on prompt rows (a step's mask has no −∞: 225–226 rows per input, all
  from B's prompts; off by one masks a step's own cell and is caught on every row), "padded to 32" misses rows whose n is already a multiple of 32 (399 of 403 on jfk_x3), "the buffer not
  cleared at a window" exists only on jfk_x3 (31,120 cache rows; asserted 0 elsewhere). Indistinguishable as
  predicted: the row converter for the CPYs (0 rows) and for the mask's cast, the sequence ignored.

## Speed (the step, before the gate)
- First fast path for a step read the f16 weights with `vcvtph2ps` (8 channels at a time): ~25 µs a product hot in
  cache. The held f32 copy (0.0.9's `wp`, the kernel's layout): 9.5 µs a product. One layer: 39 µs against 77–86 µs —
  but four layers in a row 712 µs against 408 µs: the f32 copy is 7 MB for 4 layers' three products, past the 4 MB L3,
  and the step becomes DRAM-bound. A real step streams far more (cross-attention's and the MLP's weights, and the
  logits' 40 MB table), so the step reads the f16 bytes. Revisit on Zen 3 (more conversion throughput).
- Threads for a step: a scoped spawn per call costs more than the step's three products (2 threads: 122 µs, 4: 152 µs
  against 77–86 µs at one), so `layer_into` gives a step one thread (`GEMV_MIN_MACS_PER_THREAD`); the split by output
  channels stays, unit-tested (`layer_into_split`). A persistent pool is the way to use the second core for steps.
- The prompt (226 rows): 0.0.9's panels: ~5 ms a layer at one thread; at two threads no gain on this laptop (CPU time
  doubles: the two cores' boost drops, or the scheduler pairs SMT siblings — not investigated further).

## Efficiency (gate step 15, after 4k passed; load 1.0 at the gate's start, 2.8 after; rerun at 2.2–2.6)
- block = layer 0's twelve nodes; call = all four layers' and the mask; mean over >= 1 s (gate · rerun), digests equal:
  step block 1t 120.4 · 117.5 -> 81.0 · 76.1 µs (1.49 · 1.54x); 2t 89.0 · 79.5 -> 90.0 · 72.5 (0.99 · 1.10x); 4t
  506.4 (noise) · 88.6 -> 81.5 · 73.4 (1.21x in the rerun). step call 1t 502.5 · 567.3 -> 431.2 · 399.3 (1.17 · 1.42x);
  **2t 355.8 · 366.0 -> 417.6 · 384.1 (0.85 · 0.95x); 4t 377.5 · 388.6 -> 410.8 · 386.8 (0.92 · 1.00x): not exceeded.**
  prompt block 1t 27.66 · 25.73 ms -> 5.83 · 4.96 (4.75 · 5.19x); 2t 3.88 · 2.57x; 4t 3.44 · 2.95x. prompt call 1t
  103.5 · 126.9 -> 21.7 · 20.9 (4.76 · 6.06x); 2t 3.23 · 2.74x; 4t 3.03 · 3.18x.
- memory: 198,184 B scratch per 226-row call (3,840 B a step) against the reference's 170 KiB work buffer; held 13,468
  KiB for four layers (f16 + f32 weights, the 3 MiB cache).
