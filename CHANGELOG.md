# Changelog

## 0.1.3 — 2026-10-08 — the self-attention products and the f16 self KV cache, bit-exact: every decoder layer, every call

**In every decoder layer, before self-attention itself, `whisper_build_graph_decoder` computes attn_ln, Q + bias ×
`KQscale`, K × `KQscale` (no bias), V + bias and copies K and V to f16 into `kv_self` at the slot
`whisper_kv_cache_find_slot` found; the host builds the causal KQ_mask and the graph casts it to f16. voaice computes
the same twelve nodes, keeps the same cells, builds the same mask and leaves the same cache — every cell of every
layer after every call — bit for bit on every decoder call `whisper_full` makes for the 8 inputs (620 calls, both of
0.1.2's configs, 1 and 4 threads), from the reference's layer inputs and, for layer 0, from voaice's own decoder input;
all exact on the first oracle run. Found while recording: the reference's thread dependence starts inside layer 0,
after these nodes, and only for one-row batches.**
Record: `testing/results/0.1.3.txt`; how it was read and found: `testing/selfkv/NOTES.md`.

### The op, read from the pin and the binary
- whisper.cpp:2530–2590, per layer: `cur = add(mul(norm(inpL, eps), attn_ln.weight), attn_ln.bias)` (0.0.8's three
  nodes); `Qcur = scale(add(mul_mat(attn.query.weight, cur), attn.query.bias), KQscale)` — **the bias before the
  scale**; `Kcur = scale(mul_mat(attn.key.weight, cur), KQscale)` — **no bias** ("note: no bias for Key"; the model has
  no `attn.key.bias`), **scaled here**, unlike the encoder's K; `Vcur = add(mul_mat(attn.value.weight, cur),
  attn.value.bias)`; with flash_attn, `cpy(Kcur, view_1d(kv_self.k, n_tokens · 384, 2 · 384 · (il · n_ctx + kv_head)))`
  and the same for V. `KQscale = pow(float(n_state_head), -0.25)` — 0.1.1's `0x3EB504F3`; `ggml_scale`'s b is 0 (one
  `vmulps` per 8 values). A one-row step's MUL_MAT is the encoder's code (`from_float` split by thread, then
  `ggml_vec_dot_f16` per output) as a matrix-vector product.
- `kv_self` = `whisper_kv_cache_init(F16, 384, 4, GGML_PAD(448, 256) = 512)`: k and v `[4 · 512 · 384]` f16, layer il's
  cell c at row `il · 512 + c`; cleared at init and by `whisper_kv_cache_clear` at the start of **every window**.
- Before the graph (whisper.cpp:2872): `whisper_kv_cache_find_slot` — from the head, the first `n_tokens` free cells in
  a row; the head stays at the slot it returns — then `kv_self.n = min(size, max(pad, GGML_PAD(cell_max, pad)))`, and
  **the pad is 1 on the CPU** (`whisper_kv_cache_get_padding`: 32 for Metal, 256 for CUDA, else 1), so n = cell_max and
  the mask has no padded columns; `cell_max` scans down to cell 1 (cell 0 never tested).
- KQ_mask, f32 `[n_kv, n_tokens]`, filled on the host: −∞ where `!cell.has_seq_id(seq) || cell.pos > pos`; then
  `ggml_cast(·, F16)`, a CPY by the scalar conversion — `ggml_compute_forward_dup` (with `dup_flt<float, ggml_fp16_t>`
  inlined) holds no `vcvtps2ph`. flash_attn_ext asserts only an f16, contiguous mask: no `GGML_KQ_MASK_PAD` at this pin.

### The oracle
- `whisper_oracle --selfkv`: the layout probe adds `whisper_state::kv_self` and `whisper_kv_cache`'s head, size, n and
  cells (a mirror `{int32 pos; std::set<int32> seq_id}`, its size and offsets asserted). The eval callback asks for
  every NORM and for the eleven other nodes of each layer, identified by the model tensors they read (no names); a
  call begins at layer 0's NORM (the one reading `add(get_rows, get_rows)`); a NORM is attn_ln's when the MUL by
  `attn_ln.weight` reads it, and its input — the layer's input — is kept whole. Also the KQ_mask cast and its f32
  source, the cells at the call's first node, and kv_self.k / .v whole after every call. Configs A and B at 1 and 4
  threads observed, then unobserved at 1 and 4. Self-checks yes on 8 / 8 (listed in docs/oracles.md: the scale bits
  and b = 0, the CPYs' destinations at `2 · 384 · (il · 512 + head)`, the products reading attn_ln, no ADD on K, mask
  values only 0 and −∞, **head = the batch's first position and n = head + n_tokens on every call**, observed =
  unobserved). 44 MB for 8 inputs, ~5 min.
- `oracle_selfkv_nodes_bit_exact`: all 620 calls, every layer fed its recorded input — the twelve nodes by the model
  and by the fast path, head / n / cells by `KvSelf::prepare`, the mask (model f32 and f16, fast f16), the whole cache
  after each call (from the model's CPYs, and the fast path's, which starts from a buffer of 0xFFFF): **0 differ** in
  362,112 node rows, 11,316 mask rows and 5,079,040 cache rows.
- `oracle_selfkv_layer0_from_voaice_input`: layer 0 from 0.1.2's `DecoderInput::run_batch` at 1, 2 and 4 threads, the
  cache kept across each run: **0 differ** in 2,040,432 rows (the twelve nodes and layer 0's 1,024 cache rows per call).
- `oracle_selfkv_observed_run_is_the_recorded_transcript`: config A's observed 1-thread result = 0.0.1's record (128 / 128).
- `oracle_selfkv_discriminators` (the 1-thread calls of both configs), caught on every input that decodes: the scale
  before the products (5,664–9,672 node rows per input), Q scaled before its bias, Q not scaled, K not scaled, K given
  a bias, V scaled, the scale in double, attn_ln's MUL + ADD fused, no f16 rounding, one accumulator, the accumulators
  in sequence; the mask off by one, padded to 32 columns, −∞ as −65504, −∞ as f32's lowest (caught in f32 only — its
  f16 cast is −∞); K and V a cell late, the layers 448 cells apart. The buffer left uncleared at a window: caught on
  jfk_x3 (31,120 cache rows), the only input with a second window, and asserted indistinguishable elsewhere.
  Indistinguishable, as predicted: the CPYs and the mask's cast by the row converter, the sequence ignored (one
  sequence, no free cell below n). **Found: how −∞ is written is told apart only on prompts** — a step attends
  every cell (its mask has no −∞), so only config B's 226-row prompts and jfk_x3's `[SOT, NOT]` can tell −65504 or
  f32's lowest from −∞ (the off-by-one reading masks a step's own cell, so it is caught on every row).
- **Found: where the reference's thread dependence starts.** Call by call at 1 and 4 threads while the batch is the
  same, every node here is identical wherever its input is — layer 0 on every call, and the mask and the cache.
  **Layer 1's input already differs at 4 threads on every one-row call and on no multi-row prompt** (B's 226-row
  prompts, jfk_x3's 2-row one): the dependence enters in layer 0 after these nodes — self-attention, cross-attention
  or the MLP — for single-row batches only. 0.1.4 reads the flash-attention path a single query takes first.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, no AVX-512); production's Zen 3
  library not run; `base.en` not compared; one sequence only (best-of / beam: `seq_id` ≠ 0, `seq_cp` / `seq_rm`, a
  cache made `factor` × larger, `find_slot`'s wrap — read from the source, not checked); a full cache not reached
  (the longest run ends at cell 307); GPU paddings unreachable on the CPU; layers 1–3 fed the reference's inputs.

### Faster, bits unchanged
- attn_ln computed row by row into the products' f32 → f16 conversion (never written), one conversion for Q, K and V.
- A prompt (four rows or more): 0.0.9's 4 × 3 register block over panels of 64 frames, threads by frames; the scale,
  the biases and the f16 conversions as the panel's epilogue, K and V written straight into the cache's cells.
- A step (up to three rows): a matrix-vector product from the **f16** weights, eight output channels at a time —
  `vcvtph2ps` on each weight block as `ggml_vec_dot_f16` widens it, the same four chains and lane pairing — on one
  thread: a scoped spawn costs more than a step's three products. Measured on this Zen+ core: the held f32 copy is
  ~2× faster when one layer is hot in cache but ~1.7× slower when four layers stream past the 4 MB L3, and a real step
  streams far more (the logits alone read 40 MB) — so a step reads the f16 bytes.
- The mask built from the cells once per call, in f16 directly (0 → `0x0000`, −∞ → `0xFC00`).

### Measured (gate step 15, only after 4k passed; Ryzen 3 3200U, 2 cores / 4 threads, load ≈ 1.0 at the gate's start and 2.8 after the measurements; rerun at ≈ 2.2–2.6 in `.oracle/step15_rerun.txt`)
The same input on both sides and the output digests equal (Q, the K/V cells written, the f16 mask). The reference =
the inputs set, plan and compute of exactly these nodes as a ggml graph (for **call** also the f32 mask's cast; whisper
fills the f32 mask on the host, outside the graph); voaice = `SelfAttn::layer_into` into `KvSelf`'s cells (for call
also `KvSelf::mask_into`, which builds the mask from the cells — more than the reference's measured work); the slot
found once before timing on both sides. **block** = layer 0's twelve nodes; **call** = all four layers' (each reading
the same input) and the mask. Mean per call over ≥ 1 s (gate run · rerun):

| tokens | | threads | whisper.cpp | voaice.rs | × |
|---|---|---|---|---|---|
| **1 (a step)** | block | 1 | 120.4 · 117.5 µs | **81.0 · 76.1 µs** | **1.49 · 1.54×** |
| | | 2 | 89.0 · 79.5 | 90.0 · 72.5 | 0.99 · 1.10× |
| | | 4 | 506.4 · 88.6 | 81.5 · 73.4 | (noise) · 1.21× |
| | call | 1 | 502.5 · 567.3 | **431.2 · 399.3** | **1.17 · 1.42×** |
| | | 2 | 355.8 · 366.0 | 417.6 · 384.1 | **0.85 · 0.95×** |
| | | 4 | 377.5 · 388.6 | 410.8 · 386.8 | **0.92 · 1.00×** |
| **226 (config B's prompt)** | block | 1 | 27.66 · 25.73 ms | **5.83 · 4.96 ms** | **4.75 · 5.19×** |
| | | 2 | 22.24 · 14.47 | 5.73 · 5.64 | 3.88 · 2.57× |
| | | 4 | 14.50 · 14.50 | 4.21 · 4.91 | 3.44 · 2.95× |
| | call | 1 | 103.5 · 126.9 | **21.7 · 20.9** | **4.76 · 6.06×** |
| | | 2 | 79.3 · 57.7 | 24.5 · 21.0 | 3.23 · 2.74× |
| | | 4 | 58.4 · 57.6 | 19.3 · 18.1 | 3.03 · 3.18× |

- **Read with care.** A step runs on one thread in voaice at any count; **at two and four threads the reference's
  four-layer step is as fast or faster (0.85–1.00×)** — not exceeded, and said so. One layer hot in cache is 1.5× at
  one thread; four layers in a row stream 3.5 MB of f16 weights past this 4 MB L3 and the margin shrinks to 1.2–1.4×:
  a step's products are bound by `vcvtph2ps` (voaice widens each weight block once; `ggml_vec_dot_f16` widens the
  activation's block beside it every time) and by the bytes. The 4-thread 506 µs in the gate run is the reference's noise (89 µs
  in the rerun), not claimed. The prompt — a matrix-matrix product, 0.0.9's panels — is **4.8–6.1× at one thread**,
  2.6–3.9× at two and four (at two threads voaice gains nothing over one here: measured, not explained further).
- Memory: one call at one thread allocates 198,184 bytes of panel scratch (226 rows; 3,840 bytes for a step) against
  the reference's 170 KiB work buffer. Held outside the call: per layer the three products' f16 weights and their
  widened f32 copy (the prompt's panels read the copy, a step the f16), and the 3 MiB f16 cache — 13,468 KiB for four
  layers; the reference reads the model's weights in place and keeps the same 3 MiB cache in its state.
- At the scale of a transcript: a step's self-attention products are ~0.4 ms of a decoder step whose other products
  (cross-attention's, the MLP's and above all the logits' 51,864 × 384) are v0.2.0's to measure.

## 0.1.2 — 2026-10-08 — the decoder's input, bit-exact: every decoder call `whisper_full` makes

**`whisper_build_graph_decoder` begins `add(get_rows(d_te, embd), get_rows(d_pe, position))` — the f16 token row
widened, the f32 position row added — and `whisper_decode_internal` takes its inputs from the batch whisper builds:
the prompt through `whisper_batch_prep_legacy`, every next token as a one-row batch. voaice builds the same prompts and
batches and computes the same rows, bit for bit, on every decoder call of `whisper_full` for the 8 inputs — 620 calls,
3,772 rows, at 1 and 4 threads — all exact on the first oracle run. Found while recording: every prompt the recorded
transcripts feed tiny.en is the single token `[SOT]`, so a second configuration (a 300-token prompt, no timestamps)
reaches the 226-row batch; and d_pe, stored f32, holds only f16 values, so one planned discriminator cannot fail.**
Record: `testing/results/0.1.2.txt`; how it was read and found: `testing/decin/NOTES.md`.

### The op, read from the pin and the binary
- whisper.cpp:2515: `cur = ggml_add(ggml_get_rows(model.d_te, embd), ggml_get_rows(model.d_pe, position))`. `d_te` =
  `decoder.token_embedding.weight`, the weight type: **f16** `[384, 51864]`; `d_pe` = `decoder.positional_embedding`,
  **f32** `[384, 448]` (whisper.cpp:1797). The token rows are the ADD's first operand.
- GET_ROWS from f16 (ops.cpp:4831) widens each row with `ggml_cpu_fp16_to_fp32` — objdump: `ggml_compute_forward_get_rows`
  calls it (no `vcvtph2ps` of its own); it is `vcvtph2ps` in blocks of 8 and 4 and the table for a tail (none at 384).
  Widening is exact and both agree on all 65,536 patterns (0.0.3): nothing to round. GET_ROWS from f32 copies
  (`ggml_vec_cpy_f32`); the ADD is one f32 add per value. Rows split by thread: no bit depends on it.
- The batch (whisper.cpp:511, 7456): a window's prompt as `whisper_batch_prep_legacy(prompt, n_past 0, seq 0)` —
  positions `n_past + i`, `n_seq_id` 1, logits on the **last row only**; each step one row, the decoder's last sampled
  token at position `prompt.size() + i`, logits on.
- The prompt (whisper.cpp:6975, 7106): `[SOT]`, `+ [lang, task]` for a multilingual model, `+ [NOT]` without
  timestamps; before it, when there is past text, `[PREV]` and the last `min(max_prompt_ctx − 1, |past|)` tokens
  (`max_prompt_ctx = min(n_max_text_ctx, 448 / 2)` = 224); a window starting within 500 frames of the end clears the past.

### The oracle
- `whisper_oracle --decin`: the layout probe adds `whisper_state::sched_decode` and `whisper_state::batch` (mirrored, its
  size asserted); the eval callback asks only for the two GET_ROWS (by their src0) and their ADD, and at the token rows
  reads the state's batch, self-checked on every call against the graph's own `embd` / `position` tensors. Per call: the
  batch and one digest per row of the three nodes. `whisper_full` on each input in **config A** (the 0.0.1 transcript
  record's params) observed at 1 and 4 threads and unobserved at 1 and 4, and **config B** (the same + `no_timestamps`
  + 300 `prompt_tokens`) observed at 1 and 4 and unobserved at 1. Self-checks yes on 8 / 8 (order token rows → position
  rows → ADD, the ADD's src0 the token rows, no other GET_ROWS, the types and shapes, batch = graph inputs, observed =
  unobserved). 256 KB for 8 inputs, ~4 min.
- **Found: config A's prompts are all `[SOT]`, one row.** tiny.en is English-only (no language or task token), the
  record runs with timestamps (no NOT), and jfk_x3's second window starts at 2,900 of 3,300 frames, within 500 of the
  end, so its past is cleared. A one-row prompt cannot tell "logits on the last row only" from "on every row", nor
  exercise a prompt's positions — hence config B: `[PREV, the last 223 of the prompt, SOT, NOT]`, 226 rows, positions
  0..225, the steps on to 307.
- `oracle_decin_nodes_bit_exact`: every call of both configs at both thread counts — the model's token rows, position
  rows and sum, and the fast path's sum: **0 differ** of 15,088 row comparisons (620 calls, 3,772 rows).
- `oracle_decin_batches`: **32 / 32** prompts (`Prompt::window` → `Batch::prep_legacy`) and **588 / 588** steps
  (`Batch::prep_step`) equal the recorded batches — tokens, positions, sequence ids, `n_seq_id`, logits flags. The
  past-clearing rule's input (the window's seek) belongs to the decode loop; the test supplies it for jfk_x3's second
  window and says so.
- `oracle_decin_observed_run_is_the_recorded_transcript`: the observed 1-thread result = 0.0.1's transcript record,
  **128 / 128** tokens, ids and `p` bits.
- `oracle_decin_discriminators` (the 1-thread calls of both configs, per input), each caught on every input that
  decodes (7 of 8; min_len makes no decoder call): positions off by one (every row), step positions without the prompt
  length (every step), the position row 0 (every row past 0), the multilingual ids SOT + 1 (5–20 rows), widening with
  f16 subnormals flushed (174–299 rows: SOT's own row holds 3 of d_te's 91,135 subnormals), widening through bf16 and
  the sum rounded to f16 (every row). Indistinguishable, as predicted: the operands swapped (f32 addition commutes),
  the sum in double (exact for two f32). **Found: the position rows rounded to f16 is indistinguishable too — all
  172,032 of d_pe's f32 values are exactly f16 values** (the checkpoint was f16; the converter widened it). The test
  asserts that property instead of the discriminator.
- **Found: the thread count moves the reference's decoder, not its input.** Config A feeds the same tokens at 1 and 4
  threads and every input row is identical, but the result's `p` bits differ on 5 of the 7 inputs that decode; in
  config B chirp is fed a different token at 4 threads from its third call on. The logits (v0.2.0) will be compared at
  stated thread counts, as the transcript always was.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, no AVX-512); production's Zen 3
  library not run; `base.en` (n_state 512) not compared; **multilingual prompts not exercised** (no multilingual model is
  pinned: `Prompt::init`'s language and task tokens are read from the source only); several decoders (best-of, beam:
  `seq_id` ≠ 0, several rows per step), `carry_initial_prompt` and the temperature fallback not exercised.

### Faster, bits unchanged
- The token row widened and its position row added in one pass, eight values at a time (`vcvtph2ps` — the reference's
  own instruction — and `vaddps`), straight into the caller's buffer; the batch's vectors sized once; **nothing
  allocated per call** (0 bytes after the first); one thread (a step is one row; the reference's row split cannot
  change a bit).

### Measured (gate step 14, only after 4j passed; Ryzen 3 3200U, load ≈ 4.5; rerun at ≈ 3.2 in `.oracle/step14_rerun.txt`)
The same tokens on both sides, output digests equal; the reference = the two inputs set, plan and compute of the three
nodes as a ggml graph; voaice = `Batch::prep_legacy` + `DecoderInput::run_batch`; mean per call over ≥ 1 s:

| tokens | whisper.cpp, 1 thread | whisper.cpp, 4 threads | voaice.rs (1 thread) |
|---|---|---|---|
| 1 (a step) | 1.05 · 1.00 µs | 20.1 · 4.8 µs wall (51.5 · 18.7 CPU-µs) | **0.108 · 0.107–0.114 µs** |
| 226 (config B's prompt) | 110.5 · 61.2–66.3 µs | 43.2 · 40.8–55.3 µs wall | **22.8 · 21.4–21.7 µs** |

- **Read with care.** For one token the reference's microsecond is mostly planning and dispatching a graph; in whisper
  these nodes head the whole decoder graph and share its plan and thread start, so that cost is not theirs and the
  one-token ratio is not claimed. The per-row work is comparable: ~0.095 µs a row against 0.27–0.49 µs, **2.9–3.1× at
  one thread in the rerun** (4.9× in the gate, whose reference figure was taken at load 4.5). At the scale of a
  transcript this stage is noise: well under a microsecond a token against a decoder step of milliseconds (that step
  is v0.2.0's to measure).
- Memory: voaice holds the f16 token table copied from the model (38,898 KiB) and the f32 positions (672 KiB) outside
  the call; the reference reads the model's tensors in place. The logits product (0.1.7) reads the same table.

## 0.1.1 — 2026-10-08 — cross-attention K and V, bit-exact: every node of `sched_cross` and `kv_cross` itself

**The second decade opens with the cross graph `whisper_encode_with_state` runs after the encoder: per decoder layer
K = `embd_enc` × `cross_attn.key.weight` × `Kscale` (a SCALE node), V = `embd_enc` × `cross_attn.value.weight` + bias,
both copied to f16 into `kv_cross` — every one of its 24 nodes and the whole cache, its 288 padding rows included, bit
for bit on 8 inputs, from the reference's `embd_enc` (model, 1 and 4 threads) and from voaice's own mel (1, 2 and 4
threads); all exact on the first oracle run. Then the cross stage 4.76× the reference at one thread (4.86× in a
rerun), and — like for like for the first time — the whole of `whisper_encode_with_state`, mel → `embd_enc` →
`kv_cross`, 3.70× at one thread (1,085 against 4,011 ms; 3.74× in a rerun). Found while reading: v0.1.0's encoder
ratio compared voaice's encoder against whisper's encoder + cross.**
Record: `testing/results/0.1.1.txt`; how it was read and found: `testing/cross/NOTES.md`.

### The op, read from the pin and the binary
- `whisper_build_graph_cross` (whisper.cpp:2272), on its own scheduler `sched_cross`, for each of tiny's 4 decoder
  layers: `mul_mat(cross_attn.key.weight, embd_enc)` → `ggml_scale(·, Kscale)` → `ggml_cpy` to f16;
  `mul_mat(cross_attn.value.weight, embd_enc)` → `ggml_add(·, cross_attn.value.bias)` → `ggml_cpy` to f16. **K has no
  bias** (the model has none); **only K is scaled** here.
- `Kscale = pow(float(n_state_head), -0.25)`: `std::pow(float, double)` in double, then the float initialiser —
  `0x3EB504F3` for head size 64 (as does every other way of writing 64^−0.25 in f32: the recorded op parameter, not a
  reading, pins it). `ggml_scale` passes `b = 0`, so `ggml_compute_forward_scale_f32` takes `ggml_vec_scale_f32`: one
  f32 multiply per value (6 `vmulps` in the binary; the FMA path is the `b ≠ 0` branch). The allocator runs it in place.
- The products are 0.0.9's (`from_float` split by thread, then `ggml_vec_dot_f16`); the CPYs are 0.0.9's scalar bit trick.
- `kv_cross` (flash_attn): `whisper_kv_cache_init(F16, n_text_state 384, n_text_layer 4, GGML_PAD(1500, 256) = 1536)`
  — k and v each `[4 · 1536 · 384]` f16, the buffer cleared at init; layer il's K/V written to rows `il · 1536 + [0,
  1500)` (frame-major, flash attention's layout), so **rows 1,500..1,535 of every layer are +0** for the life of the state.
- **`whisper_encode_with_state` runs this graph every call** (whisper_encode_internal: conv, encoder, then `// cross`).
  v0.1.0's step 12 timed that call as "the whole encoder" against voaice's `encode_into`, which did not compute the
  cross K/V: the reference was doing ~11 % more work than voaice's measured call. Its 3.85–4.04× was flattered; the
  like-for-like figure is below.

### The oracle
- `whisper_oracle --cross`: the layout probe adds `whisper_state::sched_cross`, `kv_cross` and the cache's `k` / `v`; the
  oracle self-checks them (kv_cross's tensors `[384 · 4 · 1536]` f16, and every CPY's destination a view of exactly those
  tensors at byte offset `2 · 384 · il · 1536`). An eval callback observes every node of `sched_cross` at 1 and 4
  threads — **24 computed nodes, 6 per layer, nothing else** — one digest per row of each; after every run (observed at
  1 and 4 threads, unobserved at 1, 2, 4) `kv_cross` is read whole and digested per row, **all 12,288 rows, the padding
  included**. Self-checks yes on 8 / 8: f16 weights in a plain CPU buffer; src1 is `embd_enc` itself; Kscale the same in
  every layer, `b` = 0; no ADD reads K; V's ADD reads `cross_attn.value.bias`; kv_cross +0 at init and its 288 padding
  rows +0 after; the CPY nodes' rows = the buffer's; 1 vs 4 threads identical; observed = unobserved; 1 = 2 = 4
  threads; `embd_enc` identical in every run; the standalone graph (what `--bench-cross` times) = the scheduler's
  kv_cross at 1 and 4 threads. 3.1 MB for 8 inputs.
- `oracle_cross_nodes_bit_exact` (fed the reference's `embd_enc`, checked against the digest the cross record took of
  the state's): the model on every frame of every layer, the fast path at 1 and 4 threads — **0 rows differ** of
  864,000 node rows and **0 of 294,912 kv_cross rows**.
- `oracle_cross_end_to_end`: voaice's mel → `Encoder::encode_into` → `Cross::run_into` at 1, 2 and 4 threads — `embd_enc`
  equal, every node **0 rows differ** (576 node comparisons), the cache (pre-filled with 0xFFFF, written twice) **0 of
  294,912 rows**; 1,158,912 rows in all.
- `oracle_cross_discriminators` (every 25th frame and the last, 4 layers, all six nodes: 1,464 rows per input; and the
  cache's layout against its 12,288 rows), each caught on every input: the scale before the product 732; folded into the
  f16 weights 732; in double 244–247; after the f16 copy 488; V scaled too 488; V without its bias 488; V's bias on K
  488; the activations not rounded to f16 1,464; the layers packed without padding 9,216; the non-flash layout (V
  transposed) 10,716; padding not +0 288. **The CPYs by the row converter: 0 on every input**, as predicted (the two
  converters agree on every finite value; 0.0.9 told them apart only with constructed NaN rows).
- **Found on the way:** compared on the f16 cache alone, "scale in double" was caught on only 6 of 8 inputs (0–3 of 488
  rows): the f16 rounding swallows most one-ulp f32 differences. The discriminators now compare every node, f32 first.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, no AVX-512). Production's Zen 3
  library not run; `base.en` (n_state 512, 6 decoder layers) not compared; `audio_ctx` ≠ 0 not compared; non-finite
  activations not compared through this graph.

### Faster, bits unchanged
- `embd_enc` converted **once** per panel of 128 frames for all eight products (whisper's eight MUL_MATs convert it
  eight times); each product 0.0.9's 4 × 3 register block; the scale, the bias and the f16 conversion as the panel's
  epilogue, written straight into the caller's cache (`KvCross`, sized on the first call); the padding rows set to +0
  on every call; threads by frames. The panel size (64, 128, 256) measured within noise: the stage is FMA-bound.

### Measured (gate step 13, only after 4i passed; Ryzen 3 3200U, 2 cores / 4 threads, load 4.5 at the gate's start, ~3
during the measurements; rerun at load 2.0–2.5 in `.oracle/step13_rerun.txt`)
jfk; each side in a fresh process; wall = best of 10 (gate run · rerun):

| | threads | whisper.cpp | voaice.rs | × |
|---|---|---|---|---|
| **cross K/V of 4 layers**, `embd_enc` → `kv_cross` | 1 | 454.2 · 451.0 ms | **95.4 · 92.8 ms** | **4.76 · 4.86×** |
| | 2 | 278.8 · 263.7 | 58.3 · 57.7 | 4.78 · 4.57× |
| | 4 | 283.2 · 274.3 | 67.5 · 69.1 | 4.19 · 3.97× |
| **the whole of `whisper_encode_with_state`**, mel → `embd_enc` → `kv_cross` | 1 | 4,010.6 · 4,021.5 | **1,085.4 · 1,074.9** | **3.70 · 3.74×** |
| | 2 | 2,440.3 · 2,457.6 | 835.8 · 825.0 | 2.92 · 2.98× |
| | 4 | 2,395.9 | 901.6 | 2.66× |

- The cross stage: 1,769,472,000 multiply-adds in 95 ms ≈ 18.5 G a second at one thread, about two thirds of this
  core's 8 FMA lanes a cycle at its 3.5 GHz boost. CPU per call 97 against 471 CPU-ms at one thread (4.8×).
- **The fair whole is 3.70–3.74× at one thread**, against step 12's 3.79× for voaice's encoder alone against the same
  call (v0.1.0's gate: 3.85×, rerun 4.04×). The reference's cross graph is ~11 % of its call (454 of 4,011 ms), and
  voaice now does that work too, faster than the rest, so the ratio barely moves — but it is now the like-for-like one.
  Subtracting, the reference's encoder alone is ≈ 3,557 ms against voaice's 1,056: ≈ 3.4× (derived, not measured).
- Memory: the cross stage holds **9,794 KiB** at one thread (the 9,216 KiB cache itself + the panels) against
  whisper's **11,466 KiB** (`sched_cross`'s 2,250 KiB compute buffer + the same 9,216 KiB cache). The whole: **26,312 KiB**
  against **40,864 KiB** (v0.1.0's 29,398 plus `sched_cross` and `kv_cross`). voaice holds the eight cross weights
  widened to f32 (4.7 MB, 2× the f16 bytes) outside the call. Both sides' `kv_cross` and `embd_enc` were digested and
  compared by the gate: `76f24e73e318b877` and `50d38ec85f2778b9` on both.
- At four threads voaice is no faster than at two (2 cores; SMT siblings share the FMA pipes); the reference's whole
  call gains as little.

## v0.1.0 — 2026-10-08 — MILESTONE: the whole encoder, bit-exact — flash attention, four blocks, ln_post, `embd_enc`

**From voaice's own mel to `embd_enc`, bit for bit what the pinned whisper.cpp's `whisper_encode_with_state` leaves in
`whisper_state::embd_enc`: every computed node of the encoder graph (96 per run) and `embd_enc` itself 0 differ on 8
inputs at 1, 2 and 4 threads. Flash attention — ggml's tiled kernel, the one whisper's graph actually takes — 0 differ
in 36,864,000 values; nine discriminators caught on every input, one of them confirmed by the reference's own other
path. Then the whole encoder, mel → `embd_enc`, 3.85× whisper.cpp at one thread in the gate run (1,038 against 3,994
ms; 4.04× in a rerun: 987 against 3,983), 3.05–3.14× at two threads, 2.95–3.01× at four, in 17.9 MB of heap against
the 29.4 MB of compute buffers and cache whisper allocates for it.** The first ten increments are done: 0.0.1 … 0.0.9
built this milestone.
Record: `testing/results/0.1.0.txt`; how it was read and found: `testing/attention/NOTES.md`.

### The op, read from the pin and the binary — and the path, before the path
- whisper's encoder: `Q = permute(reshape_3d(q_add, 64, 6, 1500))` stays **f32**; K and V are 0.0.9's f16 CPYs into
  `kv_pad`, viewed as [64, 1536, 6]; `flash_attn_ext(Q, K, V, no mask, 1.0f/sqrtf(64))`.
- **`kv_pad` has 1,536 rows** (`GGML_PAD(1500, 256)`); `whisper_kv_cache_init` clears its buffer and the CPYs write rows
  0..1499 only, so **the 36 padding rows are +0, and with no mask they are attended**: each scores exactly +0 (every
  product ±0, `+0 + −0 = +0`), adds `exp(0 − M)` to the denominator, nothing to the numerator. The record checks the
  padding is +0 in every block; leaving it out is a discriminator.
- ggml has two flash-attention kernels and a shape test picks one. TODO.md had read the one-chunk path (Q → f16,
  `ggml_vec_dot_f16` scores, V accumulated in f16). **whisper's encoder takes the tiled path**
  (`ggml_compute_forward_flash_attn_ext_tiled`: Q f32, K/V f16, 1,500 ≥ the 64-row tile, 64 % 8 = 0): scores as f32
  FMA chains from +0 (`simd_gemm`: every element its own chain) × 0.125; per tile of 64 keys the max (a `vmaxss`
  chain), `fmaxf`, and when it grows a rescale of the output and the sum by **glibc's `expf(Mold − Mnew)`**; the
  probabilities by `ggml_vec_soft_max_f32` — **ggml's own 8-lane `ggml_v_expf`** (ARM's optimized-routines expf), summed
  8 at a time in f32 as `((y0 + y4) + (y2 + y6)) + ((y1 + y5) + (y3 + y7))`, those sums added in double, then
  `S = (float)((double)S + sum)`; **the output accumulated in f32** by FMA chains over each tile's keys; × 1/S at the end.
  Confirmed in this laptop's libggml-cpu: 47 `vfmadd231ps`, one `vmaxss`, `fmaxf@plt`, `expf@plt` (GLIBC_2.27),
  `ggml_vec_soft_max_f32@plt` (8 FMA, the pairing, `vaddsd`), `vcvtss2sd / vaddsd / vcvtsd2ss` for S, no `vcvtps2ph`.
- Rows are independent (per-row softmax state; every score and output element one chain), so the reference is
  bit-identical at any thread count — checked at 1..8 — and so is `embd_enc` (1, 2, 4).

### The oracle
- `whisper_oracle --encoder`: an eval callback on `sched_encode` observing every node (0.0.9's keys + the norm chains, the
  positional ADD, FLASH_ATTN_EXT), one digest per row of each of the 97 computed nodes, `embd_enc` whole after
  unobserved runs at 1, 2 and 4 threads (read from the state through the layout probe). At each FLASH_ATTN_EXT the
  callback copies Q and all of `kv_pad` before the node runs: types, shapes, strides, op params checked, padding +0. The
  standalone attention graph (what `--bench-attn` times) = the node at 1..8 threads; the same graph with
  `cplan.use_ref` (the one-chunk path) recorded too: **it changes 2,303,363–2,303,984 of each block's 2,304,000 values.**
  28 MB for 8 inputs.
- `oracle_attention_nodes_bit_exact`: every block's attention fed Q, K, V that voaice computes from attn_ln's recorded
  input (each first equal to the record's q_add / k_cpy / v_cpy digests): the fast path at 1 and 4 threads **0 values
  differ** of 36,864,000; the model on all of block 0 and every 10th frame of blocks 1–3 **0 rows differ** of 15,600.
- `oracle_encoder_end_to_end`: the WAV → voaice's mel → the conv stage → four blocks → ln_post at 1, 2 and 4 threads:
  **every node the record names (96: all but the CONT voaice's conv stage folds into its add) 0 rows differ** — 2,304
  node comparisons, 3,456,000 rows — and `Encoder::encode_into` (no taps, caller-owned buffers, called twice)
  **`embd_enc` 0 values differ** on every input at every thread count.
- `oracle_attention_discriminators` (every 25th frame and the last, 6 heads, 4 blocks, of 244 rows per input), each
  caught on every input: glibc `expf` for the probabilities (244), `ggml_v_expf` for the rescale (226–238), no running
  max (234–242), kv_pad's zero rows excluded (244), scores without FMA (244), output without FMA (244), the softmax sums in
  f32 (215–233), out / S (244), the one-chunk path (244). **Q scaled before the dot: 0 on every input** — ×1/8 is exact,
  so this reading cannot be told apart, as predicted, and it is reported as such.
- **Found by the record:** voaice's model of the one-chunk path, checked against the reference's own `use_ref` output,
  matches it (0 of 1,952 rows) only with `S = S·ms + vs` in **two** roundings; with an FMA, 543 rows differ. GCC did not
  contract it: it split the update by branch (`vmulss`, then `vaddss`), as the disassembly then showed.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, no AVX-512) and glibc 2.35's
  `expf` as its ifunc picks it here, called through the same symbol by both sides. Production's Zen 3 library and glibc
  were not run; an AVX-512 build (`ggml_v_expf`'s 16-lane form, another softmax pairing) not compared; `base.en` (8
  heads) not compared; non-finite activations through attention not compared (the same operations, but NaN payloads
  and the max tree's choice among NaNs not checked).

### Faster, bits unchanged
- K transposed per tile and V widened **once per head per call**; the reference re-packs K for every 64-query tile.
- The softmax on the tile in two passes (no vector live across glibc's `expf`), the eight-sums of eight rows at once in
  double lanes, the rescale folded into the output product's load; 6-row × 16-column register blocks for the scores
  and the output (simd_gemm's shape), query tiles of 60 (1,500 = 25 whole tiles); threads by query tiles behind a
  per-head `std::sync::Barrier` (one scope per call), which also keeps the widened K/V to one head (0.8 MB).
- `Encoder::encode_into`: the conv stage, the blocks and ln_post chained through an `EncoderBuffers` the caller owns;
  every earlier fusion kept (attn_ln and mlp_ln inside the conversions, one conversion for Q/K/V, the MLP by panels).
- Ablations on block 0 (one thread): the softmax ≈ 14.5 ms of ~90, of which `ggml_v_expf` ≈ 11 ms (its own op count);
  scores + output ≈ 73 ms against a 65 ms floor of 256-bit FMAs on this core.

### Measured (gate step 12, only after 4h passed; Ryzen 3 3200U, 2 cores / 4 threads, load 1.5 → 3.5)
jfk; each side in a fresh process; wall = best of 10 (gate run · one rerun at load 2.1 → 3.2,
`.oracle/step12_rerun.txt`):

| | threads | whisper.cpp | voaice.rs | × |
|---|---|---|---|---|
| **the whole encoder, mel → `embd_enc`** | 1 | 3,994 · 3,983 ms | **1,038 · 987 ms** | **3.85 · 4.04×** |
| | 2 | 2,436 · 2,431 | 799 · 774 | 3.05 · 3.14× |
| | 4 | 2,378 · 2,435 | 791 · 824 | 3.01 · 2.95× |
| block 0's attention | 1 | 150.8 · 146.6 | 91.9 · 91.3 | 1.64 · 1.61× |
| | 2 | 94.7 · 90.6 | 64.1 · 59.2 | 1.48 · 1.53× |
| | 4 | 89.7 · 90.8 | 70.2 · 62.0 | 1.28 · 1.46× |

- CPU per encode: 1,149 against 4,250 CPU-ms at one thread (3.7×); 2,005 against 5,910 at two; 2,729 against 9,908 at
  four. The `embd_enc` each side measured was digested and compared by the gate: `50d38ec85f2778b9` on both.
- Memory: voaice's encode holds **17,869 KiB** at its peak at one thread (its buffers between stages, allocated by the
  first call; 21,919 at four), against whisper's **29,398 KiB** of compute buffers and `kv_pad` (allocated by
  `whisper_init_state`). Both hold the weights outside the call: whisper f16, voaice widened to f32 for the products
  (28 MB for the four blocks, 2× the f16 bytes) — said here, not hidden. Attention alone: 3,049 against 4,651 KiB.
- At four threads voaice is no faster than at two on this 2-core laptop (SMT siblings share the FMA pipes); the
  reference's thread pool gains as little.

## 0.0.9 — 2026-10-08 — the matrix products on activations, bit-exact: every block's Q, K, V, out projection and MLP

**`mul_mat`'s f32 → f16 conversion of the activations (`from_float`, split by thread) and 0.0.6's f16 dot, with every
bias, GELU, residual and the f16 copies flash attention reads — all 16 product-side nodes of all four blocks, bit for
bit as the shipped ggml-cpu computes them inside whisper's encoder scheduler, on 8 inputs at 1 and 4 threads, and block
0 from voaice's own mel; then 4.6–5.2× the reference at one thread (3.1–5.2× at two, 3.8–4.7× at four), the block in
a seventh of its memory.**
Record: `testing/results/0.0.9.txt`; how it was read and found: `testing/matmul/NOTES.md`.

### The ops, read from the pin and the binary
- Per block: Q = mul_mat(attn.query.weight, attn_ln) + query.bias; **K = mul_mat(attn.key.weight, attn_ln) with no
  bias** (the source says so; the record confirms no ADD reads K's product); V = mul_mat(attn.value.weight, attn_ln) +
  value.bias; K and V copied into flash attention's f16 cache by CPY nodes; after attention, mul_mat(attn.out.weight,
  ·) + out.bias + the block's input; mlp_ln; fc1 = mul_mat(mlp.0.weight, ·) + mlp.0.bias; GELU (0.0.3's table op);
  fc2 = mul_mat(mlp.2.weight, ·) + mlp.2.bias + that residual. Every bias, GELU and residual is its own node (the CPU
  backend fuses only RMS_NORM + MUL).
- `ggml_compute_forward_mul_mat` (ggml-cpu.c:1254) for an f16 weight: `vec_dot_type` F16, so the f32 activations are
  converted first by `from_float` = `ggml_cpu_fp32_to_fp16` (the type traits; `vcvtps2ph` on blocks of 8, then 4, the
  portable bit trick on the last `len % 4`) — **every thread converting its element range `[ith·K/nth, (ith+1)·K/nth)`
  of every row**; a barrier; then chunks of 16 × 16 outputs, each output one `ggml_vec_dot_f16(K, weight row,
  converted row)` whole in one thread. `LLAMAFILE` off; `GGML_CPU_REPACK` on but with no f16 traits (`nm -D`); the
  weights sit in a plain CPU buffer (the record checks the buffer's name). K = 384 and 1,536 are multiples of 32: no
  double tail.
- **The CPY nodes convert with the scalar bit trick** (`ggml_compute_forward_dup_flt<float, ggml_fp16_t>`, element by
  element through `GGML_CPU_FP32_TO_FP16`), not the row converter. The two agree on every finite value (0.0.3).

### The oracle
- `whisper_oracle --matmul`: an eval callback on `sched_encode` observing every node; nodes identified by what they read
  (a MUL_MAT by its weight's name, an ADD by the node it adds to and its bias's name, a CPY by K's product or V's ADD,
  GELU by fc1's ADD). **Recorded compactly**: one 64-bit FNV-1a digest per row of each node (a single changed value
  always changes its row's digest), each MUL_MAT's input digested when the scheduler asks about it, and the attention's
  output whole — 80 MB for 8 inputs, where the tensors would be ~1.8 GB. The full tensors around the products come from
  0.0.8's norm record, each checked against the digest of what the MUL_MAT read before voaice is fed it. Self-checks,
  yes on 8 / 8: 17 nodes in every block; f16 weights in a plain CPU buffer; `src1` contiguous f32; K unbiased; the
  named biases; 1 vs 4 threads identical; `embd_enc` observed = unobserved; the standalone graphs = the nodes.
- `oracle_matmul_nodes_bit_exact`: all 16 nodes (k_mm, k_cpy, v_mm, v_add, v_cpy, q_mm, q_add, o_mm, o_add, o_res,
  fc1_mm, fc1_add, gelu, fc2_mm, fc2_add, mlp_res) of all 4 blocks, by the portable model and the fast path at 1 and 4
  threads — **0 rows differ** of 2,304,000 digested rows (1,382,400,000 values); the residuals also **0 values differ**
  against the norm record.
- `oracle_block0_from_mel`: voaice's mel → conv stage → attn_ln fused into Q, K, V and the CPYs; the out projection →
  the MLP from the recorded attention output: **0 differ** at 1, 2 and 4 threads on 8 inputs.
- `whisper_oracle --mm-nan` / `oracle_mm_nan_split`: twelve rows, ten with one NaN placed in an 8-block, a 4-block or
  the bit trick's tail of some thread's range, through mul_mat(query.weight) + bias at 1..8 threads. voaice's model (at
  the same split) and fast path (told the split) **0 differ** at every count. **Found: the reference's own output
  depends on its thread count when an activation is NaN** — at 5 threads it differs from its 1-thread output in 1,920
  values (5 rows), at 7 in 1,152 (3 rows), at 1, 2, 3, 4, 6 and 8 not at all: the split decides whether `vcvtps2ph`
  (payload kept) or the bit trick (`sign | 0x7E00`) converts the NaN.
- Discriminators, each caught on every input (`oracle_matmul_discriminators`, summed over the 4 blocks): activations
  not rounded to f16, one f32 accumulator, the accumulators reduced in sequence, the bias in the first accumulator —
  **6,000 / 6,000** q_add rows each; GELU from x without the f16 table **6,000 / 6,000** rows; the residual before the
  bias **281–307 k** of 2,304,000 o_res values. The scalar bit trick for `from_float` and the split ignored change
  nothing on the 8 inputs (no NaN) and are caught on the NaN rows: 3,456 values at most thread counts and 1,920 at 5
  threads (1,152 at 7) respectively.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, no AVX-512). Production's Zen 3
  library was not run; an AVX-512 build converts 16 at a time first (another tail) — not compared; `base.en` not
  compared; the attention's output and blocks 1–3's inputs are the reference's (v0.1.0 computes attention); the NaN
  behaviour is checked on one product with constructed rows, not through a whole block.

### Faster, bits unchanged
- attn_ln and mlp_ln are computed row by row **into the conversion**: their f32 output is never written (0.0.8's TODO).
- Q, K and V share **one** conversion of attn_ln's output (the reference converts it three times).
- A panel of 64 frames is converted into a permuted layout (each accumulator's blocks contiguous), rounded through F16C
  and widened back (exact); a **4-frame × 3-row** AVX2 register block (conv2's) chains each dot's four accumulators in
  `ggml_vec_dot_f16`'s order and reduces with its pairing; bias, GELU, residual and the f16 CPYs (bit-trick semantics:
  F16C unless a block holds a NaN) are the panel's epilogue.
- The MLP runs a panel at a time from the out projection to fc2: the residual, mlp_ln's output and the 1,536-wide hidden
  layer never leave the panel. Caller-owned outputs; threads split frames.
- The weights are held widened to f32 in the kernel's layout (2× the f16 bytes: 7.1 MB per block, outside the measured
  call). Keeping them f16 and widening in the kernel measured ~30 % slower on this core.

### Measured (gate step 11, only after 4g passed; Ryzen 3 3200U, 2 cores / 4 threads, load 1.6–2.9)
Block 0 on jfk; inputs computed beforehand by each side from the WAV; the reference = the same ops as a standalone ggml
graph (equal to the scheduler's nodes), its work buffer allocated once; wall = best of 10 (gate run · one rerun):

| | reference 1t | voaice 1t | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|---|
| q: Q + bias | 54.9 ms | 10.6 ms | 5.19× · 5.07× | 3.06× · 3.91× | 4.74× · 4.16× |
| fc1 + bias + GELU | 233.1 | 49.6 | 4.70× · 4.80× | 4.68× · 4.68× | 4.27× · 3.81× |
| fc2 + bias | 210.9 | 41.0 | 5.15× · 4.98× | 5.17× · 4.79× | 4.28× · 4.17× |
| qkv: attn_ln → Q, K, V, the f16 CPYs | 168.8 | 32.6 | 5.18× · 5.15× | 4.79× · 4.34× | 4.10× · 4.10× |
| mlp: out proj → residual → mlp_ln → fc1 → GELU → fc2 → residual | 506.5 | 106.0 | 4.78× · 4.75× | 4.58× · 4.82× | 3.89× · 3.93× |
| block: qkv + mlp (V standing in for attention) | 666.8 | 146.6 | 4.55× · 4.77× | 4.36× · 4.37× | 4.14× · 3.80× |

- Q at 10.6 ms is 21 G multiply-adds per second, about three quarters of this core's eight FMA lanes per cycle at boost.
- **At 4 threads voaice is no faster than at 2** (block 93.6 → 95.0 ms; the FMA pipes are shared by SMT siblings); the
  reference gains little either (408 → 393 ms).
- **CPU:** the block at one thread 162 against 684 CPU-ms; at 4 threads 384 against 1,608.
- **Memory:** the block **10,351 KiB** (its outputs: Q, K16, V16, the V stand-in, the result, and the panels) against the
  graph's **69,751** (every node + the work buffer); q 2,444 against 5,626; fc1 9,482 against 28,126.

### Added
- `src/matmul.rs`: `Linear` (`new`, `from_parts`, `model`, `convert_model`, `run_into`, `with_split`), `Epilogue`,
  `Block` (`new`, `set_split`, `qkv_into`, `mlp_into`), `QkvTaps`, `MlpTaps`, `Variant` (the reference = default; eight
  discriminator flags), `split_ranges`, `from_float_row`, `perm`, `residual_model`, `gelu_model`, `cpy_f16_model`,
  `PANEL`; unit tests (the layout is a permutation; the fast path = the model on k = 32…1536, n = 3…12, 1…70 frames, 1
  and 3 threads, with GELU and a fused norm; a NaN follows the split; the CPY is the scalar trick).
  `LayerNorm::row_into` (one row of the norm's output, for the fusion).
- `voaice qkv`, `voaice bench-mm q|fc1|fc2|qkv|mlp|block`; `tests/matmul.rs` (4 oracle tests).
- `whisper_oracle --matmul`, `--mm-nan`, `--bench-mm`; gate steps 4g and 11; step 3 prints `ggml_cpu_fp32_to_fp16`'s
  vcvtps2ph count, `ggml_compute_forward_mul_mat`'s (no FMA: the dot is called through the traits) and the absence of
  f16 repack traits. Every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample, 5 conv1, 4 conv2, 3 norm).
  The gate's disassembly reads now let `sed` read to the end (`sed -n '1,/ret/p'`): a first run of this version
  stopped with SIGPIPE (exit 141) when `sed '/ret/q'` quit before objdump finished writing. Step 4b was run with
  `VOAICE_OPUS_HOST=unreachable.invalid` (SKIPPED: compared against the recorded answers, as 0.0.6 and 0.0.7 did).

## 0.0.8 — 2026-10-08 — the encoder's layer norms, bit-exact: all nine `norm → · w → + b` chains

**`ggml_norm` and the MUL and ADD whisper puts after it — each block's `attn_ln` and `mlp_ln`, and `ln_post` — bit for
bit as the shipped ggml-cpu computes them inside whisper's encoder scheduler, on 8 inputs at 1 and 4 threads; then
the three nodes in one pass per row: 2–3× the reference's NORM node and 4–5× its three-node chain at one thread, in a
third of its memory.** Record: `testing/results/0.0.8.txt`; how it was read and found: `testing/norm/NOTES.md`.

### The op, read from the pin and the binary
- whisper's encoder has **nine** norms, each `ggml_add(ggml_mul(ggml_norm(x, hparams.eps = 1e-5f), w), b)`: per block
  `attn_ln` (on the block input) and `mlp_ln` (on the residual after attention), then `ln_post` (whose ADD is
  `embd_enc`). MUL and ADD are separate nodes; the CPU backend's graph fusion covers only RMS_NORM + MUL.
- `ggml_compute_forward_norm_f32`, one row (frame, 384 values) whole per thread: `ggml_vec_sum_f32` adds the row to a
  **double in index order** (`vcvtss2sd` + `vaddsd`, not vectorised) and rounds it to f32; `mean = sum / (float)n` in
  f32; `ggml_vec_cvar_f32`'s **AVX2 + FMA branch, with no FMA in it** (`objdump`): per 8 values `d = x − mean`,
  `d·d` (`vmulps`), `h = p[4..8] + p[0..4]`, `(h0 + h2) + (h1 + h3)` in f32, widened and added to a double; `sum / n`
  in double, rounded to f32; `scale = 1.0f / sqrtf(var + eps)` (`vsqrtss`, `vdivss`); `y = d · scale`. Then `y · w`,
  `+ b`: two roundings.

### The oracle
- `whisper_oracle --norm`: an eval callback on `sched_encode` observing **every** encoder node (127), one at a time.
  Each NORM's input is read when the scheduler asks about the NORM (every earlier node computed, the NORM not yet);
  the NORM, its MUL and that MUL's ADD when computed. Self-checks, yes on 8 / 8: MUL reads the NORM, ADD the MUL; the
  weights are the model's own tensors by pointer; eps = 1e-5f in `op_params`; inputs contiguous f32; the reference's
  nodes identical at 1 and 4 threads; `embd_enc` the same observed and not; `ln_post`'s ADD = `embd_enc`; the
  standalone graphs the bench times = the nodes.
- Blocks 1–3 and `ln_post` read attention's and the MLP's output, which voaice.rs does not compute yet: those norms
  are fed **the reference's recorded input to that node**. Block 0's `attn_ln` is also run end to end from voaice's
  own mel.
- `oracle_norm_nodes_bit_exact`: NORM, MUL and ADD of all nine chains, by the portable model and by the fast path at 1
  and 4 threads — **0 differ** in **373,248,000** values (8 inputs).
- `oracle_attn_ln_0_from_mel`: voaice's mel → conv stage → block 0's chain at 1, 2 and 4 threads: the NORM's input and
  the three nodes **0 differ** (41,472,000 values).
- `oracle_norm_discriminators`, each caught on **every** input (ADD values that differ, of 5,184,000 per input): the
  sum in f32 **3.76–3.87 M**; the mean from the double sum, unrounded **1.17–1.22 M**; a one-pass variance
  **0.72–1.01 M**; cvar's squares added one by one in double (no 8-lane f32 reduce) **121–196 k**; eps outside the
  sqrt **5.09–5.15 M**; the scale in double **0.94–1.00 M**; dividing by the root **1.00–1.02 M**; MUL and ADD fused
  into an FMA **1.48–1.64 M**.
- **Found:** "the lane sum on every row, without its proof" changes **no** value on the 8 inputs (the 155 rows that
  cannot prove their order free still round to the same mean). The inputs could not have caught that shortcut; a
  unit test builds a row where it does change the mean and checks that the fast path refuses the lanes there.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA). Production's Zen 3 library was not
  run; an AVX-512 build takes cvar's 16-lane branch (another pairing) — not compared; `base.en` (n = 512) not compared;
  blocks 1–3 and `ln_post` only on the reference's own inputs to them.

### Faster, bits unchanged
- The three nodes are **one pass per row**: no NORM or MUL tensor, the output owned by the caller
  (`LayerNorm::run_into`).
- The reference's double sum is a chain of 384 dependent adds per row. voaice adds in vector lanes **only when the row
  proves no partial sum can round in any order** — every value a multiple of 2^q (q from the smallest non-zero |x|)
  and n · max|x| < 2^(53 + q), checked in the same pass by integer max / min over the bit patterns; then every order
  gives the exact sum, which is what the in-order sum gave. 107,845 of the 108,000 rows on the 8 inputs proved it; the
  rest take the in-order loop. cvar's 8-block sums (each with the reference's own operand pairs) are formed four at a
  time and their double sum gets the same proof.
- Threads split rows, but `run_into` starts at most one per 2,048 rows: at 1,500 × 384 the op streams ~4.5 MB (more
  than the 4 MB L3) and is bandwidth-bound here; 2 and 4 threads never beat one (also at 6,000 and 24,000 rows), and a
  scoped spawn costs ~60 µs against ~0.4 ms. `run_into_split` keeps the exact split; the oracle checks it at 4 threads.

### Measured (gate step 10, only after 4f passed; load 2.1–2.4)
jfk's encoder input [1500, 384] (each side's own conv stage computes it beforehand), block 0's `attn_ln` weights; the
reference = the same ops as a standalone ggml graph (equal to the scheduler's nodes); wall = best of 10.

| | reference (all five runs) | voaice (this gate) | gate run | three reruns (first gate's code, load 3.2–3.7) |
|---|---|---|---|---|
| NORM node, 1 thread | 0.78–1.19 ms | 0.38 ms (0.37–0.77 in all runs) | 3.11× | 1.02× · 2.30× · 2.19× |
| NORM node, 2 threads | 0.57–1.03 ms | 0.42 ms (one thread) | 2.47× | 1.22× · 1.31× · 2.08× (two threads) |
| NORM node, 4 threads | 0.38–0.54 ms | 0.42 ms (one thread) | **0.91×** | 0.94× · 0.75× · 0.83× (four threads) |
| norm → · w → + b, 1 thread | 1.87–2.51 ms | 0.40 ms (0.40–0.84 in all runs) | 4.93× | 2.60× · 3.94× · 4.39× |
| norm → · w → + b, 2 threads | 1.90–2.27 ms | 0.44 ms | 4.68× | 3.94× · 4.27× · 4.82× |
| norm → · w → + b, 4 threads | 1.67–1.86 ms | 0.45 ms | 4.11× | 3.35× · 3.44× · 3.61× |

- **The NORM node alone at 4 threads is not faster**: the reference spreads it over its persistent OpenMP pool
  (0.38 ms wall, **1.89 CPU-ms**); voaice runs it on one thread (0.42 ms wall, **0.42 CPU-ms** — 4.4× less CPU). The
  reruns were of the first gate's code, which still spawned 2 and 4 threads (slower than one: see above); that is
  why the thread cap was added and the gate run again.
- **CPU:** the chain at 4 threads 0.48 against 6.31 CPU-ms (13×); at 1 thread 0.47 against 1.81.
- **Memory:** **2,251 KiB** (the output) against the chain graph's **6,750** (NORM, MUL and ADD outputs).
- The first gate run of this version (before the thread cap; `.oracle/0.0.8.first-gate.txt`, not kept in git):
  NORM 2.10× / 1.50× / 1.13× and the chain 5.83× / 4.37× / 3.49× at 1 / 2 / 4 threads.

### Added
- `src/norm.rs`: `LayerNorm` (`new`, `from_parts`, `encoder`, `run`, `run_into`, `run_into_split`, `run_model`,
  `row_model`), `Node`, `Variant` (the reference = default; eight discriminator flags and the lane sum), `row_stats`,
  `cvar_model`, `norm_row`, `sum_lanes`, `sum_is_order_free`, `MIN_ROWS_PER_THREAD`, `EPS`; unit tests (the bound;
  shuffled sums equal in-order ones whenever it holds; a row where the proof is needed; the fast path = the model on
  n = 1, 8, 13, 384, 512, 1031 with zeros, −0, wide exponent ranges, a zero row and an infinity, 1 and 3 threads).
- `voaice norm`, `voaice bench-norm norm|chain`; `tests/norm.rs` (3 oracle tests).
- `whisper_oracle --norm`, `--bench-norm`; gate steps 4f and 10, and step 3 prints the disassembly counts of
  `ggml_vec_cvar_f32` and `ggml_compute_forward_norm` (0 FMA in either). Every earlier check kept (13 oracle, 4 opus,
  3 streamair, 3 resample, 5 conv1, 4 conv2). The gate's 4b re-asked production's opus-tools as it does by default
  (read-only, 57 / 57 answers identical).

## 0.0.7 — 2026-10-08 — encoder conv2, `embd_conv` and the positional embedding: the encoder's input, bit-exact

**conv2 (stride-2 im2col to f16, 1,152-long f16 dots), its bias and GELU (`embd_conv`), and the positional embedding
the encoder graph adds first — bit for bit as the shipped ggml-cpu computes them inside whisper's own two schedulers,
at 1, 2 and 4 threads, from voaice's own mel; then 3.4–6× faster (the whole stage 3.4–4×) in one sixth of the memory.** Record:
`testing/results/0.0.7.txt`; how it was read and found: `testing/conv2/NOTES.md`.

### The graph, read from the pin
- `whisper_build_graph_conv` ends at conv2's GELU, named **`embd_conv`** ([1500, 384], ne0 = frames). The positional
  embedding is **not** in the conv graph: it is the encoder graph's first op, `ggml_add(view_2d(e_pe, 384, n_ctx,
  offset 384·4·n_ctx·iter), ggml_cont(ggml_transpose(view of embd_conv)))`, with `static int iter = 0` — offset 0,
  and the view covers e_pe whole at n_ctx = 1500. The result ([384, 1500], frame-major) is the first block's input.
- conv2 = `im2col(w2, ·, s0 = 2, p0 = 1)` → f16 [1152, 1500] (element `3·ic + kw` of row t = input frame
  `2t + kw − 1`), `mul_mat(im2col, w2)`: each output one `ggml_vec_dot_f16(1152)` (36 × 32: no tail).
- The mel input is always 2·n_ctx frames, zero-filled past the mel, so a short input takes no other path. Only a
  non-zero `audio_ctx` changes n_ctx; that was **not** compared.

### The oracle
- `whisper_oracle --conv2`: an eval callback on `sched_conv` (every node) and one on `sched_encode` that observes up
  to the first ADD, then lets the encoder run unobserved. The 18 nodes seen are the same on all 8 inputs (…, IM2COL
  f16 [1152,1500], MUL_MAT, ADD, GELU `embd_conv`; VIEW of e_pe, TRANSPOSE, CONT, ADD). Self-checks, yes on 8 / 8:
  the reference's nodes identical at 1 and 4 threads; the last conv node = `whisper_state::embd_conv` read after an
  unobserved run; `embd_enc` (the encoder's whole output) the same observed and not; the standalone graphs the bench
  times (conv2 alone, the whole stage) = the scheduler's nodes at 1 and 4 threads.
- `oracle_conv2_im2col_bit_exact`: **13,824,000 / 13,824,000** f16 values (from voaice's conv1, itself checked equal to
  the record's conv1 GELU node).
- `oracle_conv2_bit_exact`: MUL_MAT, + bias, GELU (`embd_conv`) — **0 differ** on 8 inputs at 1 and 4 threads
  (27,648,000 values).
- `oracle_positions_bit_exact`: CONT **0 differ**; the positional ADD **0 differ** both from conv2-with-positions
  (voaice's conv1 in) and from the whole fused stage (voaice's mel in), at 1, 2 and 4 threads (27,648,000 values).
- Discriminators on JFK, all caught: stride 1 (**423,936** / 576,000 MUL_MAT values), one f32 accumulator
  (**544,848**), positions added before the transpose (**575,247** / 576,000 encoder-input values), positions one
  frame late (**510,942**), GELU before the bias (**575,992**).
- **Found:** "im2col kept in f32" cannot be caught on JFK. conv1's GELU returns the f16 table's value for x < 10 (an
  f16 already), 0 for x ≤ −10 and x itself for x ≥ 10, so conv2's f16 rounding changes only GELU outputs ≥ 10 that
  f16 cannot hold: 3 in min_len, 2 in noise_loud and odd_len, 1 in silence, none in jfk, jfk_x3, chirp, short. The
  test counts them on every input and checks the discriminator where it can: **384, 767, 384, 383** of 576,000
  MUL_MAT values differ on those four; on the other four the two readings are the same function.
- **What this holds for:** this laptop's native libggml-cpu (Zen+, AVX2 + FMA + F16C, the AVX path). Production's
  Zen 3 library was not run; an AVX-512 host, `base.en`, and `audio_ctx` ≠ 0 were not compared.

### Measured (gate step 9, only after 4e passed; 2 cores / 4 threads, load 4.7–7.8 from the operator's browser)
jfk's first 30-s window; the reference = the same ops as a standalone ggml graph on the shipped CPU backend (the
record shows it equals the scheduler's nodes); wall = best of 10. The machine was loaded, so three reruns follow:

| | reference | voaice | gate run | three reruns |
|---|---|---|---|---|
| conv2 + bias + GELU, 1 thread | 230–292 ms | 45–63 ms | 4.15× | 6.31× · 5.09× · 4.20× |
| conv2 + bias + GELU, 2 threads | 141–224 ms | 39–48 ms | 3.30× | 4.30× · 5.14× · 3.62× |
| conv2 + bias + GELU, 4 threads | 132–148 ms | 36–63 ms | 2.62× | 2.31× · 4.09× · 3.48× |
| the stage (mel → encoder input), 1 thread | 383–423 ms | 97–119 ms | 3.60× | 3.37× · 4.02× · 4.02× |
| the stage, 2 threads | 211–234 ms | 56–83 ms | 3.13× | 2.81× · 3.23× · 3.76× |
| the stage, 4 threads | 200–281 ms | 61–78 ms | 4.62× | 3.22× · 3.28× · 2.55× |

- At a lighter load earlier the same day (≈ 4): conv2 37 ms against 191–213 at one thread (5–5.7×), 22 ms at two.
- **CPU:** a third to a fifth of the reference's (gate: conv2 89 against 266 CPU-ms at 1 thread; the stage 106 against
  528 at 1, 248 against 711 at 4).
- **Memory:** conv2 holds **2,443 KiB** (its output, 2,250, + one block) against the reference graph's **10,125**; the
  whole stage **4,693 KiB** against **29,532** — no im2col, no separate add / GELU / cont outputs, and conv1's output
  kept as f16.
- What changed, bits fixed: blocks of 32 frames per thread built from conv1's output (the input span rounded once per
  channel, then spread through a column table); a **4-frame × 3-channel AVX2 register block** (7 loads per 12 FMAs);
  each row's columns **permuted so accumulator j's blocks are contiguous** (both operands, so no product and no chain
  changes; 76 → 51 ms alone); the transpose and the positional add written by the epilogue; **conv1 → conv2 through
  f16** (exact by the finding above: conv2 reads nothing of conv1's output but its f16 conversion; 6,943 → 4,693 KiB).
- 4 threads gain little over 2 (2 physical cores, FMA-bound), as in 0.0.6.

### Added
- `src/conv.rs`: `Conv2` (`new`, `from_parts`, `frames_out`, `run`, `run_into`, `run_into_f16`), `Epilogue` (`Raw`,
  `BiasGelu`, `Positions`), `ConvStage` (`new`, `run`, `run_into`), `Conv1::run_into_f16`, `im2col_strided_f16`; unit
  tests (conv2 against im2col + the model dot for every epilogue, odd lengths, padded channel blocks, a non-multiple-of-32
  k, f16 and f32 input, 1 and 3 threads; the AVX2 block and the layout's model against the model; conv1's f16 output).
- `voaice conv`, `voaice bench-conv conv2|stage`; `tests/conv2.rs` (4 oracle tests).
- `whisper_oracle --conv2`, `--bench-conv2`; gate steps 4e and 9 (every earlier check kept: 13 oracle, 4 opus, 3
  streamair, 3 resample, 5 conv1). The production Opus re-ask (4b) was skipped on purpose (production not touched).

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
