# 0.1.2 — the decoder's input: notes (written as the work goes)

## The op, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8)
- whisper_build_graph_decoder (whisper.cpp:2458): inputs `embd` (I32 [n_tokens], "embd") and `position` (I32
  [n_tokens], "position"); `cur = ggml_add(ggml_get_rows(d_te, embd), ggml_get_rows(d_pe, position))` (whisper.cpp:2515)
  — the token rows are src0 of the ADD, the position rows src1. Before them the graph builds `KQ_mask` and its f16
  cast (0.1.3's), after them block 0's norm.
- d_te = decoder.token_embedding.weight, `wtype` = **f16** [384, 51864]; d_pe = decoder.positional_embedding, **f32**
  [384, 448] (whisper.cpp:1797-1799).
- GET_ROWS from f16 (ops.cpp:4831): rows split by thread (`dr = ceil(nr/nth)`), each row `ggml_cpu_fp16_to_fp32` —
  objdump of this laptop's libggml-cpu: get_rows calls `ggml_cpu_fp16_to_fp32@plt`, whose body is `vcvtph2ps` (8 and 4
  wide) and the `ggml_table_f32_f16` tail (384 % 8 = 0: no tail). Widening is exact; table and F16C agree on all 65,536
  (0.0.3) — so the widening has no rounding to get wrong.
- GET_ROWS from f32 (ops.cpp:4913): `ggml_vec_cpy_f32`, a copy.
- ADD f32 + f32: binary-ops' op_add, one IEEE single add per element, threads by rows.
- whisper_decode_internal (whisper.cpp:2848): `embd` set from batch.token, `position` from batch.pos (one
  tensor_set per token); logits copied out only for rows whose batch.logits != 0.
- whisper_batch_prep_legacy (whisper.cpp:511): n_tokens, pos[i] = n_past + i, n_seq_id 1, seq_id[i][0] = seq_id,
  logits 0 except the last row 1.
- The prompt (whisper_full_with_state, whisper.cpp:6975-7133): prompt_init = [SOT] (+ lang, task for multilingual
  only; + NOT when no_timestamps). tiny.en is English-only and the record ran with timestamps, so **prompt_init =
  [SOT] = [50257]**. Before it, when n_max_text_ctx > 0, t < 0.5 and there is past: [PREV] + the last
  min(max_prompt_ctx − 1, |past|) tokens of prompt_past1 (max_prompt_ctx = min(16384, 448/2) = 224). no_context=1
  clears the past at the start of whisper_full only; window 2.. of a long input carries window 1's tokens. A window
  starting within 500 frames of the end clears the past.
- The prompt goes in with whisper_batch_prep_legacy(prompt, n_past 0, seq 0); each next token as a one-row batch:
  token = the decoder's last sampled, pos = prompt.size() + i, seq_id = decoder j (0 for greedy at t=0), logits 1.

## The oracle (whisper_oracle --decin), 2026-10-08
- objdump (this laptop's libggml-cpu): ggml_compute_forward_get_rows has 1 call of ggml_cpu_fp16_to_fp32@plt and no
  vcvtph2ps of its own; ggml_cpu_fp16_to_fp32 has 2 vcvtph2ps (the 8- and 4-wide loops) and the table tail.
- Layout probe: whisper_state::sched_decode at 42472, whisper_state::batch at 384, sizeof(whisper_batch) 48 (mirrored
  and static_assert'ed). The callback asks only for the three nodes (GET_ROWS whose src0 is d_te / d_pe, the ADD of
  two such), so the rest of the decoder graph is computed in runs as usual; at the token GET_ROWS it reads the state's
  batch and self-checks it against the graph's own I32 inputs (`embd` = batch.token, `position` = batch.pos,
  n_tokens = ne): yes on every call.
- **What the recorded transcripts actually feed (config A, the 0.0.1 params):** every prompt is `[SOT]` = [50257], ONE
  row — tiny.en is English-only (no language / task token) and the record runs with timestamps (no NOT). jfk_x3's
  second window starts at 2,900 of 3,300 frames, within 500 of the end, so whisper clears its past and the prompt is
  `[SOT]` again. min_len: no decoder call (too short to decode); noise_loud and short decode but produce no result
  (no-speech). So the roadmap row's "[SOT, (lang), task, NOT/BEG]" is [SOT] for every recorded input: a prompt batch
  of one row cannot tell `logits only on the last row` from `on every row`, nor test positions past 0 for a prompt.
- **So a second config (B):** the same params plus `no_timestamps` and a 300-token `prompt_tokens` (jfk's text tokens
  cycled): each first window's prompt is `[PREV, the last 223, SOT, NOT]` = 226 rows, positions 0..225, logits on the
  last row only; the steps continue at 226 + i (the last observed position 307). jfk_x3's second window again clears
  its past: `[SOT, NOT]`.
- Calls at 1 thread: A 146 (8 prompts — jfk_x3 has two windows, min_len none — and 138 steps), B 164 (8 prompts, 156
  steps); the same at 4 threads: 620 calls, 3,772 rows observed (1,886 per thread count). ~5 min for 8 inputs.
- Self-checks yes on 8 / 8: every call's three nodes observed in the order te, pe, add; the ADD's src0 is the token
  rows; no other GET_ROWS in the graph; d_te f16, d_pe f32, outputs f32 [384, n]; the ADD runs in place (the allocator
  gives it the token rows' memory: no bit consequence); config A's result at 1 and at 4 threads equal to the same run
  unobserved, and the observed 1-thread result = 0.0.1's transcript record (128 tokens, ids and p bits).
- **Found: the thread count changes the reference's decoder, not its input.** Config A feeds the same tokens at 1 and
  4 threads on all 8 inputs and every input row is identical, but the result's `p` bits differ (results 1 = 4 threads:
  NO on 5 of 7 decoding inputs) — the logits move with the thread count. Config B on chirp feeds a different token at
  4 threads from call 2 on: the sampled token itself changed. The input stage is thread-independent; the logits
  (v0.2.0) will be bit-exact "at a stated thread count", as README already says of the transcript.

## voaice side, first oracle run (2026-10-08) — ALL EXACT FIRST RUN
- src/decoder.rs: Batch (prep_legacy, prep_step), Prompt (init, window), DecoderInput (model, run_into / run_batch).
- oracle_decin_nodes_bit_exact: 620 calls (both configs, 1 and 4 threads), 3,772 rows x 4 comparisons (token rows,
  position rows, the model's sum, the fast path's sum) = 15,088 row comparisons, 0 differ.
- oracle_decin_batches: 32 prompts and 588 steps rebuilt by Prompt::window + Batch::prep_legacy / prep_step = the
  recorded batches (tokens, positions, seq ids, n_seq_id, logits flags). The past-clearing rule needs the window's
  seek, which the decode loop computes (later): jfk_x3's second window is given cleared, stated in the test.
- Discriminators (1-thread calls, both configs, per input): positions off by one, step positions without the prompt,
  position row 0, multilingual ids (SOT + 1), DAZ widening, bf16 widening, the sum to f16 — each caught on every input
  that decodes. Indistinguishable, as predicted: the operands swapped (f32 + commutes), the sum in double (exact for
  two f32). **Found: the position rows rounded to f16 is indistinguishable too** — every one of d_pe's 172,032 f32
  values is exactly an f16 (the checkpoint was f16; the converter widened it), so the test asserts that instead.
  d_te holds 91,135 f16 subnormals in 42,525 of its 51,864 rows; SOT's row has 3, so DAZ is caught on every input.

## Efficiency (gate step 14, after 4j passed; laptop Ryzen 3 3200U, 2 cores / 4 threads; load ≈ 4.5 at the
## measurements, rerun ≈ 3.2 in .oracle/step14_rerun.txt)
- Like for like as far as it goes: both sides compute the same three nodes for the same tokens (token i = (i · 7919 +
  50257) mod n_vocab at position i); output digests equal (c2d2c9882a0d9ca5 for 1 token, 13a5d8fcd95d63b5 for 226).
  The reference = the two I32 inputs memcpy'd + ggml_graph_plan + ggml_graph_compute of a 3-node graph; voaice =
  Batch::prep_legacy + DecoderInput::run_batch (a little more work: it fills the batch's five vectors).
- Mean per call over >= 1 s (gate · rerun): 1 token, 1 thread: reference 1.05 · 1.00 us, voaice 0.108 · 0.107-0.114 us;
  226 tokens, 1 thread: reference 110.5 · 61.2-66.3 us, voaice 22.8 · 21.4-21.7 us; reference at 4 threads: 1 token
  20.1 · 4.8 us wall (51.5 · 18.7 CPU-us: starting threads for 3 nodes), 226 tokens 43.2 · 40.8-55.3 us wall.
- **What this does and does not say.** For one token the reference's microsecond is mostly the cost of planning and
  dispatching a graph; in whisper these nodes head the whole decoder graph and share its plan and thread start, so that
  cost is not theirs, and the 9-10x for one token is not a claim. The per-row work is what is comparable: at 226 rows,
  voaice ~0.095 us a row against 0.27-0.49 us (2.9-4.9x at one thread; the gate's 110 us reference figure was taken at
  load 4.5 and the rerun's 61-66 us is the steadier one: 2.9-3.1x). Either way the stage is a fraction of a microsecond a
  token against a decoder step that is milliseconds (not measured here: v0.2.0's): noise at the scale of a transcript.
- Memory: nothing allocated per call (0 bytes after the first); held outside the call: the f16 token table copied
  from the model (38,898 KiB) and the f32 positions (672 KiB) — the reference reads the model's own tensors. The copy is
  the cost of the existing loader style (Linear copies its weights too); the logits (0.1.7) read the same table.
