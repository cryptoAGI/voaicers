# 0.0.7 — encoder conv2 + positions: notes (written as the work goes)

## The order, read from the pin (2026-10-08, src/whisper.cpp at 080bbbe8)
- whisper_build_graph_conv: conv1 = conv_1d_ph(w1, mel, 1, 1) + b1, GELU; conv2 = `ggml_conv_1d_ph(e_conv_2_w, cur, 2, 1)`
  + e_conv_2_b, GELU; **that GELU is `embd_conv`** ([1500, 384], ne0 = frames) — the graph's output. Nothing else.
- The positional embedding is NOT in the conv graph: it is the first op of whisper_build_graph_encoder (sched_encode):
  `cur = ggml_view_tensor(embd_conv)`; `e_pe = ggml_view_2d(model.e_pe, ne0 = 384, n_ctx, stride 384*4,
  offset 384*4*n_ctx*iter)` with `static int iter = 0` (never changed: offset 0, the view covers the whole e_pe
  [384, 1500] when n_ctx = n_audio_ctx = 1500); `cur = ggml_add(e_pe, ggml_cont(ggml_transpose(cur)))` -> [384, 1500]
  (ne0 = channels: frame-major). This is the encoder's inpL (the first layer's norm reads it).
- n_ctx = exp_n_audio_ctx > 0 ? exp_n_audio_ctx : n_audio_ctx — only `audio_ctx` (a whisper_full param) changes it;
  the default 0 gives 1500. The mel input is always 2*n_ctx frames: whisper_encode_internal copies
  mel[min(off, n_len) .. min(off + 2n_ctx, n_len)) and zero-fills the rest (conv1's window, 0.0.6). So a short mel
  takes no different path: the zeros go through conv1/conv2 like any frame. audio_ctx != 0 is NOT checked here.
- conv2 = im2col(w2, cur, s0 = 2, p0 = 1, d0 = 1) -> f16 [1152, 1500] (OW = (3000 + 2 - 2 - 1)/2 + 1 = 1500): row t,
  element 3*ic + kw = GGML_CPU_FP32_TO_FP16(gelu1[ic][2t + kw - 1]), 0 outside [0, 3000); then
  mul_mat(im2col, w2 as [1152, 384]) -> f32 [1500, 384]: out[c][t] = vec_dot_f16(1152, im2col row t, w2 row c);
  1152 = 36 * 32: no tail. + b2 (bias [1, 384] broadcast over frames), GELU (0.0.3's table op).
- tensors: encoder.positional_embedding f32 [384, 1500]; encoder.conv2.weight f16 [3, 384, 384]; conv2.bias f32 [1, 384].

## The oracle (whisper_oracle --conv2), first run 2026-10-08 (1 m 43 s for 8 inputs, load ~2)
- Two eval callbacks: conv2_cb on sched_conv (every node), enc_cb on sched_encode (observes up to the first ADD, then
  answers `ask` with false so the rest of the encoder runs as one unobserved compute).
- The 18 nodes observed (identical on all 8 inputs): conv 0-6 conv1 (0.0.6), 7 IM2COL f16 [1152,1500], 8-9 RESHAPE,
  10 MUL_MAT f32 [1500,384], 11 RESHAPE, 12 ADD, 13 GELU named embd_conv; encode 14 VIEW of a leaf f32 [384,1500]
  (e_pe), 15 TRANSPOSE (view of embd_conv), 16 CONT f32 [384,1500], 17 ADD f32 [384,1500]. Nothing else between.
- Self-checks yes on all 8: 1 vs 4 threads identical (all 7 nodes incl. conv1's GELU); the last conv node ==
  whisper_state::embd_conv read after an unobserved run (layout probe offset 42504); embd_enc (the whole encoder's
  output, offset 42512) observed == unobserved (so watching the first encoder nodes perturbs nothing downstream);
  the standalone graphs (conv2 from conv1's GELU; the whole stage from the mel) == the scheduler's nodes at 1 and 4.

## voaice side, first oracle run (2026-10-08) — ALL EXACT FIRST RUN
- src/conv.rs: im2col_strided_f16 (the node), Conv2 (weights widened once; blocks of FB = 32 frames built per
  thread, rounded 8 at a time; 4 frames x 3 channels AVX2+FMA register block, 12 chains per accumulator j; epilogues
  Raw / BiasGelu / Positions — the last writes e_pe + gelu(dot + b) frame-major, i.e. the transpose and the add fused),
  ConvStage (Conv1 into a caller-kept scratch, then Conv2 Positions). tests/conv2.rs:
  - im2col node: 8 inputs, 13,824,000 f16 identical (from voaice's own conv1, itself checked = the record's GELU node).
  - MUL_MAT, ADD, GELU (embd_conv): 0 differ, 8 inputs x 1 and 4 threads (27,648,000 values).
  - CONT = embd_conv transposed: 0 differ; pe ADD: conv2+pe from voaice's conv1, and the whole stage from voaice's
    mel, 0 differ at 1, 2, 4 threads on all 8 (27,648,000 values).
- Discriminators (jfk unless named): stride 1 423,936/576,000 differ; one f32 accumulator 544,848; pe added before the
  transpose 575,247; pe one frame late 510,942; GELU before the bias 575,992.
- FINDING: "im2col kept in f32" is NOT discriminable on jfk, jfk_x3, chirp, short: conv1's GELU output is the f16
  table's value (exactly an f16) for every x < 10 and x itself for x >= 10, so conv2's f16 rounding changes only
  GELU outputs >= 10 that f16 cannot hold. min_len 3 such inputs -> 384 MUL_MAT values differ; noise_loud 2 -> 767;
  odd_len 2 -> 384; silence 1 -> 383. The test checks every input where the count is not 0 and says which are not.

## Efficiency (jfk; conv2 always reads 3000 conv1 frames). Laptop Ryzen 3 3200U, load 3-5 (operator's browser)
- first draft (4x3 register block, accumulator j striding 32 floats over the row, tile build per element):
  conv2 1t 76 ms vs ref 213 (2.8x); 2t 61 vs 127; 4t 49 vs 128. stage 1t 120 vs 311.
- the kernel's layout: each row's columns permuted so the blocks accumulator j reads (i + 8j) are contiguous
  (col(kk) = (kk%32/8)*(k/4) + (kk/32)*8 + kk%8, on both operands: no product, no chain changes) -> 51 ms 1t.
- the tile build per input channel: the span of input frames a block reads copied and rounded once, 8 at a time,
  then spread into the rows through a column table -> 37 ms 1t (best; FMA bound ~28-32 ms at 83M ymm FMAs).
- conv1 -> conv2 through an f16 scratch: conv2's im2col reads nothing of conv1's output but its f16 conversion, so
  conv1's epilogue stores the f16 bits (F16C, the bit trick on a NaN block) and conv2 widens them exactly: the stage's
  heap 6,943 -> 4,693 KiB (scratch 2,250 KiB + output 2,250 + blocks). Unit-tested (f16 path = f32 path) and on the oracle.
- noisy machine: conv2 1t 37-54 ms across runs while the reference ran 191-213; ratio ~4-5.7x at 1 thread.
- 4 threads = 2 threads (2 physical cores, FMA-bound), as conv1.

## The gate run (testing/results/0.0.7.txt, 2026-10-08T10:23Z, load 4.7 rising to 7.4; browser; no bankml gate)
- GATE PASSED: every earlier check kept (13 oracle, 4 opus, 3 streamair, 3 resample, 5 conv1) + 4 conv2. Opus re-ask
  on production skipped on purpose (VOAICE_OPUS_HOST=unreachable.invalid).
- Step 9: conv2 1t 230.2 vs 55.5 ms (4.15x), 2t 3.30x, 4t 2.62x; stage 1t 383.3 vs 106.4 (3.60x), 2t 3.13x, 4t 4.62x.
  Heap: conv2 2,443 KiB vs 10,125; stage 4,693 vs 29,532.
- Three reruns of step 9 right after (load 6.9-7.8), .oracle/step9_reruns.txt: conv2 1t 6.31/5.09/4.20x, 2t
  4.30/5.14/3.62x, 4t 2.31/4.09/3.48x; stage 1t 3.37/4.02/4.02x, 2t 2.81/3.23/3.76x, 4t 3.22/3.28/2.55x.
