<h1 align="center">voaice.rs</h1>

<p align="center">
  <a href="https://github.com/Professor-Codephreak">Professor Codephreak</a><br>
  <a href="https://huggingface.co/Gregory-L">Gregory L. Magnusson</a><br>
  <a href="https://github.com/cryptoAGI">cryptoAGI</a>
</p>

**Speech to text in zero-dependency Rust: bit-exact against whisper.cpp first, then faster.**
Built the way [bankml](https://github.com/cryptoAGI/bankml) was built against llama.cpp.

<p>
  <img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/dependencies-0-56D364?style=flat-square" alt="zero dependencies">
  <img src="https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-2563EB?style=flat-square" alt="MIT OR Apache-2.0">
  <img src="https://img.shields.io/badge/encoder-bit--exact%20vs%20whisper.cpp%20(ggml%200.16.0)-39D3C7?style=flat-square" alt="encoder bit-exact">
  <img src="https://img.shields.io/badge/status-0.1.3%20%C2%B7%20encoder%2C%20cross--attention%20K%2FV%2C%20the%20decoder%27s%20input%2C%20self--attention%20products%20and%20KV%20cache%20bit--exact%20vs%20whisper.cpp%3B%20the%20decoder%20toward%20v0.2.0-0ECB81?style=flat-square" alt="status">
  <a href="https://github.com/cryptoAGI/voaicers/releases/latest"><img src="https://img.shields.io/github/v/release/cryptoAGI/voaicers?style=flat-square&label=release&color=0ECB81" alt="latest release"></a>
</p>

---

## What this is, and what it is not yet

mindX's production speech-to-text is whisper.cpp built against ggml 0.16.0, running `ggml-tiny.en.bin` and
`ggml-base.en.bin`. voaice.rs is a rewrite of it in Rust with **no crates at all**, held to one rule from bankml:
*a result counts only when an oracle has checked it, and the oracle is the reference's own compiled library run on
the same input, compared by bit pattern.* Speed is measured only after that.

**v0.1.0 is the first milestone: the whole encoder, bit-exact.** From voaice's own log-mel spectrogram to `embd_enc`
— conv1, conv2, the positions, four blocks of layer norm → Q, K, V → **flash attention** → out projection → MLP, then
`ln_post` — every value is the one the pinned whisper.cpp computes, checked against every node of its encoder graph
and against `embd_enc` itself on 8 inputs at 1, 2 and 4 threads, and the whole encoder runs **about four times as
fast as whisper.cpp's at one thread** (3.85–4.04×; three times at two and four threads) in less memory. It does not
transcribe yet: **the decoder is next (v0.2.0)**, then the transcript loop (v0.3.0).

**0.1.1 opens the decoder's decade with cross-attention K and V**: the third graph `whisper_encode_with_state` runs,
which turns `embd_enc` into every decoder layer's f16 `kv_cross` (K × 64^−0.25, V + bias, 36 +0 padding rows per
layer) — every node and the whole cache bit-exact, from the reference's `embd_enc` and from voaice's own mel. Reading
it showed that v0.1.0's "four times as fast" compared voaice's encoder against whisper's encoder **plus** this graph;
measured like for like — mel → `embd_enc` → `kv_cross`, the whole call — voaice is **3.70–3.74× whisper.cpp at one
thread**, and the cross stage alone 4.8×.

**0.1.2 is the decoder's input**: the token row of the f16 embedding widened, the position row added, and the batch
whisper builds around them — for a prompt and for every one-token step. It is bit-exact on **every decoder call
`whisper_full` makes** for the 8 inputs (620 calls, 3,772 rows, at 1 and 4 threads), and the prompts and batches
voaice builds are whisper's. Recording it showed that every prompt the recorded transcripts feed tiny.en is the single
token `[SOT]`, so the oracle added a run with a 300-token prompt to reach the many-row batch.

**0.1.3 is every decoder layer up to self-attention itself, and the self-attention KV cache**: attn_ln, Q + bias ×
64^−0.25, K × 64^−0.25 (no bias), V + bias, K and V copied to f16 into the 512-cell `kv_self` at the slot whisper's
cell search finds, and the causal mask with its f16 cast — every node, the cells, the mask, and the **whole cache after
every call** bit-exact on every decoder call `whisper_full` makes for the 8 inputs. Recording it located where the
reference's own thread dependence begins: inside layer 0, after these nodes, and only for one-row batches. A 226-row
prompt runs 4.8–6.1× faster than whisper.cpp at one thread; a one-token step 1.2–1.5× at one thread, and not faster
than the reference's step at two and four threads (voaice runs a step on one).

It got there one increment at a time: the front end optimized (0.0.2) with its bits unchanged, the first two encoder
kernels (0.0.3): the f32 ↔ f16 conversions and GELU, (0.0.4) the streaming Ogg/Opus reader that will bring `.opus`
input without a WAV on disk, (0.0.5) the audio reader whisper-cli itself runs: any WAV to 16 kHz mono f32 through
miniaudio's conversions, mixdown and resampler, bit for bit, (0.0.6) ggml's f16 dot product in its AVX float order and
conv1 with its bias and GELU, checked node by node against the reference's own scheduler, (0.0.7) conv2 (`embd_conv`)
and the positional embedding — the encoder's input, (0.0.8) the encoder's nine layer norms, (0.0.9) the matrix
products on activations: Q, K, V and their f16 copies, the out projection and the MLP, and (v0.1.0) flash attention —
ggml's **tiled** kernel, which is the one whisper's graph takes (Q in f32; the f16 `kv_pad` cache with its 36 zero
padding rows attended; the online softmax with glibc's `expf` for the rescale and ggml's own 8-lane `ggml_v_expf` for
the probabilities; the output in f32) — and the encoder chained end to end; then (0.1.1) the cross-attention K and V
every decoder step will read; then (0.1.2) the decoder's input; then (0.1.3) the self-attention products and the f16 KV cache.

| stage | what | oracle result (this machine, 2026-10-07) |
|---|---|---|
| 0 | the oracle harness: links the pinned `libwhisper.so` and records what it computes | built; self-checking (below) |
| 1 | the model loader + sha256 guard | **167 / 167 tensors** byte-identical to what whisper's loader holds (sha256 each, 77,110,272 bytes); hparams, filterbank (80×201, bit-exact), **51,864 / 51,864** vocab strings |
| 2 | the log-mel front end — allocation-free, threaded (0.0.2) | **2,316,640 / 2,316,640** f32 values bit-exact over 8 inputs, **max 0 ULP**; at 2, 3, 4 and 8 threads identical to 1 thread (9,266,560 values); the fused-FFT discriminator still rejected (25,242 values differ) |
| 3a | f32 ↔ f16 as ggml-cpu converts (0.0.3): the portable bit trick, the F16C row, f16 → f32 | **65,536 / 65,536** f16 patterns four ways; **all 4,294,967,296 f32 patterns** three ways (1,429,656 boundary values compared one by one, the rest by per-chunk digest); round-half-away rejected |
| 3b | GELU (0.0.3): `ggml_table_gelu_f16` and the op with its ±10 clamps | **65,536 / 65,536** table entries; the op on 1,495,192 values and on all 2³² f32 patterns; the unfused GELU rejected (it differs in one entry) |
| 4 | the streaming Ogg/Opus reader (0.0.4): pages, Ogg's CRC-32, lacing and continuation, `OpusHead` / `OpusTags`, granules and pre-skip → the exact duration, one page in memory | against **opus-tools 0.2, libopus 1.4 and libogg 1.3.5 on production**: **35 / 35** files on 18 checks each — the duration equals **opusdec's sample count** on every file, every page's granule and every packet's samples equal libogg's and libopus's; **21 / 21** corrupted files refused by name; pre-skip added, no end trim, and zlib's CRC (0 / 427 pages) all caught |
| 5 | any WAV → 16 kHz mono f32 as whisper-cli reads it (0.0.5): dr_wav's u8 / s16 / s24 / s32 / f32 conversions, miniaudio's mono average, its linear resampler with the order-4 low-pass, the length rule and its zero tail — streamed, any chunking | against **whisper-cli's own `libcommon.a`** (`read_audio_data`, miniaudio 0.11.24): **55 / 55** files, **1,955,875** samples bit-identical across 8–48 kHz, 1 / 2 / 6 channels, every format; random chunking identical; low-pass order 2 / 6, mixdown `L + R` / `L`, and the length without its extra frame all caught |
| 6 | `ggml_vec_dot_f16` and encoder conv1 + bias + GELU (0.0.6): im2col to f16, every output one f16 dot in the AVX build's order (4 × 8 lanes, pairwise reduce, a double tail) — then faster, no im2col held | the conv graph's **own nodes**, read through ggml's scheduler eval callback: im2col **5,760,000 / 5,760,000** f16 values; the product, + bias and GELU **0 differ** on all 8 inputs at 1 and 4 threads (55,296,000 values); the kernel **1,436 / 1,436** dots on real rows of all 70 f16 tensors; one accumulator, a tail in f32, a sequential reduce, im2col without its f16 rounding — all caught |
| 7 | encoder conv2 + bias + GELU (`embd_conv`) and the positional embedding (0.0.7): stride-2 im2col to f16, 1,152-long f16 dots, then `e_pe + cont(transpose(·))` — the encoder's input; then faster: no im2col held, a 4 × 3 register block over a permuted column layout, conv1 → conv2 through an f16 buffer, the transpose and the add fused into the epilogue | both schedulers' **own nodes** (the conv graph's, and the encoder graph's first four through its eval callback): im2col **13,824,000 / 13,824,000** f16 values; the product, + bias and GELU **0 differ** at 1 and 4 threads (27,648,000 values); CONT **0 differ**; the positional ADD **0 differ** from voaice's own mel at 1, 2 and 4 threads (27,648,000); stride 1, one accumulator, positions before the transpose, positions a frame late, GELU before the bias — all caught; im2col in f32 caught on the 4 inputs where it can be (below) |
| 8 | the encoder's layer norms (0.0.8): `ggml_norm` (the row summed in double in order, `mean` in f32, cvar's 8-lane f32 pairing, `1/sqrtf(var + 1e-5)`), then `· w` and `+ b` as two roundings — nine chains; then faster: the three nodes in one pass per row, the double sums in vector lanes only where the row proves every order exact | every NORM, MUL and ADD node of the encoder graph (all 127 nodes observed through its eval callback), each fed its recorded input: **0 differ** in 373,248,000 values on 8 inputs, at 1 and 4 threads and by the model; block 0 from voaice's own mel **0 differ** at 1, 2 and 4 threads; the sum in f32, the mean from the double, a one-pass variance, cvar without its f32 reduce, eps outside the sqrt, the scale in double, a division, mul + add fused — each caught on every input |
| 9 | the matrix products on activations (0.0.9): `mul_mat`'s f32 → f16 `from_float` (each of the reference's threads converting its element range of every row), then the f16 dot; Q, K (no bias), V and their f16 CPYs (the scalar bit trick), the out projection + bias + residual, fc1 + bias, GELU, fc2 + bias + residual; then faster: the norms fused into the conversion, one conversion for Q, K and V, a 4 × 3 register block over 64-frame panels, the MLP a panel at a time | all 16 product-side nodes of all 4 blocks through the encoder graph's eval callback, recorded as a digest per row: **0 differ** in 2,304,000 rows (1,382,400,000 values) on 8 inputs, by the model and at 1 and 4 threads; block 0 from voaice's own mel **0 differ** at 1, 2 and 4 threads; NaN-bearing rows at 1..8 threads **0 differ** — and the reference's own NaN output changes with its thread count (5 and 7 threads); no f16 rounding, one accumulator, a sequential reduce, the bias in the accumulator, the residual before the bias, GELU without its table — each caught on every input; the scalar converter and the split ignored caught on the NaN rows |
| 10 | **v0.1.0 — flash attention and the whole encoder**: ggml's tiled flash-attention kernel (f32 FMA-chain scores × 1/8 over the 1,536-row f16 `kv_pad` with its 36 +0 rows attended; tiles of 64 keys; glibc `expf` for the rescale, ggml's 8-lane `ggml_v_expf` for the probabilities, summed in f32 then double; the output in f32), then the encoder chained: mel → conv stage → 4 blocks → `ln_post` = `embd_enc`; then faster: K and V widened once per head, the softmax on the tile in registers, caller-owned buffers | every node of the encoder graph through its eval callback (97 digested per input) and `embd_enc` whole: from voaice's own mel **0 differ** at 1, 2 and 4 threads on 8 inputs (2,304 node comparisons, 3,456,000 rows; `embd_enc` 576,000 values each); attention **0 differ** in 36,864,000 values; the reference itself identical at 1..8 threads; glibc `expf` for the probabilities, `ggml_v_expf` for the rescale, no running max, the padding rows excluded, no FMA in the scores or the output, the sums in f32, a division, the one-chunk path — each caught on every input (the one-chunk model itself matches the reference's own `use_ref` output); Q scaled first is indistinguishable (×1/8 is exact), as predicted |
| 11 | **0.1.1 — cross-attention K and V** (`whisper_build_graph_cross` on `sched_cross`): per decoder layer K = `embd_enc` × key weight × `Kscale` (a SCALE node: one f32 multiply by `(float) pow(64, −0.25)` = `0x3EB504F3`), V = `embd_enc` × value weight + bias, both CPY'd to f16 (the scalar bit trick) into `kv_cross` at row `il · 1536`, rows 1,500..1,535 of every layer +0; then faster: `embd_enc` converted once for all eight products, the epilogue straight into the cache | every node of `sched_cross` (24) and `kv_cross` itself (12,288 rows, the padding included) through the eval callback and the layout probe: from the reference's `embd_enc` **0 differ** in 864,000 node rows and 294,912 cache rows (model, 1 and 4 threads); from voaice's own mel **0 differ** at 1, 2 and 4 threads; the scale before the product, in the weights, in double, after the f16 copy, V scaled, V unbiased, V's bias on K, no f16 rounding, no padding, the non-flash layout, padding not +0 — each caught on every input; the row converter for the CPYs indistinguishable on finite values, as predicted |
| 12 | **0.1.2 — the decoder's input** (the head of `whisper_build_graph_decoder`): `add(get_rows(d_te, embd), get_rows(d_pe, position))` — the f16 token row widened (`vcvtph2ps`, exact), the f32 position row, one f32 add — and the batch whisper builds: the prompt through `whisper_batch_prep_legacy` (positions `n_past + i`, logits on the last row only), each step one row at `prompt.size() + i`; then widened and added eight at a time into the caller's buffer | every decoder call of `whisper_full` on the 8 inputs through `sched_decode`'s eval callback, with the state's `whisper_batch` (layout probe), in two configs (the transcript record's, whose prompts are all `[SOT]`, and a 300-token prompt for the 226-row batch), at 1 and 4 threads: **0 differ** in 3,772 rows (620 calls), the batches **32 / 32** prompts and **588 / 588** steps; seven discriminators caught on every input that decodes; the operands swapped, the sum in double and (every d_pe value being an f16) the positions rounded to f16 indistinguishable |
| 13 | **0.1.3 — the self-attention products and the f16 self KV cache** (each decoder layer up to self-attention): attn_ln (0.0.8's nodes at the batch's rows), Q = (`mul_mat` + bias) × `KQscale`, K = `mul_mat` × `KQscale` (no bias), V = `mul_mat` + bias, `KQscale` = `0x3EB504F3`; K and V CPY'd to f16 into `kv_self` (512 cells per layer) at the slot `whisper_kv_cache_find_slot` finds, `n = cell_max` (no padding on the CPU); the KQ_mask (−∞ by sequence and position) and its f16 cast; then attn_ln in the conversion, a prompt in 0.0.9's panels, a step a matrix-vector product from the f16 weights, K and V straight into the cells | every decoder call of `whisper_full` on the 8 inputs through `sched_decode`'s eval callback, with `kv_self`'s cells and buffers (layout probe), both of 0.1.2's configs at 1 and 4 threads: every layer's twelve nodes, the cells, the mask and the whole cache after every call **0 differ** (362,112 node rows, 11,316 mask rows, 5,079,040 cache rows; 620 calls), layer 0 from voaice's own decoder input **0 differ** at 1, 2 and 4 threads; seventeen discriminators caught on every input that decodes (an eighteenth, the buffer left uncleared, on the one input with a second window); the row converter for the CPYs and the mask's cast and the sequence ignored indistinguishable, as predicted |

Efficiency, measured only after the oracles passed in the same gate run (0.0.2, this laptop, 4 CPUs at load ≈ 7.6,
so ±20 % is noise): the mel is **6.1× faster than 0.0.1** (rebuilt from its tag in the same run) and **6.8× faster
than the reference** at one thread (geometric means over the 8 inputs; JFK 11 s: **9.5 ms** against 60.1 ms;
JFK ×3: 24.9 ms against 198.8), at **one sixth of the reference's CPU time** (JFK 9.7 against 58.3 CPU-ms) and
**one third of its heap** (JFK 1,282 KiB — exactly the 80×4,100 output — against 3,868; the 30-s padded copy of the
audio is gone). At 4 threads it is 5.9× the reference at 4. Full record:
[`testing/results/0.0.2.txt`](testing/results/0.0.2.txt) (0.0.1's: [`0.0.1.txt`](testing/results/0.0.1.txt)).

0.0.3, in its own gate run after its oracles (same laptop, 1-minute load 7.4 at the start, falling from about 30,
so ±20 % is noise and more): the GELU op on 1,536 × 1,500 values (the encoder MLP's size) takes **6.0 ms against the
reference's 26.0** at one thread (**4.3×**; an earlier run of the same gate, at a higher load, measured 6.2 against 18.4,
2.96×). The gain is the table held widened to f32 (one lookup instead of two) and eight lanes with a gather. The
f32↔f16 rows run the same `vcvtps2ph`/`vcvtph2ps` as the reference, so their 1.24× and 1.07× are noise (0.90× and
1.01× in the earlier run); the GELU table builds in 2.5 ms against `ggml_cpu_init`'s 3.2, which also fills tables
voaice does not need. Record: [`testing/results/0.0.3.txt`](testing/results/0.0.3.txt).

0.0.4, in its own gate run after its oracles (same laptop, load ≈ 2 falling from 6.8): the reader takes **0.045 CPU-ms
for 33 s of 6 kb/s speech** (about 730,000× real time) and 0.03–0.10 ms for each 11 s JFK file, with a heap of
**66 KB whatever the file's length** — one 65,307-byte page buffer, reused; a packet that spans pages adds exactly its
length. Ogg's CRC sliced by eight runs at 1.4–1.6 GiB/s, **4.1–4.6× the byte-at-a-time table**. Record:
[`testing/results/0.0.4.txt`](testing/results/0.0.4.txt). It also found that streamair 0.0.1's writer accepts an end
trim past the last page, which opusinfo calls an error (TODO.md).

0.0.6, after its oracles (same laptop, 2 cores / 4 threads, load 1.7–3): conv1 on a 30-s window takes **21–24 ms
against the reference's 86–89** at one thread (about **4×**; 3–4× at two and four threads), with conv1 + bias + GELU
in **4,508 KiB of heap against the 14,907 KiB** the reference's graph holds — no im2col is kept, the add and GELU
are applied in place. The gate's own run of that step was noisy (2.6× at one thread); the CHANGELOG lists it with
three reruns. Record: [`testing/results/0.0.6.txt`](testing/results/0.0.6.txt).

0.0.7, after its oracles (same laptop, load 4.7–7.8 from interactive use, so the ratios swing): conv2 + bias + GELU
takes **45–63 ms against the reference's 230–292** at one thread (**4.2–6.3×**; 37 against 191–213 at a lighter load),
and the whole conv stage — mel to the encoder's input — **97–119 ms against 383–423** (3.4–4.0×), in **4,693 KiB of heap
against the 29,532 KiB** the reference's graph holds (no im2col; conv1's output kept as f16, which is exact because
conv2 reads only its f16 conversion). Record: [`testing/results/0.0.7.txt`](testing/results/0.0.7.txt); the CHANGELOG
lists the gate run with three reruns.

0.0.8, after its oracles (same laptop, load 2.1–2.4): the layer norm's three nodes (NORM, · w, + b) on the encoder's
input take **0.40 ms against the reference's 2.0** at one thread (**4.9×**; 4.1× against its four threads) in **2,251
KiB against 6,750**, and the NORM node alone 0.38 against 1.19 ms (3.1×) — but at four threads the reference's NORM
(0.38 ms over its thread pool) is level with voaice's single thread (0.42 ms, with a quarter of the CPU time). The op is
bandwidth-bound at this size here, so voaice keeps it on one thread. Record:
[`testing/results/0.0.8.txt`](testing/results/0.0.8.txt); the CHANGELOG lists three reruns.

0.0.9, after its oracles (same laptop, load 1.6–2.9): block 0's products — attn_ln → Q, K, V with the f16 copies, then
the out projection → residual → mlp_ln → fc1 → GELU → fc2 → residual, with V standing in for attention — take **147 ms
against the reference's 667** at one thread (**4.6×**; 4.4× at two threads, 4.1× at four), in **10,351 KiB of heap
against the 69,751 KiB** its graph holds; Q + bias alone 10.6 against 54.9 ms (5.2×: about 21 G multiply-adds a
second). At four threads voaice is no faster than at two on this 2-core laptop. voaice holds the weights widened to f32
(7.1 MB a block, outside the measured call). Record: [`testing/results/0.0.9.txt`](testing/results/0.0.9.txt); the
CHANGELOG lists a rerun.

**v0.1.0, the milestone, after its oracles** (same laptop, load 1.5 → 3.5): **the whole encoder, mel → `embd_enc`, takes
1,038 ms against whisper.cpp's 3,994 at one thread (3.85×; 987 against 3,983 in a rerun, 4.04×)**, 3.05–3.14× at two
threads and 2.95–3.01× at four, at **a quarter of the CPU time** (1,149 against 4,250 CPU-ms at one thread) and in
**17,869 KiB of heap against the 29,398 KiB** of compute buffers and cache whisper allocates for it; the `embd_enc`
both sides measured is the same to the bit (digest `50d38ec85f2778b9` on jfk). Block 0's attention alone: 91.9 against
150.8 ms (1.64×), near this core's FMA throughput. voaice holds the products' weights widened to f32 (28 MB for the
four blocks, outside the call). Record: [`testing/results/0.1.0.txt`](testing/results/0.1.0.txt); the CHANGELOG has the
rerun and the breakdown.

**0.1.1, after its oracles** (same laptop, load ≈ 3): the cross K/V of all four decoder layers, `embd_enc` →
`kv_cross`, take **95 ms against whisper.cpp's 454** at one thread (**4.76×**; 4.86× in a rerun), 4.6–4.8× at two
threads, in 9,794 KiB (the 9,216 KiB cache itself) against 11,466. And **the whole of `whisper_encode_with_state`,
like for like** — mel → `embd_enc` → `kv_cross`, which is what that call does — **1,085 ms against 4,011 (3.70×; 3.74×
in a rerun)**, 2.9–3.0× at two threads, in **26,312 KiB against 40,864**; both sides' `embd_enc` and `kv_cross` the same
to the bit (digests `50d38ec85f2778b9`, `76f24e73e318b877`). v0.1.0's 3.85–4.04× had timed the reference doing this
graph too while voaice's measured call did not: the reference's encoder alone is ≈ 3.4× voaice's (derived by
subtraction, not measured). Record: [`testing/results/0.1.1.txt`](testing/results/0.1.1.txt).

**0.1.2, after its oracles** (same laptop, load ≈ 4.5; rerun ≈ 3.2): the decoder's input for config B's 226-token
prompt takes **~22 µs against whisper.cpp's 61–66 µs** at one thread (2.9–3.1× in the rerun, 4.9× in the gate run),
the same tokens on both sides and the digests equal; for one token (a step) 0.11 µs against 1.0 µs, but that
microsecond is mostly ggml planning a graph, which in whisper is shared by the whole decoder — so it is not claimed.
Either way the stage is noise against a decoder step. Nothing is allocated per call; the f16 token table is held as a
38,898 KiB copy. Record: [`testing/results/0.1.2.txt`](testing/results/0.1.2.txt).

**0.1.3, after its oracles** (same laptop, load ≈ 1.0–2.8; rerun ≈ 2.2–2.6): the self-attention products and the K/V
cache writes, the same input on both sides and the digests equal, the reference timed as exactly these nodes in a ggml
graph. For config B's **226-row prompt**, one layer takes **5.0–5.8 ms against 25.7–27.7 ms** at one thread
(4.75–5.19×) and all four layers with the mask **20.9–21.7 ms against 103–127 ms** (4.76–6.06×); 2.6–3.9× at two and
four threads. For **one step**, one layer takes 76–81 µs against 117–120 µs (1.49–1.54×) and four layers 399–431 µs
against 503–567 µs (1.17–1.42×) at one thread — **but at two and four threads the reference's four-layer step is as
fast or faster (0.85–1.00×)**: voaice runs a step on one thread (a spawn costs more than the products), and the
products are bound by `vcvtph2ps` and by the bytes the step streams. Not claimed as faster there. Record:
[`testing/results/0.1.3.txt`](testing/results/0.1.3.txt).

**A determinism note on the reference itself:** whisper.cpp's transcript depends on its thread count. At 1 thread it
is identical run to run; at 4 threads the token ids and text stay the same but every token's probability differs in
its bits, and on JFK the token timestamps move. The transcript oracle is therefore pinned at 1 thread, and a
bit-exact transcript will mean "bit-exact at a stated thread count".

The encoder is done (v0.1.0) and is exact at any thread count: the reference's own encoder output does not depend on
it; so are the cross-attention K and V (0.1.1), the decoder's input (0.1.2) and every decoder layer's nodes up to
self-attention with the KV cache (0.1.3) — for a given input; the reference's decoder itself is not: past these nodes
of layer 0 its one-row steps depend on the thread count (found in 0.1.3). The rest of the decoder and the transcript loop are not written; their plan — the second decade, 0.1.1 → v0.2.0, one
increment at a time — is in [docs/ROADMAP.md](docs/ROADMAP.md) and [TODO.md](TODO.md).

## The reference

voaice.rs is measured against the whisper.cpp build that mindX production runs, pinned by commit in
[`upstream/PIN`](upstream/PIN) (ggml 0.16.0, `ggml-tiny.en.bin` pinned by sha256). Why that commit, and how it was
confirmed against production: [docs/REFERENCE.md](docs/REFERENCE.md). Credit for every external project:
[ATTRIBUTION.md](ATTRIBUTION.md).

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the modules, the oracle contract, **how to add a stage**, and every place in voaice that consumes speech-to-text today — the integration points voaice.rs is built to replace |
| [docs/oracles.md](docs/oracles.md) | every oracle, what it compares, its result, and how it sees what whisper.cpp's API hides |
| [docs/REFERENCE.md](docs/REFERENCE.md) | the pinned whisper.cpp, why a commit and not a tag, and its confirmation against production |
| [docs/ROADMAP.md](docs/ROADMAP.md) | the listening half to 1.0, and the speaking half |
| [ATTRIBUTION.md](ATTRIBUTION.md) | the external projects this stands on: references, models, standards, inspiration |
| [TODO.md](TODO.md) · [CHANGELOG.md](CHANGELOG.md) | the next stage in detail · what each version proved |

## How the oracle works

`testing/oracle/whisper_oracle` calls the shipped library in process. whisper.cpp's public API has no getter for
three of the things voaice must match — the mel it computes (`whisper_state::mel`), the tensors its loader holds
(`whisper_model::tensors`) and the filterbank — so a **layout probe** (`testing/oracle/layout_probe.cpp`) compiles
the pinned source with the same compiler and prints `offsetof()` for each; the oracle reads the shipped library's own
objects at those offsets, after checking each against a public getter (the tensor map's size against the loader's
count, the mel's `n_len_org` against `whisper_n_len_from_state`, `n_mel` against `whisper_model_n_mels`). It refuses
to run on any mismatch — the first draft of it did refuse, because `whisper_n_len_from_state` returns `n_len_org`,
not `n_len`.

It records, per run: every tensor's name, type, shape, byte count and the sha256 of the bytes in memory; every
token string; the filterbank; per WAV the f32 samples fed in, the mel at 1 and 4 threads (compared), the mel's time,
and the greedy transcript (token ids, `t0`/`t1`, the f32 bits of each token's `p`).

The oracle can fail: a variant of the front end with the FFT's multiply-adds fused (what an FMA-contracting build
computes) differs from the reference in 25,242 of 328,000 values on `jfk` (`oracle_mel_discriminates_fused_fft`).

## What the mel needed, exactly

Found by reading the source and the shipped binary (`objdump`), then confirmed by the oracle; the details are in
[`src/mel.rs`](src/mel.rs):
- `libwhisper.so` is compiled for **baseline x86-64**, not `-march=native` (only ggml-cpu gets that): **zero FMA
  instructions** in it, so every multiply and add rounds separately, in source order. Rust does not contract either.
- Its libm imports are `sincosf` (GCC fused the table's `sinf`/`cosf`), `cosf` (the Hann window) and `log10` (f64);
  voaice calls the same glibc symbols. The table argument is narrowed f64→f32 *before* `sinf`.
- The FFT is whisper.cpp's own: radix-2 recursion 400→200→100→50→25 and a naive DFT at 25 with twiddles
  `table[(k·n·step) % 400]`, f32 accumulators.
- The mel band sum is mixed precision: four f32 products summed in f32, then added to an **f64** accumulator.
  `log10` in f64, stored f32; the clamp and `(x+4)/4` in f64 against the f32 values.
- Padding: 200 samples reflected from `samples[200..=1]`, 30 s + 200 zeros after; frames beyond `(n+200)/160` are
  `log10(1e-10)` without an FFT. Fewer than 201 samples is undefined behaviour in whisper.cpp (it reads
  `samples[1..=200]`); voaice refuses it.
- Threads do not change the bits (each frame is computed whole by one thread): 1 vs 4 identical on all 8 inputs, in the
  reference and in voaice.rs (1 vs 2, 3, 4, 8).
- The input: whisper-cli's miniaudio converts s16 with the literal `0.00003051757812f`, which is exactly 2⁻¹⁵, so
  `s / 32768.0` is the same; the oracle checks the samples voaice reads equal those fed to whisper, bit for bit.

## Use

```sh
cargo build --release && cargo test --release          # unit tests (no model needed)
testing/release_gate.sh                                # the gate: reference, pins, record, compare, then speed
target/release/voaice info models/ggml-tiny.en.bin     # refuses any file that is not pinned, and says why
target/release/voaice mel  models/ggml-tiny.en.bin in.wav [out.f32] [--threads N]
target/release/voaice bench-mel models/ggml-tiny.en.bin in.wav [--threads N]   # heap, wall, CPU, RSS
target/release/voaice bench-f16 init|rows                                     # the GELU table's build; f16 rows, GELU
target/release/voaice opus info in.opus           # (0.0.4) pages, packets, headers, the exact duration; refuses by name
target/release/voaice bench-opus in.opus          # CRC sliced vs bytewise, read throughput, CPU per read, heap peak
testing/opus/oracle.sh check                      # ask opus-tools on production again about the pinned .opus files
target/release/voaice resample in.wav [out.f32]   # (0.0.5) any WAV -> 16 kHz mono f32, bit for bit as whisper-cli reads it
target/release/voaice bench-resample in.wav       # the whole read: wall, CPU per call, heap, RSS
target/release/voaice conv1 models/ggml-tiny.en.bin in.wav [out.f32] [--threads N]   # (0.0.6) conv1 + bias + GELU, 384 x 3000
target/release/voaice bench-conv1 models/ggml-tiny.en.bin in.wav conv1|gelu [--threads N]   # its wall, CPU, heap, RSS
target/release/voaice conv models/ggml-tiny.en.bin in.wav [out.f32] [--threads N]   # (0.0.7) mel -> conv1 -> conv2 -> + positions, 1500 x 384
target/release/voaice bench-conv models/ggml-tiny.en.bin in.wav conv2|stage [--threads N]   # conv2 (+ bias + GELU), or the whole stage
target/release/voaice norm models/ggml-tiny.en.bin in.wav [out.f32] [--threads N]   # (0.0.8) the encoder input through block 0's attn_ln
target/release/voaice bench-norm models/ggml-tiny.en.bin in.wav norm|chain [--threads N]   # the NORM node, or norm -> * w -> + b
target/release/voaice qkv models/ggml-tiny.en.bin in.wav [q.f32] [--threads N]   # (0.0.9) block 0's attention inputs: Q f32, K and V f16
target/release/voaice bench-mm models/ggml-tiny.en.bin in.wav q|fc1|fc2|qkv|mlp|block [--threads N]   # block 0's products
target/release/voaice encode models/ggml-tiny.en.bin in.wav [out.f32] [--threads N]   # (v0.1.0) the whole encoder: embd_enc 1500 x 384, sha256 + digest
target/release/voaice bench-attn models/ggml-tiny.en.bin in.wav [--threads N]     # block 0's flash attention: wall, CPU, heap, RSS
target/release/voaice bench-encode models/ggml-tiny.en.bin in.wav [--threads N]   # the whole encoder, mel -> embd_enc
target/release/voaice cross models/ggml-tiny.en.bin in.wav [out.f16] [--threads N]   # (0.1.1) kv_cross: 4 layers x 1536 rows x 384 f16, k then v
target/release/voaice bench-cross models/ggml-tiny.en.bin in.wav cross|whole [--threads N]   # the cross K/V, or mel -> embd_enc -> kv_cross
target/release/voaice decin models/ggml-tiny.en.bin 50257 [--pos N]   # (0.1.2) the decoder's input for tokens at positions N..: a digest per row
target/release/voaice bench-decin models/ggml-tiny.en.bin 226   # the decoder's input for n tokens: wall and CPU per call
target/release/voaice bench-selfkv models/ggml-tiny.en.bin 226 block|call [--threads N]   # (0.1.3) attn_ln -> Q, K, V -> the f16 self cache (+ the mask)
```

Input: `voaice mel` still takes 16-bit PCM, mono, 16 kHz WAV; `voaice resample` (0.0.5) takes any PCM 8/16/24/32-bit
or float 32-bit WAV at any rate and channel count up to 254 and gives the samples whisper-cli would (FLAC, MP3, Vorbis,
A-law, µ-law, ADPCM, f64 are refused by name). `.opus` files are read and measured (0.0.4) but not yet decoded.
Disk: the reference checkout and build are about 160 MB in `upstream/` (gitignored), the model 78 MB in `models/`.

## Layout

```
Cargo.toml  rust-toolchain.toml      zero dependencies; Rust 1.99.0 pinned like bankml
src/        sha256.rs model.rs wav.rs mel.rs f16.rs gelu.rs conv.rs norm.rs matmul.rs attention.rs encoder.rs cross.rs decoder.rs selfkv.rs ogg.rs resample.rs measure.rs lib.rs main.rs
tests/oracle.rs                      the oracle comparisons (#[ignore]: need the model and a recorded oracle)
testing/oracle/                      build.sh, layout_probe.cpp, whisper_oracle.cpp, resample_oracle.cpp (0.0.5)
tests/resample.rs                    (0.0.5) the resampler against whisper-cli's libcommon.a (#[ignore]: needs the record)
testing/make_resample_audio.py       (0.0.5) the 55-file resampler corpus, pinned in testing/pins/resample.sha256
testing/resample/NOTES.md            (0.0.5) what read_audio_data does, read from the pinned source line by line
tests/conv1.rs                       (0.0.6) conv1 and ggml_vec_dot_f16 against the conv graph's nodes (#[ignore])
testing/conv1/NOTES.md               (0.0.6) the order decision, the disassembly, the oracle's findings as they came
tests/conv2.rs                       (0.0.7) conv2, embd_conv and the positional add against both schedulers' nodes (#[ignore])
testing/conv2/NOTES.md               (0.0.7) the graph read from the pin, the oracle's findings, the speed steps
tests/norm.rs                        (0.0.8) the nine layer norms against every NORM, MUL and ADD node (#[ignore])
testing/norm/NOTES.md                (0.0.8) ggml_norm read from the pin and the binary, the oracle, the order-free sum
tests/matmul.rs                      (0.0.9) every block's products, biases, GELU, residuals and f16 copies (#[ignore])
testing/matmul/NOTES.md              (0.0.9) mul_mat's from_float split read from the pin, the compact oracle, the NaN finding
tests/attention.rs                   (v0.1.0) flash attention and the whole encoder against every node and embd_enc (#[ignore])
testing/attention/NOTES.md           (v0.1.0) which kernel whisper takes, kv_pad's padding, the use_ref finding, the speed steps
tests/cross.rs                       (0.1.1) the cross K/V against every node of sched_cross and kv_cross itself (#[ignore])
testing/cross/NOTES.md               (0.1.1) the cross graph read from the pin, the encode-call finding, the oracle, the speed
tests/decin.rs                       (0.1.2) the decoder's input and batches against every decoder call of whisper_full (#[ignore])
testing/decin/NOTES.md               (0.1.2) the decoder's head read from the pin, the one-row prompts, the two configs, the oracle
tests/selfkv.rs                      (0.1.3) every layer's products, the mask and kv_self against every decoder call (#[ignore])
testing/selfkv/NOTES.md              (0.1.3) the products, the cells and the mask read from the pin, the thread finding, the step's bytes
tests/opus.rs                        (0.0.4) the Ogg/Opus oracle comparisons, offline against the recorded reference
testing/make_audio.py                the 8 test WAVs, pinned in testing/pins/audio.sha256
testing/opus/                        oracle.sh record|check, reference.py, mutate.py; 35 pinned .opus files + answers
testing/release_gate.sh              the gate → testing/results/<version>.txt
upstream/PIN                         the reference (commit, ggml version, build, model sha256)
docs/                                ARCHITECTURE · oracles · REFERENCE · ROADMAP
```

Licence: MIT OR Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). External projects are credited in [ATTRIBUTION.md](ATTRIBUTION.md); none is redistributed here.

## vCLONE — the source code

vCLONE is how voaice handles cloning. It has two halves, and both are open:

| half | source |
|---|---|
| **capture and measurement** — record from the microphone, measure the recording into an 18-decimal voiceprint, write the result as a `.voaice` identity. This half captures a voice; it does not synthesise one ([why](https://github.com/cryptoAGI/voaice#vclone-captures-it-does-not-clone)) | [`web/capture.html`](https://github.com/cryptoAGI/voaice/blob/main/web/capture.html) (browser microphone capture) · [`tools/voaice.py`](https://github.com/cryptoAGI/voaice/blob/main/tools/voaice.py) · [`tools/vprint.py`](https://github.com/cryptoAGI/voaice/blob/main/tools/vprint.py) (the voiceprint) · [`voices/vclone.voaice`](https://github.com/cryptoAGI/voaice/blob/main/voices/vclone.voaice) (the template you measure into) · [`FORMAT.md`](https://github.com/cryptoAGI/voaice/blob/main/FORMAT.md) |
| **synthesis** — speak in a measured voice: stage 1 is Kokoro-82M, stage 2 is OpenVoice v2 tone-colour transfer, run on the CPU through ONNX with no torch. Without a runtime or weights it falls back to a persona-tinted render and says which path it took | [`src/NeuralVoiceEngine.js`](https://github.com/Professor-Codephreak/voaice/blob/main/src/NeuralVoiceEngine.js) · [`src/VoiceCreationEngine.js`](https://github.com/Professor-Codephreak/voaice/blob/main/src/VoiceCreationEngine.js) |

In voaice.rs, vCLONE starts with the measuring half, since a voiceprint is just a number to check against. The work
is listed below.

## TODO — the Rust crates

The full engineering plan for each stage is in [TODO.md](TODO.md) and [docs/ROADMAP.md](docs/ROADMAP.md). Versions
go up 0.0.1 at a time, with a milestone at every tenth step. The order of work:

### voaice.rs (speech to text)
- [x] **0.0.1:** the model loader with its sha256 guard, and the log-mel front end. Both are bit-exact (0 ULP)
  against whisper.cpp at ggml 0.16.0.
- [x] **0.0.2:** the mel optimised: no allocations, optional threads, CPU and memory measured, bits unchanged
  (6.1× faster than 0.0.1, 6.8× the reference at one thread, a third of its heap; [record](testing/results/0.0.2.txt)).
- [x] **0.0.3:** f32 ↔ f16 and GELU, the first encoder kernels: every f16 and every f32 bit pattern converted as
  ggml-cpu converts it (the portable bit trick and the F16C row), the GELU table 65,536 / 65,536 and the op on all
  2³² inputs; the GELU op 4.3× the reference at one thread (3.0× in an earlier run: noisy host), the rows at parity (the same `vcvtps2ph`) ([record](testing/results/0.0.3.txt)).
- [x] **0.0.4:** the streaming Ogg/Opus reader: every page checked (CRC sliced by 8, 4.1–4.6× the byte table),
  packets across pages, `OpusHead` / `OpusTags`, granules and pre-skip → the exact duration in one page of memory;
  35 / 35 files exact against production's opusdec / opusinfo / libogg / libopus, 21 / 21 corruptions refused by name
  ([record](testing/results/0.0.4.txt)).
- [x] **0.0.5:** the audio reader whisper-cli runs: dr_wav's conversions, miniaudio's mono average, its linear
  resampler with the order-4 low-pass and its length rule; 55 / 55 files, 1,955,875 samples bit-identical to
  whisper-cli's `libcommon.a`, streamed in any chunking ([record](testing/results/0.0.5.txt)).
- [x] **0.0.6:** `ggml_vec_dot_f16` (brought forward: conv1 is nothing but that dot) and encoder conv1 + bias + GELU,
  bit-exact against the conv graph's own nodes through the scheduler's eval callback, at 1 and 4 threads; then
  faster with no im2col held ([record](testing/results/0.0.6.txt)).
- [x] **0.0.7:** encoder conv2 + bias + GELU (`embd_conv`) and the positional embedding — the encoder's input,
  bit-exact against both schedulers' nodes at 1, 2 and 4 threads; then faster, with an f16 buffer between the two
  convolutions ([record](testing/results/0.0.7.txt)).
- [x] **0.0.8:** the encoder's nine layer norms (`norm → · w → + b`), bit-exact against every NORM, MUL and ADD node of
  the encoder graph; then faster in one pass per row ([record](testing/results/0.0.8.txt)).
- [x] **0.0.9:** the matrix products on activations (`from_float` split by thread, then the 0.0.6 dot), with every
  bias, GELU, residual and f16 copy of all four blocks, bit-exact against the encoder graph's nodes; then 4–5× faster
  ([record](testing/results/0.0.9.txt)).
- [ ] **Stage 0b:** pin `ggml-base.en.bin` (production's default model). Record the VPS's ISA so the oracle
  reproduces production's native ggml-cpu kernels (Zen 3, AVX2 + FMA).
- [x] **v0.1.0 — MILESTONE, the whole encoder:** flash attention (ggml's tiled kernel, the one whisper takes; kv_pad's
  zero rows attended), four blocks and `ln_post`: every node of the encoder graph and `embd_enc` bit-exact from
  voaice's own mel at 1, 2 and 4 threads; the encoder 3.85–4.04× whisper.cpp at one thread
  ([record](testing/results/0.1.0.txt)).
- [x] **0.1.1:** cross-attention K and V — every node of `sched_cross` and `kv_cross` itself (padding rows included)
  bit-exact from voaice's own mel; the cross stage 4.8× the reference, the whole of `whisper_encode_with_state` like for
  like 3.70–3.74× ([record](testing/results/0.1.1.txt)).
- [x] **0.1.2:** the decoder's input — `get_rows(d_te) + get_rows(d_pe)` and whisper's batches, bit-exact on every
  decoder call of `whisper_full` for the 8 inputs at 1 and 4 threads ([record](testing/results/0.1.2.txt)).
- [x] **0.1.3:** the self-attention products and the f16 KV cache — every layer's nodes up to self-attention, the
  cells, the mask and the whole cache after every call, bit-exact on every decoder call of `whisper_full` for the 8
  inputs at 1 and 4 threads ([record](testing/results/0.1.3.txt)).
- [ ] **Stage 4, the decoder (0.1.4 → v0.2.0):** self- and cross-attention (0.1.4, 0.1.5), the MLP (0.1.6),
  the logits (0.1.7), the incremental step (0.1.8), fast (0.1.9) — all 51,864 logits bit-exact per step through
  `whisper_get_logits_from_state` (v0.2.0; [docs/ROADMAP.md](docs/ROADMAP.md)).
- [ ] **Stage 5, the transcript (0.3.0):** the greedy loop, logit filters, timestamp rules, 30-second seek.
  `transcript.tsv` must be identical: token ids, `t0`/`t1`, and the f32 bits of `p`.
- [ ] **0.4.0:** `voaice transcribe --json` in whisper-cli's shape, plus a library entry point for voaice's call
  sites.
- [ ] **Stage 6, fast (0.5.0):** SIMD kernels, threads, a KV layout without copies. The oracle stays green
  throughout, and timing is measured against whisper.cpp on the same cores.
- [ ] **Portability:** port glibc 2.35's `sincosf`, `cosf` and `log10` in-crate, with an oracle covering every
  argument the mel uses.

### streamair (CPU → .opus)
- [x] **0.0.1:** the Ogg/Opus container. Production's opusinfo reads every test file without a warning, and
  opusdec decodes exactly the samples written.
- [ ] **0.0.2–0.0.9:** a streaming writer with bounded memory (and end trimming bounded by the last page, found by
  voaice.rs 0.0.4's reader, whose round trips now read streamair's output), the range encoder, the
  MDCT, band energies, PVQ, bit allocation.
- [ ] **0.1.0:** a mono CELT encoder (fullband, 20 ms, CBR), byte-exact against libopus 1.4 at a stated complexity.
- [ ] **0.2.0–0.3.0:** VBR at voaice's bitrates, stereo, SILK and hybrid.
- [ ] **0.4.0–0.5.0:** the speed pass, then the efficiency pass: fewer CPU-seconds per audio-second, or fewer
  bytes for the same quality, stated per change.
- [ ] **1.0.0:** voaice writes every `.opus` through streamair, with libopus needed only as the oracle.

### vCLONE in Rust — [docs/VCLONE.md](docs/VCLONE.md)
- [x] **vprint (`dvscope/1`)**, byte-identical to `tools/vprint.py`: 2,000 recorded metric sets and all 10 measured
  `.voaice` identities verify field for field (`voaice vclone check`).
- [x] **The forge log (`vclone-event/1`)**: capture, measure, ref, consent, model, actor, prompt, skill, tool,
  language, forge. Each event is hash-chained, and an edit or a dropped event is refused. `mintable()` gives the
  reasons when a voice may not be minted.
- [ ] Port the forensic print as well (what `/voicey/measure` returns today), measure in Rust, port `compare()`.
- [ ] Have ollywoo's `forgePersona()` write events as it goes, add a consent step, and record the cloning
      engine's licence. Then emit a card from a log.
- [ ] Synthesis (Kokoro + OpenVoice v2) later, after the speaking half of the [roadmap](docs/ROADMAP.md). It will
      be checked against the reference's own ONNX runtime, sample by sample.

## The voaice family — code and live links

voaice.rs is one part of a larger body of voice work. Each part does one job, and each links to the others, so a
stage written here can be swapped in where the older one runs today.

### Code

| repository | what it is |
|---|---|
| **[cryptoAGI/voaicers](https://github.com/cryptoAGI/voaicers)** — voaice.rs, this repository | speech to text in zero-dependency Rust, bit-exact against whisper.cpp |
| [voaicers/streamair](streamair/) | CPU → `.opus` in zero-dependency Rust: the Ogg/Opus container (0.0.1, proven against libopus 1.4), then the encoder |
| [cryptoAGI/voaice](https://github.com/cryptoAGI/voaice) | what a voice is, written down: `.voaice` identities, the 18-decimal vprint, the pronunciation table every engine speaks through |
| [Professor-Codephreak/voaice](https://github.com/Professor-Codephreak/voaice) | the voice engine: in-house DSP, scientific and forensic voiceprints, the non-destructive editor, WAV/OGG export, torch-free neural TTS and cloning |
| [Professor-Codephreak/playdocs](https://github.com/Professor-Codephreak/playdocs) | an instrument with a document inside it: point it at a URL, hear it read in the DeltaVerse cast, zoom the waveform to the sample |
| [Professor-Codephreak/docsreader](https://github.com/Professor-Codephreak/docsreader) | the mindX and DeltaVerse document readers: speak a page aloud and light the words as they are read |
| [Professor-Codephreak/faicey](https://github.com/Professor-Codephreak/faicey) | the face of AI, voaice's peer: what speaks, seen |
| [Professor-Codephreak/aivatar](https://github.com/Professor-Codephreak/aivatar) | the `.persona` tool that joins them: looks (faicey), speaks (voaice), rigs and thinks |
| [cryptoAGI/bankml](https://github.com/cryptoAGI/bankml) | the method this follows: bit-exact against llama.cpp's compiled library, then faster ([thesis](https://github.com/cryptoAGI/bankml/blob/main/docs/thesis.md)) |

### Live

| | |
|---|---|
| [**the mindX thesis**](https://mindx.pythai.net/doc/THESIS) · [listen](https://mindx.pythai.net/listen/THESIS) | the argument mindX is built on, read aloud by the voices this family makes |
| [**rage.pythai.net**](https://rage.pythai.net/) — the WordPress player | every article is playable in the pre-rendered cast. For example: [the bankML thesis](https://rage.pythai.net/bankML-thesis/) (neural voice) and [OVERLORD of the DeltaVerse](https://rage.pythai.net/overlord-of-the-deltaverse/) (the OVERLORD voice) |
| [playdocs](https://deltaverse.pythai.net/playdocs) | the playdocs instrument, live |
| [docsplayer](https://deltaverse.pythai.net/docsplayer) | the document player: playlist, oscilloscope, spectrum, the cast |
| [docsreader](https://deltaverse.pythai.net/docsreader) | the reader that lights each word as it is spoken |
| [listen](https://deltaverse.pythai.net/listen) · [voices](https://deltaverse.pythai.net/voices) | the DeltaVerse listening room and the cast |
| [ollywoo](https://deltaverse.pythai.net/ollywoo) | the stage where the cast performs: wardrobe, scenes, lip-sync |
| [PYTHAI/voaice on Hugging Face](https://huggingface.co/PYTHAI/voaice) | the voice library: 70 open-licensed Piper voices, for anyone to use, each with Piper's attribution and its own licence |

---

<p align="center">
  <a href="https://github.com/Professor-Codephreak">Professor Codephreak</a><br>
  <a href="https://huggingface.co/Gregory-L">Gregory L. Magnusson</a><br>
  <a href="https://github.com/cryptoAGI">cryptoAGI</a>
</p>
