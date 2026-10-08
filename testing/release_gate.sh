#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The release gate: nothing is a result until the oracle has checked it in the same run.
#   testing/release_gate.sh            -> testing/results/<version>.txt
# Steps, each must pass:
#   1. the reference: upstream/whisper.cpp at upstream/PIN's commit, built; the oracle harness built against it
#   2. the inputs: the model's sha256 equals its pin; the test audio equals testing/pins/audio.sha256
#   3. the record: the shipped library's model view, mel and transcripts (testing/oracle/bin/whisper_oracle); and
#      (0.0.3) libggml-base / libggml-cpu's f32<->f16 conversions and GELU, on every f16 and every f32 pattern
#   4. voaice.rs: cargo build, unit tests, clippy, then the oracle comparisons (bit patterns, never tolerances)
#   5. efficiency, only after 4 passed: the mel's wall time (best of 10), CPU time per call, heap and RSS peaks, voaice
#      against the reference (each in a fresh process: `voaice bench-mel`, `whisper_oracle --bench-mel`), at 1 thread
#      and at N = nproc threads, and against 0.0.1's code path rebuilt from its tag in the same run; then (0.0.3) the
#      GELU table's build, the f32<->f16 rows and the GELU op against the reference's (`voaice bench-f16`,
#      `whisper_oracle --bench-f16`)
#   4b. (0.0.4) the Ogg/Opus reader against opus-tools 0.2 / libopus 1.4 / libogg 1.3.5: the pinned files
#      (testing/opus/files.sha256) and the reference's recorded answers (testing/opus/reference.jsonl) compared offline
#      (tests/opus.rs), streamair's writer read back (streamair/tests/roundtrip.rs); then the reference asked again on
#      mindX production (testing/opus/oracle.sh check) when the host answers — skipped, and said so, when it does not
#   6. (0.0.4) the reader's efficiency, only after 4b passed: the CRC sliced against one byte at a time, the whole
#      read from memory and from the file, CPU per read, heap peak (`voaice bench-opus`)
#   4c. (0.0.5) the audio reader + mixdown + resampler against whisper-cli's read_audio_data (the build's libcommon.a,
#      testing/oracle/bin/resample_oracle) on the pinned corpus (testing/make_resample_audio.py), bit for bit; chunked
#      streaming equal to the whole; discriminators (another low-pass order, another mixdown, another length rule)
#   7. (0.0.5) its efficiency, only after 4c passed: each side reads the same file whole, in a fresh process (`voaice
#      bench-resample`, `resample_oracle --bench`)
#   4d. (0.0.6) encoder conv1 and ggml_vec_dot_f16: the conv graph's IM2COL, MUL_MAT, ADD and GELU nodes as the shipped
#      scheduler computed them (read through its eval callback, `whisper_oracle --conv1`), and the kernel on real rows of
#      every f16 tensor, compared bit for bit (tests/conv1.rs); discriminators (other dot orders, im2col without its f16)
#   8. (0.0.6) its efficiency, only after 4d passed: conv1, and conv1 + bias + GELU, each side in a fresh process
#      (`voaice bench-conv1`, `whisper_oracle --bench-conv1`: the same ops as a ggml graph on the shipped CPU backend,
#      which the record shows equal to the scheduler's nodes), at 1, 2 and nproc threads
#   4e. (0.0.7) encoder conv2 and the positional embedding: the conv graph's second IM2COL, MUL_MAT, ADD and GELU
#      (embd_conv) and the encoder graph's CONT(TRANSPOSE) and ADD(e_pe, ·), read through both schedulers' eval callbacks
#      (`whisper_oracle --conv2`), compared bit for bit from voaice's own mel (tests/conv2.rs); discriminators (stride 1,
#      one accumulator, im2col in f32 where an input can tell, positions before the transpose / one frame late, GELU
#      before the bias)
#   9. (0.0.7) its efficiency, only after 4e passed: conv2 (+ bias + GELU) from conv1's output, and the whole conv stage
#      mel -> encoder input, each side in a fresh process (`voaice bench-conv`, `whisper_oracle --bench-conv2`: the same
#      ops as a ggml graph, which the record shows equal to the schedulers' nodes), at 1, 2 and nproc threads
#   4f. (0.0.8) the encoder's nine layer norms (attn_ln and mlp_ln of each block, ln_post): every NORM, MUL and ADD node
#      read through the encoder scheduler's eval callback with every node observed (`whisper_oracle --norm`), each NORM's
#      input read before it ran; voaice fed that recorded input (blocks 1-3 and ln_post follow attention and the MLP,
#      not ported yet) and block 0's from its own mel, compared bit for bit (tests/norm.rs); discriminators (f32 sum,
#      mean from the double, one-pass variance, cvar without its f32 reduce, eps outside the sqrt, scale in double,
#      a division, mul + add fused)
#   10. (0.0.8) its efficiency, only after 4f passed: the NORM node, and norm -> * w -> + b, on jfk's encoder input,
#      each side in a fresh process (`voaice bench-norm`, `whisper_oracle --bench-norm`: the same ops as a ggml graph,
#      which the record shows equal to the scheduler's nodes), at 1, 2 and nproc threads
#   4g. (0.0.9) the matrix products on activations: every block's Q, K, V, K's and V's f16 copies, the out projection, its
#      bias and residual, fc1, its bias, GELU, fc2, its bias and residual, read through the encoder scheduler's eval
#      callback (`whisper_oracle --matmul`: one digest per row of every node, each product's input digested before it
#      ran, the attention's output whole) and from_float on NaN-bearing rows at 1..8 threads (`--mm-nan`); voaice fed
#      the recorded inputs (with the norm record of 4f) and block 0 from its own mel, compared bit for bit
#      (tests/matmul.rs); discriminators (no f16 rounding, one accumulator, the accumulators in sequence, the bias in the
#      accumulator, the residual before the bias, GELU without its table, the scalar converter, the split ignored)
#   11. (0.0.9) its efficiency, only after 4g passed: block 0's products on jfk — q, fc1 (+ GELU), fc2, and the block's
#      two halves and whole (qkv = attn_ln -> Q, K, V; mlp = out proj -> MLP with V standing in for the attention) — each
#      side in a fresh process (`voaice bench-mm`, `whisper_oracle --bench-mm`: the same ops as a ggml graph, which the
#      record shows equal to the scheduler's nodes), at 1, 2 and nproc threads
#   4h. (v0.1.0) flash attention and the WHOLE encoder: every node of the encoder graph read through its eval callback
#      (`whisper_oracle --encoder`: one digest per row of every computed node, attention's inputs and kv_pad's padding
#      rows checked, embd_enc whole after unobserved runs at 1, 2 and 4 threads, the attention node recomputed by the
#      standalone graph at 1..8 threads and by the use_ref path); voaice's attention fed Q, K, V it computes from the
#      recorded attn_ln input (each checked against the record) and compared value by value at 1 and 4 threads, the
#      model on every frame of block 0 and every 10th of blocks 1-3; then the whole encoder from voaice's own mel at
#      1, 2 and 4 threads, every node by digest and embd_enc value by value (tests/attention.rs); discriminators (glibc
#      expf for the probabilities, ggml_v_expf for the rescale, no running max, kv_pad's zero rows excluded, no FMA in
#      the scores or the output, the softmax sums in f32, a division by S, the one-chunk path — itself checked against
#      the reference's use_ref output — and Q scaled before the dot, which ×1/8 makes indistinguishable)
#   12. (v0.1.0) its efficiency, only after 4h passed: block 0's attention, and THE WHOLE ENCODER (the mel ->
#      embd_enc: `voaice bench-encode` against `whisper_oracle --bench-encode`, i.e. whisper_encode_with_state, the
#      digest of each side's embd_enc printed and compared), each in a fresh process, at 1, 2 and nproc threads
#   4i. (0.1.1) cross-attention K and V: every node of the cross graph read through sched_cross's eval callback
#      (`whisper_oracle --cross`: one digest per row of its 24 nodes, and of the kv_cross buffer itself — every layer's
#      1,536 rows, the padding included — after observed runs at 1 and 4 threads and unobserved runs at 1, 2 and 4, the
#      standalone graph checked against it); voaice fed the reference's embd_enc (the model and the fast path at 1 and 4
#      threads) and its own, from its own mel at 1, 2 and 4 threads, compared bit for bit (tests/cross.rs);
#      discriminators (the scale before the product, in the weights, in double, after the f16 copy; V scaled, V
#      unbiased, V's bias on K; no f16 rounding; the layer stride without padding, the non-flash layout, padding not +0;
#      the CPY by the row converter, which finite activations make indistinguishable)
#   13. (0.1.1) its efficiency, only after 4i passed: the cross K/V from embd_enc (`voaice bench-cross ... cross` against
#      `whisper_oracle --bench-cross`, the same ops as a ggml graph, which the record shows equal to the scheduler's), and
#      the WHOLE of whisper_encode_with_state — which runs the cross graph too, so v0.1.0's step 12 timed the reference
#      doing more than voaice did — mel -> embd_enc -> kv_cross (`voaice bench-cross ... whole` against
#      `whisper_oracle --bench-encode`), the digests of embd_enc and kv_cross compared, at 1, 2 and nproc threads
#   4j. (0.1.2) the decoder's input: on EVERY decoder call whisper_full makes (each window's prompt and every one-token
#      step) for the 8 inputs — with the transcript record's params (config A) and with a 300-token prompt and no
#      timestamps (config B: the 226-row batch and positions past 225) — sched_decode's GET_ROWS(d_te, embd),
#      GET_ROWS(d_pe, position) and their ADD read through its eval callback with the state's whisper_batch
#      (`whisper_oracle --decin`, at 1 and 4 threads, then unobserved); voaice's prompt and batch builders and its
#      three nodes (the model and the fast path) compared bit for bit (tests/decin.rs); the observed run equal to 0.0.1's
#      transcript record; discriminators (positions off by one, step positions without the prompt, position row 0, the
#      multilingual ids, DAZ widening, bf16 widening, the sum to f16; the position rows to f16, the operands swapped and
#      the sum in double, which the data and IEEE make indistinguishable)
#   14. (0.1.2) its efficiency, only after 4j passed: the three nodes for 1 token (a step) and 226 (config B's prompt),
#      `voaice bench-decin` (batch prep + rows into the caller's buffer, one thread) against `whisper_oracle --bench-decin`
#      (the inputs set, plan and compute of the same three nodes as a ggml graph) at 1 and nproc threads, digests compared
#   4k. (0.1.3) the self-attention products and the f16 self KV cache: on EVERY decoder call whisper_full makes for the 8
#      inputs (configs A and B of 4j, at 1 and 4 threads, then unobserved at 1 and 4), every decoder layer's attn_ln
#      (NORM, MUL, ADD), Q (MUL_MAT, ADD, SCALE), K (MUL_MAT, SCALE), V (MUL_MAT, ADD) and the two CPYs into kv_self,
#      read through sched_decode's eval callback with each layer's input whole, the KQ_mask's f32 rows and its f16 cast,
#      kv_self's head, n and cells, and kv_self.k / .v after every call — every cell of every layer (`whisper_oracle
#      --selfkv`); voaice fed each layer's recorded input (the model and the fast path) and layer 0 from its own decoder
#      input at 1, 2 and 4 threads, its KvSelf kept across each run's calls, compared bit for bit (tests/selfkv.rs);
#      the thread behaviour of every node recorded; discriminators (the scale before the products, Q scaled before its
#      bias, Q or K unscaled, K biased, V scaled, the scale in double, attn_ln fused, no f16 rounding, one accumulator,
#      the accumulators in sequence; the mask off by one, padded to 32, -inf as a finite value; K/V a cell late, the
#      layers unpadded, the buffer not cleared at a window; the CPYs and the mask's cast by the row converter and the
#      sequence ignored, which finite values, one sequence and no free cell make indistinguishable)
#   15. (0.1.3) its efficiency, only after 4k passed: block = decoder layer 0's twelve nodes, call = all four layers' and
#      the mask, for 1 token (a step at cell 226) and 226 (config B's prompt), `voaice bench-selfkv` against
#      `whisper_oracle --bench-selfkv` (the inputs set, plan and compute of the same nodes as a ggml graph), at 1, 2 and
#      nproc threads, digests compared
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
version=$(awk -F'"' '/^version/{print $2; exit}' Cargo.toml)
out=testing/results/$version.txt
mkdir -p testing/results .oracle
model=models/ggml-tiny.en.bin
pin_sha=$(awk -F'\t' '$1=="model" && $2=="ggml-tiny.en.bin"{print $4}' upstream/PIN)
pin_url=$(awk -F'\t' '$1=="model" && $2=="ggml-tiny.en.bin"{print $5}' upstream/PIN)

log() { echo "$*" | tee -a "$out"; }
: > "$out"
log "# voaice.rs $version release gate — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "# host: $(uname -m), $(nproc) cpus, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); load $(cut -d' ' -f1-3 /proc/loadavg)"
log "# rustc: $(rustc --version); gcc: $(gcc --version | head -1)"

log "## 1. reference"
testing/oracle/build.sh > .oracle/build.log   # whole, then its first line (`| head -1` under pipefail can SIGPIPE it)
head -1 .oracle/build.log | tee -a "$out"

log "## 2. inputs"
if [ ! -f "$model" ]; then
  mkdir -p models
  curl -sfL -o "$model.part" "$pin_url" && mv "$model.part" "$model"
fi
have=$(sha256sum "$model" | cut -d' ' -f1)
[ "$have" = "$pin_sha" ] || { log "FAIL: $model sha256 $have != pin $pin_sha"; exit 1; }
log "model $model sha256 $have (pinned)"
python3 testing/make_audio.py | tee -a "$out"
python3 testing/make_resample_audio.py | tee -a "$out"

log "## 3. record (the shipped libwhisper.so, in process)"
rm -rf .oracle/tiny.en
testing/oracle/bin/whisper_oracle "$model" .oracle/tiny.en .audio/*.wav 2>&1 | tee -a "$out"
rm -rf .oracle/f16
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --f16 .oracle/f16 2>&1 | tee -a "$out"
log "(the f16 record took $(( $(date +%s) - t0 )) s: all 2^32 f32 patterns, three conversions and the GELU op, one thread)"

rm -rf .oracle/resample && mkdir -p .oracle/resample
testing/oracle/bin/resample_oracle .oracle/resample .audio/resample/*.wav > .oracle/resample.log
log "resample_oracle (libcommon.a's read_audio_data): $(wc -l < .oracle/resample.log) files recorded, $(awk '{s+=$2} END{print s}' .oracle/resample.log) samples"
log "libcommon.a's common-whisper.cpp.o (miniaudio inside): $(objdump -d upstream/whisper.cpp/build/examples/CMakeFiles/common.dir/common-whisper.cpp.o | grep -cE 'vfn?m(add|sub)') FMA instructions, $(objdump -d upstream/whisper.cpp/build/examples/CMakeFiles/common.dir/common-whisper.cpp.o | grep -c '%ymm') ymm uses"

rm -rf .oracle/conv1
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --conv1 "$model" .oracle/conv1 .audio/*.wav 2>&1 | tee -a "$out"
log "(the conv1 record took $(( $(date +%s) - t0 )) s: each input encoded three times, observed at 1 and 4 threads and not observed)"
lib=upstream/whisper.cpp/build/bin/libggml-cpu.so
a=$(nm -D --defined-only "$lib" | awk '$3=="ggml_vec_dot_f16"{print "0x"$1}')
dis=$(objdump -d --no-show-raw-insn "$lib" --start-address="$a" --stop-address=$(printf '0x%x' $((a + 0x100))) | sed -n '1,/ret/p')   # sed reads to the end: quitting early can SIGPIPE objdump
log "libggml-cpu's ggml_vec_dot_f16 ($a): $(echo "$dis" | grep -c vcvtph2ps) vcvtph2ps, $(echo "$dis" | grep -c 'vfmadd231ps.*ymm') ymm vfmadd231ps, $(echo "$dis" | grep -c vhaddps) vhaddps, $(echo "$dis" | grep -c vaddsd) vaddsd (the double tail); GGML_LLAMAFILE $(awk -F= '/^GGML_LLAMAFILE:/{print $2}' upstream/whisper.cpp/build/CMakeCache.txt) in the build"
rm -rf .oracle/conv2
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --conv2 "$model" .oracle/conv2 .audio/*.wav 2>&1 | tee -a "$out"
log "(the conv2 record took $(( $(date +%s) - t0 )) s: each input encoded three times, observed at 1 and 4 threads and not observed)"
rm -rf .oracle/norm
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --norm "$model" .oracle/norm .audio/*.wav 2>&1 | tee -a "$out"
log "(the norm record took $(( $(date +%s) - t0 )) s: each input encoded three times, every encoder node observed at 1 and 4 threads, and not observed)"
nfn() { local a; a=$(nm -D --defined-only "$lib" | awk -v f="$1" '$3==f{print "0x"$1}'); objdump -d --no-show-raw-insn "$lib" --start-address="$a" --stop-address=$(printf '0x%x' $((a + $2))) | sed -n '1,/ret *$/p'; }   # to the end: no SIGPIPE under pipefail
cv=$(nfn ggml_vec_cvar_f32 0x400); fn=$(objdump -d --no-show-raw-insn "$lib" | sed -n '/<ggml_compute_forward_norm>:/,/^$/p')
log "libggml-cpu's ggml_vec_cvar_f32: $(echo "$cv" | grep -c 'vmulps') vmulps, $(echo "$cv" | grep -c 'vaddsd') vaddsd, $(echo "$cv" | grep -cE 'vfn?m(add|sub)') FMA; ggml_compute_forward_norm: $(echo "$fn" | grep -c vsqrtss) vsqrtss, $(echo "$fn" | grep -c vcvtsd2ss) vcvtsd2ss, $(echo "$fn" | grep -cE 'vfn?m(add|sub)') FMA, $(echo "$fn" | grep -c 'vaddsd') vaddsd (the in-order double sum)"
rm -rf .oracle/matmul
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --matmul "$model" .oracle/matmul .audio/*.wav 2>&1 | tee -a "$out"
testing/oracle/bin/whisper_oracle --mm-nan "$model" .oracle/matmul 2>&1 | tee -a "$out"
log "(the matmul record took $(( $(date +%s) - t0 )) s: each input encoded three times, every encoder node observed at 1 and 4 threads, and not observed; $(du -sh .oracle/matmul | cut -f1) of digests and attention outputs)"
ff=$(nfn ggml_cpu_fp32_to_fp16 0x200); mm=$(objdump -d --no-show-raw-insn "$lib" | sed -n '/<ggml_compute_forward_mul_mat>:/,/^$/p')
log "libggml-cpu's ggml_cpu_fp32_to_fp16 (mul_mat's from_float): $(echo "$ff" | grep -c vcvtps2ph) vcvtps2ph; ggml_compute_forward_mul_mat: $(echo "$mm" | grep -c 'call') calls, $(echo "$mm" | grep -cE 'vfn?m(add|sub)') FMA (the dot is ggml_vec_dot_f16 through the traits); GGML_CPU_REPACK $(awk -F= '/^GGML_CPU_REPACK:/{print $2}' upstream/whisper.cpp/build/CMakeCache.txt) (no f16 repack: $(nm -D --defined-only "$lib" | grep -c 'repack.*ggml_type1EE') f16 traits)"
rm -rf .oracle/encoder
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --encoder "$model" .oracle/encoder .audio/*.wav 2>&1 | tee -a "$out"
log "(the encoder record took $(( $(date +%s) - t0 )) s: each input encoded five times — observed at 1 and 4 threads, not observed at 1, 2, 4 — and each attention node recomputed at 1..8 threads and by use_ref; $(du -sh .oracle/encoder | cut -f1) of digests and embd_enc)"
fa=$(objdump -d --no-show-raw-insn -C "$lib" | sed -n '/<ggml_compute_forward_flash_attn_ext_tiled(.*)>:$/,/^$/p'); sm=$(nfn ggml_vec_soft_max_f32 0x300)
log "libggml-cpu's ggml_compute_forward_flash_attn_ext_tiled (the path whisper's encoder takes): $(echo "$fa" | grep -c 'vfmadd231ps') vfmadd231ps (simd_gemm's chains), $(echo "$fa" | grep -c 'vmaxss') vmaxss (the tile max), $(echo "$fa" | grep -c 'call.*<expf@plt>') calls of glibc expf, $(echo "$fa" | grep -c 'call.*<fmaxf@plt>') of fmaxf, $(echo "$fa" | grep -c 'call.*<ggml_vec_soft_max_f32@plt>') of ggml_vec_soft_max_f32, $(echo "$fa" | grep -c 'vcvtps2ph') vcvtps2ph (Q is not converted), $(echo "$fa" | grep -c 'vaddsd') vaddsd (S += the double sum); ggml_vec_soft_max_f32: $(echo "$sm" | grep -cE 'vfn?madd') FMA (ggml_v_expf), $(echo "$sm" | grep -c vaddsd) vaddsd, $(echo "$sm" | grep -c 'call.*expf') call of expf (the n % 8 tail); expf from $(objdump -T "$lib" | awk '$NF=="expf"{print $(NF-1)}')"
rm -rf .oracle/cross
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --cross "$model" .oracle/cross .audio/*.wav 2>&1 | tee -a "$out"
log "(the cross record took $(( $(date +%s) - t0 )) s: each input encoded five times — the cross graph observed at 1 and 4 threads, not observed at 1, 2, 4 — and the standalone cross graph at 1 and 4 threads; $(du -sh .oracle/cross | cut -f1) of digests)"
sc=$(objdump -d --no-show-raw-insn "$lib" | sed -n '/<ggml_compute_forward_scale>:/,/^$/p')
log "libggml-cpu's ggml_compute_forward_scale: $(echo "$sc" | grep -c 'vmulps') vmulps (ggml_vec_scale_f32: b == 0, whisper's Kscale), $(echo "$sc" | grep -cE 'vfn?madd') FMA (ggml_vec_mad1_f32, the b != 0 branch)"
rm -rf .oracle/decin
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --decin "$model" .oracle/decin .audio/*.wav 2>&1 | tee -a "$out"
log "(the decin record took $(( $(date +%s) - t0 )) s: whisper_full seven times per input — config A observed at 1 and 4 threads and not observed at 1 and 4, config B observed at 1 and 4 and not observed at 1; $(du -sh .oracle/decin | cut -f1) of batches and digests)"
gr=$(objdump -d --no-show-raw-insn "$lib" | sed -n '/<ggml_compute_forward_get_rows>:/,/^$/p'); cf=$(nfn ggml_cpu_fp16_to_fp32 0x200)
log "libggml-cpu's ggml_compute_forward_get_rows: $(echo "$gr" | grep -c 'call.*<ggml_cpu_fp16_to_fp32@plt>') call of ggml_cpu_fp16_to_fp32 (the f16 rows), $(echo "$gr" | grep -c vcvtph2ps) vcvtph2ps inline; ggml_cpu_fp16_to_fp32: $(echo "$cf" | grep -c vcvtph2ps) vcvtph2ps (blocks of 8 and 4; the table for a tail, none at n_state 384)"
rm -rf .oracle/selfkv
t0=$(date +%s)
testing/oracle/bin/whisper_oracle --selfkv "$model" .oracle/selfkv .audio/*.wav 2>&1 | cut -c1-600 | tee -a "$out"
log "(the selfkv record took $(( $(date +%s) - t0 )) s: whisper_full eight times per input — configs A and B observed at 1 and 4 threads and not observed at 1 and 4; $(du -sh .oracle/selfkv | cut -f1) of digests, layer inputs and cache digests)"
dp=$(objdump -d --no-show-raw-insn "$lib" | sed -n '/<ggml_compute_forward_dup>:$/,/^$/p')
log "libggml-cpu's ggml_compute_forward_dup (dup_flt<float, ggml_fp16_t> inlined: the KQ_mask cast and the K/V CPYs): $(echo "$dp" | grep -c vcvtps2ph) vcvtps2ph (the conversion is the scalar bit trick, 0.0.9's)"
log "this CPU: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx|avx2|fma|f16c|avx512f' | paste -sd' ') (production: Zen 3, the same extensions; its library is not the one checked here)"

log "## 4. voaice.rs"
cargo build --release 2>&1 | tail -1 | tee -a "$out"
cargo clippy --release --all-targets -q -- -D warnings 2>&1 | tee -a "$out"
cargo test --release 2>&1 | grep -E "^test result" | tee -a "$out"
cargo test --release --test oracle -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" | tee -a "$out"
grep -q "test result: ok. 13 passed" "$out" || { log "FAIL: the oracle comparisons did not all pass"; exit 1; }

log "## 4b. the Ogg/Opus reader (0.0.4): opus-tools 0.2 (opusinfo, opusdec), libopus 1.4, libogg 1.3.5 on mindX production"
(cd testing/opus/files && sha256sum -c --quiet ../files.sha256) && log "$(wc -l < testing/opus/files.sha256) pinned files: sha256 ok"
log "reference: $(cut -c1-220 testing/opus/reference.meta.json)…"
step=.oracle/opus_step.log   # each check reads its own step's output, not the whole record (earlier steps print counts too)
cargo test --release --test opus -- --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 4 passed" "$step" || { log "FAIL: the Ogg/Opus oracle comparisons did not all pass"; exit 1; }
cargo test --release --manifest-path streamair/Cargo.toml --test roundtrip 2>&1 | grep -E "^test |^test result" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 3 passed" "$step" || { log "FAIL: streamair -> voaice round trips"; exit 1; }
log "voaice opus info on one pinned file:"
target/release/voaice opus info testing/opus/files/e_six_family1_96k.opus | sed 's/^/    /' | tee -a "$out" >/dev/null
opus_host=${VOAICE_OPUS_HOST:-root@168.231.126.58}
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$opus_host" true 2>/dev/null; then
  testing/opus/oracle.sh check "$opus_host" 2>&1 | tee -a "$out"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "FAIL: the reference's answers on $opus_host changed"; exit 1; }
else
  log "SKIPPED: $opus_host unreachable — the reference was not asked again in this run; the comparisons above are"
  log "         against its answers recorded in testing/opus/reference.jsonl ($(grep -o '"date": "[^"]*"' testing/opus/reference.meta.json))"
fi

log "## 4c. the resampler (0.0.5): whisper-cli's read_audio_data (miniaudio 0.11.24 in the build's libcommon.a)"
step=.oracle/resample_step.log
cargo test --release --test resample -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 3 passed" "$step" || { log "FAIL: the resampler oracle comparisons did not all pass"; exit 1; }

log "## 4d. encoder conv1 and ggml_vec_dot_f16 (0.0.6): the conv graph's nodes through the scheduler's eval callback"
step=.oracle/conv1_step.log
cargo test --release --test conv1 -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 5 passed" "$step" || { log "FAIL: the conv1 oracle comparisons did not all pass"; exit 1; }

log "## 4e. encoder conv2 and the positional embedding (0.0.7): both schedulers' nodes through their eval callbacks"
step=.oracle/conv2_step.log
cargo test --release --test conv2 -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 4 passed" "$step" || { log "FAIL: the conv2 oracle comparisons did not all pass"; exit 1; }

log "## 4f. the encoder's layer norms (0.0.8): every NORM, MUL and ADD node through the encoder scheduler's eval callback"
step=.oracle/norm_step.log
cargo test --release --test norm -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 3 passed" "$step" || { log "FAIL: the layer norm oracle comparisons did not all pass"; exit 1; }

log "## 4g. the matrix products on activations (0.0.9): every block's products, biases, GELU, residuals and f16 copies"
step=.oracle/matmul_step.log
cargo test --release --test matmul -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 4 passed" "$step" || { log "FAIL: the matmul oracle comparisons did not all pass"; exit 1; }

log "## 4h. flash attention and the whole encoder (v0.1.0): every node of the encoder graph through its eval callback, embd_enc"
step=.oracle/attention_step.log
cargo test --release --test attention -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 3 passed" "$step" || { log "FAIL: the attention / encoder oracle comparisons did not all pass"; exit 1; }

log "## 4i. cross-attention K and V (0.1.1): every node of sched_cross through its eval callback, and kv_cross whole"
step=.oracle/cross_step.log
cargo test --release --test cross -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 3 passed" "$step" || { log "FAIL: the cross K/V oracle comparisons did not all pass"; exit 1; }

log "## 4j. the decoder's input (0.1.2): every decoder call's GET_ROWS d_te, GET_ROWS d_pe and ADD, and its batch"
step=.oracle/decin_step.log
cargo test --release --test decin -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 5 passed" "$step" || { log "FAIL: the decoder input oracle comparisons did not all pass"; exit 1; }

log "## 4k. the self-attention products and kv_self (0.1.3): every layer's nodes before self-attention, the mask, the cache"
step=.oracle/selfkv_step.log
cargo test --release --test selfkv -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" > "$step" || true
tee -a "$out" < "$step"
grep -q "test result: ok. 5 passed" "$step" || { log "FAIL: the self-attention / kv_self oracle comparisons did not all pass"; exit 1; }
nt=$(nproc)
log "## 5. efficiency (only now): log-mel; wall = best of 10 calls, cpu = CPU ms per call (utime+stime, all threads,"
log "##    loop >= 1 s), heap = bytes live at the first call's peak (KiB; voaice's counting allocator, the reference's"
log "##    operator new counted), rss = VmHWM of the first call minus VmRSS before it (KiB; heap reuse can hide growth)"
# 0.0.1's code path, rebuilt from its tag in this run so both see the same load (scratch under .oracle/, removed after)
old=.oracle/v0.0.1
rm -rf "$old" && mkdir -p "$old"
git archive v0.0.1 | tar -x -C "$old"
cargo build --release -q --manifest-path "$old/Cargo.toml" --target-dir "$old/target"
log "0.0.1 rebuilt from tag v0.0.1 ($(git rev-parse --short 'v0.0.1^{commit}')): its \`voaice mel\` times the mel call alone, 1 thread, best of 10"
field() { sed -E "s/.* $1 ([^ ]+).*/\1/"; }
log "$(printf '%-11s %7s | %8s %8s %8s | %8s %8s | %7s | %8s %8s | %8s %8s | %6s %6s | %6s %6s' wav samples \
  ref_1t v001_1t vo_1t "ref_${nt}t" "vo_${nt}t" x_v001 cpu_ref1 cpu_vo1 cpu_ref"$nt" cpu_vo"$nt" heap_r heap_v rss_r rss_v)"
sum_l001=0; sum_lref=0; sum_lrefn=0; k=0
for d in .oracle/tiny.en/*/; do
  w=$(basename "$d")
  r1=$(testing/oracle/bin/whisper_oracle --bench-mel "$model" ".audio/$w.wav" 1)
  rn=$(testing/oracle/bin/whisper_oracle --bench-mel "$model" ".audio/$w.wav" "$nt")
  v1=$(target/release/voaice bench-mel "$model" ".audio/$w.wav" --threads 1)
  vn=$(target/release/voaice bench-mel "$model" ".audio/$w.wav" --threads "$nt")
  o=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ms=$("$old/target/release/voaice" mel "$model" ".audio/$w.wav" | sed -E 's/.* ([0-9.]+) ms$/\1/')
    o=$(awk -v a="$o" -v b="$ms" 'BEGIN{print (a=="" || b<a) ? b : a}')
  done
  n=$(echo "$v1" | field samples)
  r1w=$(echo "$r1" | field wall_best_ms); rnw=$(echo "$rn" | field wall_best_ms)
  v1w=$(echo "$v1" | field wall_best_ms); vnw=$(echo "$vn" | field wall_best_ms)
  x001=$(awk -v a="$o" -v b="$v1w" 'BEGIN{printf "%.2fx", a/b}')
  log "$(printf '%-11s %7s | %8.3f %8.3f %8.3f | %8.3f %8.3f | %7s | %8s %8s | %8s %8s | %6s %6s | %6s %6s' "$w" "$n" \
    "$r1w" "$o" "$v1w" "$rnw" "$vnw" "$x001" \
    "$(echo "$r1" | field cpu_ms_per_call)" "$(echo "$v1" | field cpu_ms_per_call)" \
    "$(echo "$rn" | field cpu_ms_per_call)" "$(echo "$vn" | field cpu_ms_per_call)" \
    "$(echo "$r1" | field heap_peak_kb)" "$(echo "$v1" | field heap_peak_kb)" \
    "$(echo "$r1" | field rss_peak_delta_kb)" "$(echo "$v1" | field rss_peak_delta_kb)")"
  sum_l001=$(awk -v s="$sum_l001" -v a="$o" -v b="$v1w" 'BEGIN{print s + log(a/b)}')
  sum_lref=$(awk -v s="$sum_lref" -v a="$r1w" -v b="$v1w" 'BEGIN{print s + log(a/b)}')
  sum_lrefn=$(awk -v s="$sum_lrefn" -v a="$rnw" -v b="$vnw" 'BEGIN{print s + log(a/b)}')
  k=$((k + 1))
done
rm -rf "$old"
log "$(awk -v a="$sum_l001" -v b="$sum_lref" -v c="$sum_lrefn" -v k="$k" -v nt="$nt" 'BEGIN{printf "geometric mean of wall-time ratios over %d inputs: voaice at 1 thread is %.2fx faster than 0.0.1 and %.2fx faster than the reference at 1 thread; at %d threads, %.2fx faster than the reference at %d", k, exp(a/k), exp(b/k), nt, exp(c/k), nt}')"
log "## 5b. efficiency (only now): f32<->f16 and GELU. init = the first GELU table build in a fresh process (best of"
log "##     5 processes; the reference's ggml_cpu_init also fills its quick-GELU and f32<-f16 tables and two 256-entry"
log "##     ones, voaice only the GELU table it needs); rows = best of 10 calls on 384x1500 values; gelu = the op on"
log "##     1536x1500 (the encoder MLP's size), 1 thread; the reference's through a ggml graph, whose two tensor copies"
log "##     are measured alone and taken off (gelu_net)"
ri=1e30; vi=1e30
for _ in 1 2 3 4 5; do
  ri=$(awk -v a="$ri" -v b="$(testing/oracle/bin/whisper_oracle --bench-f16 init | field init_ms)" 'BEGIN{print (b<a)?b:a}')
  vi=$(awk -v a="$vi" -v b="$(target/release/voaice bench-f16 init | field init_ms)" 'BEGIN{print (b<a)?b:a}')
done
rr=$(testing/oracle/bin/whisper_oracle --bench-f16 rows); vr=$(target/release/voaice bench-f16 rows)
log "reference: $rr"
log "voaice:    $vr"
gnet=$(awk -v a="$(echo "$rr" | field gelu_ms)" -v b="$(echo "$rr" | field gelu_copies_ms)" 'BEGIN{printf "%.3f", a-b}')
log "$(printf '%-30s %10s %10s %8s' measure reference voaice ratio)"
row() { log "$(awk -v n="$1" -v a="$2" -v b="$3" 'BEGIN{printf "%-30s %10.3f %10.3f %7.2fx", n, a, b, a/b}')"; }
row "gelu table, first build (ms)" "$ri" "$vi"
row "fp32_to_fp16 row 576k (ms)" "$(echo "$rr" | field fp32_to_fp16_row_ms)" "$(echo "$vr" | field fp32_to_fp16_row_ms)"
row "fp16_to_fp32 row 576k (ms)" "$(echo "$rr" | field fp16_to_fp32_row_ms)" "$(echo "$vr" | field fp16_to_fp32_row_ms)"
row "gelu op 2.3M, net (ms)" "$gnet" "$(echo "$vr" | field gelu_ms)"
row "gelu op 2.3M, scalar path (ms)" "$gnet" "$(echo "$vr" | field gelu_scalar_ms)"
log "## 6. efficiency (only now): the Ogg/Opus reader. crc = 16 MiB, best of 7, sliced-by-8 against one byte at a time;"
log "##    read = Reader::new + every packet + the end checks, best of 10 (mem: from a slice; file: open + read), cpu ="
log "##    CPU ms per read over >= 1 s; heap = bytes live at one read's peak (the counting allocator), reading from the file"
for f in e_jfk_x3_m_6k_comp0 e_jfk_m_2.5ms_24k e_jfk_m_60ms_24k sa_silence_60 sa_continuation e_picture_24k; do
  log "$(printf '%-22s ' "$f") $(target/release/voaice bench-opus "testing/opus/files/$f.opus" | sed 's/^bench-opus //')"
done
log "## 7. efficiency (only now): the resampler. One call = the whole read of a file (open, parse, convert, mix,"
log "##    resample, the vector out); wall = best of 10, cpu = CPU ms per call over >= 1 s, heap = bytes live at the first"
log "##    call's peak (voaice: the counting allocator; reference: malloc/calloc/realloc interposed), rss = VmHWM delta"
log "$(printf '%-18s %8s %8s | %8s %8s %6s | %8s %8s | %6s %6s | %5s %5s | %8s' file in_s samples ref_ms vo_ms x cpu_ref cpu_vo heap_r heap_v rss_r rss_v cpu_ms/as)"
sum=0; k=0
for f in jfk_48k bench_44k1_s_60s r48000_s_s16 r44100_s_s16 r22050_m_s16 r8000_s_s16 r16000_s_s16 r48000_c6_s16_ext; do
  r=$(testing/oracle/bin/resample_oracle --bench ".audio/resample/$f.wav")
  v=$(target/release/voaice bench-resample ".audio/resample/$f.wav")
  secs=$(awk -v n="$(echo "$v" | field samples)" 'BEGIN{printf "%.2f", n/16000}')
  rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
  log "$(printf '%-18s %8s %8s | %8.3f %8.3f %5.2fx | %8s %8s | %6s %6s | %5s %5s | %8s' "$f" "$secs" "$(echo "$v" | field samples)" \
    "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
    "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
    "$(echo "$r" | field heap_peak_kb)" "$(echo "$v" | field heap_peak_kb)" \
    "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)" \
    "$(awk -v c="$(echo "$v" | field cpu_ms_per_call)" -v s="$secs" 'BEGIN{printf "%.3f", c/s}')")"
  sum=$(awk -v s="$sum" -v a="$rw" -v b="$vw" 'BEGIN{print s + log(a/b)}'); k=$((k + 1))
done
log "$(awk -v s="$sum" -v k="$k" 'BEGIN{printf "geometric mean over %d files: voaice reads %.2fx faster than whisper-cli'"'"'s read_audio_data (cpu_ms/as = voaice CPU ms per second of output audio)", k, exp(s/k)}')"
log "## 8. efficiency (only now): encoder conv1 (0.0.6), the first 30-s window of jfk (3000 frames, any input costs the"
log "##    same). wall = best of 10, cpu = CPU ms per call over >= 1 s; heap: voaice = bytes live at the first call's peak"
log "##    (its output allocated by that call, then reused, as the reference's graph keeps its tensors); the reference ="
log "##    the bytes of the graph's own tensors (im2col, the product, and for +gelu the add and gelu outputs), from"
log "##    ggml_nbytes; rss = VmHWM delta of the first call"
log "$(printf '%-12s %3s | %8s %8s %6s | %8s %8s | %7s %7s | %6s %6s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v)"
for what in conv1 gelu; do
  for th in 1 2 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-conv1 "$model" .audio/jfk.wav "$th" "$what")
    v=$(target/release/voaice bench-conv1 "$model" .audio/jfk.wav "$what" --threads "$th")
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    log "$(printf '%-12s %3s | %8.3f %8.3f %5.2fx | %8s %8s | %7s %7s | %6s %6s' "$( [ "$what" = gelu ] && echo conv1+b+gelu || echo conv1)" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$(echo "$r" | field op_mem_kb)" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)")"
  done
done
log "## 9. efficiency (only now): encoder conv2 and the conv stage (0.0.7), jfk's first 30-s window. conv2 = conv1's GELU"
log "##    output -> conv2 + bias + GELU (embd_conv); stage = the mel -> conv1 -> conv2 -> + positions (the encoder's"
log "##    input). wall = best of 10, cpu = CPU ms per call over >= 1 s; heap: voaice = bytes live at the first call's peak"
log "##    (its buffers allocated by that call, then reused); the reference = the bytes of the graph's own tensors (every"
log "##    node that is not a view: im2cols, products, adds, GELUs, the cont), from ggml_nbytes; rss = VmHWM delta"
log "$(printf '%-6s %3s | %8s %8s %6s | %8s %8s | %7s %7s | %6s %6s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v)"
for what in conv2 stage; do
  for th in 1 2 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-conv2 "$model" .audio/jfk.wav "$th" "$what")
    v=$(target/release/voaice bench-conv "$model" .audio/jfk.wav "$what" --threads "$th")
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    log "$(printf '%-6s %3s | %8.3f %8.3f %5.2fx | %8s %8s | %7s %7s | %6s %6s' "$what" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$(echo "$r" | field op_mem_kb)" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)")"
  done
done
log "## 10. efficiency (only now): the encoder's layer norm (0.0.8) on jfk's encoder input ([1500, 384]; block 0's"
log "##    attn_ln weights). norm = the NORM node alone; chain = norm -> * w -> + b (the three nodes). wall = best of 10,"
log "##    cpu = CPU ms per call over >= 1 s; heap: voaice = bytes live at the first call's peak (its output, then reused);"
log "##    the reference = the bytes of the graph's non-view nodes (ggml_nbytes); rss = VmHWM delta"
log "$(printf '%-6s %3s | %8s %8s %6s | %8s %8s | %7s %7s | %6s %6s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v)"
for what in norm chain; do
  for th in 1 2 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-norm "$model" .audio/jfk.wav "$th" "$what")
    v=$(target/release/voaice bench-norm "$model" .audio/jfk.wav "$what" --threads "$th")
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    log "$(printf '%-6s %3s | %8.4f %8.4f %5.2fx | %8s %8s | %7s %7s | %6s %6s' "$what" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$(echo "$r" | field op_mem_kb)" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)")"
  done
done
log "## 11. efficiency (only now): block 0's products (0.0.9) on jfk (inputs computed beforehand from the WAV by each"
log "##    side: X = the encoder's input; q = Q + b and fc1 = fc1 + b + GELU on attn_ln_0(X); fc2 = fc2 + b on that GELU;"
log "##    qkv = attn_ln -> Q + b, K, V + b, K and V to f16; mlp = out proj + b + X -> mlp_ln -> fc1 + b -> GELU -> fc2 + b"
log "##    + residual, V's output standing in for the attention; block = qkv then mlp). wall = best of 10, cpu = CPU ms per"
log "##    call over >= 1 s; heap: voaice = bytes live at the first call's peak (outputs, then reused; the widened weights"
log "##    are held with the model, outside the call); the reference = the graph's non-view nodes + its work buffer"
log "$(printf '%-6s %3s | %9s %9s %6s | %9s %9s | %7s %7s | %6s %6s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v)"
for what in q fc1 fc2 qkv mlp block; do
  for th in 1 2 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-mm "$model" .audio/jfk.wav "$th" "$what")
    v=$(target/release/voaice bench-mm "$model" .audio/jfk.wav "$what" --threads "$th")
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    log "$(printf '%-6s %3s | %9.3f %9.3f %5.2fx | %9s %9s | %7s %7s | %6s %6s' "$what" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$(echo "$r" | field op_mem_kb)" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)")"
  done
done
log "## 12. efficiency (only now): block 0's attention (v0.1.0) on jfk (Q, K, V computed beforehand from the WAV by each"
log "##    side), then THE WHOLE ENCODER: the mel (computed beforehand, as whisper's state holds it) -> conv1 -> conv2 ->"
log "##    positions -> 4 blocks -> ln_post = embd_enc (voaice: Encoder::encode_into; the reference: whisper_encode_with_state,"
log "##    its conv graph then its encoder graph). wall = best of 10, cpu = CPU ms per call over >= 1 s (attention) or"
log "##    >= 2 s (encoder); heap: voaice = bytes live at the first call's peak (its buffers between stages, allocated by"
log "##    that call and then reused; the widened weights are held with the model, outside the call); the reference ="
log "##    attention: the node's output + the K/V buffers + the work buffer; encoder: the two schedulers' compute buffers +"
log "##    kv_pad (allocated by whisper_init_state); rss = VmHWM delta of the first call. digest = 64-bit FNV-1a of embd_enc"
log "$(printf '%-7s %3s | %9s %9s %6s | %9s %9s | %7s %7s | %6s %6s | %s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v digests)"
for what in attn encode; do
  for th in 1 2 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-"$what" "$model" .audio/jfk.wav "$th")
    v=$(target/release/voaice bench-"$what" "$model" .audio/jfk.wav --threads "$th")
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    dg="-"
    if [ "$what" = encode ]; then
      rd=$(echo "$r" | field embd_enc_digest); vd=$(echo "$v" | field embd_enc_digest)
      [ "$rd" = "$vd" ] || { log "FAIL: embd_enc digests differ in the benchmark ($rd vs $vd)"; exit 1; }
      dg="$vd = $rd"
    fi
    log "$(printf '%-7s %3s | %9.3f %9.3f %5.2fx | %9s %9s | %7s %7s | %6s %6s | %s' "$what" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$(echo "$r" | field op_mem_kb)" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)" "$dg")"
  done
done
log "## 13. efficiency (only now): cross-attention K and V (0.1.1) on jfk. cross = every decoder layer's K (x Kscale) and V"
log "##    (+ b) from embd_enc (computed beforehand by each side) into the f16 cache; whole = the mel -> embd_enc -> kv_cross,"
log "##    i.e. what whisper_encode_with_state does (its conv, encoder AND cross graphs) against voaice's encode_into +"
log "##    Cross::run_into. wall = best of 10, cpu = CPU ms per call over >= 1 s (cross) or >= 2 s (whole); heap: voaice ="
log "##    bytes live at the first call's peak (the cache and buffers, allocated by that call and then reused; the widened"
log "##    weights are held outside the call); the reference: cross = sched_cross's compute buffer + kv_cross, whole = the"
log "##    conv and encoder schedulers' buffers + kv_pad + sched_cross's buffer + kv_cross (v0.1.0's step 12 counted the"
log "##    first three only); rss ="
log "##    VmHWM delta of the first call (the reference's cross: its standalone graph's tensors were allocated before)"
log "$(printf '%-7s %3s | %9s %9s %6s | %9s %9s | %7s %7s | %6s %6s | %s' op thr ref_ms vo_ms x cpu_ref cpu_vo mem_ref heap_vo rss_r rss_v digests)"
for what in cross whole; do
  for th in 1 2 "$nt"; do
    if [ "$what" = cross ]; then r=$(testing/oracle/bin/whisper_oracle --bench-cross "$model" .audio/jfk.wav "$th")
    else r=$(testing/oracle/bin/whisper_oracle --bench-encode "$model" .audio/jfk.wav "$th"); fi
    v=$(target/release/voaice bench-cross "$model" .audio/jfk.wav "$what" --threads "$th")
    rk=$(echo "$r" | field kv_cross_digest); vk=$(echo "$v" | field kv_cross_digest)
    [ "$rk" = "$vk" ] || { log "FAIL: kv_cross digests differ in the benchmark ($rk vs $vk)"; exit 1; }
    dg="kv $vk"
    if [ "$what" = whole ]; then
      rd=$(echo "$r" | field embd_enc_digest); vd=$(echo "$v" | field embd_enc_digest)
      [ "$rd" = "$vd" ] || { log "FAIL: embd_enc digests differ in the benchmark ($rd vs $vd)"; exit 1; }
      dg="$dg, enc $vd"
    fi
    rw=$(echo "$r" | field wall_best_ms); vw=$(echo "$v" | field wall_best_ms)
    rm_kb=$(echo "$r" | field op_mem_kb)
    [ "$what" = whole ] && rm_kb=$(( rm_kb + $(echo "$r" | field cross_mem_kb) ))
    log "$(printf '%-7s %3s | %9.3f %9.3f %5.2fx | %9s %9s | %7s %7s | %6s %6s | %s' "$what" "$th" \
      "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_ms_per_call)" "$(echo "$v" | field cpu_ms_per_call)" \
      "$rm_kb" "$(echo "$v" | field heap_peak_kb)" \
      "$(echo "$r" | field rss_peak_delta_kb)" "$(echo "$v" | field rss_peak_delta_kb)" "$dg")"
  done
done
log "## 14. efficiency (only now): the decoder's input (0.1.2): the same tokens on both sides (token i = (i * 7919 + 50257)"
log "##    mod n_vocab at position i); n = 1 is one decoding step's input, n = 226 config B's prompt. The reference: the two"
log "##    I32 inputs set, ggml_graph_plan and ggml_graph_compute of the three nodes as a ggml graph on the shipped CPU"
log "##    backend (in whisper these three are the head of the whole decoder graph, planned once with it: the plan and the"
log "##    threads' start are a per-graph cost this standalone graph pays for three nodes alone); voaice: Batch::prep_legacy"
log "##    + DecoderInput::run_batch into the caller's buffer, always one thread. One call is microseconds: wall = the mean"
log "##    over >= 1 s (>= 1000 calls), cpu = CPU us per call over the same loop; the output's digest compared"
log "$(printf '%-6s %8s | %10s %10s %7s | %10s %10s | %s' tokens ref_thr ref_us vo_us x cpu_ref cpu_vo digest)"
for n in 1 226; do
  v=$(target/release/voaice bench-decin "$model" "$n")
  for th in 1 "$nt"; do
    r=$(testing/oracle/bin/whisper_oracle --bench-decin "$model" "$th" "$n")
    rd=$(echo "$r" | field out_digest); vd=$(echo "$v" | field out_digest)
    [ "$rd" = "$vd" ] || { log "FAIL: decoder input digests differ in the benchmark ($rd vs $vd)"; exit 1; }
    rw=$(echo "$r" | field wall_us_per_call); vw=$(echo "$v" | field wall_us_per_call)
    log "$(printf '%-6s %8s | %10.4f %10.4f %6.2fx | %10s %10s | %s' "$n" "$th" "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
      "$(echo "$r" | field cpu_us_per_call)" "$(echo "$v" | field cpu_us_per_call)" "$vd")"
  done
done
log "voaice: $(target/release/voaice bench-decin "$model" 1 | sed -E 's/.*(heap_per_call_bytes [0-9]+ held_kb [0-9]+).*/\1/') (the f16 token table and the f32 positions, held outside the call; the reference's are the model's)"
log "## 15. efficiency (only now): the self-attention products and kv_self (0.1.3). The same input on both sides (x[i] ="
log "##    ((i * 7919) mod 2001 - 1000) / 256 for every row); n = 1 is one step (its K/V into cell 226), n = 226 config B's"
log "##    prompt (cells 0..225). block = decoder layer 0's twelve nodes: attn_ln (norm, * w, + b), Q + b, x KQscale, K x"
log "##    KQscale, V + b, the two CPYs into the f16 cache; call = all four layers' (each reading the same input: in whisper"
log "##    layers 1-3 read the previous layer's output, which is not this increment's) and the mask. The reference: the inputs"
log "##    set (x; for call the f32 mask, which whisper fills on the host outside the graph), ggml_graph_plan and"
log "##    ggml_graph_compute of exactly those nodes as a ggml graph on the shipped CPU backend; voaice: SelfAttn::layer_into"
log "##    per layer into KvSelf's cells (and for call KvSelf::mask_into, which builds the mask from the cells: more than"
log "##    the reference's measured cast), the slot found once before timing on both sides. wall = the mean over >= 1 s"
log "##    (>= 20 calls), cpu = CPU us per call over the same loop; the output's digest (Q, the K/V cells written, the f16"
log "##    mask) compared. voaice runs a step on one thread at any count (a spawn costs more than its products)"
log "$(printf '%-6s %-5s %3s | %10s %10s %7s | %10s %10s | %s' tokens what thr ref_us vo_us x cpu_ref cpu_vo digest)"
for n in 1 226; do
  for what in block call; do
    for th in 1 2 "$nt"; do
      r=$(testing/oracle/bin/whisper_oracle --bench-selfkv "$model" "$th" "$n" "$what")
      v=$(target/release/voaice bench-selfkv "$model" "$n" "$what" --threads "$th")
      rd=$(echo "$r" | field out_digest); vd=$(echo "$v" | field out_digest)
      [ "$rd" = "$vd" ] || { log "FAIL: self-attention digests differ in the benchmark ($rd vs $vd)"; exit 1; }
      rw=$(echo "$r" | field wall_us_per_call); vw=$(echo "$v" | field wall_us_per_call)
      log "$(printf '%-6s %-5s %3s | %10.3f %10.3f %6.2fx | %10s %10s | %s' "$n" "$what" "$th" "$rw" "$vw" "$(awk -v a="$rw" -v b="$vw" 'BEGIN{print a/b}')" \
        "$(echo "$r" | field cpu_us_per_call)" "$(echo "$v" | field cpu_us_per_call)" "$vd")"
    done
  done
done
log "voaice: $(target/release/voaice bench-selfkv "$model" 226 call | sed -E 's/.*(heap_per_call_bytes [0-9]+ held_kb [0-9]+).*/\1/') at one thread (the scratch per call; held: the three products' f16 weights and their widened f32 copy for every layer, the 3 MiB f16 cache); the reference's work buffer: $(testing/oracle/bin/whisper_oracle --bench-selfkv "$model" 1 226 call | field work_kb) KiB, its weights the model's, its cache the state's"
log "load after the measurements: $(cut -d' ' -f1-3 /proc/loadavg)"
log "## transcripts recorded (not yet reproduced by voaice.rs: the encoder is v0.1.0's, the cross K/V 0.1.1's, the decoder's input 0.1.2's, the self-attention products and kv_self 0.1.3's, the decoder is v0.2.0's)"
for d in .oracle/tiny.en/*/; do log "$(basename "$d"): $(tr '\n' ' ' < "$d/transcript.txt")"; done
log "GATE PASSED"
