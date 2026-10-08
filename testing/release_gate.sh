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
log "## transcripts recorded (not yet reproduced by voaice.rs: encoder and decoder are later stages)"
for d in .oracle/tiny.en/*/; do log "$(basename "$d"): $(tr '\n' ' ' < "$d/transcript.txt")"; done
log "GATE PASSED"
