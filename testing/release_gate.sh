#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The release gate: nothing is a result until the oracle has checked it in the same run.
#   testing/release_gate.sh            -> testing/results/<version>.txt
# Steps, each must pass:
#   1. the reference: upstream/whisper.cpp at upstream/PIN's commit, built; the oracle harness built against it
#   2. the inputs: the model's sha256 equals its pin; the test audio equals testing/pins/audio.sha256
#   3. the record: the shipped library's model view, mel and transcripts (testing/oracle/bin/whisper_oracle)
#   4. voaice.rs: cargo build, unit tests, clippy, then the oracle comparisons (bit patterns, never tolerances)
#   5. efficiency, only after 4 passed: the mel's wall time (best of 10), CPU time per call, heap and RSS peaks, voaice
#      against the reference (each in a fresh process: `voaice bench-mel`, `whisper_oracle --bench-mel`), at 1 thread
#      and at N = nproc threads, and against 0.0.1's code path rebuilt from its tag in the same run
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

log "## 3. record (the shipped libwhisper.so, in process)"
rm -rf .oracle/tiny.en
testing/oracle/bin/whisper_oracle "$model" .oracle/tiny.en .audio/*.wav 2>&1 | tee -a "$out"

log "## 4. voaice.rs"
cargo build --release 2>&1 | tail -1 | tee -a "$out"
cargo clippy --release --all-targets -q -- -D warnings 2>&1 | tee -a "$out"
cargo test --release 2>&1 | grep -E "^test result" | tee -a "$out"
cargo test --release --test oracle -- --ignored --nocapture --test-threads=1 2>&1 \
  | grep -vE "^\s*(Compiling|Finished|Running)|^$|^running" | tee -a "$out"
grep -q "test result: ok. 6 passed" "$out" || { log "FAIL: the oracle comparisons did not all pass"; exit 1; }

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
log "## transcripts recorded (not yet reproduced by voaice.rs: encoder and decoder are later stages)"
for d in .oracle/tiny.en/*/; do log "$(basename "$d"): $(tr '\n' ' ' < "$d/transcript.txt")"; done
log "GATE PASSED"
