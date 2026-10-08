#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The release gate: nothing is a result until the oracle has checked it in the same run.
#   testing/release_gate.sh            -> testing/results/<version>.txt
# Steps, each must pass:
#   1. the reference: upstream/whisper.cpp at upstream/PIN's commit, built; the oracle harness built against it
#   2. the inputs: the model's sha256 equals its pin; the test audio equals testing/pins/audio.sha256
#   3. the record: the shipped library's model view, mel and transcripts (testing/oracle/bin/whisper_oracle)
#   4. voaice.rs: cargo build, unit tests, clippy, then the oracle comparisons (bit patterns, never tolerances)
#   5. speed, only after 4 passed: voaice's mel time against the reference's, same input, one thread each
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
testing/oracle/build.sh | head -1 | tee -a "$out"

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
grep -q "test result: ok. 5 passed" "$out" || { log "FAIL: the oracle comparisons did not all pass"; exit 1; }

log "## 5. speed (only now): log-mel, one thread, best of 5"
log "wav          samples   reference ms   voaice ms"
for d in .oracle/tiny.en/*/; do
  w=$(basename "$d")
  ref=$(awk -F'\t' '$1=="reference_ms_1_thread_best_of_5"{print $2}' "$d/mel.tsv")
  best=""
  for _ in 1 2 3 4 5; do
    ms=$(target/release/voaice mel "$model" ".audio/$w.wav" | sed -E 's/.* ([0-9.]+) ms$/\1/')
    best=$(awk -v a="$best" -v b="$ms" 'BEGIN{print (a=="" || b<a) ? b : a}')
  done
  n=$(awk -F'\t' '$1=="n_samples"{print $2}' "$d/mel.tsv")
  log "$(printf '%-11s %8s   %12s   %9s' "$w" "$n" "$ref" "$best")"
done
log "## transcripts recorded (not yet reproduced by voaice.rs: encoder and decoder are later stages)"
for d in .oracle/tiny.en/*/; do log "$(basename "$d"): $(tr '\n' ' ' < "$d/transcript.txt")"; done
log "GATE PASSED"
