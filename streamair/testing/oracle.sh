#!/usr/bin/env bash
# streamair container oracle: libopus 1.4's own tools (opusinfo, opusdec from opus-tools 0.2) must read every
# generated file without a warning and decode exactly the samples written, pre-skip and end trim included.
# usage: testing/oracle.sh [host]   (host defaults to local; the pinned reference is mindX production)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
host=${1:-}
work=$(mktemp -d)
cargo build --release -q --manifest-path "$root/Cargo.toml"
for t in 0.001 1 2.5 7.3333 60; do "$root/target/release/streamair" silence "$t" "$work/s_$t.opus" >/dev/null; done
run() { if [ -n "$host" ]; then ssh -o BatchMode=yes "$host" "$@"; else bash -c "$*"; fi; }
remote=/tmp/streamair_oracle
run "mkdir -p $remote"
if [ -n "$host" ]; then scp -q "$work"/*.opus "$host:$remote/"; else cp "$work"/*.opus "$remote/"; fi
fail=0
for t in 0.001 1 2.5 7.3333 60; do
  want=$(python3 -c "print(round($t*48000))")
  f=$remote/s_$t.opus
  warn=$(run "opusinfo $f 2>&1 | grep -ciE 'warn|error|hole|corrupt' || true")
  got=$(run "opusdec --quiet --rate 48000 $f $f.wav >/dev/null 2>&1; python3 -c 'import wave,sys;print(wave.open(sys.argv[1]).getnframes())' $f.wav")
  if [ "$warn" = 0 ] && [ "$got" = "$want" ]; then echo "PASS $t s: $got samples"; else echo "FAIL $t s: want $want got $got, $warn warnings"; fail=1; fi
done
rm -rf "$work"
exit $fail
