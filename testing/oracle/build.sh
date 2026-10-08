#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Build the pinned whisper.cpp (if needed), the layout probe, and the oracle harness.
#   testing/oracle/build.sh            -> testing/oracle/bin/whisper_oracle
# The reference is upstream/whisper.cpp at the commit in upstream/PIN, CPU only, Release, shared libraries, GGML_NATIVE
# (as mindX production builds it). The oracle links the shipped libwhisper.so / libggml*.so; the probe only computes
# struct offsets (see layout_probe.cpp).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
up=$root/upstream/whisper.cpp
pin_commit=$(awk -F'\t' '$1=="commit"{print $2}' "$root/upstream/PIN")
pin_url=$(awk -F'\t' '$1=="url"{print $2}' "$root/upstream/PIN")

if [ ! -d "$up/.git" ]; then
  git clone -q --filter=blob:none "$pin_url" "$up"
fi
have=$(git -C "$up" rev-parse HEAD)
if [ "$have" != "$pin_commit" ]; then
  git -C "$up" fetch -q origin "$pin_commit" 2>/dev/null || true
  git -C "$up" checkout -q "$pin_commit"
fi
[ "$(git -C "$up" rev-parse HEAD)" = "$pin_commit" ] || { echo "upstream is not at the pinned commit" >&2; exit 1; }
# the ggml version the pin promises
gv=$(grep -E 'set\(GGML_VERSION_(MAJOR|MINOR|PATCH) ' "$up/ggml/CMakeLists.txt" | grep -oE '[0-9]+\)' | tr -d ')' | paste -sd.)
[ "$gv" = "$(awk -F'\t' '$1=="ggml"{print $2}' "$root/upstream/PIN")" ] || { echo "bundled ggml is $gv, not the pinned version" >&2; exit 1; }

if [ ! -f "$up/build/bin/libwhisper.so" ]; then
  cmake -S "$up" -B "$up/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DWHISPER_SDL2=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=ON >/dev/null
  cmake --build "$up/build" -j"$(nproc)" --target whisper-cli >/dev/null
fi

out=$root/testing/oracle/bin
mkdir -p "$out"
inc="-I$up/include -I$up/ggml/include -I$up/src"
lib="-L$up/build/bin -lwhisper -lggml -lggml-base -lggml-cpu -Wl,-rpath,$up/build/bin"
defs="-DGGML_SHARED -DWHISPER_SHARED -DGGML_USE_CPU"

# 1. the probe: the same compiler and language standard as libwhisper; it is linked only so it can run
g++ -std=gnu++17 -O0 -w $defs -DWHISPER_VERSION=\"1.9.1\" $inc "$root/testing/oracle/layout_probe.cpp" -o "$out/layout_probe" $lib -lpthread
"$out/layout_probe" > "$out/layout.h"
# 2. the oracle
g++ -std=gnu++17 -O2 -Wall $defs $inc -I"$out" "$root/testing/oracle/whisper_oracle.cpp" -o "$out/whisper_oracle" $lib
echo "built ${out#"$root"/}/whisper_oracle (reference $pin_commit, ggml $gv)"   # repo-relative: a record is not a map of this machine
cat "$out/layout.h"
