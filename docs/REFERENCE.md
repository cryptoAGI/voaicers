# The reference — what voaice.rs is measured against, and why that commit

voaice.rs is bit-exact against **the whisper.cpp that mindX production runs**, not against whatever whisper.cpp is
newest. Matching production is what lets voaice.rs replace it call site by call site.

## The pin

| | |
|---|---|
| whisper.cpp | [`080bbbe85230f624f0b52127f1ae1218247989f9`](https://github.com/ggml-org/whisper.cpp/commit/080bbbe85230f624f0b52127f1ae1218247989f9) (`v1.9.1-154-g080bbbe8`), `WHISPER_VERSION` 1.9.1 |
| ggml | 0.16.0 |
| model | `ggml-tiny.en.bin`, 77,704,715 bytes, sha256 `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f` (equal to Hugging Face's `X-Linked-ETag`) |
| local build | CPU-only, `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON`, gcc 11.4.0, glibc 2.35 |

**Why a commit and not a tag.** No whisper.cpp release tag bundles ggml 0.16.0: v1.9.0 and v1.9.1 ship 0.15.1 and
v1.9.2 jumps to 0.18.1. ggml 0.16.0 was on master from 2026-07-10 to 2026-07-31.

## Confirmed against production (2026-10-07, read-only)

- Production's library is `/opt/whisper.cpp/build/bin/libwhisper.so.1.9.1` with `libggml-*.so.0.16.0`, built
  **2026-07-22 00:13 UTC**.
- With full history, the first master commit after `080bbbe8` (2026-07-11) is `97c56f1d` (2026-07-28), so master's
  head at the production build was this pin.
- Production's `libwhisper` contains **zero FMA instructions** (`objdump -d | grep -c vfmadd` = 0), like the local
  build, so the mel's bit-exactness holds on production's library too.
- Production's `libggml-cpu` is a native build for its CPU (AMD EPYC 7543P, Zen 3): 652 `vfmadd` and AVX2 code. The
  encoder and decoder run there, so their oracle must match **that** build's kernels, not only a laptop's native
  build. See [TODO.md](../TODO.md).

## Credit

The model and its design are OpenAI's [Whisper](https://github.com/openai/whisper) (MIT). The reference
implementation is [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), by Georgi Gerganov and its
contributors. voaice.rs uses whisper.cpp only as the oracle; it is not redistributed.
