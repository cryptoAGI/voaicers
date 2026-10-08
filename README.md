<h1 align="center">voaice.rs</h1>

<p align="center">
  <b>Speech to text in zero-dependency Rust: bit-exact against whisper.cpp first, then faster.</b><br>
  Built the way <a href="https://github.com/cryptoAGI/bankml">bankml</a> was built against llama.cpp.<br><br>
  <a href="https://github.com/Professor-Codephreak">Professor Codephreak</a> &middot; Gregory L. Magnusson &middot; <a href="https://github.com/cryptoAGI">cryptoAGI</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/dependencies-0-56D364?style=flat-square" alt="zero dependencies">
  <img src="https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-2563EB?style=flat-square" alt="MIT OR Apache-2.0">
  <img src="https://img.shields.io/badge/log--mel-bit--exact%20vs%20whisper.cpp%20(ggml%200.16.0)-39D3C7?style=flat-square" alt="log-mel bit-exact">
  <img src="https://img.shields.io/badge/status-0.0.1%20%C2%B7%20loader%20%2B%20front%20end%3B%20no%20transcript%20yet-F59E0B?style=flat-square" alt="status">
</p>

---

## What this is, and what it is not yet

mindX's production speech-to-text is whisper.cpp built against ggml 0.16.0, running `ggml-tiny.en.bin` and
`ggml-base.en.bin`. voaice.rs is a rewrite of it in Rust with **no crates at all**, held to one rule from bankml:
*a result counts only when an oracle has checked it, and the oracle is the reference's own compiled library run on
the same input, compared by bit pattern.* Speed is measured only after that.

**0.0.1 does not transcribe.** It is the first three stages:

| stage | what | oracle result (this machine, 2026-10-07) |
|---|---|---|
| 0 | the oracle harness: links the pinned `libwhisper.so` and records what it computes | built; self-checking (below) |
| 1 | the model loader + sha256 guard | **167 / 167 tensors** byte-identical to what whisper's loader holds (sha256 each, 77,110,272 bytes); hparams, filterbank (80×201, bit-exact), **51,864 / 51,864** vocab strings |
| 2 | the log-mel front end | **2,316,640 / 2,316,640** f32 values bit-exact over 8 inputs, **max 0 ULP** |

Speed, measured only after the oracles passed in the same gate run: the mel is at about **parity** with the
reference, one thread each (JFK 11 s: 58.7 ms against 57.7 ms; the laptop was loaded, so ±20 % is noise). Nothing
is optimized yet. Full record: [`testing/results/0.0.1.txt`](testing/results/0.0.1.txt).

**A determinism note on the reference itself:** whisper.cpp's transcript depends on its thread count. At 1 thread it
is identical run to run; at 4 threads the token ids and text stay the same but every token's probability differs in
its bits, and on JFK the token timestamps move. The transcript oracle is therefore pinned at 1 thread, and a
bit-exact transcript will mean "bit-exact at a stated thread count".

The encoder, decoder and the transcript loop are not written. Their plan, with the ggml ops, their float order and
how the oracle will check each, is in [TODO.md](TODO.md).

## The reference

[`upstream/PIN`](upstream/PIN) pins it by commit, because **no whisper.cpp release tag bundles ggml 0.16.0**
(v1.9.1 has 0.15.1, v1.9.2 has 0.18.1). ggml 0.16.0 was on master from 2026-07-10 to 2026-07-31; production was
built on 2026-07-22 00:13 UTC from master, whose head then was **`080bbbe85230f624f0b52127f1ae1218247989f9`**
(`v1.9.1-154-g080bbbe8`, `WHISPER_VERSION` 1.9.1, ggml 0.16.0). **Confirmed against production** (read-only,
2026-10-07): production's `libwhisper.so.1.9.1` + ggml 0.16.0 was built then, and the next master commit is dated
2026-07-28; production's `libwhisper` has no FMA instructions either, so the mel's bit-exactness holds there. Details:
[docs/REFERENCE.md](docs/REFERENCE.md).

Built CPU-only: `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON`, gcc 11.4.0, glibc 2.35,
an AVX2+FMA host. The model: `ggml-tiny.en.bin`, 77,704,715 bytes (production's size),
sha256 `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f` (equal to Hugging Face's `X-Linked-ETag`).

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the modules, the oracle contract, **how to add a stage**, and every place in voaice that consumes speech-to-text today — the integration points voaice.rs is built to replace |
| [docs/oracles.md](docs/oracles.md) | every oracle, what it compares, its result, and how it sees what whisper.cpp's API hides |
| [docs/REFERENCE.md](docs/REFERENCE.md) | the pinned whisper.cpp, why a commit and not a tag, and its confirmation against production |
| [docs/ROADMAP.md](docs/ROADMAP.md) | the listening half to 1.0, and the speaking half — inspired by Kitten TTS v1 and v2 |
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
- Threads do not change the bits (each frame is computed whole by one thread): 1 vs 4 identical on all 8 inputs.
- The input: whisper-cli's miniaudio converts s16 with the literal `0.00003051757812f`, which is exactly 2⁻¹⁵, so
  `s / 32768.0` is the same; the oracle checks the samples voaice reads equal those fed to whisper, bit for bit.

## Use

```sh
cargo build --release && cargo test --release          # unit tests (no model needed)
testing/release_gate.sh                                # the gate: reference, pins, record, compare, then speed
target/release/voaice info models/ggml-tiny.en.bin     # refuses any file that is not pinned, and says why
target/release/voaice mel  models/ggml-tiny.en.bin in.wav [out.f32]
```

Input: 16-bit PCM, mono, 16 kHz WAV; anything else is refused, not converted (resampling is a later stage).
Disk: the reference checkout and build are about 160 MB in `upstream/` (gitignored), the model 78 MB in `models/`.

## Layout

```
Cargo.toml  rust-toolchain.toml      zero dependencies; Rust 1.99.0 pinned like bankml
src/        sha256.rs model.rs wav.rs mel.rs lib.rs main.rs
tests/oracle.rs                      the oracle comparisons (#[ignore]: need the model and a recorded oracle)
testing/oracle/                      build.sh, layout_probe.cpp, whisper_oracle.cpp
testing/make_audio.py                the 8 test WAVs, pinned in testing/pins/audio.sha256
testing/release_gate.sh              the gate → testing/results/<version>.txt
upstream/PIN                         the reference (commit, ggml version, build, model sha256)
docs/                                ARCHITECTURE · oracles · REFERENCE · ROADMAP
```

Licence: MIT OR Apache-2.0. whisper.cpp (MIT) is used only as the oracle and is not redistributed.

## Where it lives

- **This repository:** [cryptoAGI/voaicers](https://github.com/cryptoAGI/voaicers) — voaice.rs, the speech-to-text half of voaice in Rust.
- **[cryptoAGI/voaice](https://github.com/cryptoAGI/voaice):** what a voice is, written down — `.voaice` identities, the 18-decimal vprint, the pronunciation table every engine speaks through.
- **[cryptoAGI/bankml](https://github.com/cryptoAGI/bankml):** the method this follows — a zero-dependency Rust runtime bit-exact against llama.cpp's compiled library, then faster. Its [thesis](https://github.com/cryptoAGI/bankml/blob/main/docs/thesis.md) is the argument for exactness first.
- **[PYTHAI/voaice](https://huggingface.co/PYTHAI/voaice) on Hugging Face:** the voice library — 70 open-licensed Piper voices, for anyone to use.
- **The reference:** [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), pinned in [`upstream/PIN`](upstream/PIN) at the commit mindX production runs. voaice.rs reproduces its compiled output; credit for the model and its design belongs to OpenAI Whisper and to whisper.cpp's authors.
