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
  <img src="https://img.shields.io/badge/log--mel-bit--exact%20vs%20whisper.cpp%20(ggml%200.16.0)-39D3C7?style=flat-square" alt="log-mel bit-exact">
  <img src="https://img.shields.io/badge/status-0.0.5%20%C2%B7%20loader%2C%20front%20end%2C%20f16%20%2B%20GELU%2C%20resampler%20bit--exact%2C%20Ogg%2FOpus%20reader%3B%20no%20transcript%20yet-F59E0B?style=flat-square" alt="status">
  <a href="https://github.com/cryptoAGI/voaicers/releases/latest"><img src="https://img.shields.io/github/v/release/cryptoAGI/voaicers?style=flat-square&label=release&color=0ECB81" alt="latest release"></a>
</p>

---

## What this is, and what it is not yet

mindX's production speech-to-text is whisper.cpp built against ggml 0.16.0, running `ggml-tiny.en.bin` and
`ggml-base.en.bin`. voaice.rs is a rewrite of it in Rust with **no crates at all**, held to one rule from bankml:
*a result counts only when an oracle has checked it, and the oracle is the reference's own compiled library run on
the same input, compared by bit pattern.* Speed is measured only after that.

**0.0.5 does not transcribe yet.** It is the first three stages, the front end optimized (0.0.2) with its bits unchanged,
the first two encoder kernels (0.0.3): the f32 ↔ f16 conversions and GELU, (0.0.4) the streaming Ogg/Opus reader
that will bring `.opus` input to them without a WAV on disk, and (0.0.5) the audio reader whisper-cli itself runs:
any WAV to 16 kHz mono f32 through miniaudio's conversions, mixdown and resampler, bit for bit.

| stage | what | oracle result (this machine, 2026-10-07) |
|---|---|---|
| 0 | the oracle harness: links the pinned `libwhisper.so` and records what it computes | built; self-checking (below) |
| 1 | the model loader + sha256 guard | **167 / 167 tensors** byte-identical to what whisper's loader holds (sha256 each, 77,110,272 bytes); hparams, filterbank (80×201, bit-exact), **51,864 / 51,864** vocab strings |
| 2 | the log-mel front end — allocation-free, threaded (0.0.2) | **2,316,640 / 2,316,640** f32 values bit-exact over 8 inputs, **max 0 ULP**; at 2, 3, 4 and 8 threads identical to 1 thread (9,266,560 values); the fused-FFT discriminator still rejected (25,242 values differ) |
| 3a | f32 ↔ f16 as ggml-cpu converts (0.0.3): the portable bit trick, the F16C row, f16 → f32 | **65,536 / 65,536** f16 patterns four ways; **all 4,294,967,296 f32 patterns** three ways (1,429,656 boundary values compared one by one, the rest by per-chunk digest); round-half-away rejected |
| 3b | GELU (0.0.3): `ggml_table_gelu_f16` and the op with its ±10 clamps | **65,536 / 65,536** table entries; the op on 1,495,192 values and on all 2³² f32 patterns; the unfused GELU rejected (it differs in one entry) |
| 4 | the streaming Ogg/Opus reader (0.0.4): pages, Ogg's CRC-32, lacing and continuation, `OpusHead` / `OpusTags`, granules and pre-skip → the exact duration, one page in memory | against **opus-tools 0.2, libopus 1.4 and libogg 1.3.5 on production**: **35 / 35** files on 18 checks each — the duration equals **opusdec's sample count** on every file, every page's granule and every packet's samples equal libogg's and libopus's; **21 / 21** corrupted files refused by name; pre-skip added, no end trim, and zlib's CRC (0 / 427 pages) all caught |
| 5 | any WAV → 16 kHz mono f32 as whisper-cli reads it (0.0.5): dr_wav's u8 / s16 / s24 / s32 / f32 conversions, miniaudio's mono average, its linear resampler with the order-4 low-pass, the length rule and its zero tail — streamed, any chunking | against **whisper-cli's own `libcommon.a`** (`read_audio_data`, miniaudio 0.11.24): **55 / 55** files, **1,955,875** samples bit-identical across 8–48 kHz, 1 / 2 / 6 channels, every format; random chunking identical; low-pass order 2 / 6, mixdown `L + R` / `L`, and the length without its extra frame all caught |

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

**A determinism note on the reference itself:** whisper.cpp's transcript depends on its thread count. At 1 thread it
is identical run to run; at 4 threads the token ids and text stay the same but every token's probability differs in
its bits, and on JFK the token timestamps move. The transcript oracle is therefore pinned at 1 thread, and a
bit-exact transcript will mean "bit-exact at a stated thread count".

The encoder, decoder and the transcript loop are not written. Their plan, with the ggml ops, their float order and
how the oracle will check each, is in [TODO.md](TODO.md).

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
```

Input: `voaice mel` still takes 16-bit PCM, mono, 16 kHz WAV; `voaice resample` (0.0.5) takes any PCM 8/16/24/32-bit
or float 32-bit WAV at any rate and channel count up to 254 and gives the samples whisper-cli would (FLAC, MP3, Vorbis,
A-law, µ-law, ADPCM, f64 are refused by name). `.opus` files are read and measured (0.0.4) but not yet decoded.
Disk: the reference checkout and build are about 160 MB in `upstream/` (gitignored), the model 78 MB in `models/`.

## Layout

```
Cargo.toml  rust-toolchain.toml      zero dependencies; Rust 1.99.0 pinned like bankml
src/        sha256.rs model.rs wav.rs mel.rs f16.rs gelu.rs ogg.rs resample.rs measure.rs lib.rs main.rs
tests/oracle.rs                      the oracle comparisons (#[ignore]: need the model and a recorded oracle)
testing/oracle/                      build.sh, layout_probe.cpp, whisper_oracle.cpp, resample_oracle.cpp (0.0.5)
tests/resample.rs                    (0.0.5) the resampler against whisper-cli's libcommon.a (#[ignore]: needs the record)
testing/make_resample_audio.py       (0.0.5) the 55-file resampler corpus, pinned in testing/pins/resample.sha256
testing/resample/NOTES.md            (0.0.5) what read_audio_data does, read from the pinned source line by line
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
- [ ] **0.0.6–0.0.9:** conv1, conv2 + positions, layer norm, the f16 dot in AVX2 lane order — one per step, each
  bit-exact ([plan](docs/ROADMAP.md)).
- [ ] **Stage 0b:** pin `ggml-base.en.bin` (production's default model). Record the VPS's ISA so the oracle
  reproduces production's native ggml-cpu kernels (Zen 3, AVX2 + FMA).
- [ ] **Stage 3, the encoder (0.1.0):** ~~the GELU f16 table, f32↔f16 conversion~~ (0.0.3), `vec_dot_f16` in AVX2
  lane order, conv1 and conv2 through im2col, layer norm, flash attention, four blocks, then `embd_enc` bit-exact. Each intermediate is observed through ggml's scheduler callback.
- [ ] **Stage 4, the decoder (0.2.0):** cross-attention, the f16 KV cache, and all 51,864 logits bit-exact per
  step through `whisper_get_logits_from_state`.
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
