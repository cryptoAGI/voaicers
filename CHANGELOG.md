# Changelog

## Unreleased — vclone

- `src/vclone.rs`: the vprint (`dvscope/1`) byte-identical to cryptoAGI/voaice `tools/vprint.py`. Checked on 2,000
  recorded metric sets (`testing/vclone/make_oracle.py`, which names the vprint.py it ran by sha256) and on all 10
  measured `.voaice` identities, every field (`tests/vclone.rs`).
- The forge log (`vclone-event/1`): hash-chained events for capture, measure, ref, consent, model, actor, prompt,
  skill, tool, language and forge, and `mintable()`.
- `src/json.rs` (an order-keeping JSON reader and writer) and `src/sha512.rs` (FIPS 180-4), in-crate: still zero
  dependencies.
- `voaice vclone check | print | log`. Plan and TODO: [docs/VCLONE.md](docs/VCLONE.md).

## 0.0.2 — 2026-10-08 — the mel, six times faster, still 0 ULP

**The log-mel front end without the waste: no allocation per frame, no padded copy of the audio, vectors only
across independent lanes, threads by frames — and every float operation, in its order, still the reference's.**
The oracle says so on every value of every input. Record: `testing/results/0.0.2.txt`.

### Measured (testing/release_gate.sh, the same laptop: 4 CPUs, load ≈ 7.6 during the run, so ±20 % is noise)
- `oracle_mel_bit_exact`: **2,316,640 / 2,316,640** values identical, **max 0 ULP**, 8 inputs (unchanged from 0.0.1).
- `oracle_mel_threads_bit_identical` (new): at 2, 3, 4 and 8 threads, **9,266,560** values identical to 1 thread
  and to the reference.
- `oracle_mel_discriminates_fused_fft`: the fused variant still differs in **25,242 of 328,000** values on JFK.
- The other three oracles unchanged (167/167 tensors, 51,864/51,864 tokens, PCM identical, guard refuses).
- Efficiency, only after the above; wall = best of 10, CPU per call over ≥ 1 s, heap = bytes live at the first
  call's peak; 0.0.1 **rebuilt from its tag and run in the same gate** (not its recorded numbers):

  | input | reference 1 thread | 0.0.1 | **0.0.2, 1 thread** | reference 4 threads | 0.0.2, 4 threads | CPU ms ref / 0.0.2 | heap KiB ref / 0.0.2 |
  |---|---|---|---|---|---|---|---|
  | JFK, 11 s | 60.1 ms | 62.9 ms | **9.5 ms** | 52.0 ms | 8.8 ms | 58.3 / **9.7** | 3,868 / **1,282** |
  | JFK ×3, 33 s | 198.8 ms | 167.8 ms | **24.9 ms** | 116.7 ms | 15.8 ms | 162.0 / **25.8** | 5,925 / **1,969** |
  | chirp, 2.5 s | 13.2 ms | 14.1 ms | **2.1 ms** | 8.9 ms | 1.6 ms | 13.7 / **2.2** | 3,068 / **1,016** |
  | 201 samples | 1.36 ms | 1.8 ms | **0.34 ms** | 1.52 ms | 0.34 ms | 1.6 / **0.38** | 2,836 / **938** |

  Geometric means over the 8 inputs: **6.10× faster than 0.0.1** and **6.78× faster than the reference** at one
  thread; at 4 threads 5.92× the reference at 4. CPU per call is about **one sixth** of the reference's, and the heap
  **one third**: what is left is the output itself (80 × 4,100 × 4 bytes = 1,281 KiB on JFK) — the reference also
  holds the audio padded with 30 s of zeros. Threads help less than they could here: the host was already running
  ≈ 7.6 runnable tasks on 4 CPUs, so 4 threads bought 1.1–1.6× on the long inputs; inputs under 128 frames per thread
  stay on one (their spawn costs more than it saves).

### Changed — `src/mel.rs`
- `MelPlan`: the invariants built once (window, each butterfly level's twiddles, the 25-point DFT's gathered
  `table[(k*n*16) % 400]`, each band's non-zero span of the filterbank); `MelPlan::run(samples, threads)` allocates
  only its output.
- The FFT recursion unrolled bottom-up and in place: the 16 leaf DFTs read the frame by stride, the butterflies of
  each level work on [even | odd] halves exactly where the recursion would have put them, with the spectrum split
  into re and im so butterfly k is lane k. The DFT computes its 25 outputs side by side (lane k accumulates over n in
  the reference's order). AVX2 code path chosen at run time (`is_x86_feature_detected!`), mul and add only.
  Tried and measured slower, so not kept: lanes across the 16 leaves instead of across k.
- Band sums skip the all-zero groups outside each band's span (an exact +0), except in a frame with a non-finite
  power, which takes the full loop (`inf * 0 = NaN`).
- No padded copy: a frame reads the reflected head and the audio directly.
- Threads (`log_mel_spectrogram_threads`, `MelPlan::run`): contiguous runs of frames, each frame whole in one thread;
  the clamp pass split by range. Default 1.
- Fixed: `log10(std::max(sum, 1e-10))` now keeps C++'s `max` — a NaN sum stays NaN, where Rust's `f64::max` gave
  1e-10 (only reachable with non-finite samples; unit-tested).

### Added
- `src/measure.rs`: CPU seconds (`/proc/self/stat`), RSS and peak RSS (`/proc/self/status`, peak reset through
  `/proc/self/clear_refs`), and the `bench` loop — no crate, no libc binding.
- `voaice bench-mel <model> <wav> [--threads N]` (with a counting allocator for the heap peak); `voaice mel --threads N`.
- `whisper_oracle --bench-mel <model> <wav> <threads>`: the reference measured the same way (operator new counted).
- Gate step 5 rewritten: 1 and nproc threads, both sides, plus 0.0.1 rebuilt from its tag; the record keeps only
  repo-relative paths. `testing/oracle/build.sh | head -1` replaced by a log file (under `pipefail` the head could
  SIGPIPE the build script and fail the gate).
- Unit test `same_bits_as_the_0_0_1_port_at_every_thread_count`: 0.0.1's port kept verbatim (test-only) as a second
  witness at 1, 2, 3, 4, 7 threads and in the fused variant.
- docs/ROADMAP.md: the first decade (0.0.3 the streaming Ogg/Opus reader … v0.1.0 the encoder), the milestones to
  v1.0.0, and the Opus efficiency thread.

## 0.0.1 — 2026-10-07 — the oracle, the loader, the log-mel front end (stages 0–2)

**voaice.rs begins: the reference pinned, an oracle that reads the shipped whisper.cpp library, a model loader whose
167 tensors are byte-identical to what whisper's loader holds, and a log-mel front end bit-exact on every value of
every test input.** It does not transcribe yet. Record: `testing/results/0.0.1.txt`.

### Measured (testing/release_gate.sh, this laptop: 4 CPUs, AVX2+FMA, gcc 11.4.0, glibc 2.35)
- `oracle_model_hparams_tensors_vocab_filters`: 11 hparams equal; **167 / 167 tensors** equal in name, type, shape,
  byte count and the sha256 of the bytes in the loader's memory (77,110,272 bytes); filterbank 80×201 bit-exact;
  **51,864 / 51,864** token strings equal as `whisper_token_to_str` shows them (token 188 is the byte `0x00`, which
  that C API cannot show; voaice holds it).
- `oracle_mel_bit_exact`: **2,316,640 / 2,316,640** f32 values identical, **max 0 ULP**, over 8 inputs: JFK (11 s), JFK
  ×3 (33 s, past one 30-s window), a chirp, full-scale noise with ±32767/−32768, silence, 0.3 s, a length off the hop
  (12,345), and the 201-sample minimum. `n_len` and `n_len_org` equal on all 8.
- `oracle_mel_discriminates_fused_fft`: the same pipeline with the FFT's multiply-adds fused differs in **25,242 of
  328,000** values on JFK — the oracle can tell float orders apart.
- `oracle_pcm_input_identical`: the samples voaice reads from each WAV are the f32 bits fed to whisper.
- `guard_refuses_a_modified_model`: one flipped bit in `decoder.token_embedding.weight` still parses and is refused by
  the pin, with the reason.
- Reference determinism, recorded: the mel at 1 and 4 threads is bit-identical (all 8 inputs); `whisper_full` at 1
  thread is identical run to run (all 8); **at 4 threads against 1 the token ids and text are the same but each
  token's `p` differs, and on JFK the token timestamps too** — the decoder's arithmetic depends on the thread count,
  so the transcript oracle is pinned at 1 thread (production's whisper-cli runs min(4, cores) threads).
- Speed, only after the above (mel, one thread each, best of 5, the laptop at load 9.6 so ±20 % is noise): about
  **parity** — voaice 0.82–1.22× the reference's time per input (JFK 58.7 ms against 57.7; JFK ×3 198.1 against
  175.8; chirp 14.4 against 16.1). Nothing has been optimized: this is the exact port, allocations and all.

### Added
- `upstream/PIN`: whisper.cpp `080bbbe85230f624f0b52127f1ae1218247989f9` (`v1.9.1-154-g080bbbe8`), ggml 0.16.0 — no
  release tag bundles 0.16.0 — and `ggml-tiny.en.bin` by size and sha256.
- `testing/oracle/`: `build.sh` (reference at the pin, CPU-only shared build, refuses another commit or ggml version),
  `layout_probe.cpp` (offsets of the internals the API does not expose, from the pinned source), `whisper_oracle.cpp`
  (records model, vocab, filterbank, mel at 1 and 4 threads with its time, and greedy transcripts with token ids,
  timestamps and `p` bits; self-checks every probed offset against a public getter).
- `testing/make_audio.py` + `testing/pins/audio.sha256`: 8 deterministic test WAVs (JFK from whisper.cpp's samples,
  the rest synthetic), pinned.
- `src/sha256.rs` (FIPS 180-4 vectors), `src/model.rs` (format, guard, the tensor set whisper creates for the hparams,
  extra-token names including `[_LANG_xx]`), `src/wav.rs`, `src/mel.rs`, `src/main.rs` (`voaice info | mel | version`).
- `tests/oracle.rs`, `testing/release_gate.sh`, README, TODO (stage 3–6 plan).

### Fixed (found on the way)
- `sha256::update` reset a partial block's length when its input fitted in the buffer (a hang on any input that was
  not a multiple of 64 bytes: the FIPS-vector test hung).
- The oracle's first layout self-check was wrong (`whisper_n_len_from_state` returns `n_len_org`); the check refused
  to run, which is what it is for.
