# Changelog

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
