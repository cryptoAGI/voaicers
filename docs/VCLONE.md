# vclone — voaice's voice identities in Rust

`src/vclone.rs` is the Rust side of vCLONE. A **voice identity** is a `.voaice` file: what a voice is rendered
with, and what it measures. The **vprint** is the measurement's fingerprint. The **forge log** records how a
person's voice became part of a `.persona`. This page records what is built, what is checked, and what is next.

## Built

| | what | oracle |
|---|---|---|
| **vprint (`dvscope/1`)** | eight metrics → `floor(v·1e18)` with 1e18 as an f64 → canonical JSON in a fixed order with no whitespace → SHA-256, SHA-512, and the digest as a uint256 | **byte-identical** to cryptoAGI/voaice `tools/vprint.py` on 2,000 recorded metric sets (zero, subnormals, NaN and ±inf, negatives, values that round differently under the f64 multiply) and on every shipped identity |
| **`.voaice` check** | `voaice vclone check <file>…` recomputes each identity's print from `measured.metrics` and compares **every** stored field: hash, hash512, short, uint256, version, each metric's wei and decimal, canonical | **10 / 10** measured identities verify (NEURAL, JAIMLA, OVERLORD, LEADER ×2, and the five eSpeak NG renders). `vclone.voaice` reports *unmeasured*, its honest state |
| **forge log (`vclone-event/1`)** | an append-only JSONL log of one persona's forge. Each event is canonical JSON chained to the previous one by SHA-256 (genesis = 64 zeros). Kinds: `capture`, `measure`, `ref`, `consent`, `model`, `actor`, `prompt`, `skill`, `tool`, `language`, `forge` | unit tests: a round trip is identical; an edited event, a dropped event, or a reordered sequence is refused |
| **`mintable(log)`** | whether the voice in a log may become an iNFT, and every reason when it may not | unit tests |
| **JSON, SHA-512** | in-crate (zero dependencies): an order-keeping reader and writer, and FIPS 180-4 SHA-512 | FIPS vectors; JSON.stringify's escaping |

The mint rule applies the voice cards' rule ([PYTHAI/voaice cards/](https://huggingface.co/PYTHAI/voaice/tree/main/cards))
to a voice that belongs to a person. A voice may be minted only when all of these hold:
- it was measured;
- the speaker gave consent with scope `mint`, naming the reference that was cloned;
- the consent was not revoked later in the log;
- the cloning engine is recorded as cleared for the use (`licence_cleared` on the `ref` event).

Until the licence of pocket_tts (the cloning model inside audio.cpp) is recorded, **no cloned voice is mintable**.

```sh
voaice vclone check tests/fixtures/voaice/*.voaice        # recompute and compare every identity
voaice vclone print 0.166 96.8 2618 5840 0.145 2831 0 32.9 # the print of eight values
voaice vclone log forge.jsonl                             # verify a chain; say whether it is mintable
python3 testing/vclone/make_oracle.py ~/voaice/tools/vprint.py 2000 > tests/fixtures/vprint_oracle.jsonl
```

## Why a log, and where the events come from

ollywoo's `forgePersona()` (in the DeltaVerse) downloads a `.persona.json` that **states** its history: face, voice
print, ref, the model facet pending. The forge log turns that statement into a record. Every step that made the
persona is an event, the chain makes the record tamper-evident, and the `.persona` becomes a **projection** of the
log, which can be rebuilt by replaying it. That is the same doctrine mindX uses for its own data: the data
directory holds projections, rebuilt from the logs.

The events map onto what ollywoo already does:

| event | produced today by |
|---|---|
| `capture` | ollywoo "⊙ capture voice": MediaRecorder, then the browser's WAV encoder (`toMonoWav`, 24 kHz) |
| `measure` | `POST /voicey/measure`: the forensic 18-decimal print. **Not yet `dvscope/1`**, see below |
| `ref` | `POST /voicey/ref` → `refs/<sha256-32>.wav` |
| `consent` | **nothing yet**: the forge has no consent step |
| `model` · `actor` · `prompt` · `skill` · `tool` · `language` | the `.persona`'s facets: `model` (pending imprint, from mindXtrain), `actor` (face, theme), `mind.prompt`, `agent.skills`, tools, the pronunciation table's language |
| `forge` | `forgePersona()`: the sealed bundle |

## TODO

### vclone.rs
- [ ] **The two prints.** `/voicey/measure` returns voaice's *forensic* print (six measures, `features`, `precision`,
      `integrity`, `snr`). The `.voaice` files and the cards carry the *dvscope/1* print (eight metrics). Port the
      forensic one as well, with its own oracle (the voaice server on the same WAV). Then record the mapping between
      the two prints, or decide that one of them is the identity.
- [ ] **Measure in Rust.** The eight metrics from audio (`rms`, the spectral metrics over a 2048-point FFT with hop
      512 and a Hann window, frames averaged, a silence floor of 0.01 RMS): the `params` block of a `.voaice` file.
      Oracle: vprint.py's `measure()` on the same frames. Reuse the mel stage's exact-FFT discipline.
- [ ] **`compare`.** Port vprint.py's `compare()`, including its refusal when parameters differ.
- [ ] **Write `.voaice`.** Fill `vclone.voaice` from a capture plus a measure, and round-trip it byte for byte
      against `tools/voaice.py`.
- [ ] **Events from ollywoo.** Have `forgePersona()` append events to a log as it goes (capture → measure → ref →
      consent → facets → forge), download the log beside the `.persona`, and stamp the bundle with the head event's
      id. That change lives in the private DeltaVerse.
- [ ] **A consent step in the forge.** Who consents, to which scope (render, clone, mint), for which ref, and
      revocable. Signed with the speaker's wallet when there is one: an EIP-712 consent, so a contract can check it.
      A consent event without a signature records a claim, not proof, and the card should say which it is.
- [ ] **Record the cloning engine's licence.** pocket_tts inside audio.cpp: confirm the weights' licence, then
      record `licence_cleared` on the `ref` event, or record why it is not cleared.
- [ ] **Retention for refs.** A reference clip is biometric. Add a delete route and a retention limit on the host,
      and append a `consent` event with `revoked: true` when a speaker withdraws.
- [ ] **A card from a log.** Emit a voice card in the `voaice-card/1` shape from a forge log, with
      `inft.content` committing to the head event id and the vprint hash.
- [ ] **Time.** Events record `at` as the producer states it. Bind it to verified time (Chronos), as mindX does for
      its own events.

### Kept honest
- A vprint fingerprints a **measurement**, not a speaker. Two recordings of one person measured differently give
  different prints. It is evidence that two files came from the same measured signal. **It is not a biometric
  and must not be used as one.**
- Nothing here mints. A mint is the OVERLORD's signature, never a program's.
