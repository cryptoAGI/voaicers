# fCLONE and vCLONE in streamair — the identities a stream carries

streamair turns what a CPU produces into `.opus`. What it produces belongs to someone: a voice, and in ollywoo
a face too. streamair therefore carries both identities. Each is a print of a **measurement**: evidence that a
file came from a measured signal, never a biometric.

| | module | what | oracle |
|---|---|---|---|
| **vCLONE** | `streamair::vclone` (re-exported from voaice.rs `src/vclone.rs`) | the voice: the `dvscope/1` vprint, the `.voaice` check, the hash-chained forge log (`vclone-event/1`) and `mintable()` | byte-identical to cryptoAGI/voaice `vprint.py` (2,000 recorded sets, all 10 measured identities), and to ollywoo's `forgelog.js` |
| **fCLONE** | `streamair::fclone` (`src/fclone.rs`) | the face: the twelve-ratio `faceprint/1` in `faice/1` files, symmetry, and MediaPipe's triangulation, read and checked | byte-identical to faicey's `geometry.js` + `faceprint.js` on **1,000** recorded landmark sets: the canonical string and hash, and symmetry bit for bit. A `.faice` written by ollywoo's `fclone.js` verifies |

```sh
streamair fclone check tests/fixtures/*.faice.json     # verified / unmeasured / MISMATCH (which fields)
streamair vclone check ../tests/fixtures/voaice/*.voaice
node testing/fclone/make_oracle.mjs <faicey>/src/face_clone 1000 > tests/fixtures/faceprint_oracle.jsonl
```

## What the face port had to get exactly right

- **`Math.hypot` is V8's, not libm's.** V8 scales by the largest argument and sums the squares with Kahan
  compensation (`builtins/math.tq`). Every landmark distance goes through it. `js_hypot` reproduces it step for
  step. The test `the_oracle_can_fail_a_naive_hypot_disagrees` shows the stakes: plain `sqrt(dx²+dy²+dz²)` gives a
  different face width on **379 of 1,000** recorded sets.
- **`toFixed18` is not the vprint's encoding.** The face keeps nine real decimals,
  `floor(v)·10¹⁸ + round(frac·10⁹)·10⁹`. The voice uses `floor(v·1e18)` with an f64 multiply. The two are kept
  apart, by name.
- **`x || 1e-6`.** A zero (or NaN) face width or height becomes 1e-6, as JavaScript's falsy `||` does. The oracle
  includes a face with coincident cheeks and an all-zero face.
- **The triangulation is read, not computed.** MediaPipe's tessellation is 852 consecutive edge triples. The check:
  all triples close, the mesh is manifold, and V − E + F = −2 (the outline, the eyes and the mouth are holes).
  468 vertices, 1,322 edges, 88 on the boundary.

## Done since the first cut

- [x] **The capture pipeline.** `pose_normalize`, `aggregate` and `frontality`, plus `fclone_frames()`, the whole
      path from frames to faceprint. On 60 generated frame sets they match faicey and ollywoo's `fclone.js`, compared
      as digests of the raw f64 bits; 30 of the sets go all the way to the faceprint. Inputs are regenerated on both
      sides from an exact Park–Miller LCG, so the fixture holds only digests (`testing/fclone/make_pose_oracle.mjs`).
- [x] **V8's `Math.atan2`.** fdlibm's `atan2` and `atan` (V8 `base/ieee754`), step for step: 324 special pairs bit
      for bit (zeros, subnormals, infinities, NaN), and 200,000 generated pairs by digest. The system libm's `atan2`
      differs on **29,545 of 200,000**, and a test asserts that it does. `Math.max` keeps NaN where `f64::max` drops
      it, so `js_max` is ported too.
- [x] **The persona print** (faicey `persona.js`): face and voice bound into one hash. 200 / 200 cases match:
      face only, voice only (the forensic print), both, and both with a `dvscope/1` vprint (which `persona.js`
      reads as precision 0). Neither modality is refused, as there.
- [x] **Identities in the stream** (`src/identity.rs`). `VOAICE_VPRINT=dvscope/1:…`, `FAICE_FPRINT=faceprint/1:0x…`,
      `PERSONA_PRINT=0x…` and `VCLONE_FORGE_HEAD=…` go into OpusTags. They are read back case-insensitively; a
      malformed or contradictory tag is refused, and each tag is verified against the `.voaice` and `.faice` files
      it names. End to end in `tests/identity.rs`: production's `opusinfo` lists the four tags with no warning, and
      `opusdec` decodes the file to exactly 48,000 samples.
- [x] **The face mint rule** (`mintable_face`): a face measurement with this faceprint, a capture that recorded
      `image_kept: false`, and the person's latest face consent for `mint` naming this faceprint and not revoked.
      A voice consent does not cover the face. ollywoo's `forgelog.js` has the same rule (`mintableFace`), and the
      set has "✋ consent: my face, to mint" and "⊘ revoke face consent".

## TODO

- [x] **The forensic voice print's encoding** (`src/forensic.rs`): the print `/voicey/measure` returns, from its six
      feature means, sample rate and frame count. It matches voaice's `Forensic.js` on 12 / 12 cases (four real
      WAVs, three frame budgets each), the hash and every measure.
- [ ] **The forensic DSP**: voice-activity detection, the analyser's FFT, pitch, flatness and HNR, which produce
      those features. V8 takes `Math.log` and `Math.exp` from fdlibm, so they get the treatment `atan2` got.
- [ ] **`framesUsed` from the server.** `/voicey/measure` returns the features and the hash but not `framesUsed`,
      so a print it hands out cannot be recomputed from its own response. One field on voaice's server (mindX
      `voaice/server.js` `_measure`) closes it.
- [ ] **Read identities from any `.opus`.** The test reads the tags from streamair's own single-page OpusTags. The
      general reader is voaice.rs 0.0.4's streaming Ogg reader; wire `Identity::from_comments` to it when it lands,
      and add `streamair identity <file.opus>`.
- [ ] **fdlibm `atan2` for any caller.** It lives in `fclone`; if anything else needs V8-exact trigonometry, move it
      to its own module rather than copying it.
- [ ] **A real camera.** Every check so far is on recorded or generated landmarks. Record a few real fCLONE captures
      (landmarks and matrices only, with consent) as fixtures.
