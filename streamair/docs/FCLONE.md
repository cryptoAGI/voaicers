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

## TODO

- [ ] **Pose normalisation and frame averaging** (geometry.js `poseNormalize`, `aggregate`): arithmetic and means
      in a fixed order, so they are portable bit for bit, with their own oracle on recorded frames.
- [ ] **Frontality.** It calls `Math.atan2`, which V8 takes from fdlibm and Rust from the system libm. Port
      fdlibm's `atan2` (it is freely licensed) and check it on every recorded pose before claiming the quality
      figures.
- [ ] **The persona print.** faicey's `persona.js` fuses face and voice into one print, `{v:1, kind:"persona",
      modalities, faceHash, voiceHash, measures}`. It belongs here, since streamair carries both.
- [ ] **Identities in the stream.** Write the vprint, the faceprint and the forge log's head into OpusTags
      (`VOAICE_VPRINT=`, `FAICE_FPRINT=`, `VCLONE_FORGE_HEAD=`), so a `.opus` names the identities it was made
      from and `streamair` can check them on read.
- [ ] **A face mint rule.** `mintable()` covers a voice: measured, consented for that ref, engine cleared. A face
      needs its own rule: measured, consented with scope `mint` for that faceprint, and no image retained (the
      capture event already records `image_kept: false`).
- [ ] **The forensic voice print** that `/voicey/measure` returns (six measures) alongside `dvscope/1`. The
      tracking item is in voaice.rs `docs/VCLONE.md`.
