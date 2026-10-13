# Pointer: the rage / DeltaVerse render chain in Rust

The design and a measured proof of concept are in browseML:
[`browseML/docs/RENDER_RUST.md`](https://github.com/browseML/browseML/blob/main/docs/RENDER_RUST.md). The scratch crate
`overlord-dsp` is at `browseML/render/` and is not wired into anything.

Why this note is here: the design proposes **this family** (voaice.rs, beside `streamair`) as the crate's long-term
home. Rendering is "text → sound → .opus", which is streamair's stated purpose, and streamair already owns the Ogg/Opus
container.

What the PoC established (2026-10-12):

* **Bit-exact natively**: WSOLA, DTW, interp, linspace and pitch_shift against mindX's `overlord_dsp.py` (numba).
* **FFT stages at 281–303 dB SNR**: zero-dependency radix-2 / real / Bluestein against numpy's pocketfft.
* **wasm32 is exact only when the window bits are fixed.** wasm32's libm `cos` moves 14 of 704 window values by one
  ulp, and WSOLA's argmax then diverges. Every transcendental that feeds a decision must come from one
  target-independent source.
* **Speed: parity with numba**, not a speedup.
* **Server-side, the DSP is ~8% of OVERLORD's cost.** The rest is the two piper (VITS) inferences.

**Licence line.** alan, jenny, lessac and lessac-derived voices never ship to browsers. Only open voices may, for
example `en_US-kristin-medium` (SAVANTE).

Status: exploration, uncommitted.
