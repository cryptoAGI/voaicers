# streamair roadmap — 0.0.1 at a time, milestones counting by ten

Each increment is one 0.0.1 step. Every step either makes the output exact against libopus 1.4 or makes it cheaper,
and the oracle stays green across both. Milestones fall at 0.1.0, 0.2.0 and so on; the ten steps before each one
build it.

| milestone | what it delivers | oracle |
|---|---|---|
| **0.0.1** ✓ | the Ogg/Opus container: CRC, pager, lacing, continuation, OpusHead/OpusTags, granules, pre-skip, end trim | opusinfo clean, opusdec exact sample count |
| (reader) | the round-trip reader is voaice.rs 0.0.4's `voaice::ogg` (proven against opus-tools 0.2): `tests/roundtrip.rs` reads back 400 random streams exactly, and `examples/opus_corpus.rs` writes 9 of the 14 streamair files in its oracle corpus (the other 5 are `streamair silence`) | voaice.rs `testing/opus/` |
| 0.0.2–0.0.9 | a streaming writer (bounded memory, flush policy from latency; and end trimming bounded by the last page's samples — voaice.rs 0.0.4's reader found that 0.0.1's `mux` accepts a trim past them, which opusinfo calls an error), the range encoder (RFC 6716 §4.1 / §5.1), CELT's MDCT, band energies, PVQ, the bit allocation | each stage's output against libopus 1.4 internals, through a compiled harness |
| **0.1.0** | **a CELT encoder**, mono, fullband 20 ms, constant bitrate, bit-exact against libopus 1.4 at one stated complexity | the packets byte for byte; opusdec decodes them |
| 0.2.0 | VBR, the bitrates voaice uses (24 kbps speech, 64–96 kbps music), stereo | the same, per configuration |
| 0.3.0 | SILK and hybrid for narrow-band speech | the same |
| 0.4.0 | the speed pass: SIMD kernels chosen at run time, allocation-free frames, threads only where they pay | unchanged bytes, then CPU-seconds per audio-second against libopus |
| 0.5.0 | the efficiency pass: encoder choices that **cost less CPU for the same bytes**, or fewer bytes for the same quality (measured by an objective metric, stated), when a decision may diverge from libopus | opusdec decodes them; quality and CPU stated per change |
| 1.0.0 | voaice writes every `.opus` through streamair; libopus is needed only as the oracle | all of the above on production |

The rule from bankml carries over: a step that cannot show its oracle result is not finished.
