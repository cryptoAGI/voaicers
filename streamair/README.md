# streamair — CPU → .opus in zero-dependency Rust

streamair is the **compression half of [voaice.rs](../README.md)**: the path from PCM produced on a CPU to a
`.opus` file, written in Rust with no dependencies. It is built the same way as voaice.rs and
[bankml](https://github.com/cryptoAGI/bankml): **exact first, fast second**. Each stage is checked against a pinned,
compiled reference before any work on speed starts.

The reference is **libopus 1.4**, the encoder mindX production runs (`libopus0 1.4-1build1`, `opus-tools 0.2`).
voaice already writes every `.opus` it serves through that library. streamair aims to replace it one stage at a
time, using less CPU at every step.

## State — 0.0.1: the container

| | |
|---|---|
| **Ogg pages** (RFC 3533) | Ogg's CRC-32 (0x04C11DB7, unreflected, init 0, which is not zlib's), lacing (an exact multiple of 255 ends with a 0-length segment), packets continued across pages, granule −1 on a page that completes no packet |
| **Opus encapsulation** (RFC 7845) | `OpusHead` alone on the BOS page; `OpusTags` on the second; audio starts on a fresh page; granule = 48 kHz samples decoded through the page's last packet, **pre-skip included**; end trimming on the EOS page |
| **TOC parsing** (RFC 6716 §3.1) | samples per packet for every configuration, frame-count codes 0–3, the 120 ms limit |
| **oracle** | `testing/oracle.sh`: production's `opusinfo` reads every file with no warning, and `opusdec` decodes **exactly** the samples written, 48 to 2,880,000 ([result](testing/results/0.0.1.txt)) |

There is no encoder yet. `streamair silence <seconds> <out.opus>` writes 20 ms CELT DTX frames (TOC `0xF8`, no
payload) of exact length. That exercises every container rule without an encoder, so the container is proven before
anything is put in it.

```bash
cargo test                                   # 6 unit tests: CRC check value, TOC, lacing, continuation, granules
cargo run --release -- silence 2.5 out.opus  # 466 bytes, 120,000 samples
testing/oracle.sh user@host                  # against that host's libopus 1.4 tools (local if no host)
```

## Where it is going

The full plan, counting by tens, is in [docs/ROADMAP.md](docs/ROADMAP.md). In short: the CELT encoder first
(fullband speech and music at the bitrates voaice uses), then SILK and hybrid. Each stage is first bit-exact against
libopus 1.4 at a stated complexity, then made cheaper. The measure is **bytes per second of audio at a stated
quality, and CPU-seconds per second of audio**. For the container, efficiency is already measurable: the page
overhead at 50 packets per page is under 1% of a speech stream at 24 kbps.

## Credit

Opus is [RFC 6716](https://www.rfc-editor.org/rfc/rfc6716), by Jean-Marc Valin, Koen Vos, Timothy B. Terriberry
and contributors. The reference implementation is [libopus](https://opus-codec.org) (BSD-3-Clause), from Xiph.Org
and contributors. Ogg is [RFC 3533](https://www.rfc-editor.org/rfc/rfc3533). streamair uses libopus only as the
oracle; it copies no code from it.

Licence: MIT OR Apache-2.0 (the repository's [LICENSE-MIT](../LICENSE-MIT) and [LICENSE-APACHE](../LICENSE-APACHE)).

[Professor Codephreak](https://github.com/Professor-Codephreak) · [cryptoAGI](https://github.com/cryptoAGI)
