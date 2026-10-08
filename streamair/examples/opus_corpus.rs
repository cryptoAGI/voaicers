//! `cargo run --release --example opus_corpus -- <dir>` — the files streamair contributes to voaice.rs 0.0.4's
//! Ogg/Opus reader oracle (testing/opus/ in voaice.rs): each exercises one container rule with no encoder at all.
//! Every packet is TOC-only (or TOC + padding), which libopus decodes as concealment of exactly the TOC's duration,
//! so opusdec's sample count is the container's arithmetic and nothing else.
use streamair::ogg::{mux, Stream};

/// A code-3 packet of one frame of `toc`'s configuration whose length is `len` bytes, all of it padding
/// (RFC 6716 §3.2.5: each padding-length byte of 255 adds 254 and continues).
fn padded(toc: u8, len: usize) -> Vec<u8> {
    let mut p = vec![toc | 3, 0x40 | 1];
    // n padding-length bytes carry pad = len − 2 − n bytes: pad / 254 bytes of 255, then pad % 254
    let n = (1..len).find(|&n| (len - 2 - n) / 254 + 1 == n).expect("a length encoding");
    let pad = len - 2 - n;
    p.extend(std::iter::repeat_n(255, pad / 254));
    p.push((pad % 254) as u8);
    p.resize(len, 0);
    p
}

#[allow(clippy::too_many_arguments)] // one call per file, every argument named at the call site's position
fn write(dir: &str, name: &str, channels: u8, pre_skip: u16, input_rate: u32, packets: Vec<Vec<u8>>, per_page: usize, trim: u64) {
    let decoded: u64 = packets.iter().map(|p| streamair::ogg::packet_samples(p).expect("TOC") as u64).sum();
    let s = Stream { serial: 0x5354_4D41, channels, pre_skip, input_rate, vendor: "streamair 0.0.1",
                     comments: &["ENCODER=streamair 0.0.1 opus_corpus"], samples_48k: decoded - pre_skip as u64 - trim,
                     packets_per_page: per_page };
    let f = mux(&s, &packets).unwrap_or_else(|e| panic!("{name}: {e}"));
    std::fs::write(format!("{dir}/{name}.opus"), &f).unwrap();
    println!("{name}.opus {} bytes {} packets", f.len(), packets.len());
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: opus_corpus <dir>");
    std::fs::create_dir_all(&dir).unwrap();
    let dtx = |n: usize| -> Vec<Vec<u8>> { (0..n).map(|_| vec![0xF8]).collect() };
    // packets per page: one per page, exactly 255 (every lacing value used), 300 (the writer cuts at 255)
    write(&dir, "sa_ppp1", 1, 312, 48000, dtx(51), 1, 648);
    write(&dir, "sa_ppp255", 1, 312, 48000, dtx(510), 255, 300);
    write(&dir, "sa_ppp300", 1, 312, 48000, dtx(600), 300, 1);
    write(&dir, "sa_ppp7_trim", 1, 312, 44100, dtx(130), 7, 959);
    // continuation: a 70,000-byte packet (more than a page's 65,025 body bytes) and one of exactly 255 × 255 bytes,
    // which ends with a zero lacing value on the next page
    let mut p = dtx(3);
    p.push(padded(0xF8, 70_000));
    p.extend(dtx(2));
    p.push(padded(0xF8, 255 * 255));
    p.extend(dtx(2));
    write(&dir, "sa_continuation", 1, 312, 48000, p, 3, 100);
    // every frame size and code: CELT 2.5/5/10/20 ms, codes 1, 2 and 3, SILK 40 and 60 ms, hybrid 10 and 20 ms
    let mixed: Vec<Vec<u8>> = (0..20)
        .flat_map(|_| [vec![0xE0], vec![0xE8], vec![0xF0], vec![0xF8], vec![0xF9], vec![0xFA, 0], vec![0xFB, 3],
                       vec![0x18], vec![0x10], vec![0x60], vec![0x68]])
        .collect();
    write(&dir, "sa_mixed_toc", 1, 312, 48000, mixed, 5, 37);
    // stereo (the TOC's stereo bit, OpusHead channels 2); pre-skip 0; a large pre-skip (80 ms)
    write(&dir, "sa_stereo", 2, 312, 48000, (0..60).map(|_| vec![0xFC]).collect(), 50, 500);
    write(&dir, "sa_preskip0", 1, 0, 48000, dtx(50), 50, 0);
    write(&dir, "sa_preskip3840", 1, 3840, 16000, dtx(60), 20, 5);
}
