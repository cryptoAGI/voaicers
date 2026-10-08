//! Ogg pages (RFC 3533) and Opus encapsulation (RFC 7845), zero dependencies.
//!
//! The rules that are easy to get wrong, each enforced here (learned in mindX's ogg-opus.js, which was verified
//! bit-exact through opusdec):
//!   - CRC-32 is polynomial 0x04C11DB7, NOT reflected, init 0, no final xor, over the whole page with the CRC
//!     field zeroed. It is not zlib's CRC-32.
//!   - OpusHead is alone on the first page (BOS, granule 0). OpusTags begins on the second page (granule 0).
//!     Audio starts on a fresh page.
//!   - A page's granule is the number of 48 kHz samples DECODED through the last packet completed on it, whatever
//!     the input rate. Pre-skip is part of those decoded samples, not added on top: playable length = last
//!     granule − pre-skip. The last page's granule may be short of the packet total, which trims the encoder's
//!     padding. (Adding pre-skip to every page made opusinfo report a file 6 ms short and opusdec stall on it.)
//!   - Lacing: a packet is 255-byte segments; a segment < 255 ends it; an exact multiple of 255 ends with a
//!     0-length segment. A page carries at most 255 segments.

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut k = 0;
        while k < 8 {
            r = if r & 0x8000_0000 != 0 { (r << 1) ^ 0x04c1_1db7 } else { r << 1 };
            k += 1;
        }
        t[i] = r;
        i += 1;
    }
    t
}
static CRC: [u32; 256] = crc_table();

/// Ogg's CRC-32 (0x04C11DB7, unreflected, init 0, no final xor).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0u32;
    for &b in bytes {
        c = (c << 8) ^ CRC[((c >> 24) as u8 ^ b) as usize];
    }
    c
}

/// Samples at 48 kHz decoded from one Opus packet, from its TOC byte and frame count (RFC 6716 §3.1–3.2).
/// Returns None for a packet that is empty or whose frame-count code is malformed.
pub fn packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    // frame size in 48 kHz samples, by configuration (Table 2): SILK 10/20/40/60, hybrid 10/20, CELT 2.5/5/10/20 ms
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][(config & 3) as usize],
        12..=15 => [480, 960][(config & 1) as usize],
        _ => [120, 240, 480, 960][(config & 3) as usize],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => (*packet.get(1)? & 0x3f) as u32,
    };
    if frames == 0 || frames * frame > 5760 {
        return None; // RFC 6716 §3.2.5: at most 120 ms
    }
    Some(frames * frame)
}

/// One Ogg logical stream being written, packet by packet.
///
/// `push` laces a packet onto the pending page and records the granule reached when it completes; a page is cut
/// when its 255 lacing values are used (a packet may continue onto the next page, which then carries the
/// continued flag, and a page that completes no packet carries granule −1). `flush` closes the pending page.
pub struct PageWriter {
    serial: u32,
    seq: u32,
    lacing: Vec<u8>,
    body: Vec<u8>,
    granule: u64,
    continued: bool,
    out: Vec<u8>,
}

pub const BOS: u8 = 0x02;
pub const EOS: u8 = 0x04;
const CONTINUED: u8 = 0x01;

impl PageWriter {
    pub fn new(serial: u32) -> Self {
        PageWriter { serial, seq: 0, lacing: Vec::with_capacity(255), body: Vec::new(), granule: u64::MAX,
                     continued: false, out: Vec::new() }
    }

    /// Add one packet; `granule` is the stream position once this packet is decoded.
    pub fn push(&mut self, packet: &[u8], granule: u64) {
        let mut off = 0;
        loop {
            if self.lacing.len() == 255 {
                self.emit(0);
            }
            let n = (packet.len() - off).min(255);
            self.lacing.push(n as u8);
            self.body.extend_from_slice(&packet[off..off + n]);
            off += n;
            if n < 255 {
                break;
            }
        }
        self.granule = granule;
    }

    /// Close the pending page (if any) with `flags` (EOS on the last page). The first page always carries BOS.
    pub fn flush(&mut self, flags: u8) {
        if !self.lacing.is_empty() || flags & EOS != 0 {
            self.emit(flags & EOS);
        }
    }

    fn emit(&mut self, eos: u8) {
        let mut f = eos;
        if self.continued {
            f |= CONTINUED;
        }
        if self.seq == 0 {
            f |= BOS;
        }
        let page_start = self.out.len();
        let o = &mut self.out;
        o.extend_from_slice(b"OggS");
        o.push(0);
        o.push(f);
        o.extend_from_slice(&self.granule.to_le_bytes());
        o.extend_from_slice(&self.serial.to_le_bytes());
        o.extend_from_slice(&self.seq.to_le_bytes());
        o.extend_from_slice(&[0, 0, 0, 0]);
        o.push(self.lacing.len() as u8);
        o.extend_from_slice(&self.lacing);
        o.extend_from_slice(&self.body);
        let c = crc32(&o[page_start..]);
        o[page_start + 22..page_start + 26].copy_from_slice(&c.to_le_bytes());
        self.continued = self.lacing.last() == Some(&255);
        self.seq += 1;
        self.lacing.clear();
        self.body.clear();
        self.granule = u64::MAX;
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

/// The OpusHead identification header (RFC 7845 §5.1), mapping family 0 (mono or stereo).
pub fn opus_head(channels: u8, pre_skip: u16, input_rate: u32, gain_q8: i16) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_rate.to_le_bytes());
    h.extend_from_slice(&gain_q8.to_le_bytes());
    h.push(0);
    h
}

/// The OpusTags comment header (RFC 7845 §5.2).
pub fn opus_tags(vendor: &str, comments: &[&str]) -> Vec<u8> {
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor.as_bytes());
    t.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for c in comments {
        t.extend_from_slice(&(c.len() as u32).to_le_bytes());
        t.extend_from_slice(c.as_bytes());
    }
    t
}

/// What an Ogg Opus file needs besides its packets.
pub struct Stream<'a> {
    pub serial: u32,
    pub channels: u8,
    pub pre_skip: u16,
    pub input_rate: u32,
    pub vendor: &'a str,
    pub comments: &'a [&'a str],
    /// how many 48 kHz samples the audio really has (excluding pre-skip); the last granule trims to it
    pub samples_48k: u64,
    /// packets per page (one page of audio every `packets_per_page` packets); latency against overhead
    pub packets_per_page: usize,
}

/// Write a complete Ogg Opus file. Granules follow RFC 7845: decoded samples through the last packet on each
/// page, pre-skip included; the final page's granule is `pre_skip + samples_48k` (end trimming).
pub fn mux(s: &Stream, packets: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    if packets.is_empty() {
        return Err("no audio packets".into());
    }
    let mut w = PageWriter::new(s.serial);
    w.push(&opus_head(s.channels, s.pre_skip, s.input_rate, 0), 0);
    w.flush(0);
    w.push(&opus_tags(s.vendor, s.comments), 0);
    w.flush(0);
    let end = s.pre_skip as u64 + s.samples_48k;
    let per = s.packets_per_page.max(1);
    // decoded samples through each packet
    let mut cum = Vec::with_capacity(packets.len());
    let mut decoded: u64 = 0;
    for p in packets {
        decoded += packet_samples(p).ok_or("malformed Opus packet (TOC)")? as u64;
        cum.push(decoded);
    }
    if decoded < end {
        return Err(format!("packets decode to {} samples, fewer than pre-skip + audio ({})", decoded, end));
    }
    if decoded - end >= 960 * 6 {
        return Err(format!("{} samples of trailing padding: more than a packet's worth", decoded - end));
    }
    // End trimming happens on the last page only (RFC 7845 §4.4): the page before it must not claim more samples
    // than the stream ends at, or the final granule would go backwards ("more than one page of end trimming", as
    // opusinfo says). So the last page starts where the trim begins, at the latest: its first packet is the earliest
    // one that the page boundaries of `packets_per_page` would leave before a granule larger than `end`.
    let mut last_start = (packets.len() - 1) / per * per;
    while last_start > 0 && cum[last_start - 1] > end {
        last_start -= 1;
    }
    for (i, p) in packets.iter().enumerate() {
        if i + 1 == packets.len() {
            w.push(p, end);
            w.flush(EOS);
        } else {
            w.push(p, cum[i]);
            if i < last_start && ((i + 1) % per == 0 || i + 1 == last_start) {
                w.flush(0);
            }
        }
    }
    Ok(w.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_is_ogg_not_zlib() {
        // RFC 3533's polynomial, unreflected: "123456789" → 0x89A1897F (CRC-32/MPEG-2 would be 0x0376E6E7 with
        // init ~0 and no xorout; Ogg uses init 0)
        assert_eq!(crc32(b"123456789"), 0x89A1_897F);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn toc_samples() {
        assert_eq!(packet_samples(&[0xF8]), Some(960)); // config 31: CELT FB 20 ms, one frame
        assert_eq!(packet_samples(&[0xF8 | 1]), Some(1920)); // two frames
        assert_eq!(packet_samples(&[0x08]), Some(960)); // config 1: SILK NB 20 ms
        assert_eq!(packet_samples(&[0xF8 | 3, 6]), Some(5760)); // 6 × 20 ms = 120 ms
        assert_eq!(packet_samples(&[0xF8 | 3, 7]), None); // 140 ms: too long
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn lacing_exact_multiple_of_255_gets_a_zero_segment() {
        let mut w = PageWriter::new(1);
        let p = vec![0xAAu8; 510];
        w.push(&p, 7);
        w.flush(0);
        let b = w.into_bytes();
        assert_eq!(b[26], 3); // 255, 255, 0
        assert_eq!(&b[27..30], &[255, 255, 0]);
        assert_eq!(b.len(), 27 + 3 + 510);
    }

    #[test]
    fn page_crc_verifies() {
        let mut w = PageWriter::new(0x1234);
        w.push(&opus_head(1, 312, 48000, 0), 0);
        w.flush(0);
        let mut b = w.into_bytes();
        let stored = u32::from_le_bytes([b[22], b[23], b[24], b[25]]);
        b[22..26].copy_from_slice(&[0, 0, 0, 0]);
        assert_eq!(crc32(&b), stored);
        assert_eq!(b[5], BOS);
    }

    #[test]
    fn granule_trims_and_counts_pre_skip() {
        // 10 packets of 20 ms = 9600 samples; pre-skip 312; audio 9000 → last granule 9312
        let pk: Vec<Vec<u8>> = (0..10).map(|_| vec![0xF8]).collect();
        let s = Stream { serial: 9, channels: 1, pre_skip: 312, input_rate: 48000, vendor: "streamair",
                         comments: &[], samples_48k: 9000, packets_per_page: 4 };
        let f = mux(&s, &pk).unwrap();
        // walk pages, collect granules
        let mut i = 0;
        let mut g = vec![];
        while i < f.len() {
            assert_eq!(&f[i..i + 4], b"OggS");
            g.push(u64::from_le_bytes(f[i + 6..i + 14].try_into().unwrap()));
            let n = f[i + 26] as usize;
            let body: usize = f[i + 27..i + 27 + n].iter().map(|&x| x as usize).sum();
            i += 27 + n + body;
        }
        assert_eq!(g, vec![0, 0, 3840, 7680, 9312]);
        assert!(mux(&Stream { samples_48k: 9600, ..s }, &pk).is_err()); // needs 9912 > 9600 decoded
    }

    #[test]
    fn a_packet_longer_than_a_page_continues() {
        // 300 segments' worth: the first page fills 255 lacing values and completes nothing (granule −1),
        // the second carries the continued flag and the granule
        let mut w = PageWriter::new(2);
        let p = vec![1u8; 255 * 299 + 10];
        w.push(&p, 960);
        w.flush(EOS);
        let b = w.into_bytes();
        assert_eq!(b[26], 255);
        assert_eq!(u64::from_le_bytes(b[6..14].try_into().unwrap()), u64::MAX);
        let second = 27 + 255 + 255 * 255;
        assert_eq!(&b[second..second + 4], b"OggS");
        assert_eq!(b[second + 5], CONTINUED | EOS);
        assert_eq!(u64::from_le_bytes(b[second + 6..second + 14].try_into().unwrap()), 960);
        assert_eq!(b[second + 26], 45); // 44 × 255 + one 10-byte segment
    }
}
