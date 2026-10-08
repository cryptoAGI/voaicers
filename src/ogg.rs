// SPDX-License-Identifier: MIT OR Apache-2.0
//! The streaming Ogg/Opus reader (0.0.4): Ogg pages (RFC 3533) and the Opus encapsulation (RFC 7845), read from any
//! `std::io::Read` with memory bounded by one page and the longest packet that spans pages — the file is never held.
//!
//! It is the reading twin of streamair's writer (streamair/src/ogg.rs, proven against libopus 1.4) and keeps the same
//! rules; this module does not depend on streamair (streamair depends on this crate), so the page format is written
//! out again here and the two are checked against each other by round trips (streamair/tests/roundtrip.rs) and
//! against opus-tools 0.2 and libogg 1.3.5 by the oracle (testing/opus/).
//!
//! What every page is checked for: the capture pattern `OggS`, version 0, header-type flags (only BOS / EOS /
//! continued; BOS on the first page only; continued exactly when a packet is carried over), one serial (a second
//! logical stream is refused, by name), page sequence numbers without a gap or a repeat, and the CRC-32 —
//! polynomial 0x04C11DB7, **not reflected, init 0, no final xor**, over the page with its CRC field zeroed. It is
//! not zlib's CRC-32 (that one is reflected, with init and xorout 0xFFFFFFFF), and the oracle's discriminator shows
//! that zlib's verifies none of the pinned pages.
//!
//! What the stream is checked for (RFC 7845 §3–§5): `OpusHead` alone on the BOS page with granule 0; `OpusTags`
//! next, on one or more pages, the page completing it carrying nothing else and granule 0; then audio. A page's
//! granule is the 48 kHz sample count decoded through the last packet completed on it, pre-skip included, and a page
//! completing no packet carries −1. The first audio page may start the stream past zero (its granule larger than the
//! samples on it: the stream's start is the difference, §4.3), never before it unless it is also the last page.
//! Mid-stream, each granule must equal the previous one plus the samples completed on the page. The last page (EOS)
//! may claim fewer samples than its packets decode (end trimming, §4.4) but never more, and never fewer than the
//! pre-skip. The playable length is `last granule − start − pre-skip`, in 48 kHz samples, whatever the input rate.
//!
//! Every refusal is an [`Error`] with a named [`Kind`], the byte offset in the file and the page index.
use std::io::Read;

/// The largest Ogg page: a 27-byte header, 255 lacing values and 255 segments of 255 bytes.
pub const MAX_PAGE: usize = 27 + 255 + 255 * 255;
/// Header-type flags (RFC 3533 §6).
pub const CONTINUED: u8 = 0x01;
pub const BOS: u8 = 0x02;
pub const EOS: u8 = 0x04;
/// The default bound on one packet: an audio packet longer than this is refused; an `OpusTags` longer than this is
/// kept up to it and marked truncated (pictures in comments can be megabytes; the duration does not need them).
pub const DEFAULT_MAX_PACKET: usize = 1 << 20;

// ---------------------------------------------------------------------------------------------------------------
// The CRC

const fn crc_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut r = (i as u32) << 24;
        let mut k = 0;
        while k < 8 {
            r = if r & 0x8000_0000 != 0 { (r << 1) ^ 0x04c1_1db7 } else { r << 1 };
            k += 1;
        }
        t[0][i] = r;
        i += 1;
    }
    // t[k][i]: the CRC of byte i followed by k zero bytes — what slicing-by-8 looks up for byte 7 − k of a block
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            let p = t[k - 1][i];
            t[k][i] = (p << 8) ^ t[0][(p >> 24) as usize];
            i += 1;
        }
        k += 1;
    }
    t
}
static CRC: [[u32; 256]; 8] = crc_tables();

/// Continue Ogg's CRC-32 over `bytes`, eight bytes per step (slicing-by-8; the same value as [`crc32_bytewise`]).
pub fn crc32_update(mut c: u32, bytes: &[u8]) -> u32 {
    let (blocks, tail) = bytes.as_chunks::<8>();
    for b in blocks {
        let h = c ^ u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        c = CRC[7][(h >> 24) as usize]
            ^ CRC[6][(h >> 16) as u8 as usize]
            ^ CRC[5][(h >> 8) as u8 as usize]
            ^ CRC[4][h as u8 as usize]
            ^ CRC[3][b[4] as usize]
            ^ CRC[2][b[5] as usize]
            ^ CRC[1][b[6] as usize]
            ^ CRC[0][b[7] as usize];
    }
    for &x in tail {
        c = (c << 8) ^ CRC[0][((c >> 24) as u8 ^ x) as usize];
    }
    c
}

/// Ogg's CRC-32 (0x04C11DB7, unreflected, init 0, no final xor).
pub fn crc32(bytes: &[u8]) -> u32 {
    crc32_update(0, bytes)
}

/// The same CRC one byte at a time with one table, as streamair's writer and libogg compute it: the baseline the
/// sliced form is measured against, and its second witness in the tests.
pub fn crc32_bytewise(bytes: &[u8]) -> u32 {
    let mut c = 0u32;
    for &b in bytes {
        c = (c << 8) ^ CRC[0][((c >> 24) as u8 ^ b) as usize];
    }
    c
}

/// A page's CRC: over the whole page with bytes 22..26 (the CRC field) taken as zero, without writing to it.
pub fn page_crc(page: &[u8]) -> u32 {
    let c = crc32_update(0, &page[..22]);
    let c = crc32_update(c, &[0; 4]);
    crc32_update(c, &page[26..])
}

// ---------------------------------------------------------------------------------------------------------------
// The packet's duration

/// Samples at 48 kHz in one Opus packet, from its TOC byte and, for code 3, its frame-count byte (RFC 6716 §3.1–3.2):
/// the frame size by configuration (SILK 10/20/40/60 ms, hybrid 10/20, CELT 2.5/5/10/20) times the frame count.
/// `None` for an empty packet, a code-3 packet without its count byte, zero frames, or more than 120 ms.
pub fn packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let frame: u32 = match config {
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
        return None;
    }
    Some(frames * frame)
}

// ---------------------------------------------------------------------------------------------------------------
// Errors

/// What was wrong. Each refusal names one of these; the oracle's adversarial files are checked against them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// the reader's source failed
    Io,
    /// the file ends inside a page
    TruncatedPage,
    /// the file ends after a whole page, but before a page with the end-of-stream flag
    NoEndOfStream,
    /// a page does not start with `OggS`
    CapturePattern,
    /// a page's stream structure version is not 0
    Version,
    /// a page's stored CRC is not the CRC of its bytes
    Crc,
    /// undefined header-type bits, BOS anywhere but the first page, or the first page without BOS
    HeaderFlags,
    /// the continued flag does not match whether a packet was carried over from the previous page
    Continuation,
    /// a page of another logical stream (multiplexed streams are not read)
    Serial,
    /// page sequence numbers not consecutive: a page lost, repeated or reordered
    Sequence,
    /// bytes after the end-of-stream page (chained streams are not read)
    DataAfterEnd,
    /// the identification header is missing, malformed, or not alone on its page
    OpusHead,
    /// the comment header is missing or malformed, or shares its last page with audio
    OpusTags,
    /// an audio packet of zero bytes, or a TOC whose frame count is malformed or longer than 120 ms
    Packet,
    /// an audio packet longer than the reader's bound
    PacketTooLarge,
    /// a header page's granule is not 0 (or −1 on a page that completes no packet)
    HeaderGranule,
    /// a page completing no packet carries a granule, or one completing a packet carries −1 or a negative value
    GranuleMissing,
    /// a granule smaller than the previous page's
    GranuleBackwards,
    /// a mid-stream granule other than the previous one plus the samples completed on the page
    GranuleMismatch,
    /// a granule claiming more samples than the packets completed so far decode to
    GranuleBeyondSamples,
    /// the stream's playable end is before its pre-skip
    PreSkip,
    /// the first audio page claims fewer samples than complete on it, and it is not the last page: the stream would
    /// start before zero (RFC 7845 §4.5)
    GranuleBeforeStart,
    /// headers but no audio packet
    NoAudio,
}

/// A refusal: what, where (byte offset of the page, or of the packet within it) and on which page (0-based).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: Kind,
    pub offset: u64,
    pub page: u64,
    pub detail: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "byte {} (page {}): {:?}: {}", self.offset, self.page, self.kind, self.detail)
    }
}
impl std::error::Error for Error {}

// ---------------------------------------------------------------------------------------------------------------
// The headers

/// `OpusHead` (RFC 7845 §5.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpusHead {
    pub version: u8,
    pub channels: u8,
    pub pre_skip: u16,
    /// the original input rate, informational only (playback is at 48 kHz)
    pub input_rate: u32,
    /// output gain in Q7.8 dB
    pub gain_q8: i16,
    pub mapping_family: u8,
    /// family ≠ 0: the stream count, the coupled count and one entry per channel (255 = silent)
    pub streams: u8,
    pub coupled: u8,
    pub mapping: Vec<u8>,
}

impl OpusHead {
    /// Parse and validate an identification header.
    pub fn parse(p: &[u8]) -> Result<OpusHead, String> {
        if p.len() < 19 || &p[..8] != b"OpusHead" {
            return Err(format!("not an OpusHead packet ({} bytes)", p.len()));
        }
        let version = p[8];
        if version >> 4 != 0 {
            return Err(format!("version {version}: major version not 0"));
        }
        let channels = p[9];
        if channels == 0 {
            return Err("0 channels".into());
        }
        let mut h = OpusHead {
            version,
            channels,
            pre_skip: u16::from_le_bytes([p[10], p[11]]),
            input_rate: u32::from_le_bytes([p[12], p[13], p[14], p[15]]),
            gain_q8: i16::from_le_bytes([p[16], p[17]]),
            mapping_family: p[18],
            streams: 1,
            coupled: (channels == 2) as u8,
            mapping: Vec::new(),
        };
        if h.mapping_family == 0 {
            if channels > 2 {
                return Err(format!("mapping family 0 with {channels} channels"));
            }
            return Ok(h);
        }
        if h.mapping_family == 1 && channels > 8 {
            return Err(format!("mapping family 1 with {channels} channels"));
        }
        let need = 21 + channels as usize;
        if p.len() < need {
            return Err(format!("mapping family {} needs {need} bytes, has {}", h.mapping_family, p.len()));
        }
        h.streams = p[19];
        h.coupled = p[20];
        if h.streams == 0 || h.coupled > h.streams || h.streams as u32 + h.coupled as u32 > 255 {
            return Err(format!("{} streams, {} coupled", h.streams, h.coupled));
        }
        h.mapping = p[21..need].to_vec();
        let decoded = h.streams as u32 + h.coupled as u32;
        if let Some(&m) = h.mapping.iter().find(|&&m| m != 255 && m as u32 >= decoded) {
            return Err(format!("channel mapping entry {m} beyond {decoded} decoded channels"));
        }
        Ok(h)
    }
}

/// `OpusTags` (RFC 7845 §5.2): the vendor string and the user comments, as far as the packet bound kept them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpusTags {
    pub vendor: String,
    pub comments: Vec<String>,
    /// the comment count the header declares (more than `comments.len()` only when `truncated`)
    pub declared: u32,
    /// the packet was longer than the reader's bound; what is here is what fitted
    pub truncated: bool,
    /// the whole packet's length in bytes
    pub bytes: u64,
}

impl OpusTags {
    /// Parse a comment header; `truncated` says the packet was cut at the reader's bound, so a length running past
    /// the end stops the parse instead of refusing it.
    pub fn parse(p: &[u8], truncated: bool, bytes: u64) -> Result<OpusTags, String> {
        if p.len() < 16 || &p[..8] != b"OpusTags" {
            return Err(format!("not an OpusTags packet ({} bytes)", p.len()));
        }
        let u32_at = |o: usize| p.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let mut t = OpusTags { truncated, bytes, ..Default::default() };
        let vlen = u32_at(8).unwrap() as usize;
        let Some(v) = p.get(12..12 + vlen) else {
            return if truncated { Ok(t) } else { Err(format!("vendor length {vlen} past the packet")) };
        };
        t.vendor = String::from_utf8_lossy(v).into_owned();
        let mut o = 12 + vlen;
        let Some(n) = u32_at(o) else {
            return if truncated { Ok(t) } else { Err("no comment count".into()) };
        };
        t.declared = n;
        o += 4;
        for i in 0..n {
            let Some(len) = u32_at(o) else {
                return if truncated { Ok(t) } else { Err(format!("comment {i}: no length")) };
            };
            let Some(c) = p.get(o + 4..o + 4 + len as usize) else {
                return if truncated { Ok(t) } else { Err(format!("comment {i}: length {len} past the packet")) };
            };
            t.comments.push(String::from_utf8_lossy(c).into_owned());
            o += 4 + len as usize;
        }
        Ok(t)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The reader

/// One audio packet, borrowed from the reader until the next call.
#[derive(Debug)]
pub struct Packet<'a> {
    pub data: &'a [u8],
    /// 48 kHz samples the packet decodes to (its TOC)
    pub samples: u32,
    /// samples decoded before this packet, from the stream's first audio packet (pre-skip not removed)
    pub decoded_before: u64,
    /// of this packet's samples, how many to discard from its front (pre-skip) and how many to play after that
    /// (end trimming); a decoder plays `samples[skip .. skip + keep]`
    pub skip: u32,
    pub keep: u32,
    /// the page this packet completed on (0-based), the page it began on, and the file offset of its first byte
    pub page: u64,
    pub first_page: u64,
    pub offset: u64,
}

/// What a whole stream came to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// pages in the stream, header pages included
    pub pages: u64,
    /// audio packets (the two header packets not counted)
    pub packets: u64,
    /// bytes read, the whole file
    pub bytes: u64,
    /// samples all audio packets decode to, at 48 kHz
    pub decoded: u64,
    /// the granule the stream starts from (0 unless it began mid-broadcast, RFC 7845 §4.3)
    pub start_granule: u64,
    /// the end-of-stream page's granule
    pub last_granule: u64,
    /// the playable length at 48 kHz: `last_granule − start_granule − pre_skip`
    pub duration: u64,
    /// samples trimmed from the end (decoded − pre-skip − duration)
    pub end_trim: u64,
    /// the shortest and longest audio packet in samples, and the largest in bytes
    pub min_packet_samples: u32,
    pub max_packet_samples: u32,
    pub max_packet_bytes: usize,
    /// FNV-1a 64 over every page's (sequence u32, granule u64, flags u8), little-endian: what the oracle compares page
    /// by page against libogg without the reader keeping a list
    pub page_fnv: u64,
}

/// FNV-1a, 64-bit: the digest the oracle records (testing/opus/reference.py) for pages and packets.
pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// The streaming reader. `new` reads and checks the two header packets; `next_packet` hands out audio packets one
/// at a time; `None` means the end-of-stream page was read whole and checked, and [`Reader::summary`] is final.
///
/// Memory: one page buffer of [`MAX_PAGE`] bytes, allocated once; a packet that lies within one page is handed out
/// as a slice of it; one that spans pages is assembled in a second buffer, reused, which grows to the longest such
/// packet and no further (bounded by `max_packet`).
pub struct Reader<R: Read> {
    src: R,
    page: Box<[u8]>,
    /// the current page: its length, its lacing count and where its body starts
    page_len: usize,
    nseg: usize,
    /// the next lacing value and body byte to hand out
    seg: usize,
    bpos: usize,
    /// the bytes of a packet begun on an earlier page; `carry_len` counts bytes beyond the bound as well
    carry: Vec<u8>,
    carry_len: u64,
    /// where the carried packet began (byte offset, page)
    carry_at: (u64, u64),
    clear_carry: bool,
    /// samples decoded through the packets handed out so far
    handed: u64,
    max_packet: usize,
    head: OpusHead,
    tags: OpusTags,
    serial: u32,
    next_seq: u32,
    page_index: u64,
    page_offset: u64,
    offset: u64,
    eos: bool,
    finished: bool,
    /// the granule of the last page that completed a packet (None before the first audio one)
    prev_granule: Option<u64>,
    /// the end of the playable region, in decoded samples from the start (known once the last page is read)
    end: u64,
    sum: Summary,
}

fn err(kind: Kind, offset: u64, page: u64, detail: impl Into<String>) -> Error {
    Error { kind, offset, page, detail: detail.into() }
}

impl<R: Read> Reader<R> {
    /// Read the headers with the default packet bound.
    pub fn new(src: R) -> Result<Self, Error> {
        Self::with_max_packet(src, DEFAULT_MAX_PACKET)
    }

    /// Read the headers; audio packets longer than `max_packet` bytes are refused, `OpusTags` is kept up to it.
    pub fn with_max_packet(src: R, max_packet: usize) -> Result<Self, Error> {
        let mut r = Reader {
            src,
            page: vec![0u8; MAX_PAGE].into_boxed_slice(),
            page_len: 0,
            nseg: 0,
            seg: 0,
            bpos: 0,
            carry: Vec::new(),
            carry_len: 0,
            carry_at: (0, 0),
            clear_carry: false,
            handed: 0,
            max_packet: max_packet.max(64),
            head: OpusHead::parse(b"OpusHead\x01\x01\0\0\x80\xbb\0\0\0\0\0\0\0\0\0").unwrap(),
            tags: OpusTags::default(),
            serial: 0,
            next_seq: 0,
            page_index: 0,
            page_offset: 0,
            offset: 0,
            eos: false,
            finished: false,
            prev_granule: None,
            end: u64::MAX,
            sum: Summary { min_packet_samples: u32::MAX, page_fnv: FNV_OFFSET, ..Default::default() },
        };
        r.read_headers()?;
        Ok(r)
    }

    pub fn head(&self) -> &OpusHead {
        &self.head
    }
    pub fn tags(&self) -> &OpusTags {
        &self.tags
    }
    /// The counts so far; final once `next_packet` has returned `None`.
    pub fn summary(&self) -> &Summary {
        &self.sum
    }

    /// Read every remaining packet and return the summary: the duration without decoding anything.
    pub fn finish(mut self) -> Result<Summary, Error> {
        while self.next_packet()?.is_some() {}
        Ok(self.sum)
    }

    fn body(&self) -> usize {
        27 + self.nseg
    }

    /// Read into `buf` until full or end of input; the count read.
    fn fill(&mut self, at: usize, n: usize) -> Result<usize, Error> {
        let mut got = 0;
        while got < n {
            match self.src.read(&mut self.page[at + got..at + n]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(err(Kind::Io, self.offset + got as u64, self.page_index, e.to_string())),
            }
        }
        self.offset += got as u64;
        Ok(got)
    }

    /// Read and check one page's framing (capture, version, flags, serial, sequence, CRC, continuation).
    /// `Ok(false)` at a clean end of input.
    fn read_page(&mut self) -> Result<bool, Error> {
        let at = self.offset;
        let pi = self.sum.pages;
        let got = self.fill(0, 27)?;
        if got == 0 {
            return Ok(false);
        }
        if got < 27 {
            return Err(err(Kind::TruncatedPage, at, pi, format!("{got} of 27 header bytes")));
        }
        let p = &self.page;
        if &p[..4] != b"OggS" {
            return Err(err(Kind::CapturePattern, at, pi, format!("{:02x?} where OggS should be", &p[..4])));
        }
        if p[4] != 0 {
            return Err(err(Kind::Version, at + 4, pi, format!("stream structure version {}", p[4])));
        }
        let nseg = p[26] as usize;
        let got = self.fill(27, nseg)?;
        if got < nseg {
            return Err(err(Kind::TruncatedPage, at, pi, format!("{got} of {nseg} lacing values")));
        }
        let blen: usize = self.page[27..27 + nseg].iter().map(|&l| l as usize).sum();
        let got = self.fill(27 + nseg, blen)?;
        if got < blen {
            return Err(err(Kind::TruncatedPage, at, pi, format!("{got} of {blen} body bytes")));
        }
        let len = 27 + nseg + blen;
        let p = &self.page[..len];
        let stored = u32::from_le_bytes([p[22], p[23], p[24], p[25]]);
        let computed = page_crc(p);
        if stored != computed {
            return Err(err(Kind::Crc, at + 22, pi, format!("stored {stored:08x}, computed {computed:08x}")));
        }
        let flags = p[5];
        if flags & !(CONTINUED | BOS | EOS) != 0 {
            return Err(err(Kind::HeaderFlags, at + 5, pi, format!("undefined header-type bits {flags:#04x}")));
        }
        let serial = u32::from_le_bytes([p[14], p[15], p[16], p[17]]);
        let seq = u32::from_le_bytes([p[18], p[19], p[20], p[21]]);
        if pi == 0 {
            if flags & BOS == 0 {
                return Err(err(Kind::HeaderFlags, at + 5, pi, "the first page has no BOS flag"));
            }
            self.serial = serial;
        } else {
            if flags & BOS != 0 {
                return Err(err(Kind::HeaderFlags, at + 5, pi, "BOS on a page after the first"));
            }
            if serial != self.serial {
                return Err(err(Kind::Serial, at + 14, pi, format!("serial {serial:08x} in a stream of {:08x}", self.serial)));
            }
            if seq != self.next_seq {
                return Err(err(Kind::Sequence, at + 18, pi, format!("page sequence {seq}, expected {}", self.next_seq)));
            }
        }
        let carried = self.carry_len > 0;
        if (flags & CONTINUED != 0) != carried {
            let why = if carried { "a packet continues but the page lacks the continued flag" } else { "continued flag with no packet to continue" };
            return Err(err(Kind::Continuation, at + 5, pi, why));
        }
        self.next_seq = seq.wrapping_add(1);
        let mut rec = [0u8; 13];
        rec[..4].copy_from_slice(&p[18..22]);
        rec[4..12].copy_from_slice(&p[6..14]);
        rec[12] = flags;
        self.sum.page_fnv = fnv1a(self.sum.page_fnv, &rec);
        self.page_len = len;
        self.nseg = nseg;
        self.seg = 0;
        self.bpos = 27 + nseg;
        self.page_offset = at;
        self.page_index = pi;
        self.sum.pages += 1;
        self.sum.bytes = self.offset;
        Ok(true)
    }

    fn granule(&self) -> u64 {
        u64::from_le_bytes(self.page[6..14].try_into().unwrap())
    }
    fn flags(&self) -> u8 {
        self.page[5]
    }

    /// Append `page[from..to]` to the carried packet, counting past the bound without keeping.
    fn carry_append(&mut self, from: usize, to: usize) {
        if self.carry_len == 0 {
            self.carry_at = (self.page_offset + from as u64, self.page_index);
        }
        let room = self.max_packet.saturating_sub(self.carry.len());
        let keep = (to - from).min(room);
        // exactly: the buffer ends at the longest spanning packet, not at twice it (one realloc per page it spans)
        self.carry.reserve_exact(keep);
        self.carry.extend_from_slice(&self.page[from..from + keep]);
        self.carry_len += (to - from) as u64;
    }

    /// The header pages: `OpusHead` alone on the BOS page, then `OpusTags` to the end of a page.
    fn read_headers(&mut self) -> Result<(), Error> {
        if !self.read_page()? {
            return Err(err(Kind::OpusHead, 0, 0, "empty input"));
        }
        let body = self.body();
        let lacing = &self.page[27..body];
        if self.nseg == 0 || lacing[..self.nseg - 1].iter().any(|&l| l < 255) || lacing[self.nseg - 1] == 255 {
            return Err(err(Kind::OpusHead, self.page_offset, 0, "the first page must hold exactly one whole packet"));
        }
        self.head = OpusHead::parse(&self.page[body..self.page_len])
            .map_err(|e| err(Kind::OpusHead, self.page_offset + body as u64, 0, e))?;
        if self.granule() != 0 {
            return Err(err(Kind::HeaderGranule, self.page_offset + 6, 0, format!("OpusHead page granule {}", self.granule() as i64)));
        }
        if self.flags() & EOS != 0 {
            return Err(err(Kind::NoAudio, self.page_offset + 5, 0, "end of stream on the OpusHead page"));
        }
        // OpusTags: assembled in the carry buffer (bounded), across as many pages as it takes
        loop {
            if !self.read_page()? {
                return Err(err(Kind::OpusTags, self.offset, self.sum.pages, "the file ends before OpusTags is complete"));
            }
            let (body, n) = (self.body(), self.nseg);
            let lacing = &self.page[27..body];
            let done = lacing.iter().position(|&l| l < 255);
            let g = self.granule();
            match done {
                None => {
                    if g != 0 && g != u64::MAX {
                        return Err(err(Kind::HeaderGranule, self.page_offset + 6, self.page_index, format!("OpusTags page granule {}", g as i64)));
                    }
                    let len = self.page_len;
                    self.carry_append(body, len);
                }
                Some(i) => {
                    if i + 1 != n {
                        return Err(err(Kind::OpusTags, self.page_offset, self.page_index, "audio on the page that completes OpusTags"));
                    }
                    if g != 0 {
                        return Err(err(Kind::HeaderGranule, self.page_offset + 6, self.page_index, format!("OpusTags page granule {}", g as i64)));
                    }
                    let len = self.page_len;
                    self.carry_append(body, len);
                    let truncated = self.carry_len > self.carry.len() as u64;
                    self.tags = OpusTags::parse(&self.carry, truncated, self.carry_len)
                        .map_err(|e| err(Kind::OpusTags, self.page_offset, self.page_index, e))?;
                    self.carry.clear();
                    self.carry.shrink_to(4096); // a large comment block (a picture) is not kept for the audio
                    self.carry_len = 0;
                    self.seg = n;
                    if self.flags() & EOS != 0 {
                        return Err(err(Kind::NoAudio, self.page_offset + 5, self.page_index, "end of stream on the OpusTags page"));
                    }
                    return Ok(());
                }
            }
        }
    }

    /// The byte `i` of the packet that begins with the carried bytes and continues at `page[from..]`.
    fn packet_byte(&self, carried: bool, from: usize, i: usize) -> Option<u8> {
        let c = if carried { self.carry.len() } else { 0 };
        if i < c { Some(self.carry[i]) } else { self.page[..self.page_len].get(from + i - c).copied() }
    }

    /// Check an audio page's packets and granule before any of its packets is handed out.
    fn check_audio_page(&mut self) -> Result<(), Error> {
        let (body, at, pi) = (self.body(), self.page_offset, self.page_index);
        let mut carried = self.carry_len > 0;
        let mut plen = self.carry_len;
        let mut start = body; // where the packet being measured starts on this page
        let mut pos = body;
        let mut completed = 0u64;
        let mut samples = 0u64;
        for s in 0..self.nseg {
            let l = self.page[27 + s] as usize;
            pos += l;
            plen += l as u64;
            if l == 255 {
                continue;
            }
            if plen == 0 {
                return Err(err(Kind::Packet, at + start as u64, pi, "an audio packet of zero bytes"));
            }
            if plen > self.max_packet as u64 {
                return Err(err(Kind::PacketTooLarge, at + start as u64, pi, format!("{plen} bytes, bound {}", self.max_packet)));
            }
            let toc = [self.packet_byte(carried, start, 0), self.packet_byte(carried, start, 1)];
            let n = match toc {
                [Some(t), b] => packet_samples(&[t, b.unwrap_or(0)][..1 + b.is_some() as usize]),
                _ => None,
            }
            .ok_or_else(|| err(Kind::Packet, at + start as u64, pi, format!("TOC {:02x?}: malformed frame count or longer than 120 ms", toc)))?;
            samples += n as u64;
            completed += 1;
            carried = false;
            plen = 0;
            start = pos;
        }
        if plen > self.max_packet as u64 {
            return Err(err(Kind::PacketTooLarge, at + start as u64, pi, format!("{plen} bytes so far, bound {}", self.max_packet)));
        }
        let g = self.granule();
        let eos = self.flags() & EOS != 0;
        if eos && plen > 0 {
            return Err(err(Kind::TruncatedPage, at, pi, "the end-of-stream page ends inside a packet"));
        }
        self.sum.decoded += samples;
        if completed == 0 {
            if g != u64::MAX && !(eos && Some(g) == self.prev_granule) {
                return Err(err(Kind::GranuleMissing, at + 6, pi, format!("granule {} on a page that completes no packet", g as i64)));
            }
        } else if g == u64::MAX || g > i64::MAX as u64 {
            return Err(err(Kind::GranuleMissing, at + 6, pi, format!("granule {} on a page that completes {completed} packet(s)", g as i64)));
        }
        let g = if completed == 0 { self.prev_granule.unwrap_or(self.sum.start_granule) } else { g };
        match self.prev_granule {
            None if completed > 0 => {
                // the first audio page with a completed packet (RFC 7845 §4.5)
                if eos {
                    if g > samples {
                        return Err(err(Kind::GranuleBeyondSamples, at + 6, pi, format!("granule {g}, but the packets decode to {samples}")));
                    }
                    self.sum.start_granule = 0;
                } else {
                    if g < samples {
                        return Err(err(Kind::GranuleBeforeStart, at + 6, pi, format!("first audio granule {g} is less than the {samples} samples completed on its page (only the last page may trim)")));
                    }
                    self.sum.start_granule = g - samples;
                }
                self.prev_granule = Some(g);
            }
            None => {}
            Some(prev) => {
                let expect = prev + samples;
                if g < prev {
                    return Err(err(Kind::GranuleBackwards, at + 6, pi, format!("granule {g} after {prev}")));
                }
                if eos {
                    if g > expect {
                        return Err(err(Kind::GranuleBeyondSamples, at + 6, pi, format!("granule {g}, but the packets decode only to {expect}")));
                    }
                } else if g != expect {
                    return Err(err(Kind::GranuleMismatch, at + 6, pi, format!("granule {g}, expected {prev} + {samples} = {expect}")));
                }
                self.prev_granule = Some(g);
            }
        }
        if eos {
            let Some(last) = self.prev_granule else {
                return Err(err(Kind::NoAudio, at, pi, "end of stream before any audio packet"));
            };
            let pre = self.head.pre_skip as u64;
            let span = last - self.sum.start_granule;
            if span < pre {
                return Err(err(Kind::PreSkip, at + 6, pi, format!("the stream ends at {span} samples, before its pre-skip of {pre}")));
            }
            self.sum.last_granule = last;
            self.sum.duration = span - pre;
            self.sum.end_trim = self.sum.decoded - span;
            self.end = span;
            self.eos = true;
            // nothing may follow (chained streams are not read): one byte is enough to say so
            let mut one = [0u8; 1];
            loop {
                match self.src.read(&mut one) {
                    Ok(0) => break,
                    Ok(_) => return Err(err(Kind::DataAfterEnd, self.offset, pi + 1, "bytes after the end-of-stream page (a chained or appended stream)")),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(err(Kind::Io, self.offset, pi + 1, e.to_string())),
                }
            }
        }
        Ok(())
    }

    /// The next audio packet, or `None` after the end-of-stream page.
    pub fn next_packet(&mut self) -> Result<Option<Packet<'_>>, Error> {
        if self.clear_carry {
            self.carry.clear();
            self.carry_len = 0;
            self.clear_carry = false;
        }
        loop {
            if self.seg == self.nseg {
                if self.eos {
                    if !self.finished {
                        self.finished = true;
                        if self.sum.packets == 0 {
                            return Err(err(Kind::NoAudio, self.page_offset, self.page_index, "no audio packet"));
                        }
                    }
                    return Ok(None);
                }
                if !self.read_page()? {
                    let what = if self.carry_len > 0 { "the file ends inside a packet, with no end-of-stream page" } else { "the file ends with no end-of-stream page" };
                    return Err(err(Kind::NoEndOfStream, self.offset, self.sum.pages, what));
                }
                self.check_audio_page()?;
                continue;
            }
            let start = self.bpos;
            let mut complete = false;
            while self.seg < self.nseg {
                let l = self.page[27 + self.seg] as usize;
                self.seg += 1;
                self.bpos += l;
                if l < 255 {
                    complete = true;
                    break;
                }
            }
            let end = self.bpos;
            if !complete {
                self.carry_append(start, end);
                continue;
            }
            let from_carry = self.carry_len > 0;
            let (offset, first_page) = if from_carry { self.carry_at } else { (self.page_offset + start as u64, self.page_index) };
            if from_carry {
                self.carry_append(start, end);
                self.clear_carry = true;
            }
            let len = if from_carry { self.carry.len() } else { end - start };
            let data: &[u8] = if from_carry { &self.carry } else { &self.page[start..end] };
            let samples = packet_samples(data).unwrap_or(0); // checked with the page
            let before = self.handed;
            let after = before + samples as u64;
            let pre = self.head.pre_skip as u64;
            let lo = before.max(pre).min(after);
            let hi = after.min(self.end).max(lo);
            self.handed = after;
            self.sum.packets += 1;
            self.sum.min_packet_samples = self.sum.min_packet_samples.min(samples);
            self.sum.max_packet_samples = self.sum.max_packet_samples.max(samples);
            self.sum.max_packet_bytes = self.sum.max_packet_bytes.max(len);
            return Ok(Some(Packet {
                data,
                samples,
                decoded_before: before,
                skip: (lo - before) as u32,
                keep: (hi - lo) as u32,
                page: self.page_index,
                first_page,
                offset,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal writer for the tests (streamair's is the real one; streamair/tests/roundtrip.rs reads its output):
    /// pages of whole packets, `granules[i]` on page i, flags as given, lacing as RFC 3533 says.
    fn page(seq: u32, flags: u8, granule: u64, packets: &[&[u8]], continues: Option<usize>) -> Vec<u8> {
        let mut lacing = Vec::new();
        let mut body = Vec::new();
        for (i, p) in packets.iter().enumerate() {
            let mut n = p.len();
            while n >= 255 {
                lacing.push(255);
                n -= 255;
            }
            if !(continues == Some(i)) {
                lacing.push(n as u8);
            }
            body.extend_from_slice(p);
        }
        let mut o = b"OggS".to_vec();
        o.push(0);
        o.push(flags);
        o.extend_from_slice(&granule.to_le_bytes());
        o.extend_from_slice(&7u32.to_le_bytes());
        o.extend_from_slice(&seq.to_le_bytes());
        o.extend_from_slice(&[0; 4]);
        o.push(lacing.len() as u8);
        o.extend_from_slice(&lacing);
        o.extend_from_slice(&body);
        let c = crc32(&o);
        o[22..26].copy_from_slice(&c.to_le_bytes());
        o
    }
    fn head(pre_skip: u16) -> Vec<u8> {
        let mut h = b"OpusHead\x01\x01".to_vec();
        h.extend_from_slice(&pre_skip.to_le_bytes());
        h.extend_from_slice(&48000u32.to_le_bytes());
        h.extend_from_slice(&[0, 0, 0]);
        h
    }
    fn tags() -> Vec<u8> {
        let mut t = b"OpusTags".to_vec();
        t.extend_from_slice(&4u32.to_le_bytes());
        t.extend_from_slice(b"test");
        t.extend_from_slice(&1u32.to_le_bytes());
        t.extend_from_slice(&5u32.to_le_bytes());
        t.extend_from_slice(b"A=b c");
        t
    }
    /// headers + audio pages of 20 ms packets (`per_page` each), the last granule `pre + audio`
    fn stream(pre: u16, packets: usize, per_page: usize, audio: u64) -> Vec<u8> {
        let mut f = page(0, BOS, 0, &[&head(pre)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        let pk = [0xF8u8];
        let mut seq = 2;
        let mut done = 0;
        while done < packets {
            let n = per_page.min(packets - done);
            done += n;
            let last = done == packets;
            let g = if last { pre as u64 + audio } else { done as u64 * 960 };
            let ps: Vec<&[u8]> = (0..n).map(|_| &pk[..]).collect();
            f.extend(page(seq, if last { EOS } else { 0 }, g, &ps, None));
            seq += 1;
        }
        f
    }
    fn read(f: &[u8]) -> Result<Summary, Error> {
        Reader::new(f)?.finish()
    }

    #[test]
    fn crc_is_ogg_not_zlib_and_sliced_equals_bytewise() {
        assert_eq!(crc32(b"123456789"), 0x89A1_897F);
        assert_eq!(crc32_bytewise(b"123456789"), 0x89A1_897F);
        assert_eq!(crc32(b""), 0);
        // every length 0..300 at every alignment 0..8, pseudo-random bytes
        let mut s = 1u64;
        let buf: Vec<u8> = (0..320).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as u8 }).collect();
        for a in 0..8 {
            for n in 0..300 {
                assert_eq!(crc32(&buf[a..a + n]), crc32_bytewise(&buf[a..a + n]), "align {a} len {n}");
            }
        }
        // split anywhere: update composes
        let c = crc32_update(crc32(&buf[..77]), &buf[77..300]);
        assert_eq!(c, crc32(&buf[..300]));
    }

    #[test]
    fn toc_samples() {
        assert_eq!(packet_samples(&[0xF8]), Some(960));
        assert_eq!(packet_samples(&[0xF8 | 1]), Some(1920));
        assert_eq!(packet_samples(&[0xF8 | 2, 0]), Some(1920));
        assert_eq!(packet_samples(&[0x80]), Some(120)); // config 16: CELT NB 2.5 ms
        assert_eq!(packet_samples(&[0x18]), Some(2880)); // config 3: SILK NB 60 ms
        assert_eq!(packet_samples(&[0x60]), Some(480)); // config 12: hybrid SWB 10 ms
        assert_eq!(packet_samples(&[0xF8 | 3, 6]), Some(5760));
        assert_eq!(packet_samples(&[0xF8 | 3, 7]), None);
        assert_eq!(packet_samples(&[0xF8 | 3, 0]), None);
        assert_eq!(packet_samples(&[0xF8 | 3]), None);
        assert_eq!(packet_samples(&[0x18 | 1]), Some(5760)); // 2 × 60 ms: the 120 ms limit itself
        assert_eq!(packet_samples(&[0x18 | 3, 3]), None); // 3 × 60 ms
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn a_stream_reads_and_trims() {
        let f = stream(312, 10, 4, 9000);
        let mut r = Reader::new(&f[..]).unwrap();
        assert_eq!(r.head().pre_skip, 312);
        assert_eq!(r.tags().vendor, "test");
        assert_eq!(r.tags().comments, vec!["A=b c".to_string()]);
        let mut played = 0u64;
        let mut n = 0;
        while let Some(p) = r.next_packet().unwrap() {
            assert_eq!(p.data, &[0xF8]);
            played += p.keep as u64;
            n += 1;
        }
        let s = r.summary().clone();
        assert_eq!((n, s.packets, s.pages, s.decoded, s.last_granule, s.duration, s.end_trim), (10, 10, 5, 9600, 9312, 9000, 288));
        assert_eq!(played, 9000);
        assert_eq!(s.bytes, f.len() as u64);
    }

    #[test]
    fn continuation_across_pages_and_the_exact_multiple_of_255() {
        // a 70,000-byte code-3 packet with padding (TOC + count byte with the padding flag + padding length bytes):
        // more than one page's 65,025 body bytes, so it must continue
        let mut big = vec![0xF8 | 3, 0x40 | 1];
        let pad = 70_000 - 2;
        let mut left = pad;
        while left > 0 { let b = left.min(254); big.push(if left > 254 { 255 } else { b as u8 }); left -= b; }
        big.resize(70_000, 0);
        let exact = vec![0xF8u8; 255 * 255]; // ends with a zero lacing value on the next page
        let mut f = page(0, BOS, 0, &[&head(0)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        // page 2: 255 segments of `big`, no packet completes → granule −1; page 3: the rest of it (275 segments...)
        let first = &big[..255 * 255];
        f.extend(page(2, 0, u64::MAX, &[first], Some(0)));
        let rest = &big[255 * 255..];
        f.extend(page(3, CONTINUED, 960, &[rest], None));
        f.extend(page(4, 0, u64::MAX, &[&exact[..]], Some(0)));
        f.extend(page(5, CONTINUED | EOS, 1920, &[&[][..]], None));
        let mut r = Reader::new(&f[..]).unwrap();
        let p = r.next_packet().unwrap().unwrap();
        assert_eq!((p.data.len(), p.samples, p.first_page, p.page), (70_000, 960, 2, 3));
        let p = r.next_packet().unwrap().unwrap();
        assert_eq!((p.data.len(), p.samples, p.first_page, p.page), (65_025, 960, 4, 5));
        assert!(r.next_packet().unwrap().is_none());
        assert_eq!(r.summary().duration, 1920);
        // the same with a bound below the big packet: refused, by name
        let e = Reader::with_max_packet(&f[..], 65_536).unwrap().finish().unwrap_err();
        assert_eq!(e.kind, Kind::PacketTooLarge);
    }

    #[test]
    fn the_stream_may_start_past_zero_but_not_before_it() {
        let mut f = page(0, BOS, 0, &[&head(100)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        f.extend(page(2, 0, 48_000 + 1920, &[&[0xF8], &[0xF8]], None));
        f.extend(page(3, EOS, 48_000 + 2880 - 10, &[&[0xF8]], None));
        let s = read(&f).unwrap();
        assert_eq!((s.start_granule, s.duration, s.end_trim), (48_000, 2880 - 10 - 100, 10));
        let mut f = page(0, BOS, 0, &[&head(100)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        f.extend(page(2, 0, 1000, &[&[0xF8], &[0xF8]], None));
        f.extend(page(3, EOS, 1960, &[&[0xF8]], None));
        assert_eq!(read(&f).unwrap_err().kind, Kind::GranuleBeforeStart);
        // a single audio page that is also the last may trim: granule ≤ its samples, ≥ pre-skip
        let mut f = page(0, BOS, 0, &[&head(100)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        f.extend(page(2, EOS, 700, &[&[0xF8]], None));
        assert_eq!(read(&f).unwrap().duration, 600);
    }

    #[test]
    fn refusals_are_named() {
        let good = stream(312, 30, 4, 28_000);
        assert!(read(&good).is_ok());
        // page offsets
        let mut offs = vec![];
        let mut i = 0;
        while i < good.len() {
            offs.push(i);
            let n = good[i + 26] as usize;
            i += 27 + n + good[i + 27..i + 27 + n].iter().map(|&l| l as usize).sum::<usize>();
        }
        let recrc = |f: &mut Vec<u8>, at: usize| {
            let n = f[at + 26] as usize;
            let len = 27 + n + f[at + 27..at + 27 + n].iter().map(|&l| l as usize).sum::<usize>();
            f[at + 22..at + 26].copy_from_slice(&[0; 4]);
            let c = crc32(&f[at..at + len]);
            f[at + 22..at + 26].copy_from_slice(&c.to_le_bytes());
        };
        let kind = |f: &[u8]| read(f).unwrap_err().kind;
        let mut f = good.clone();
        f[offs[3] + 22] ^= 1;
        assert_eq!(kind(&f), Kind::Crc);
        let e = read(&f).unwrap_err();
        assert_eq!((e.offset, e.page), (offs[3] as u64 + 22, 3));
        let mut f = good.clone();
        f[offs[3] + 30] ^= 0x80;
        assert_eq!(kind(&f), Kind::Crc);
        assert_eq!(kind(&good[..offs[5] + 10]), Kind::TruncatedPage);
        assert_eq!(kind(&good[..offs[5]]), Kind::NoEndOfStream);
        let mut f = good[..offs[4]].to_vec();
        f.extend_from_slice(&good[offs[5]..]);
        assert_eq!(kind(&f), Kind::Sequence);
        let mut f = good.clone();
        f[offs[3] + 3] = b'X';
        assert_eq!(kind(&f), Kind::CapturePattern);
        let mut f = good.clone();
        f[offs[3] + 4] = 1;
        recrc(&mut f, offs[3]);
        assert_eq!(kind(&f), Kind::Version);
        let mut f = good.clone();
        f[offs[4] + 6..offs[4] + 14].copy_from_slice(&(2 * 4 * 960u64 - 960).to_le_bytes());
        recrc(&mut f, offs[4]);
        assert_eq!(kind(&f), Kind::GranuleBackwards);
        let mut f = good.clone();
        f[offs[4] + 6..offs[4] + 14].copy_from_slice(&(3 * 4 * 960u64 + 960).to_le_bytes());
        recrc(&mut f, offs[4]);
        assert_eq!(kind(&f), Kind::GranuleMismatch);
        let last = *offs.last().unwrap();
        let mut f = good.clone();
        f[last + 6..last + 14].copy_from_slice(&(30 * 960u64 + 1).to_le_bytes());
        recrc(&mut f, last);
        assert_eq!(kind(&f), Kind::GranuleBeyondSamples);
        let mut f = good.clone();
        f[offs[3] + 14] ^= 1;
        recrc(&mut f, offs[3]);
        assert_eq!(kind(&f), Kind::Serial);
        let mut f = good.clone();
        f[offs[3] + 5] |= BOS;
        recrc(&mut f, offs[3]);
        assert_eq!(kind(&f), Kind::HeaderFlags);
        let mut f = good.clone();
        f[offs[3] + 5] |= CONTINUED;
        recrc(&mut f, offs[3]);
        assert_eq!(kind(&f), Kind::Continuation);
        let mut f = good.clone();
        f[27 + 1 + 10] = 0xff; // pre-skip 65,535 > the stream
        f[27 + 1 + 11] = 0xff;
        recrc(&mut f, 0);
        assert_eq!(kind(&f), Kind::PreSkip);
        let mut f = good.clone();
        f[27 + 1 + 7] = b'X';
        recrc(&mut f, 0);
        assert_eq!(kind(&f), Kind::OpusHead);
        let mut f = good.clone();
        f.extend_from_slice(&good[..offs[1]]);
        assert_eq!(kind(&f), Kind::DataAfterEnd);
        let mut f = good.clone();
        f[offs[1] + 28 + 3] = b'X'; // OpusTags → OpuXTags
        recrc(&mut f, offs[1]);
        assert_eq!(kind(&f), Kind::OpusTags);
        assert_eq!(kind(&[]), Kind::OpusHead);
        // an audio packet with a code-3 TOC and no count byte
        let mut f = page(0, BOS, 0, &[&head(0)], None);
        f.extend(page(1, 0, 0, &[&tags()], None));
        f.extend(page(2, EOS, 960, &[&[0xFB]], None));
        assert_eq!(kind(&f), Kind::Packet);
    }

    #[test]
    fn head_families_and_tags_bounds() {
        let mut h = b"OpusHead\x01\x06\x38\x01\x80\xbb\0\0\0\0\x01".to_vec();
        h.extend_from_slice(&[4, 2, 0, 4, 1, 2, 3, 5]);
        let p = OpusHead::parse(&h).unwrap();
        assert_eq!((p.channels, p.mapping_family, p.streams, p.coupled, p.pre_skip), (6, 1, 4, 2, 312));
        assert_eq!(p.mapping, vec![0, 4, 1, 2, 3, 5]);
        h[21 + 5] = 6; // beyond the 6 decoded channels
        assert!(OpusHead::parse(&h).is_err());
        assert!(OpusHead::parse(&h[..24]).is_err());
        assert!(OpusHead::parse(b"OpusHead\x01\x03\0\0\0\0\0\0\0\0\0").is_err()); // family 0, 3 channels
        // a tags packet cut at a bound keeps what fitted
        let t = tags();
        let cut = OpusTags::parse(&t[..20], true, t.len() as u64).unwrap();
        assert_eq!((cut.vendor.as_str(), cut.declared, cut.comments.len(), cut.truncated), ("test", 1, 0, true));
        assert!(OpusTags::parse(&t[..20], false, 20).is_err());
        // and the reader applies the bound to OpusTags without refusing
        let mut f = page(0, BOS, 0, &[&head(0)], None);
        let mut big = tags();
        big.extend(std::iter::repeat_n(b'x', 100_000));
        let (a, b) = big.split_at(255 * 255);
        f.extend(page(1, 0, u64::MAX, &[a], Some(0)));
        f.extend(page(2, CONTINUED, 0, &[b], None));
        f.extend(page(3, EOS, 960, &[&[0xF8]], None));
        let r = Reader::with_max_packet(&f[..], 4096).unwrap();
        assert_eq!((r.tags().truncated, r.tags().bytes, r.tags().vendor.as_str()), (true, big.len() as u64, "test"));
        assert_eq!(r.tags().comments, vec!["A=b c".to_string()]); // what fitted in 4 KiB is kept
        assert_eq!(r.finish().unwrap().duration, 960);
    }
}
