// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.4's oracle comparisons: the streaming Ogg/Opus reader against what opus-tools 0.2, libopus 1.4 and libogg
//! 1.3.5 on mindX production said about the same files (testing/opus/oracle.sh record; the files and the answers are
//! pinned in git, so these run offline under plain `cargo test`; `oracle.sh check` re-asks the reference).
//!
//! The good files: 14 written by streamair (silence of five lengths, packets per page 1 / 7 / 255 / 300, a 70,000-byte
//! packet and one of exactly 255 × 255 bytes across pages, every frame size and frame-count code, stereo, pre-skip 0
//! and 3,840) and 21 encoded by opusenc (JFK at 2.5 / 5 / 10 / 20 / 40 / 60 ms, 6 to 64 kb/s, CBR, complexity 0,
//! stereo, downmix, six channels in mapping family 1, a picture that makes OpusTags span pages, UTF-8 comments, the
//! 201-sample file whose only audio page is also its last). The adversarial files: one corruption each, made again
//! here by the rules in testing/opus/mutate.py and checked byte for byte (sha256) against what the reference saw.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use voaice::json::{self, Value};
use voaice::ogg::{self, fnv1a, Kind, Reader, FNV_OFFSET};
use voaice::sha256;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testing/opus")
}
fn lines(name: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    text.lines().filter(|l| !l.is_empty()).map(|l| json::parse(l).unwrap()).collect()
}
fn reference() -> BTreeMap<String, Value> {
    lines("reference.jsonl").into_iter().map(|v| (v.get("file").unwrap().as_str().unwrap().to_string(), v)).collect()
}
fn num(v: &Value, path: &[&str]) -> Option<u64> {
    v.path(path).and_then(Value::as_f64).map(|f| f as u64)
}
fn text<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
    v.path(path).and_then(Value::as_str)
}
/// The pinned good files, in files.sha256's order, each checked against its pin.
fn good_files() -> Vec<(String, Vec<u8>)> {
    let pins = std::fs::read_to_string(dir().join("files.sha256")).unwrap();
    pins.lines()
        .map(|l| {
            let (sha, name) = l.split_once("  ").unwrap();
            let bytes = std::fs::read(dir().join("files").join(name)).unwrap();
            assert_eq!(sha256::hex(&sha256::digest(&bytes)), sha, "{name}: not the pinned bytes");
            (name.to_string(), bytes)
        })
        .collect()
}

/// What voaice.rs reads from one file: the summary, the headers, and the packet digest the reference records.
struct Read {
    sum: ogg::Summary,
    head: ogg::OpusHead,
    tags: ogg::OpusTags,
    packet_fnv: u64,
    kept: u64,
}
fn read(bytes: &[u8]) -> Result<Read, ogg::Error> {
    let mut r = Reader::new(bytes)?;
    let (head, tags) = (r.head().clone(), r.tags().clone());
    let mut h = FNV_OFFSET;
    let mut kept = 0;
    while let Some(p) = r.next_packet()? {
        let mut rec = [0u8; 12];
        rec[..8].copy_from_slice(&(p.data.len() as u64).to_le_bytes());
        rec[8..].copy_from_slice(&p.samples.to_le_bytes());
        h = fnv1a(h, &rec);
        kept += p.keep as u64;
    }
    Ok(Read { sum: r.summary().clone(), head, tags, packet_fnv: h, kept })
}

fn playback_length(samples: u64) -> String {
    let t = samples as f64 / 48000.0;
    let m = (t as u64) / 60;
    let s = t as u64 - m * 60;
    let ms = ((t - (m * 60) as f64 - s as f64) * 1000.0) as u64;
    format!("{m}m:{s:02}.{ms:03}s")
}
fn packet_duration(sum: &ogg::Summary) -> String {
    let ms = |s: f64| s / 48.0;
    format!(
        "{:.1}ms (max), {:>6.1}ms (avg), {:>6.1}ms (min)",
        ms(sum.max_packet_samples as f64),
        ms(sum.decoded as f64 / sum.packets as f64),
        ms(sum.min_packet_samples as f64)
    )
}

/// A comment as opusinfo prints it: `METADATA_BLOCK_PICTURE` decoded from base64 and summarised as
/// `type|mime|description|WxHxDEPTH[/colors]|<N bytes of image data>`; any other comment as it is.
fn as_opusinfo_prints(c: &str) -> String {
    let Some(b64) = c.strip_prefix("METADATA_BLOCK_PICTURE=") else { return c.to_string() };
    let val = |ch: u8| match ch { b'A'..=b'Z' => ch - b'A', b'a'..=b'z' => ch - b'a' + 26, b'0'..=b'9' => ch - b'0' + 52, b'+' => 62, _ => 63 };
    let digits: Vec<u8> = b64.bytes().filter(|&ch| ch != b'=').map(val).collect();
    let b: Vec<u8> = digits.chunks(4).flat_map(|q| {
        let v = q.iter().enumerate().fold(0u32, |a, (i, &d)| a | (d as u32) << (18 - 6 * i));
        [(v >> 16) as u8, (v >> 8) as u8, v as u8].into_iter().take(q.len() - 1)
    }).collect();
    fn u32be(b: &[u8], o: &mut usize) -> u32 {
        *o += 4;
        u32::from_be_bytes(b[*o - 4..*o].try_into().unwrap())
    }
    let mut o = 0;
    let kind = u32be(&b, &mut o);
    let ml = u32be(&b, &mut o) as usize;
    let mime = String::from_utf8_lossy(&b[o..o + ml]).into_owned();
    o += ml;
    let dl = u32be(&b, &mut o) as usize;
    let desc = String::from_utf8_lossy(&b[o..o + dl]).into_owned();
    o += dl;
    let [w, h, depth, colors, n] = [(); 5].map(|_| u32be(&b, &mut o));
    assert_eq!(o + n as usize, b.len(), "picture block length");
    let colors = if colors > 0 { format!("/{colors}") } else { String::new() };
    format!("METADATA_BLOCK_PICTURE={kind}|{mime}|{desc}|{w}x{h}x{depth}{colors}|<{n} bytes of image data>")
}

/// opusinfo warnings that are advice, not a finding about the container: a page holding more than a second of audio
/// ("high muxing delay": streamair's 255- and 300-packet pages, opusenc's --max-delay 0 file) and a pre-skip of 0.
fn advisory(line: &str) -> bool {
    line.contains("high muxing delay") || line.contains("Implausibly low preskip")
}

#[test]
fn oracle_opus_good_files() {
    let refs = reference();
    let files = good_files();
    let mut tally: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut fail = Vec::new();
    for (name, bytes) in &files {
        let r = refs.get(name).unwrap_or_else(|| panic!("{name}: no recorded reference"));
        assert_eq!(text(r, &["sha256"]), Some(sha256::hex(&sha256::digest(bytes)).as_str()), "{name}: recorded for other bytes");
        let ours = read(bytes).unwrap_or_else(|e| panic!("{name}: refused: {e}"));
        let s = &ours.sum;
        let gain = format!("{} dB", ours.head.gain_q8 as f64 / 256.0);
        let comments: Vec<&str> = r.path(&["opusinfo", "comments"]).map(|c| match c {
            Value::Arr(a) => a.iter().map(|x| x.as_str().unwrap()).collect(),
            _ => vec![],
        }).unwrap_or_default();
        let warnings: Vec<&str> = match r.path(&["opusinfo", "warning_lines"]) {
            Some(Value::Arr(a)) => a.iter().map(|x| x.as_str().unwrap()).collect(),
            _ => vec![],
        };
        let checks: Vec<(&str, bool, String)> = vec![
            ("duration = opusdec's samples", Some(s.duration) == num(r, &["opusdec", "frames"]), format!("{} vs {:?}", s.duration, num(r, &["opusdec", "frames"]))),
            ("samples kept by packet = duration", ours.kept == s.duration, format!("{} vs {}", ours.kept, s.duration)),
            ("pre-skip", Some(ours.head.pre_skip as u64) == num(r, &["opusinfo", "pre_skip"]), format!("{}", ours.head.pre_skip)),
            ("channels (opusinfo, opusdec)", Some(ours.head.channels as u64) == num(r, &["opusinfo", "channels"]) && Some(ours.head.channels as u64) == num(r, &["opusdec", "channels"]), format!("{}", ours.head.channels)),
            ("input rate", Some(ours.head.input_rate as u64) == num(r, &["opusinfo", "input_rate"]), format!("{}", ours.head.input_rate)),
            ("output gain", Some(gain.as_str()) == text(r, &["opusinfo", "gain"]), gain.clone()),
            ("vendor", Some(ours.tags.vendor.as_str()) == text(r, &["opusinfo", "vendor"]), ours.tags.vendor.clone()),
            ("comments (a picture as opusinfo summarises it)", ours.tags.comments.iter().map(|c| as_opusinfo_prints(c)).collect::<Vec<_>>() == comments, format!("{:?}", ours.tags.comments.iter().map(|c| c.chars().take(40).collect::<String>()).collect::<Vec<_>>())),
            ("playback length (opusinfo)", Some(playback_length(s.duration).as_str()) == text(r, &["opusinfo", "playback_length"]), playback_length(s.duration)),
            ("packet duration max/avg/min (opusinfo)", Some(packet_duration(s).as_str()) == text(r, &["opusinfo", "packet_duration"]), packet_duration(s)),
            ("bytes (opusinfo total data length)", Some(s.bytes) == num(r, &["opusinfo", "data_length"]), format!("{}", s.bytes)),
            ("pages (libogg)", Some(s.pages) == num(r, &["libogg", "pages"]), format!("{}", s.pages)),
            ("audio packets (libogg)", Some(s.packets) == num(r, &["libogg", "audio_packets"]), format!("{}", s.packets)),
            ("decoded samples (libopus per packet)", Some(s.decoded) == num(r, &["libogg", "libopus_samples"]), format!("{}", s.decoded)),
            ("every page's sequence, granule, flags (libogg)", Some(format!("{:016x}", s.page_fnv).as_str()) == text(r, &["libogg", "page_fnv"]), format!("{:016x}", s.page_fnv)),
            ("every packet's bytes, samples (libogg + libopus)", Some(format!("{:016x}", ours.packet_fnv).as_str()) == text(r, &["libogg", "packet_fnv"]), format!("{:016x}", ours.packet_fnv)),
            ("last granule (libogg)", Some(s.last_granule) == num(r, &["libogg", "last_granule"]), format!("{}", s.last_granule)),
            ("the reference found no fault", num(r, &["opusdec", "exit"]) == Some(0) && warnings.iter().all(|w| advisory(w)) && num(r, &["libogg", "sync_lost"]) == Some(0) && num(r, &["libogg", "holes"]) == Some(0), format!("{warnings:?}")),
        ];
        for (what, ok, ours_says) in checks {
            let t = tally.entry(what).or_default();
            t.1 += 1;
            if ok {
                t.0 += 1;
            } else {
                fail.push(format!("{name}: {what}: ours {ours_says}"));
            }
        }
    }
    println!("oracle_opus_good_files: {} files ({} by streamair, {} by opusenc)", files.len(),
             files.iter().filter(|f| f.0.starts_with("sa_")).count(), files.iter().filter(|f| f.0.starts_with("e_")).count());
    for (what, (ok, n)) in &tally {
        println!("  {ok:>3} / {n:<3} {what}");
    }
    assert!(fail.is_empty(), "{} mismatches:\n{}", fail.len(), fail.join("\n"));
}

// ---------------------------------------------------------------------------------------------------------------
// The adversarial files, made by the rules of testing/opus/mutate.py

fn pages(f: &[u8]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut i = 0;
    while i < f.len() {
        let n = f[i + 26] as usize;
        let len = 27 + n + f[i + 27..i + 27 + n].iter().map(|&l| l as usize).sum::<usize>();
        out.push((i, len));
        i += len;
    }
    out
}
fn recrc(f: &mut [u8], (at, len): (usize, usize)) {
    f[at + 22..at + 26].copy_from_slice(&[0; 4]);
    let c = ogg::crc32(&f[at..at + len]);
    f[at + 22..at + 26].copy_from_slice(&c.to_le_bytes());
}
fn granule(f: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(f[at + 6..at + 14].try_into().unwrap())
}
fn set_granule(f: &mut [u8], p: (usize, usize), g: u64) {
    f[p.0 + 6..p.0 + 14].copy_from_slice(&g.to_le_bytes());
    recrc(f, p);
}
fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn mutation(name: &str, a: &[u8], b: &[u8]) -> Vec<u8> {
    let p = pages(a);
    let (p3, p4, pl) = (p[3], p[4], *p.last().unwrap());
    let mut f = a.to_vec();
    match name {
        "m_crc_field" => f[p3.0 + 22] ^= 1,
        "m_body_bit" => f[p3.0 + 27 + a[p3.0 + 26] as usize + 100] ^= 0x80,
        "m_truncated_mid_page" => f.truncate(pl.0 + pl.1 / 2),
        "m_truncated_last_byte" => f.truncate(a.len() - 1),
        "m_truncated_header" => f.truncate(pl.0 + 10),
        "m_no_eos" => f.truncate(pl.0),
        "m_dropped_page" => f = cat(&[&a[..p4.0], &a[p4.0 + p4.1..]]),
        "m_duplicated_page" => f = cat(&[&a[..p4.0 + p4.1], &a[p4.0..]]),
        "m_capture" => f[p3.0 + 3] = b'X',
        "m_version" => { f[p3.0 + 4] = 1; recrc(&mut f, p3) }
        "m_granule_backwards" => set_granule(&mut f, p4, granule(a, p3.0) - 960),
        "m_granule_mismatch" => set_granule(&mut f, p4, granule(a, p4.0) + 960),
        "m_granule_beyond_eos" => set_granule(&mut f, pl, granule(a, pl.0) + 5760),
        "m_serial" => { f[p3.0 + 14] ^= 1; recrc(&mut f, p3) }
        "m_bos_again" => { f[p3.0 + 5] |= ogg::BOS; recrc(&mut f, p3) }
        "m_continued_flag" => { f[p3.0 + 5] ^= ogg::CONTINUED; recrc(&mut f, p3) }
        "m_eos_removed" => { f[pl.0 + 5] &= !ogg::EOS; recrc(&mut f, pl) }
        "m_head_magic" => { f[p[0].0 + 28 + 7] = b'X'; recrc(&mut f, p[0]) }
        "m_tags_magic" => { f[p[1].0 + 27 + a[p[1].0 + 26] as usize + 7] = b'X'; recrc(&mut f, p[1]) }
        "m_appended_stream" => f = cat(&[a, a]),
        "m_start_offset" => {
            for &q in &p[2..] {
                if granule(a, q.0) != u64::MAX {
                    set_granule(&mut f, q, granule(a, q.0) + 48000);
                }
            }
        }
        "m_preskip_over" => {
            f = b.to_vec();
            f[28 + 10..28 + 12].copy_from_slice(&2000u16.to_le_bytes());
            recrc(&mut f, pages(b)[0]);
        }
        _ => panic!("unknown mutation {name}"),
    }
    f
}

#[test]
fn oracle_opus_adversarial_files_refused() {
    let refs = reference();
    let files: BTreeMap<String, Vec<u8>> = good_files().into_iter().collect();
    let base_frames = |n: &str| num(&refs[n], &["opusdec", "frames"]);
    let mut n = 0;
    let mut noticed = 0;
    for m in lines("mutations.jsonl") {
        let name = m.get("name").unwrap().as_str().unwrap();
        let base = m.get("base").unwrap().as_str().unwrap();
        let expect = m.get("expect").unwrap().as_str().unwrap();
        let f = mutation(name, &files["e_jfk_m_20ms_24k.opus"], &files["e_min_len_24k.opus"]);
        assert_eq!(Some(sha256::hex(&sha256::digest(&f)).as_str()), m.get("sha256").unwrap().as_str(), "{name}: not the bytes the reference saw");
        let r = &refs[&format!("{name}.opus")];
        // did the reference notice? a warning from opusinfo, a failure of opusdec, or a different sample count
        let warned = num(r, &["opusinfo", "warnings"]).unwrap_or(0) > 0 || num(r, &["opusinfo", "exit"]) != Some(0);
        let dec_differs = num(r, &["opusdec", "exit"]) != Some(0) || num(r, &["opusdec", "frames"]) != base_frames(base);
        let got = read(&f);
        if expect == "ok" {
            let s = got.unwrap_or_else(|e| panic!("{name}: refused a valid stream: {e}")).sum;
            assert_eq!(Some(s.duration), num(r, &["opusdec", "frames"]), "{name}: duration");
            assert_eq!(s.start_granule, 48000, "{name}");
            println!("  accepted   {name:<24} start granule {}, duration {} = opusdec's {} (reference warned: {warned})", s.start_granule, s.duration, s.duration);
        } else {
            let e = match got {
                Ok(r) => panic!("{name}: accepted (duration {}), expected {expect}", r.sum.duration),
                Err(e) => e,
            };
            assert_eq!(format!("{:?}", e.kind), expect, "{name}: {e}");
            noticed += (warned || dec_differs) as usize;
            println!("  refused    {name:<24} {e}  | reference: opusinfo {} warning(s), opusdec {}",
                     num(r, &["opusinfo", "warnings"]).unwrap_or(0),
                     if dec_differs { format!("{:?} samples (exit {:?})", num(r, &["opusdec", "frames"]), num(r, &["opusdec", "exit"])) } else { "the base's samples".into() });
        }
        n += 1;
    }
    let refused = n - 1;
    println!("oracle_opus_adversarial_files_refused: {refused} / {refused} refused with the expected kind, 1 / 1 valid variant accepted; \
              the reference noticed {noticed} of the {refused} (opusinfo warned or opusdec's output changed)");
    assert_eq!(n, lines("mutations.jsonl").len());
}

// ---------------------------------------------------------------------------------------------------------------
// Discriminators: readers that are wrong in plausible ways, and the oracle catching each

#[test]
fn oracle_opus_discriminators() {
    let refs = reference();
    let files = good_files();
    let (mut add, mut untrimmed, mut one_frame, mut zlib_pages, mut ogg_pages, mut total_pages) = (0, 0, 0, 0, 0, 0);
    for (name, bytes) in &files {
        let ours = read(bytes).unwrap();
        let s = &ours.sum;
        let want = num(&refs[name], &["opusdec", "frames"]).unwrap();
        let pre = ours.head.pre_skip as u64;
        // 1. pre-skip added to the granule instead of subtracted
        add += (s.last_granule - s.start_granule + pre != want) as usize;
        // 2. no end trimming: every decoded sample after the pre-skip played
        untrimmed += (s.decoded - pre != want) as usize;
        // 3. the frame-count byte ignored: a code-3 packet counted as one frame
        let mut r = Reader::new(&bytes[..]).unwrap();
        let mut d = 0u64;
        while let Some(p) = r.next_packet().unwrap() {
            d += if p.data[0] & 3 == 3 { p.samples as u64 / (p.data[1] & 0x3f) as u64 } else { p.samples as u64 };
        }
        one_frame += (d != s.decoded) as usize;
        // 4. zlib's CRC-32 (reflected, init and xorout 0xFFFFFFFF) against Ogg's, on every page
        for (at, len) in pages(bytes) {
            let mut pg = bytes[at..at + len].to_vec();
            let stored = u32::from_le_bytes(pg[22..26].try_into().unwrap());
            pg[22..26].copy_from_slice(&[0; 4]);
            zlib_pages += (zlib_crc32(&pg) == stored) as usize;
            ogg_pages += (ogg::crc32(&pg) == stored) as usize;
            total_pages += 1;
        }
    }
    let n = files.len();
    println!("oracle_opus_discriminators over {n} files, {total_pages} pages:");
    println!("  pre-skip added instead of subtracted: wrong on {add} / {n} files (right only where pre-skip is 0)");
    println!("  no end trimming:                      wrong on {untrimmed} / {n} files");
    println!("  code-3 frame count ignored:           wrong on {one_frame} / {n} files (those with multi-frame packets)");
    println!("  zlib's CRC-32:                        verifies {zlib_pages} / {total_pages} pages; Ogg's verifies {ogg_pages} / {total_pages}");
    assert!(add >= n - 1 && untrimmed > n / 2 && one_frame >= 3, "a discriminator was not caught");
    assert_eq!((zlib_pages, ogg_pages), (0, total_pages));
}

/// zlib's CRC-32, bit by bit (the discriminator only).
fn zlib_crc32(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for &x in b {
        c ^= x as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

#[test]
fn bounded_memory_one_byte_at_a_time() {
    // a source that yields one byte per read: the reader must not need more than it asks for, and gives the same answer
    struct Trickle<'a>(&'a [u8]);
    impl std::io::Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0.is_empty() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.0[0];
            self.0 = &self.0[1..];
            Ok(1)
        }
    }
    for (name, bytes) in good_files().iter().filter(|(n, _)| n.contains("continuation") || n.contains("picture") || n.contains("2.5ms")) {
        let a = Reader::new(&bytes[..]).unwrap().finish().unwrap();
        let b = Reader::new(Trickle(bytes)).unwrap().finish().unwrap();
        assert_eq!(a, b, "{name}");
    }
    let _ = Kind::Io;
}
