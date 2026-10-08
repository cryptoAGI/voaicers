//! Round trips: what streamair's writer muxes, voaice.rs's streaming reader (voaice::ogg, 0.0.4) reads back — every
//! packet's bytes in order, the page count, and the duration the writer was asked for, to the sample. The writer is
//! proven against libopus 1.4 (testing/oracle.sh) and the reader against opus-tools 0.2 (voaice.rs testing/opus/);
//! this holds the two to the same page format across many more shapes than either oracle runs.
use streamair::ogg::{mux, packet_samples, Stream};
use voaice::ogg::Reader;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

#[test]
fn writer_to_reader_round_trips() {
    let tocs: [&[u8]; 8] = [&[0xF8], &[0xE0], &[0xF0], &[0x18], &[0xF9], &[0xFA, 0], &[0xFB, 3], &[0x68]];
    let mut s = 4004u64;
    let mut n = 0;
    for case in 0..400 {
        let count = 1 + (lcg(&mut s) % 300) as usize;
        let per_page = 1 + (lcg(&mut s) % 260) as usize;
        let channels = 1 + (case % 2) as u8;
        let pre_skip = (lcg(&mut s) % 4000) as u16;
        // packets: a TOC from the table, then a payload of 0..2000 bytes (a few of them past one page)
        let packets: Vec<Vec<u8>> = (0..count)
            .map(|_| {
                let mut p = tocs[(lcg(&mut s) % 8) as usize].to_vec();
                let extra = match lcg(&mut s) % 50 { 0 => 70_000, 1 => 255 * 255 - p.len(), 2..=5 => 255 * (1 + lcg(&mut s) as usize % 4) - p.len(), _ => (lcg(&mut s) % 2000) as usize };
                p.extend((0..extra).map(|i| i as u8));
                p
            })
            .collect();
        let decoded: u64 = packets.iter().map(|p| packet_samples(p).unwrap() as u64).sum();
        if decoded <= pre_skip as u64 {
            continue;
        }
        // end trimming within the last packet (RFC 7845 §4.4's SHOULD; past the last page it is an error, below)
        let last = packet_samples(packets.last().unwrap()).unwrap() as u64;
        let trim = (lcg(&mut s) % last).min(decoded - pre_skip as u64 - 1);
        let audio = decoded - pre_skip as u64 - trim;
        let st = Stream { serial: case, channels, pre_skip, input_rate: 48000, vendor: "streamair round trip",
                          comments: &["A=1", "B=two"], samples_48k: audio, packets_per_page: per_page };
        let f = mux(&st, &packets).unwrap();
        let mut r = Reader::new(&f[..]).unwrap_or_else(|e| panic!("case {case}: {e}"));
        assert_eq!((r.head().channels, r.head().pre_skip), (channels, pre_skip));
        assert_eq!(r.tags().vendor, "streamair round trip");
        assert_eq!(r.tags().comments, vec!["A=1".to_string(), "B=two".to_string()]);
        let mut i = 0;
        let mut kept = 0u64;
        while let Some(p) = r.next_packet().unwrap_or_else(|e| panic!("case {case}: {e}")) {
            assert_eq!(p.data, &packets[i][..], "case {case} packet {i}");
            kept += p.keep as u64;
            i += 1;
        }
        let sum = r.summary();
        assert_eq!((i, sum.duration, kept, sum.decoded, sum.bytes), (count, audio, audio, decoded, f.len() as u64), "case {case}");
        n += 1;
    }
    assert!(n > 350, "{n} cases ran");
}

#[test]
fn the_silence_cli_lengths_read_back_exactly() {
    // the five lengths streamair's own oracle checks against opusdec, now read by voaice.rs
    for secs in [0.001f64, 1.0, 2.5, 7.3333, 60.0] {
        let samples = (secs * 48000.0).round() as u64;
        let n = (312 + samples).div_ceil(960) as usize;
        let packets: Vec<Vec<u8>> = (0..n).map(|_| vec![0xF8]).collect();
        let st = Stream { serial: 1, channels: 1, pre_skip: 312, input_rate: 48000, vendor: "v", comments: &[],
                          samples_48k: samples, packets_per_page: 50 };
        let f = mux(&st, &packets).unwrap();
        assert_eq!(Reader::new(&f[..]).unwrap().finish().unwrap().duration, samples, "{secs} s");
    }
}

#[test]
fn end_trimming_stays_on_the_last_page() {
    // Was a known issue (found by the 0.0.4 oracle): streamair 0.0.1 bounded the trim by 5,760 samples, not by the
    // samples on the last page. Here the natural last page would hold one 2.5 ms packet (120 samples) against a trim of
    // 500, so its EOS granule (9,220) fell below the previous page's (9,600), and opusinfo 0.2 called the file an
    // ERROR. The writer now starts the last page early enough that the trim lies inside it, and voaice reads the
    // exact length back.
    let mut p: Vec<Vec<u8>> = (0..10).map(|_| vec![0xF8]).collect();
    p.push(vec![0xE0]);
    let st = Stream { serial: 1, channels: 1, pre_skip: 312, input_rate: 48000, vendor: "v", comments: &[],
                      samples_48k: 9600 + 120 - 312 - 500, packets_per_page: 10 };
    let f = mux(&st, &p).unwrap();
    let sum = Reader::new(&f[..]).unwrap().finish().unwrap();
    assert_eq!(sum.duration, 9600 + 120 - 312 - 500);
    if let Ok(out) = std::env::var("STREAMAIR_TRIM_OUT") {
        std::fs::write(out, &f).unwrap(); // for opusinfo, the external check
    }
}
