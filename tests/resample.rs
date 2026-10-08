// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.5's oracle comparisons: voaice's WAV reader + mixdown + resampler against what whisper-cli's own
//! `read_audio_data` produced from the same files (testing/oracle/resample_oracle, which links the pinned build's
//! libcommon.a — miniaudio 0.11.24 as whisper-cli compiles it — run by testing/release_gate.sh). `#[ignore]`d
//! because the corpus (testing/make_resample_audio.py, sha256-pinned) and the record are not in git:
//!
//!   testing/release_gate.sh
//!   cargo test --release --test resample -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_AUDIO_RESAMPLE (default .audio/resample), VOAICE_ORACLE_RESAMPLE (default .oracle/resample).
//! Bit patterns only; where values differ the test reports how many and the largest distance in ULPs, then fails.
use std::path::{Path, PathBuf};
use voaice::resample::{self, Converter, Linear, Sample, WavFormat};
use voaice::ulp_distance;

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn audio() -> PathBuf {
    env_path("VOAICE_AUDIO_RESAMPLE", ".audio/resample")
}
fn record() -> PathBuf {
    env_path("VOAICE_ORACLE_RESAMPLE", ".oracle/resample")
}
fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e} (run testing/release_gate.sh first)", p.display()));
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}
/// The pinned corpus, in pin order, each recorded by the reference.
fn corpus() -> Vec<String> {
    let pins = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("testing/pins/resample.sha256"))
        .expect("testing/pins/resample.sha256");
    pins.lines().map(|l| l.split_once("  ").unwrap().1.trim_end_matches(".wav").to_string()).collect()
}
/// (differing values, max ULP distance); a length mismatch counts every value of the longer as differing
fn diff(ours: &[f32], theirs: &[f32]) -> (usize, u64) {
    let mut n = ours.len().abs_diff(theirs.len());
    let mut worst = 0;
    for (a, b) in ours.iter().zip(theirs) {
        if a.to_bits() != b.to_bits() {
            n += 1;
            worst = worst.max(ulp_distance(*a, *b));
        }
    }
    (n, worst)
}
fn wav_bytes(stem: &str) -> Vec<u8> {
    std::fs::read(audio().join(format!("{stem}.wav"))).unwrap()
}
fn header(b: &[u8]) -> WavFormat {
    resample::parse_header(b.len() as u64, |o, buf| {
        buf.copy_from_slice(&b[o as usize..o as usize + buf.len()]);
        Ok(())
    })
    .unwrap()
}

#[test]
#[ignore]
fn oracle_resample_bit_exact() {
    let (mut files, mut samples, mut padded) = (0, 0usize, 0);
    let mut bad = Vec::new();
    for stem in corpus() {
        let theirs = read_f32(&record().join(format!("{stem}.f32")));
        let (fmt, ours) = resample::read(&audio().join(format!("{stem}.wav"))).unwrap();
        let (n, worst) = diff(&ours, &theirs);
        // the length rule's promise beyond what the resampler made: the zero tail whisper-cli's vector carries
        let mut conv = Converter::new(&fmt);
        let mut made = Vec::new();
        let b = wav_bytes(&stem);
        conv.push(&b[fmt.data_offset as usize..(fmt.data_offset + fmt.data_len) as usize], &mut made);
        let tail = theirs.len() - made.len();
        println!(
            "{stem:<22} {:>5} Hz {} ch {:<3} {:>7} frames -> {:>7} samples (zero tail {tail}): {}",
            fmt.rate,
            fmt.channels,
            fmt.sample.name(),
            fmt.frames(),
            theirs.len(),
            if n == 0 { "identical".to_string() } else { format!("{n} differ, max {worst} ULP") }
        );
        files += 1;
        samples += theirs.len();
        padded += (tail > 0) as usize;
        if n != 0 {
            bad.push(stem);
        }
    }
    println!(
        "resampler: {}/{files} files, {samples} samples bit-identical to whisper-cli's read_audio_data ({padded} files end in the length rule's zero tail)",
        files - bad.len()
    );
    assert!(bad.is_empty(), "differ: {bad:?}");
}

#[test]
#[ignore]
fn oracle_resample_streaming_equals_whole() {
    // the corpus pushed in pseudo-random pieces (1 byte .. 9,000, splitting frames) gives the whole run's bits
    let mut s = 0x9e37_79b9u32;
    let mut checked = 0;
    for stem in corpus() {
        let b = wav_bytes(&stem);
        let fmt = header(&b);
        let theirs = read_f32(&record().join(format!("{stem}.f32")));
        let mut conv = Converter::new(&fmt);
        let mut out = Vec::new();
        let mut rest = &b[fmt.data_offset as usize..(fmt.data_offset + fmt.data_len) as usize];
        while !rest.is_empty() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let k = (1 + s % 9000) as usize;
            let k = k.min(rest.len());
            conv.push(&rest[..k], &mut out);
            rest = &rest[k..];
        }
        conv.finish(&mut out);
        assert_eq!(diff(&out, &theirs).0, 0, "{stem}: chunked run differs from the reference");
        checked += 1;
    }
    println!("streaming: {checked}/{checked} files pushed in random 1..9000-byte pieces equal the reference bit for bit");
}

/// What a wrong reading of miniaudio would produce, and on how many files the oracle catches it.
#[test]
#[ignore]
fn oracle_resample_discriminates() {
    let files = corpus();
    let resampled: Vec<&String> = files.iter().filter(|f| !f.starts_with("r16000")).collect();
    // 1. another low-pass order (2 = one biquad; 6 = three), everything else the same
    for order in [2u32, 6] {
        let mut caught = 0;
        for stem in &resampled {
            let b = wav_bytes(stem);
            let fmt = header(&b);
            let theirs = read_f32(&record().join(format!("{stem}.f32")));
            let mut c = Converter::with_order(&fmt, order);
            let mut out = Vec::with_capacity(c.output_len());
            c.push(&b[fmt.data_offset as usize..(fmt.data_offset + fmt.data_len) as usize], &mut out);
            c.finish(&mut out);
            caught += (diff(&out, &theirs).0 > 0) as usize;
        }
        println!("discriminator: low-pass order {order} instead of 4 -> {caught}/{} resampled files differ", resampled.len());
        // files of 1-2 frames emit only the leading 0 (and the tail) and cannot tell any filter apart
        assert!(caught + 8 >= resampled.len() && caught > 30, "order {order} not caught");
    }
    // 2. the stereo mixdown as whisper-cli's --diarize path writes it (L + R, no division) or as the left channel
    //    alone, on the stereo s16 files: decode by hand, then the same resampler
    let stereo: Vec<&String> = files.iter().filter(|f| f.contains("_s_s16")).collect();
    for (name, mix) in [("L + R", (|l: f32, r: f32| l + r) as fn(f32, f32) -> f32), ("L only", |l, _| l)] {
        let mut caught = 0;
        for stem in &stereo {
            let b = wav_bytes(stem);
            let fmt = header(&b);
            assert_eq!((fmt.channels, fmt.sample), (2, Sample::S16));
            let theirs = read_f32(&record().join(format!("{stem}.f32")));
            let d = &b[fmt.data_offset as usize..(fmt.data_offset + fmt.data_len) as usize];
            let mono = d.as_chunks::<4>().0.iter().map(|f| {
                let l = i16::from_le_bytes([f[0], f[1]]) as f32 / 32768.0;
                let r = i16::from_le_bytes([f[2], f[3]]) as f32 / 32768.0;
                mix(l, r)
            });
            let mut out = Vec::new();
            if fmt.rate == 16000 {
                out.extend(mono);
            } else {
                Linear::new(fmt.rate, 16000).feed(mono, &mut out, fmt.output_len() as usize);
            }
            out.resize(fmt.output_len() as usize, 0.0);
            caught += (diff(&out, &theirs).0 > 0) as usize;
        }
        println!("discriminator: stereo mixdown as {name} -> {caught}/{} stereo s16 files differ", stereo.len());
        assert_eq!(caught, stereo.len(), "mixdown {name} not caught");
    }
    // 3. the length as the resampler's own count (no promised extra frame): every file with a zero tail is caught
    let mut caught = 0;
    for stem in &files {
        let b = wav_bytes(stem);
        let fmt = header(&b);
        let theirs = read_f32(&record().join(format!("{stem}.f32")));
        let mut c = Converter::new(&fmt);
        let mut out = Vec::new();
        c.push(&b[fmt.data_offset as usize..(fmt.data_offset + fmt.data_len) as usize], &mut out);
        caught += (out.len() != theirs.len()) as usize;
    }
    println!("discriminator: output length = samples the resampler made (no zero tail) -> {caught}/{} files differ", files.len());
    assert!(caught > 0, "the length rule was not exercised");
}
