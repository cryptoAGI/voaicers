// SPDX-License-Identifier: MIT OR Apache-2.0
//! The oracle comparisons: voaice.rs against what the shipped whisper.cpp library recorded
//! (testing/oracle/whisper_oracle, run by testing/release_gate.sh). `#[ignore]`d because they need the model and
//! the recorded outputs, which are not in git:
//!
//!   testing/release_gate.sh                      # builds the reference and the oracle, records, then runs these
//!   cargo test --release -- --ignored --nocapture
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE (default .oracle/tiny.en),
//! VOAICE_AUDIO (default .audio). Every comparison is of bit patterns, never within a tolerance; where values
//! differ the test reports how many and the largest distance in ULPs before failing.
use std::path::{Path, PathBuf};
use voaice::{mel, model::Model, sha256, ulp_distance, wav};

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn oracle_dir() -> PathBuf {
    env_path("VOAICE_ORACLE", ".oracle/tiny.en")
}
fn model() -> Model {
    let p = env_path("VOAICE_MODEL", "models/ggml-tiny.en.bin");
    Model::load_pinned(&p).unwrap_or_else(|e| panic!("{e}"))
}
fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e} (run testing/release_gate.sh first)", p.display()));
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}
fn hex(b: &[u8]) -> String {
    sha256::hex(b)
}
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// (count of differing values, max ULP distance, index of the worst)
fn compare(ours: &[f32], theirs: &[f32]) -> (usize, u64, usize) {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    let mut n = 0;
    let mut worst = (0u64, 0usize);
    for (i, (a, b)) in ours.iter().zip(theirs).enumerate() {
        if a.to_bits() != b.to_bits() {
            n += 1;
            let d = ulp_distance(*a, *b);
            if d > worst.0 {
                worst = (d, i);
            }
        }
    }
    (n, worst.0, worst.1)
}

fn wavs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(oracle_dir())
        .expect("oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("mel.f32").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded WAVs in the oracle dir");
    v
}

#[test]
#[ignore]
fn oracle_model_hparams_tensors_vocab_filters() {
    let m = model();
    let tsv = std::fs::read_to_string(oracle_dir().join("model.tsv")).unwrap();
    let h = &m.hparams;
    let mut tensor_lines = Vec::new();
    for line in tsv.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let int = |i: usize| f[i].parse::<i64>().unwrap();
        match f[0] {
            "n_vocab" => assert_eq!(int(1), h.n_vocab as i64, "n_vocab"),
            "n_audio_ctx" => assert_eq!(int(1), h.n_audio_ctx as i64),
            "n_audio_state" => assert_eq!(int(1), h.n_audio_state as i64),
            "n_audio_head" => assert_eq!(int(1), h.n_audio_head as i64),
            "n_audio_layer" => assert_eq!(int(1), h.n_audio_layer as i64),
            "n_text_ctx" => assert_eq!(int(1), h.n_text_ctx as i64),
            "n_text_state" => assert_eq!(int(1), h.n_text_state as i64),
            "n_text_head" => assert_eq!(int(1), h.n_text_head as i64),
            "n_text_layer" => assert_eq!(int(1), h.n_text_layer as i64),
            "n_mels" => assert_eq!(int(1), h.n_mels as i64),
            "ftype" => assert_eq!(int(1), h.ftype as i64),
            "model_type" => assert_eq!(f[1], h.model_type()),
            "filters" => assert_eq!((int(1), int(2)), (m.filters_n_mel as i64, m.filters_n_fft as i64)),
            "n_tensors" => assert_eq!(int(1) as usize, m.tensors.len(), "tensor count"),
            "tensor" => tensor_lines.push(f),
            other => panic!("unknown model.tsv key {other}"),
        }
    }
    // the reference's map is sorted by name; so is this list
    let mut ours: Vec<_> = m.tensors.iter().collect();
    ours.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(ours.len(), tensor_lines.len());
    let mut bytes = 0usize;
    for (t, f) in ours.iter().zip(&tensor_lines) {
        assert_eq!(t.name, f[1]);
        assert_eq!(t.dtype.name(), f[2], "{}: type", t.name);
        let ne: Vec<i64> = f[3..7].iter().map(|x| x.parse().unwrap()).collect();
        assert_eq!(&t.ne[..], &ne[..], "{}: shape", t.name);
        assert_eq!(t.nbytes, f[7].parse::<usize>().unwrap(), "{}: nbytes", t.name);
        assert_eq!(hex(&sha256::digest(m.tensor_bytes(t))), f[8], "{}: the bytes the loader holds", t.name);
        bytes += t.nbytes;
    }
    // the filterbank, bit for bit
    let fil = read_f32(&oracle_dir().join("filters.f32"));
    let (n, ulp, _) = compare(&m.filters, &fil);
    assert_eq!(n, 0, "filters differ in {n} values (max {ulp} ULP)");
    // the vocabulary: every id's string as whisper_token_to_str returns it
    let voc = std::fs::read_to_string(oracle_dir().join("vocab.tsv")).unwrap();
    let mut ids = 0;
    let mut nul = 0;
    for line in voc.lines() {
        let (a, b) = line.split_once('\t').unwrap();
        if a == "special" {
            let s = voaice::model::Specials::for_hparams(h);
            assert_eq!(b, format!("eot={} sot={} prev={} solm={} not={} beg={}", s.eot, s.sot, s.prev, s.solm, s.not, s.beg));
            continue;
        }
        let id: usize = a.parse().unwrap();
        // whisper_token_to_str returns std::string::c_str(), so the reference's API ends a token at its first NUL;
        // whisper.cpp itself holds the full bytes (token 188 is the single byte 0x00). Compare what the API can show.
        let ours = &m.vocab[id];
        let shown = &ours[..ours.iter().position(|&c| c == 0).unwrap_or(ours.len())];
        if shown.len() != ours.len() {
            nul += 1;
        }
        assert_eq!(shown, &unhex(b)[..], "token {id}");
        ids += 1;
    }
    assert_eq!(ids, m.vocab.len());
    eprintln!(
        "oracle_model: {} tensors ({} bytes) identical by sha256, filterbank {}x{} bit-exact, {} vocab strings identical ({} hold a NUL the C API cannot show past)",
        ours.len(),
        bytes,
        m.filters_n_mel,
        m.filters_n_fft,
        ids,
        nul
    );
}

#[test]
#[ignore]
fn oracle_pcm_input_identical() {
    for w in wavs() {
        let ours = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
        let theirs = read_f32(&oracle_dir().join(&w).join("pcm.f32"));
        let (n, _, _) = compare(&ours, &theirs);
        assert_eq!(n, 0, "{w}: the f32 samples differ from what was fed to whisper");
    }
}

#[test]
#[ignore]
fn oracle_mel_bit_exact() {
    let m = model();
    let t = mel::Tables::new();
    let mut total = 0usize;
    let mut report = String::new();
    let mut failed = false;
    for w in wavs() {
        let dir = oracle_dir().join(&w);
        let pcm = read_f32(&dir.join("pcm.f32"));
        let meta = std::fs::read_to_string(dir.join("mel.tsv")).unwrap();
        let get = |k: &str| meta.lines().find_map(|l| l.strip_prefix(&format!("{k}\t"))).unwrap().to_string();
        let ours = mel::log_mel_spectrogram(&t, &pcm, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
        assert_eq!(ours.n_len.to_string(), get("n_len"), "{w}: n_len");
        assert_eq!(ours.n_len_org.to_string(), get("n_len_org"), "{w}: n_len_org");
        assert_eq!(get("threads_1_vs_4_bit_identical"), "yes", "{w}: the reference's own mel depends on its thread count");
        let theirs = read_f32(&dir.join("mel.f32"));
        let (n, ulp, at) = compare(&ours.data, &theirs);
        total += theirs.len();
        report += &format!("  {w:<11} {:>2} x {:>4} = {:>7} values: {n} differ, max {ulp} ULP", ours.n_mel, ours.n_len, theirs.len());
        if n > 0 {
            failed = true;
            report += &format!(" (worst at mel {} frame {})", at / ours.n_len, at % ours.n_len);
        }
        report += "\n";
    }
    eprint!("oracle_mel ({total} values):\n{report}");
    assert!(!failed, "the mel is not bit-exact against the reference");
}

/// The oracle must be able to fail: the same pipeline with the FFT's multiply-adds fused (what a build with FMA
/// contraction computes) has to be caught on real audio.
#[test]
#[ignore]
fn oracle_mel_discriminates_fused_fft() {
    let m = model();
    let t = mel::Tables::new();
    let dir = oracle_dir().join("jfk");
    let pcm = read_f32(&dir.join("pcm.f32"));
    let theirs = read_f32(&dir.join("mel.f32"));
    let fused = mel::log_mel_spectrogram_fused_fft(&t, &pcm, &m.filters, 80, 201).unwrap();
    let (n, ulp, _) = compare(&fused.data, &theirs);
    eprintln!("oracle_mel_discriminates: fused-FFT variant on jfk: {n} of {} values differ, max {ulp} ULP", theirs.len());
    assert!(n > 0, "the oracle did not tell a fused FFT from the reference");
}

/// The guard: one flipped bit in a tensor still parses (the format cannot see it), and is refused by the pin.
#[test]
#[ignore]
fn guard_refuses_a_modified_model() {
    let p = env_path("VOAICE_MODEL", "models/ggml-tiny.en.bin");
    let mut b = std::fs::read(&p).unwrap();
    let m = Model::parse(b.clone()).unwrap();
    let t = m.tensor("decoder.token_embedding.weight").unwrap();
    b[t.offset + 12345] ^= 1;
    let changed = Model::parse(b).expect("a flipped weight bit is still a well-formed file");
    assert!(changed.pin.is_none());
    let why = changed.guard("modified tiny.en").err().expect("the guard must refuse it");
    eprintln!("guard: {why}");
    assert!(why.contains("has the size of ggml-tiny.en.bin") && why.contains("not the pinned 921e4cf8"));
}
