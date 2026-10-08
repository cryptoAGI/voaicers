// SPDX-License-Identifier: MIT OR Apache-2.0
//! The oracle comparisons: voaice.rs against what the shipped whisper.cpp library recorded
//! (testing/oracle/whisper_oracle, run by testing/release_gate.sh). `#[ignore]`d because they need the model and
//! the recorded outputs, which are not in git:
//!
//!   testing/release_gate.sh                      # builds the reference and the oracle, records, then runs these
//!   cargo test --release -- --ignored --nocapture
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE (default .oracle/tiny.en),
//! VOAICE_AUDIO (default .audio), VOAICE_ORACLE_F16 (default .oracle/f16: `whisper_oracle --f16`, 0.0.3). Every comparison is of bit patterns, never within a tolerance; where values
//! differ the test reports how many and the largest distance in ULPs before failing.
use std::path::{Path, PathBuf};
use voaice::{f16, gelu, mel, model::Model, sha256, ulp_distance, wav};

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

/// Threads split frames, each frame computed whole by one thread: the mel at 2, 3, 4 and 8 threads must have the
/// 1-thread bits on every input (and so the reference's, which `oracle_mel_bit_exact` checks for 1 thread).
#[test]
#[ignore]
fn oracle_mel_threads_bit_identical() {
    let m = model();
    let t = mel::Tables::new();
    let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
    let mut checked = 0usize;
    for w in wavs() {
        let pcm = read_f32(&oracle_dir().join(&w).join("pcm.f32"));
        let theirs = read_f32(&oracle_dir().join(&w).join("mel.f32"));
        let one = plan.run(&pcm, 1).unwrap();
        for threads in [2, 3, 4, 8] {
            let many = plan.run(&pcm, threads).unwrap();
            let (n, ulp, _) = compare(&many.data, &one.data);
            assert_eq!(n, 0, "{w}: {threads} threads differ from 1 thread in {n} values (max {ulp} ULP)");
            let (n, ulp, _) = compare(&many.data, &theirs);
            assert_eq!(n, 0, "{w}: {threads} threads differ from the reference in {n} values (max {ulp} ULP)");
            checked += many.data.len();
        }
    }
    eprintln!("oracle_mel_threads: 8 inputs x threads {{2, 3, 4, 8}}: {checked} values, all identical to 1 thread and to the reference");
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

// ---- 0.0.3: f32 <-> f16 and GELU against the shipped libggml-base / libggml-cpu (`whisper_oracle --f16`) ----------

fn f16_dir() -> PathBuf {
    env_path("VOAICE_ORACLE_F16", ".oracle/f16")
}
fn read_raw(name: &str) -> Vec<u8> {
    let p = f16_dir().join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run testing/release_gate.sh first)", p.display()))
}
fn read_u16(name: &str) -> Vec<u16> {
    read_raw(name).as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}
fn read_u32(name: &str) -> Vec<u32> {
    read_raw(name).as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}
fn read_u64(name: &str) -> Vec<u64> {
    read_raw(name).as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
}
fn f16_meta(key: &str) -> String {
    let t = std::fs::read_to_string(f16_dir().join("f16.tsv")).unwrap();
    t.lines().find_map(|l| l.strip_prefix(&format!("{key}\t"))).unwrap_or_else(|| panic!("f16.tsv: no {key}")).to_string()
}
/// The oracle's digest: 64-bit FNV-1a over u64 words, f16 outputs packed four to a word, f32 two (little-endian).
fn digest16(y: &[u16]) -> u64 {
    y.as_chunks::<4>().0.iter().fold(0xcbf2_9ce4_8422_2325, |h, w| {
        let w = w[0] as u64 | (w[1] as u64) << 16 | (w[2] as u64) << 32 | (w[3] as u64) << 48;
        (h ^ w).wrapping_mul(0x0100_0000_01b3)
    })
}
fn digest32(y: &[f32]) -> u64 {
    y.as_chunks::<2>().0.iter().fold(0xcbf2_9ce4_8422_2325, |h, w| {
        (h ^ (w[0].to_bits() as u64 | (w[1].to_bits() as u64) << 32)).wrapping_mul(0x0100_0000_01b3)
    })
}
/// For every chunk c of 65,536 f32 patterns (c = the high 16 bits), `each(c, chunk)` -> its digests; spread over the
/// cores (each chunk is independent, so the order of work cannot change a digest).
fn every_f32_chunk<const K: usize>(each: impl Fn(&[f32]) -> [u64; K] + Sync) -> Vec<[u64; K]> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut out = vec![[0u64; K]; 1 << 16];
    let per = (1usize << 16).div_ceil(threads);
    std::thread::scope(|s| {
        for (t, part) in out.chunks_mut(per).enumerate() {
            let each = &each;
            s.spawn(move || {
                let mut x = vec![0f32; 1 << 16];
                for (k, d) in part.iter_mut().enumerate() {
                    let c = (t * per + k) as u32;
                    for (i, v) in x.iter_mut().enumerate() {
                        *v = f32::from_bits(c << 16 | i as u32);
                    }
                    *d = each(&x);
                }
            });
        }
    });
    out
}
fn count_ne<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len(), "lengths differ");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}
fn first_ne<T: PartialEq + std::fmt::Debug>(inputs: &[u32], a: &[T], b: &[T]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        None => String::new(),
        Some(i) => format!(" (first: input {:#010x} -> ours {:?}, reference {:?})", inputs[i], a[i], b[i]),
    }
}

/// Every f16 pattern widened: the port against libggml-base's `ggml_fp16_to_fp32`, ggml-cpu's table, its F16C row
/// and its scalar tail.
#[test]
#[ignore]
fn oracle_f16_to_f32_all_65536() {
    let ours: Vec<u32> = (0..=u16::MAX).map(|h| f16::fp16_to_fp32(h).to_bits()).collect();
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let mut row = vec![0f32; all.len()];
    f16::fp16_to_fp32_row(&all, &mut row);
    let row: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
    let inputs: Vec<u32> = all.iter().map(|&h| h as u32).collect();
    let mut report = String::new();
    for (name, mine) in [("base", &ours), ("table", &ours), ("cpu_tail", &ours), ("cpu_row", &row)] {
        let theirs = read_u32(&format!("f16_to_f32.{name}.u32"));
        let n = count_ne(mine, &theirs);
        report += &format!("  ggml {name:<8}: {} / 65536 identical{}\n", 65536 - n, first_ne(&inputs, mine, &theirs));
        assert_eq!(n, 0, "f16 -> f32 differs from the reference's {name} in {n} patterns\n{report}");
    }
    eprint!("oracle_f16_to_f32 (port: scalar for base/table/tail, row for the F16C row):\n{report}");
}

/// The boundary set narrowed: the portable scalar against libggml-base and ggml-cpu's inlined copy (the row tail);
/// the row (F16C) and the software model of `vcvtps2ph` against ggml-cpu's row.
#[test]
#[ignore]
fn oracle_f32_to_f16_boundary_set() {
    let inputs = read_u32("f32_inputs.u32");
    assert_eq!(inputs.len().to_string(), f16_meta("f32_inputs"));
    assert_eq!(f16_meta("f16c"), "yes", "the recorded reference was not an F16C build: its row is not the one ported");
    assert_eq!(f16_meta("from_float_f16_is_ggml_cpu_fp32_to_fp16"), "yes", "mul_mat's F16 from_float is not ggml_cpu_fp32_to_fp16");
    let x: Vec<f32> = inputs.iter().map(|&u| f32::from_bits(u)).collect();
    let scalar: Vec<u16> = x.iter().map(|&v| f16::fp32_to_fp16(v)).collect();
    let model: Vec<u16> = x.iter().map(|&v| f16::fp32_to_fp16_f16c(v)).collect();
    let mut row = vec![0u16; x.len()];
    f16::fp32_to_fp16_row(&x, &mut row);
    let mut report = String::new();
    for (what, mine, name) in [
        ("scalar   vs libggml-base ggml_fp32_to_fp16      ", &scalar, "base"),
        ("scalar   vs ggml-cpu GGML_CPU_FP32_TO_FP16 (tail)", &scalar, "cpu_tail"),
        ("row      vs ggml-cpu ggml_cpu_fp32_to_fp16 (F16C)", &row, "cpu_row"),
        ("vcvtps2ph model vs ggml_cpu_fp32_to_fp16 (F16C)  ", &model, "cpu_row"),
    ] {
        let theirs = read_u16(&format!("f32_to_f16.{name}.u16"));
        let n = count_ne(mine, &theirs);
        report += &format!("  {what}: {} / {} identical{}\n", x.len() - n, x.len(), first_ne(&inputs, mine, &theirs));
        assert_eq!(n, 0, "f32 -> f16 differs in {n} values\n{report}");
    }
    let nan = x.iter().filter(|v| v.is_nan()).count();
    let nan_differ = count_ne(&scalar, &model);
    eprint!("oracle_f32_to_f16_boundary_set ({} inputs, {nan} NaN; the portable and F16C conversions differ on {nan_differ}, all NaN):\n{report}", x.len());
    assert!(nan_differ > 0 && nan_differ <= nan);
}

/// All 2^32 f32 patterns, by per-chunk digest: the scalar against libggml-base and ggml-cpu's tail, the row and the
/// `vcvtps2ph` model against ggml-cpu's row.
#[test]
#[ignore]
fn oracle_f32_to_f16_every_pattern() {
    let base = read_u64("f32_to_f16.base.digest");
    let tail = read_u64("f32_to_f16.cpu_tail.digest");
    let rowd = read_u64("f32_to_f16.cpu_row.digest");
    let t = std::time::Instant::now();
    let ours = every_f32_chunk::<3>(|x| {
        let mut y = vec![0u16; x.len()];
        for (o, &v) in y.iter_mut().zip(x) {
            *o = f16::fp32_to_fp16(v);
        }
        let s = digest16(&y);
        f16::fp32_to_fp16_row(x, &mut y);
        let r = digest16(&y);
        for (o, &v) in y.iter_mut().zip(x) {
            *o = f16::fp32_to_fp16_f16c(v);
        }
        [s, r, digest16(&y)]
    });
    let col = |k: usize| ours.iter().map(|d| d[k]).collect::<Vec<_>>();
    let (s, r, m) = (col(0), col(1), col(2));
    let checks = [
        ("scalar vs libggml-base", count_ne(&s, &base)),
        ("scalar vs ggml-cpu tail", count_ne(&s, &tail)),
        ("row vs ggml-cpu row (F16C)", count_ne(&r, &rowd)),
        ("vcvtps2ph model vs ggml-cpu row", count_ne(&m, &rowd)),
    ];
    let mut report = String::new();
    for (what, n) in checks {
        report += &format!("  {what:<32}: {} / 65536 chunks identical ({} of 4,294,967,296 patterns)\n", 65536 - n, if n == 0 { "all" } else { "NOT all" });
    }
    eprint!("oracle_f32_to_f16_every_pattern ({:.1} s here; the reference's own scalar copies agree on {} chunks, its row and scalar on {}):\n{report}",
        t.elapsed().as_secs_f64(), f16_meta("chunks_base_eq_cpu_tail"), f16_meta("chunks_base_eq_cpu_row"));
    assert!(checks.iter().all(|c| c.1 == 0), "not every f32 pattern narrows to the reference's f16");
}

/// The oracle must be able to fail: rounding exact ties away from zero instead of to even.
#[test]
#[ignore]
fn oracle_f16_discriminates_round_half_away() {
    let inputs = read_u32("f32_inputs.u32");
    let theirs = read_u16("f32_to_f16.cpu_row.u16");
    let rna: Vec<u16> = inputs.iter().map(|&u| f16::fp32_to_fp16_round_half_away(f32::from_bits(u))).collect();
    let n = count_ne(&rna, &theirs);
    let rowd = read_u64("f32_to_f16.cpu_row.digest");
    let ours = every_f32_chunk::<1>(|x| {
        let y: Vec<u16> = x.iter().map(|&v| f16::fp32_to_fp16_round_half_away(v)).collect();
        [digest16(&y)]
    });
    let chunks = ours.iter().zip(&rowd).filter(|(a, b)| a[0] != **b).count();
    eprintln!("oracle_f16_discriminates: round-half-away differs in {n} of {} boundary values{}, and in {chunks} of 65536 chunks of all 2^32",
        inputs.len(), first_ne(&inputs, &rna, &theirs));
    assert!(n > 0 && chunks > 0, "the oracle did not tell round-half-away from round-to-even");
}

/// `ggml_table_gelu_f16`, all 65,536 entries, against the port's table built the way ggml_cpu_init builds it.
#[test]
#[ignore]
fn oracle_gelu_table_all_65536() {
    let theirs = read_u16("gelu_table.u16");
    let t = std::time::Instant::now();
    let g = gelu::Gelu::new();
    let took = t.elapsed();
    let idx: Vec<u32> = (0..65536).collect();
    let n = count_ne(&g.f16, &theirs);
    eprintln!("oracle_gelu_table: {} / 65536 entries identical{} (built in {:.2} ms)", 65536 - n, first_ne(&idx, &g.f16, &theirs), took.as_secs_f64() * 1e3);
    assert_eq!(n, 0, "the GELU table differs in {n} entries");
}

/// The oracle must be able to fail: the source's order without the FMA GCC formed (a build without contraction).
#[test]
#[ignore]
fn oracle_gelu_table_discriminates_unfused() {
    let theirs = read_u16("gelu_table.u16");
    let unfused = gelu::table_with(gelu::gelu_f32_unfused);
    let idx: Vec<u32> = (0..65536).collect();
    let n = count_ne(&unfused, &theirs);
    eprintln!("oracle_gelu_discriminates: the unfused GELU differs in {n} of 65536 table entries{}", first_ne(&idx, &unfused, &theirs));
    assert!(n > 0, "the oracle did not tell an unfused GELU from the shipped one");
}

/// The GELU op as the encoder runs it (`ggml_gelu` through the shipped CPU backend): the boundary set and every f16
/// value, then all 2^32 f32 patterns by digest — the vector path and the scalar loop both.
#[test]
#[ignore]
fn oracle_gelu_op() {
    assert_eq!(f16_meta("gelu_threads_1_vs_4_bit_identical"), "yes", "the reference's gelu depends on its thread count");
    let g = gelu::Gelu::new();
    let mut x: Vec<f32> = read_u32("f32_inputs.u32").iter().map(|&u| f32::from_bits(u)).collect();
    x.extend((0..=u16::MAX).map(f16::fp16_to_fp32));
    let theirs = read_u32("gelu.u32");
    assert_eq!(x.len().to_string(), f16_meta("gelu_inputs"));
    let inputs: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
    let mut report = String::new();
    for (what, vector) in [("vector", true), ("scalar", false)] {
        let mut y = vec![0f32; x.len()];
        if vector { g.row(&x, &mut y) } else { g.row_scalar(&x, &mut y) }
        let y: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
        let n = count_ne(&y, &theirs);
        report += &format!("  {what}: {} / {} identical{}\n", x.len() - n, x.len(), first_ne(&inputs, &y, &theirs));
        assert_eq!(n, 0, "the GELU op differs in {n} values\n{report}");
    }
    let dig = read_u64("gelu.digest");
    let t = std::time::Instant::now();
    let ours = every_f32_chunk::<2>(|x| {
        let mut y = vec![0f32; x.len()];
        g.row(x, &mut y);
        let v = digest32(&y);
        g.row_scalar(x, &mut y);
        [v, digest32(&y)]
    });
    let nv = ours.iter().zip(&dig).filter(|(a, b)| a[0] != **b).count();
    let ns = ours.iter().zip(&dig).filter(|(a, b)| a[1] != **b).count();
    report += &format!("  every f32 pattern ({:.1} s here): vector {} / 65536 chunks identical, scalar {} / 65536\n", t.elapsed().as_secs_f64(), 65536 - nv, 65536 - ns);
    eprint!("oracle_gelu_op:\n{report}");
    assert!(nv == 0 && ns == 0, "the GELU op differs from the reference on some f32 pattern");
}
