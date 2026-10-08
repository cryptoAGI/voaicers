// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.6: the encoder's conv1 and `ggml_vec_dot_f16` against what the shipped library computed
//! (`whisper_oracle --conv1`, run by testing/release_gate.sh into .oracle/conv1). `#[ignore]`d: they need the model
//! and the record, which are not in git.
//!
//!   cargo test --release --test conv1 -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_CONV1 (default .oracle/conv1), VOAICE_AUDIO
//! (default .audio). The conv graph's nodes were read through ggml's scheduler eval callback; every comparison is of
//! bit patterns. voaice's side starts from the WAV: its own samples, its own mel (0.0.1's oracle), its own conv1.
use std::path::{Path, PathBuf};
use voaice::conv::{self, Conv1};
use voaice::f16::{fp16_to_fp32, fp32_to_fp16};
use voaice::{mel, model::Model, ulp_distance, wav};

const N_FRAMES: usize = 3000; // 2 · n_audio_ctx: the window whisper_encode slices from the mel

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn dir() -> PathBuf {
    env_path("VOAICE_ORACLE_CONV1", ".oracle/conv1")
}
fn model() -> Model {
    Model::load_pinned(&env_path("VOAICE_MODEL", "models/ggml-tiny.en.bin")).unwrap_or_else(|e| panic!("{e}"))
}
fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e} (run testing/release_gate.sh first)", p.display()))
}
fn read_f32(p: &Path) -> Vec<f32> {
    read(p).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}
fn read_u16(p: &Path) -> Vec<u16> {
    read(p).as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}
fn meta(p: &Path, key: &str) -> String {
    let s = String::from_utf8(read(p)).unwrap();
    s.lines().find_map(|l| l.strip_prefix(&format!("{key}\t"))).unwrap_or_else(|| panic!("{}: no {key}", p.display())).to_string()
}
fn wavs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir())
        .expect("conv1 oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("conv1.f32").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the conv1 oracle dir");
    v
}
/// (count of differing values, max ULP distance)
fn compare(ours: &[f32], theirs: &[f32]) -> (usize, u64) {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    ours.iter().zip(theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).fold((0, 0), |(n, m), (a, b)| (n + 1, m.max(ulp_distance(*a, *b))))
}
/// voaice's mel for a recorded input, from the WAV (0.0.1's oracle shows it equals the reference's)
fn our_mel(m: &Model, w: &str) -> mel::Mel {
    let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
    let t = mel::Tables::new();
    mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap().run(&pcm, 4).unwrap()
}

/// One kernel record: (x, y, the shipped vec_dot's result).
fn vecdot_records() -> Vec<(Vec<u16>, Vec<u16>, u32)> {
    let b = read(&dir().join("vecdot.bin"));
    let mut out = Vec::new();
    let mut p = 0;
    let u32_at = |p: usize| u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
    let halves = |p: usize, n: usize| (0..n).map(|i| u16::from_le_bytes([b[p + 2 * i], b[p + 2 * i + 1]])).collect::<Vec<u16>>();
    while p < b.len() {
        let n = u32_at(p) as usize;
        let x = halves(p + 4, n);
        let y = halves(p + 4 + 2 * n, n);
        out.push((x, y, u32_at(p + 4 + 4 * n)));
        p += 8 + 4 * n;
    }
    assert_eq!(out.len().to_string(), meta(&dir().join("vecdot.tsv"), "records"));
    out
}

/// The kernel: `type_traits_cpu[F16].vec_dot` of the shipped libggml-cpu on real rows of every f16 tensor of the
/// model, every length 1..300 and random finite patterns — bit for bit.
#[test]
#[ignore]
fn oracle_vec_dot_f16_kernel() {
    let m = dir().join("vecdot.tsv");
    assert_eq!(meta(&m, "vec_dot_is_ggml_vec_dot_f16"), "yes", "the F16 traits do not point at ggml_vec_dot_f16");
    assert_eq!(meta(&m, "vec_dot_type"), "f16");
    assert_eq!(meta(&m, "nrows"), "1");
    let recs = vecdot_records();
    let mut bad = 0;
    let mut lens = std::collections::BTreeSet::new();
    for (x, y, want) in &recs {
        lens.insert(x.len());
        let got = conv::vec_dot_f16(x, y).to_bits();
        if got != *want {
            bad += 1;
            if bad <= 3 {
                eprintln!("  n = {}: ours {got:08x}, the reference's {want:08x}", x.len());
            }
        }
    }
    eprintln!(
        "oracle_vec_dot_f16: {} / {} dots identical ({} distinct lengths, 1..={}); the reference's ISA here: avx2 {} fma {} f16c {} avx512f {}",
        recs.len() - bad,
        recs.len(),
        lens.len(),
        lens.iter().max().unwrap(),
        meta(&m, "avx2"),
        meta(&m, "fma"),
        meta(&m, "f16c"),
        meta(&m, "avx512f")
    );
    assert_eq!(bad, 0, "ggml_vec_dot_f16 is not reproduced");
}

/// The oracle must be able to fail: three other float orders for the same dot, each caught on the kernel record.
#[test]
#[ignore]
fn oracle_vec_dot_f16_discriminators() {
    let recs = vecdot_records();
    for (name, f) in [
        ("one f32 accumulator in index order", conv::wrong::single_accumulator as fn(&[u16], &[u16]) -> f32),
        ("the tail added in f32, not double", conv::wrong::tail_in_f32),
        ("the accumulators reduced in sequence", conv::wrong::sequential_reduce),
    ] {
        let n = recs.iter().filter(|(x, y, want)| f(x, y).to_bits() != *want).count();
        eprintln!("oracle_vec_dot_f16_discriminates: {name}: {n} / {} dots differ", recs.len());
        assert!(n > 0, "the oracle did not tell '{name}' from the shipped kernel");
    }
}

/// The im2col node: f16 [3000][240] from voaice's own mel, as ggml-cpu wrote it.
#[test]
#[ignore]
fn oracle_conv1_im2col_bit_exact() {
    let m = model();
    let mut total = 0;
    for w in wavs() {
        let mel = our_mel(&m, &w);
        assert_eq!(meta(&dir().join(&w).join("conv1.tsv"), "n_len"), mel.n_len.to_string(), "{w}: n_len");
        let ours = conv::im2col_f16(&mel.data, mel.n_mel, mel.n_len, 0, N_FRAMES);
        let theirs = read_u16(&dir().join(&w).join("im2col.u16"));
        assert_eq!(ours.len(), theirs.len(), "{w}: im2col size");
        let n = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        assert_eq!(n, 0, "{w}: im2col differs in {n} values");
        total += theirs.len();
    }
    eprintln!("oracle_conv1_im2col: {} inputs, {total} f16 values identical", wavs().len());
}

/// The MUL_MAT node (conv1 without its bias), the ADD and the GELU after it: voaice's fast path at 1 and 4 threads.
#[test]
#[ignore]
fn oracle_conv1_bit_exact() {
    let m = model();
    let c = Conv1::new(&m).unwrap();
    let mut report = String::new();
    let mut failed = false;
    let mut total = 0usize;
    for w in wavs() {
        let d = dir().join(&w);
        let t = d.join("conv1.tsv");
        for k in ["threads_1_vs_4_bit_identical", "embd_conv_observed_eq_unobserved", "standalone_graph_eq_sched"] {
            assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
        }
        let mel = our_mel(&m, &w);
        report += &format!("  {w:<11}");
        for (node, file) in [("mul_mat", "conv1.f32"), ("+ bias", "conv1_bias.f32"), ("gelu", "conv1_gelu.f32")] {
            let theirs = read_f32(&d.join(file));
            for threads in [1, 4] {
                let ours = match node {
                    "mul_mat" => c.run(&mel.data, mel.n_len, 0, N_FRAMES, threads),
                    "+ bias" => c.run_bias(&mel.data, mel.n_len, 0, N_FRAMES, threads),
                    _ => c.run_gelu(&mel.data, mel.n_len, 0, N_FRAMES, threads),
                };
                let (n, ulp) = compare(&ours, &theirs);
                failed |= n > 0;
                total += theirs.len();
                if threads == 1 {
                    report += &format!(" | {node}: {n} differ (max {ulp} ULP)");
                } else if n > 0 {
                    report += &format!(" [at 4 threads: {n} differ]");
                }
            }
        }
        report += "\n";
    }
    eprint!("oracle_conv1 ({total} f32 values: 3 nodes x 384 x 3000 x 2 thread counts per input):\n{report}");
    assert!(!failed, "conv1 is not bit-exact against the reference");
}

/// The oracle must be able to fail on the real node too: conv1 with the mel kept in f32 (im2col without its f16
/// rounding), and with a single-accumulator dot, each differs from the shipped MUL_MAT on jfk.
#[test]
#[ignore]
fn oracle_conv1_discriminators() {
    let m = model();
    let w = "jfk";
    let mel = our_mel(&m, w);
    let theirs = read_f32(&dir().join(w).join("conv1.f32"));
    let wt = m.tensor("encoder.conv1.weight").unwrap();
    let wh: Vec<u16> = m.tensor_bytes(wt).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let k = 240;
    let im = conv::im2col_f16(&mel.data, mel.n_mel, mel.n_len, 0, N_FRAMES);
    // unrounded: the same window, f32 values straight from the mel (0 outside)
    let raw = |t: usize, kk: usize| -> f32 {
        let (ic, kw) = (kk / 3, kk % 3);
        if t + kw >= 1 && t + kw - 1 < N_FRAMES { mel.data[ic * mel.n_len + t + kw - 1] } else { 0.0 }
    };
    let ww: Vec<f32> = wh.iter().map(|&h| fp16_to_fp32(h)).collect();
    let mut no_round = vec![0f32; theirs.len()];
    let mut single = vec![0f32; theirs.len()];
    for c in 0..384 {
        for t in 0..N_FRAMES {
            let x: Vec<f32> = (0..k).map(|kk| raw(t, kk)).collect();
            no_round[c * N_FRAMES + t] = conv::dot_f16_model(&x, &ww[c * k..(c + 1) * k]);
            single[c * N_FRAMES + t] = conv::wrong::single_accumulator(&im[t * k..(t + 1) * k], &wh[c * k..(c + 1) * k]);
        }
    }
    // the check that these variants are not vacuous: the rounding does change some inputs
    let changed = (0..N_FRAMES * k).filter(|&i| fp16_to_fp32(fp32_to_fp16(raw(i / k, i % k))) != raw(i / k, i % k)).count();
    for (name, v) in [("im2col kept in f32 (no f16 rounding)", &no_round), ("one f32 accumulator", &single)] {
        let (n, ulp) = compare(v, &theirs);
        eprintln!("oracle_conv1_discriminates: {name}: {n} of {} values differ on {w} (max {ulp} ULP)", theirs.len());
        assert!(n > 0, "the oracle did not tell '{name}' from the shipped conv1");
    }
    eprintln!("  ({changed} of {} im2col inputs on {w} change when rounded to f16)", N_FRAMES * k);
}
