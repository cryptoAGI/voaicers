// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.7: the encoder's conv2, its bias and GELU (`embd_conv`), and the positional embedding the encoder adds first,
//! against what the shipped library computed (`whisper_oracle --conv2`, run by testing/release_gate.sh into
//! .oracle/conv2). `#[ignore]`d: they need the model and the record, which are not in git.
//!
//!   cargo test --release --test conv2 -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_CONV2 (default .oracle/conv2), VOAICE_AUDIO
//! (default .audio). The nodes were read through ggml's scheduler eval callback on the conv graph and on the encoder
//! graph's first nodes; every comparison is of bit patterns. voaice's side starts from the WAV: its own samples, its
//! own mel (0.0.1's oracle), its own conv1 (0.0.6's), its own conv2.
use std::path::{Path, PathBuf};
use voaice::conv::{self, Conv1, Conv2, ConvStage, Epilogue};
use voaice::f16::{fp16_to_fp32, fp32_to_fp16};
use voaice::gelu::Gelu;
use voaice::{mel, model::Model, ulp_distance, wav};

const N_FRAMES: usize = 3000; // 2 · n_audio_ctx: the mel window
const N_CTX: usize = 1500; // conv2's output frames
const N: usize = 384; // n_audio_state (tiny)

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn dir() -> PathBuf {
    env_path("VOAICE_ORACLE_CONV2", ".oracle/conv2")
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
        .expect("conv2 oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("pe_add.f32").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the conv2 oracle dir");
    v
}
/// (count of differing values, max ULP distance)
fn compare(ours: &[f32], theirs: &[f32]) -> (usize, u64) {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    ours.iter().zip(theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).fold((0, 0), |(n, m), (a, b)| (n + 1, m.max(ulp_distance(*a, *b))))
}
fn our_mel(m: &Model, w: &str) -> mel::Mel {
    let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
    let t = mel::Tables::new();
    mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap().run(&pcm, 4).unwrap()
}
/// voaice's conv1 + bias + GELU from the WAV (conv2's input), checked against the record's copy of that node
fn our_conv1(m: &Model, c1: &Conv1, w: &str) -> Vec<f32> {
    let mel = our_mel(m, w);
    assert_eq!(meta(&dir().join(w).join("conv2.tsv"), "n_len"), mel.n_len.to_string(), "{w}: n_len");
    let y = c1.run_gelu(&mel.data, mel.n_len, 0, N_FRAMES, 4);
    let (n, _) = compare(&y, &read_f32(&dir().join(w).join("conv1_gelu.f32")));
    assert_eq!(n, 0, "{w}: conv2's input (conv1's GELU) differs");
    y
}
fn check_record(w: &str) {
    let t = dir().join(w).join("conv2.tsv");
    for k in [
        "threads_1_vs_4_bit_identical",
        "last_conv_node_eq_embd_conv_unobserved",
        "embd_enc_observed_eq_unobserved",
        "standalone_conv2_eq_sched",
        "standalone_stage_eq_sched",
    ] {
        assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
    }
    assert_eq!(meta(&t, "n_ctx"), N_CTX.to_string());
}

/// conv2's im2col node: f16 [1500][1152] (stride 2, pad 1) from voaice's own conv1 output.
#[test]
#[ignore]
fn oracle_conv2_im2col_bit_exact() {
    let m = model();
    let c1 = Conv1::new(&m).unwrap();
    let mut total = 0;
    for w in wavs() {
        check_record(&w);
        let x = our_conv1(&m, &c1, &w);
        let ours = conv::im2col_strided_f16(&x, N, N_FRAMES, 2, N_CTX);
        let theirs = read_u16(&dir().join(&w).join("im2col2.u16"));
        assert_eq!(ours.len(), theirs.len(), "{w}: im2col size");
        let n = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        assert_eq!(n, 0, "{w}: im2col differs in {n} values");
        total += theirs.len();
    }
    eprintln!("oracle_conv2_im2col: {} inputs, {total} f16 values identical (conv2's input, voaice's conv1, = the node on every input)", wavs().len());
}

/// The MUL_MAT node (conv2 without its bias), the ADD and the GELU (`embd_conv`): voaice's fast path at 1 and 4 threads.
#[test]
#[ignore]
fn oracle_conv2_bit_exact() {
    let m = model();
    let c1 = Conv1::new(&m).unwrap();
    let c2 = Conv2::new(&m).unwrap();
    let mut report = String::new();
    let mut failed = false;
    let mut total = 0usize;
    for w in wavs() {
        let d = dir().join(&w);
        let x = our_conv1(&m, &c1, &w);
        report += &format!("  {w:<11}");
        for (node, file) in [("mul_mat", "conv2.f32"), ("+ bias", "conv2_bias.f32"), ("gelu", "conv2_gelu.f32")] {
            let theirs = read_f32(&d.join(file));
            for threads in [1, 4] {
                let ours = match node {
                    "mul_mat" => c2.run(&x, N_FRAMES, threads, Epilogue::Raw),
                    "+ bias" => {
                        // the fast path's product, then the add (one rounding, as the node)
                        let mut y = c2.run(&x, N_FRAMES, threads, Epilogue::Raw);
                        for (row, &b) in y.as_chunks_mut::<N_CTX>().0.iter_mut().zip(&c2.bias) {
                            row.iter_mut().for_each(|v| *v += b);
                        }
                        y
                    }
                    _ => c2.run(&x, N_FRAMES, threads, Epilogue::BiasGelu),
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
    eprint!("oracle_conv2 ({total} f32 values: 3 nodes x 384 x 1500 x 2 thread counts per input):\n{report}");
    assert!(!failed, "conv2 is not bit-exact against the reference");
}

/// The encoder's first nodes: CONT(TRANSPOSE(embd_conv)) and ADD(e_pe view, ·) — voaice's conv2 with the positions
/// fused into its epilogue (from voaice's conv1), and the whole fused stage from voaice's mel, at 1, 2 and 4 threads.
#[test]
#[ignore]
fn oracle_positions_bit_exact() {
    let m = model();
    let st = ConvStage::new(&m).unwrap();
    let mut report = String::new();
    let mut failed = false;
    let mut total = 0usize;
    for w in wavs() {
        let d = dir().join(&w);
        let x = our_conv1(&m, &st.conv1, &w);
        // the CONT node is embd_conv transposed: frame-major
        let e = st.conv2.run(&x, N_FRAMES, 4, Epilogue::BiasGelu);
        let mut tr = vec![0f32; e.len()];
        for c in 0..N {
            for t in 0..N_CTX {
                tr[t * N + c] = e[c * N_CTX + t];
            }
        }
        let (nc, _) = compare(&tr, &read_f32(&d.join("cont.f32")));
        let theirs = read_f32(&d.join("pe_add.f32"));
        let mel = our_mel(&m, &w);
        report += &format!("  {w:<11} | cont: {nc} differ");
        failed |= nc > 0;
        for threads in [1, 2, 4] {
            let (n1, u1) = compare(&st.conv2.run(&x, N_FRAMES, threads, Epilogue::Positions), &theirs);
            let (n2, u2) = compare(&st.run(&mel.data, mel.n_len, 0, N_FRAMES, threads), &theirs);
            failed |= n1 > 0 || n2 > 0;
            total += 2 * theirs.len();
            report += &format!(" | {threads}t: conv2+pe {n1} (max {u1} ULP), stage from the mel {n2} (max {u2} ULP)");
        }
        report += "\n";
    }
    eprint!("oracle_positions ({total} f32 values: 384 x 1500, 2 paths x 3 thread counts per input; + cont):\n{report}");
    assert!(!failed, "the encoder's input is not bit-exact against the reference");
}

/// The oracle must be able to fail: other readings of the same graph, each differing from the shipped nodes on jfk.
#[test]
#[ignore]
fn oracle_conv2_discriminators() {
    let m = model();
    let st = ConvStage::new(&m).unwrap();
    let w = "jfk";
    let d = dir().join(w);
    let x = our_conv1(&m, &st.conv1, w);
    let mm = read_f32(&d.join("conv2.f32"));
    let pe_add = read_f32(&d.join("pe_add.f32"));
    let wt = m.tensor("encoder.conv2.weight").unwrap();
    let wh: Vec<u16> = m.tensor_bytes(wt).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    let ww: Vec<f32> = wh.iter().map(|&h| fp16_to_fp32(h)).collect();
    let k = 3 * N;
    let im = conv::im2col_strided_f16(&x, N, N_FRAMES, 2, N_CTX);
    let raw = |stride: usize, t: usize, kk: usize| -> f32 {
        let (ic, kw) = (kk / 3, kk % 3);
        let src = stride * t + kw;
        if src >= 1 && src - 1 < N_FRAMES { x[ic * N_FRAMES + src - 1] } else { 0.0 }
    };
    let mut single = vec![0f32; mm.len()];
    let mut stride1 = vec![0f32; mm.len()];
    for c in 0..N {
        for t in 0..N_CTX {
            single[c * N_CTX + t] = conv::wrong::single_accumulator(&im[t * k..(t + 1) * k], &wh[c * k..(c + 1) * k]);
            let x1: Vec<f32> = (0..k).map(|kk| fp16_to_fp32(fp32_to_fp16(raw(1, t, kk)))).collect();
            stride1[c * N_CTX + t] = conv::dot_f16_model(&x1, &ww[c * k..(c + 1) * k]); // the first 1500 frames of a stride-1 conv
        }
    }
    for (name, v) in [("conv2 with stride 1", &stride1), ("one f32 accumulator", &single)] {
        let (n, ulp) = compare(v, &mm);
        eprintln!("oracle_conv2_discriminates: {name}: {n} of {} MUL_MAT values differ on {w} (max {ulp} ULP)", mm.len());
        assert!(n > 0, "the oracle did not tell '{name}' from the shipped conv2");
    }
    // im2col kept in f32: conv1's GELU output is a widened f16 (the table) for every x < 10, and x itself above, so the
    // f16 rounding of conv2's im2col changes only values >= 10 that f16 cannot hold. Where an input has none (jfk), the
    // two readings are the same function on it and no oracle can tell them apart; on every input that has some, they
    // must differ. Counted on all 8; checked where the count is not 0.
    let mut checked = 0;
    for w2 in wavs() {
        let x = if w2 == w { x.clone() } else { our_conv1(&m, &st.conv1, &w2) };
        let changed = x.iter().filter(|&&v| fp16_to_fp32(fp32_to_fp16(v)).to_bits() != v.to_bits()).count();
        if changed == 0 {
            eprintln!("oracle_conv2_discriminates: im2col kept in f32: {w2}: no conv2 input changes when rounded to f16 (not discriminable)");
            continue;
        }
        let mm2 = read_f32(&dir().join(&w2).join("conv2.f32"));
        let mut no_round = vec![0f32; mm2.len()];
        let r2 = |t: usize, kk: usize| -> f32 {
            let (ic, kw) = (kk / 3, kk % 3);
            let src = 2 * t + kw;
            if src >= 1 && src - 1 < N_FRAMES { x[ic * N_FRAMES + src - 1] } else { 0.0 }
        };
        for c in 0..N {
            for t in 0..N_CTX {
                let xr: Vec<f32> = (0..k).map(|kk| r2(t, kk)).collect();
                no_round[c * N_CTX + t] = conv::dot_f16_model(&xr, &ww[c * k..(c + 1) * k]);
            }
        }
        let (n, ulp) = compare(&no_round, &mm2);
        eprintln!("oracle_conv2_discriminates: im2col kept in f32 (no f16 rounding): {w2}: {changed} conv2 inputs >= 10 change when rounded; {n} of {} MUL_MAT values differ (max {ulp} ULP)", mm2.len());
        assert!(n > 0, "the oracle did not tell 'im2col kept in f32' from the shipped conv2 on {w2}");
        checked += 1;
    }
    assert!(checked > 0, "no input exercises conv2's f16 rounding");
    // the encoder's input, read three wrong ways from the right embd_conv
    let g = Gelu::new();
    let e = st.conv2.run(&x, N_FRAMES, 4, Epilogue::BiasGelu);
    let raw_mm = st.conv2.run(&x, N_FRAMES, 4, Epilogue::Raw);
    let pe = &st.conv2.pe;
    let no_transpose: Vec<f32> = (0..N * N_CTX).map(|i| pe[i] + e[i]).collect(); // pe added to the channel-major buffer
    let shifted: Vec<f32> = (0..N * N_CTX).map(|i| pe[(i + N) % (N * N_CTX)] + e[(i % N) * N_CTX + i / N]).collect(); // positions one frame late
    let mut gelu_then_bias = vec![0f32; N * N_CTX];
    for c in 0..N {
        let mut gg = vec![0f32; N_CTX];
        g.row(&raw_mm[c * N_CTX..(c + 1) * N_CTX], &mut gg);
        for t in 0..N_CTX {
            gelu_then_bias[t * N + c] = pe[t * N + c] + (gg[t] + st.conv2.bias[c]);
        }
    }
    for (name, v) in [
        ("positions added before the transpose (to embd_conv's layout)", &no_transpose),
        ("positions one frame late (another e_pe view offset)", &shifted),
        ("GELU before the bias", &gelu_then_bias),
    ] {
        let (n, ulp) = compare(v, &pe_add);
        eprintln!("oracle_conv2_discriminates: {name}: {n} of {} encoder-input values differ on {w} (max {ulp} ULP)", pe_add.len());
        assert!(n > 0, "the oracle did not tell '{name}' from the shipped graph");
    }
}
