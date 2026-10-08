// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.8: the encoder's nine layer norms (each block's attn_ln and mlp_ln, then ln_post) — the NORM, MUL and ADD
//! nodes — against what the shipped library computed (`whisper_oracle --norm`, run by testing/release_gate.sh into
//! .oracle/norm). `#[ignore]`d: they need the model and the record, which are not in git.
//!
//!   cargo test --release --test norm -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_NORM (default .oracle/norm), VOAICE_AUDIO
//! (default .audio). The nodes were read through ggml's scheduler eval callback on the encoder graph, every node
//! observed; each NORM's input was read before the NORM ran. Blocks 1-3 and ln_post read the output of attention and
//! the MLP, which voaice.rs does not compute yet (0.0.9, v0.1.0), so those norms are fed the reference's recorded
//! input to that node; block 0's attn_ln is also computed end to end from voaice's own mel. Every comparison is of
//! bit patterns.
use std::path::{Path, PathBuf};
use voaice::conv::ConvStage;
use voaice::norm::{self, LayerNorm, Node, Variant};
use voaice::{mel, model::Model, ulp_distance, wav};

const N_FRAMES: usize = 3000;
const N: usize = 384;
const CHAINS: [&str; 9] = ["attn_ln_0", "mlp_ln_0", "attn_ln_1", "mlp_ln_1", "attn_ln_2", "mlp_ln_2", "attn_ln_3", "mlp_ln_3", "ln_post"];

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn dir() -> PathBuf {
    env_path("VOAICE_ORACLE_NORM", ".oracle/norm")
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
fn meta(p: &Path, key: &str) -> String {
    let s = String::from_utf8(read(p)).unwrap();
    s.lines().find_map(|l| l.strip_prefix(&format!("{key}\t"))).unwrap_or_else(|| panic!("{}: no {key}", p.display())).to_string()
}
fn wavs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir())
        .expect("norm oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("norm.tsv").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the norm oracle dir");
    v
}
/// (count of differing values, max ULP distance)
fn compare(ours: &[f32], theirs: &[f32]) -> (usize, u64) {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    ours.iter().zip(theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).fold((0, 0), |(n, m), (a, b)| (n + 1, m.max(ulp_distance(*a, *b))))
}
fn check_record(w: &str) {
    let t = dir().join(w).join("norm.tsv");
    for k in [
        "eps_is_1e-5f",
        "inputs_contiguous_f32",
        "weights_are_the_named_tensors",
        "threads_1_vs_4_bit_identical",
        "embd_enc_observed_eq_unobserved",
        "ln_post_add_eq_embd_enc",
        "standalone_eq_sched",
    ] {
        assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
    }
    assert_eq!(meta(&t, "chains"), "9");
    assert_eq!(meta(&t, "n_state"), N.to_string());
}
fn rec(w: &str, chain: &str, node: &str) -> Vec<f32> {
    read_f32(&dir().join(w).join(format!("{chain}.{node}.f32")))
}
/// rows whose double sum is provably order-free (the fast path's four lanes) and rows that take the reference's order
fn order_free_rows(x: &[f32]) -> (usize, usize) {
    x.as_chunks::<N>().0.iter().fold((0, 0), |(a, b), r| {
        let mx = r.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).max().unwrap();
        let mn = r.iter().map(|v| v.to_bits() & 0x7FFF_FFFF).filter(|&v| v != 0).min().unwrap_or(u32::MAX);
        if norm::sum_is_order_free(N, mx, mn) { (a + 1, b) } else { (a, b + 1) }
    })
}

/// Every chain's NORM, MUL and ADD nodes from the recorded input: voaice's fast path at 1 and 4 threads and the
/// portable model.
#[test]
#[ignore]
fn oracle_norm_nodes_bit_exact() {
    let m = model();
    let norms = LayerNorm::encoder(&m).unwrap();
    assert_eq!(norms.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), CHAINS);
    let mut report = String::new();
    let (mut failed, mut total, mut free, mut ordered) = (false, 0usize, 0usize, 0usize);
    for w in wavs() {
        check_record(&w);
        let mut line = format!("  {w:<11}");
        let mut bad = 0usize;
        for (name, ln) in &norms {
            let x = rec(&w, name, "in");
            let (a, b) = order_free_rows(&x);
            free += a;
            ordered += b;
            for (node, file) in [(Node::Norm, "norm"), (Node::Mul, "mul"), (Node::Add, "add")] {
                let theirs = rec(&w, name, file);
                let model = ln.run_model(&x, Variant::default(), node);
                let (n, ulp) = compare(&model, &theirs);
                if n > 0 {
                    line += &format!(" [{name} {file} model: {n} differ, max {ulp} ULP]");
                }
                bad += n;
                total += theirs.len();
                for threads in [1, 4] {
                    let mut ours = vec![0.0f32; x.len()];
                    ln.run_into_split(&x, threads, node, &mut ours); // the split as asked, not the one-thread policy
                    let (n, ulp) = compare(&ours, &theirs);
                    if n > 0 {
                        line += &format!(" [{name} {file} {threads}t: {n} differ, max {ulp} ULP]");
                    }
                    bad += n;
                    total += theirs.len();
                }
            }
        }
        failed |= bad > 0;
        report += &format!("{line} | 9 chains x norm, mul, add (model, 1 and 4 threads): {bad} differ\n");
    }
    eprint!(
        "oracle_norm_nodes ({total} f32 values: 9 chains x 3 nodes x 384 x 1500 x 3 paths per input; rows whose sum is \
         provably order-free {free}, rows in the reference's order {ordered}):\n{report}"
    );
    assert!(!failed, "a norm node differs");
}

/// Block 0's attn_ln end to end from voaice's own mel: the conv stage (0.0.7) gives the NORM's input, then the chain.
#[test]
#[ignore]
fn oracle_attn_ln_0_from_mel() {
    let m = model();
    let st = ConvStage::new(&m).unwrap();
    let ln = LayerNorm::new(&m, "encoder.blocks.0.attn_ln").unwrap();
    let t = mel::Tables::new();
    let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
    let mut report = String::new();
    let (mut failed, mut total) = (false, 0usize);
    for w in wavs() {
        let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
        let mel = plan.run(&pcm, 4).unwrap();
        report += &format!("  {w:<11}");
        for threads in [1, 2, 4] {
            let x = st.run(&mel.data, mel.n_len, 0, N_FRAMES, threads);
            let (n_in, _) = compare(&x, &rec(&w, "attn_ln_0", "in"));
            let mut y = vec![0.0f32; x.len()];
            let mut n_nodes = 0;
            for (node, file) in [(Node::Norm, "norm"), (Node::Mul, "mul"), (Node::Add, "add")] {
                ln.run_into_split(&x, threads, node, &mut y);
                n_nodes += compare(&y, &rec(&w, "attn_ln_0", file)).0;
                total += y.len();
            }
            failed |= n_in > 0 || n_nodes > 0;
            report += &format!(" | {threads}t: input {n_in} differ, norm/mul/add {n_nodes} differ");
        }
        report += "\n";
    }
    eprint!("oracle_attn_ln_0_from_mel ({total} f32 values: 3 nodes x 384 x 1500 x 3 thread counts per input, from the WAV):\n{report}");
    assert!(!failed, "block 0's norm from voaice's mel differs");
}

/// Every other reading of the source the oracle must reject, on every chain of every input: how many of the ADD
/// node's values (or the NORM node's, for the norm-only readings) each changes.
#[test]
#[ignore]
fn oracle_norm_discriminators() {
    let m = model();
    let norms = LayerNorm::encoder(&m).unwrap();
    let variants: [(&str, Variant); 9] = [
        ("sum in f32", Variant { sum_f32: true, ..Variant::default() }),
        ("mean from the double sum", Variant { mean_double: true, ..Variant::default() }),
        ("one-pass variance", Variant { single_pass: true, ..Variant::default() }),
        ("cvar without the 8-lane f32 reduce", Variant { cvar_sequential: true, ..Variant::default() }),
        ("eps outside the sqrt", Variant { eps_outside: true, ..Variant::default() }),
        ("scale in double", Variant { scale_double: true, ..Variant::default() }),
        ("divide by the root", Variant { divide: true, ..Variant::default() }),
        ("mul + add fused (FMA)", Variant { fma: true, ..Variant::default() }),
        ("lane sum on every row (no proof)", Variant { sum_lanes: true, ..Variant::default() }),
    ];
    let mut counts = vec![vec![0usize; wavs().len()]; variants.len()];
    let mut values = 0usize;
    for (wi, w) in wavs().iter().enumerate() {
        for (name, ln) in &norms {
            let x = rec(w, name, "in");
            let theirs = rec(w, name, "add");
            values += theirs.len();
            for (vi, (_, v)) in variants.iter().enumerate() {
                counts[vi][wi] += compare(&ln.run_model(&x, *v, Node::Add), &theirs).0;
            }
        }
    }
    let mut report = format!("  {:<36} {}\n", "reading", wavs().iter().map(|w| format!("{w:>11}")).collect::<String>());
    let mut missed = Vec::new();
    for (vi, (label, _)) in variants.iter().enumerate() {
        report += &format!("  {label:<36} {}\n", counts[vi].iter().map(|c| format!("{c:>11}")).collect::<String>());
        if counts[vi].iter().all(|&c| c == 0) {
            missed.push(*label);
        }
    }
    eprint!("oracle_norm_discriminators (ADD-node values that differ, of {} per input: 9 chains x 384 x 1500):\n{report}", values / wavs().len());
    // the fast path's lane sum without its proof: say on how many inputs it would have been wrong (it may be none —
    // then the proof is what makes the fast path safe, not what makes these inputs pass)
    let lanes = counts[variants.len() - 1].iter().filter(|&&c| c > 0).count();
    eprintln!("  the lane sum without the proof changes the ADD node on {lanes} of {} inputs", wavs().len());
    missed.retain(|l| !l.starts_with("lane sum"));
    assert!(missed.is_empty(), "not rejected on any input: {missed:?}");
}
