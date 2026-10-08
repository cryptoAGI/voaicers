// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.0.9: the encoder's matrix products on activations — every block's Q, K, V (and K's and V's f16 copies), the out
//! projection, its bias and residual, fc1, its bias, GELU, fc2, its bias and residual — against what the shipped
//! library computed (`whisper_oracle --matmul` and `--mm-nan`, run by testing/release_gate.sh into .oracle/matmul; the
//! norm record of 0.0.8 in .oracle/norm supplies the full tensors around them). `#[ignore]`d: they need the model and
//! the records, which are not in git.
//!
//!   cargo test --release --test matmul -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_MATMUL (default .oracle/matmul),
//! VOAICE_ORACLE_NORM (default .oracle/norm), VOAICE_AUDIO (default .audio).
//!
//! The record is compact: each node as one 64-bit FNV-1a digest per row (frame) of its bits, as the oracle wrote them
//! (a single changed value always changes its row's digest: every step of the digest is a bijection), plus the
//! attention's output whole (voaice.rs does not compute attention yet). Each product's input is checked against the
//! digest of what the reference's MUL_MAT read before voaice is fed it: attn_ln's and mlp_ln's outputs and the
//! residual stream come whole from the norm record, the attention's output from this one; fc2's input is voaice's
//! own GELU, fed only after its every row matched the reference's. The residual adds are also compared value by
//! value against the norm record (they are the next norm's input).
use std::path::{Path, PathBuf};
use voaice::conv::ConvStage;
use voaice::gelu::Gelu;
use voaice::matmul::{self, Block, Linear, MlpTaps, QkvTaps, Variant};
use voaice::{mel, model::Model, wav};

const N_FRAMES: usize = 3000;
const NS: usize = 384;
const NH: usize = 1536;
const ROWS: usize = 1500;
const LAYERS: usize = 4;

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn dir() -> PathBuf {
    env_path("VOAICE_ORACLE_MATMUL", ".oracle/matmul")
}
fn norm_dir() -> PathBuf {
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
fn read_u64(p: &Path) -> Vec<u64> {
    read(p).as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
}
fn meta(p: &Path, key: &str) -> String {
    let s = String::from_utf8(read(p)).unwrap();
    s.lines().find_map(|l| l.strip_prefix(&format!("{key}\t"))).unwrap_or_else(|| panic!("{}: no {key}", p.display())).to_string()
}
fn wavs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir())
        .expect("matmul oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("matmul.tsv").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the matmul oracle dir");
    v
}
fn check_record(w: &str) {
    let t = dir().join(w).join("matmul.tsv");
    for k in [
        "every_block_17_nodes",
        "weights_f16_named",
        "weights_in_plain_cpu_buffers",
        "src1_contiguous_f32",
        "biases_named",
        "k_has_no_bias",
        "threads_1_vs_4_bit_identical",
        "embd_enc_observed_eq_unobserved",
        "standalone_eq_sched",
    ] {
        assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
    }
    assert_eq!(meta(&t, "n_state"), NS.to_string());
    assert_eq!(meta(&t, "n_layer"), LAYERS.to_string());
}

// the oracle's digest32 / digest16: 64-bit FNV-1a over u64 words, f32 two to a word, f16 four (little-endian packing)
fn digest32(x: &[f32]) -> u64 {
    x.as_chunks::<2>().0.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
        (h ^ (p[0].to_bits() as u64 | (p[1].to_bits() as u64) << 32)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
fn digest16(x: &[u16]) -> u64 {
    x.as_chunks::<4>().0.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
        (h ^ (p[0] as u64 | (p[1] as u64) << 16 | (p[2] as u64) << 32 | (p[3] as u64) << 48)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
fn rows32(x: &[f32], n: usize) -> Vec<u64> {
    x.chunks_exact(n).map(digest32).collect()
}
fn rows16(x: &[u16], n: usize) -> Vec<u64> {
    x.chunks_exact(n).map(digest16).collect()
}
/// rows whose digest differs from the record's
fn diff_rows(ours: &[u64], theirs: &[u64]) -> usize {
    assert_eq!(ours.len(), theirs.len(), "row counts differ");
    ours.iter().zip(theirs).filter(|(a, b)| a != b).count()
}
fn values_differ(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}
fn d(w: &str, il: usize, key: &str) -> Vec<u64> {
    read_u64(&dir().join(w).join(format!("b{il}.{key}.d64")))
}
fn norm_rec(w: &str, chain: &str, node: &str) -> Vec<f32> {
    read_f32(&norm_dir().join(w).join(format!("{chain}.{node}.f32")))
}
/// the block's input: attn_ln_il's input; the block's output: the next norm's input (ln_post's after the last)
fn block_in(w: &str, il: usize) -> Vec<f32> {
    norm_rec(w, &format!("attn_ln_{il}"), "in")
}
fn block_out(w: &str, il: usize) -> Vec<f32> {
    if il + 1 == LAYERS { norm_rec(w, "ln_post", "in") } else { norm_rec(w, &format!("attn_ln_{}", il + 1), "in") }
}
fn fa(w: &str, il: usize) -> Vec<f32> {
    read_f32(&dir().join(w).join(format!("b{il}.fa.f32")))
}

/// Every node of a block computed one way, as row digests (and the block's output whole).
struct Nodes {
    rows: std::collections::BTreeMap<&'static str, Vec<u64>>,
    out: Vec<f32>,
    o_res: Vec<f32>,
}
impl Nodes {
    /// count of differing rows per node against the record, and of differing values of o_res and the block's output
    fn compare(&self, w: &str, il: usize) -> (usize, String) {
        let mut bad = 0;
        let mut s = String::new();
        for (k, v) in &self.rows {
            let n = diff_rows(v, &d(w, il, k));
            if n > 0 {
                s += &format!(" [b{il} {k}: {n} rows differ]");
            }
            bad += n;
        }
        let n = values_differ(&self.o_res, &norm_rec(w, &format!("mlp_ln_{il}"), "in")) + values_differ(&self.out, &block_out(w, il));
        if n > 0 {
            s += &format!(" [b{il} residuals: {n} values differ from the norm record]");
        }
        (bad + n, s)
    }
}

/// The block by the model, each node fed the reference's recorded input to it.
fn model_block(b: &Block, g: &Gelu, w: &str, il: usize) -> Nodes {
    let v = Variant::default();
    let mut rows = std::collections::BTreeMap::new();
    let x = block_in(w, il);
    let ln = norm_rec(w, &format!("attn_ln_{il}"), "add");
    for k in ["q", "k", "v"] {
        assert_eq!(rows32(&ln, NS), d(w, il, &format!("{k}.in")), "{w} b{il}: attn_ln's output is not what {k}'s MUL_MAT read");
    }
    let k_mm = b.k.model(&ln, 1, v, false);
    rows.insert("k_cpy", rows16(&matmul::cpy_f16_model(&k_mm), NS));
    rows.insert("k_mm", rows32(&k_mm, NS));
    let v_mm = b.v.model(&ln, 1, v, false);
    let v_add = b.v.model(&ln, 1, v, true);
    rows.insert("v_cpy", rows16(&matmul::cpy_f16_model(&v_add), NS));
    rows.insert("v_mm", rows32(&v_mm, NS));
    rows.insert("v_add", rows32(&v_add, NS));
    rows.insert("q_mm", rows32(&b.q.model(&ln, 1, v, false), NS));
    rows.insert("q_add", rows32(&b.q.model(&ln, 1, v, true), NS));
    let att = fa(w, il);
    assert_eq!(rows32(&att, NS), d(w, il, "o.in"), "{w} b{il}: the attention's output is not what o's MUL_MAT read");
    rows.insert("o_mm", rows32(&b.o.model(&att, 1, v, false), NS));
    let o_add = b.o.model(&att, 1, v, true);
    rows.insert("o_add", rows32(&o_add, NS));
    let o_res = matmul::residual_model(&o_add, &x);
    rows.insert("o_res", rows32(&o_res, NS));
    let ml = norm_rec(w, &format!("mlp_ln_{il}"), "add");
    assert_eq!(rows32(&ml, NS), d(w, il, "fc1.in"), "{w} b{il}: mlp_ln's output is not what fc1's MUL_MAT read");
    rows.insert("fc1_mm", rows32(&b.fc1.model(&ml, 1, v, false), NH));
    let fc1_add = b.fc1.model(&ml, 1, v, true);
    rows.insert("fc1_add", rows32(&fc1_add, NH));
    let ge = matmul::gelu_model(g, &fc1_add, v);
    let ge_rows = rows32(&ge, NH);
    assert_eq!(ge_rows, d(w, il, "fc2.in"), "{w} b{il}: voaice's GELU is not what fc2's MUL_MAT read (fc2 would not be fed the reference's input)");
    rows.insert("gelu", ge_rows);
    rows.insert("fc2_mm", rows32(&b.fc2.model(&ge, 1, v, false), NS));
    let fc2_add = b.fc2.model(&ge, 1, v, true);
    rows.insert("fc2_add", rows32(&fc2_add, NS));
    let rec_o_res = norm_rec(w, &format!("mlp_ln_{il}"), "in");
    let out = matmul::residual_model(&fc2_add, &rec_o_res);
    rows.insert("mlp_res", rows32(&out, NS));
    Nodes { rows, out, o_res }
}

/// The block by the fast path: qkv from the block's input (attn_ln fused), the MLP half from the recorded attention
/// output and the block's input, every intermediate tapped.
fn fast_block(b: &Block, x: &[f32], att: &[f32], threads: usize) -> Nodes {
    let mut rows = std::collections::BTreeMap::new();
    let z = || vec![0.0f32; ROWS * NS];
    let (mut q, mut k16, mut v16) = (z(), vec![0u16; ROWS * NS], vec![0u16; ROWS * NS]);
    let (mut k_mm, mut v_mm, mut v_add, mut q_mm) = (z(), z(), z(), z());
    b.qkv_into(x, threads, &mut q, &mut k16, &mut v16, QkvTaps { k_mm: Some(&mut k_mm), v_mm: Some(&mut v_mm), v_add: Some(&mut v_add), q_mm: Some(&mut q_mm) });
    rows.insert("k_mm", rows32(&k_mm, NS));
    rows.insert("k_cpy", rows16(&k16, NS));
    rows.insert("v_mm", rows32(&v_mm, NS));
    rows.insert("v_add", rows32(&v_add, NS));
    rows.insert("v_cpy", rows16(&v16, NS));
    rows.insert("q_mm", rows32(&q_mm, NS));
    rows.insert("q_add", rows32(&q, NS));
    let zh = || vec![0.0f32; ROWS * NH];
    let (mut o_mm, mut o_add, mut o_res, mut fc2_mm, mut fc2_add, mut out) = (z(), z(), z(), z(), z(), z());
    let (mut fc1_mm, mut fc1_add, mut ge) = (zh(), zh(), zh());
    b.mlp_into(
        att,
        x,
        threads,
        &mut out,
        MlpTaps {
            o_mm: Some(&mut o_mm),
            o_add: Some(&mut o_add),
            o_res: Some(&mut o_res),
            fc1_mm: Some(&mut fc1_mm),
            fc1_add: Some(&mut fc1_add),
            gelu: Some(&mut ge),
            fc2_mm: Some(&mut fc2_mm),
            fc2_add: Some(&mut fc2_add),
        },
    );
    for (k, v, n) in [("o_mm", &o_mm, NS), ("o_add", &o_add, NS), ("o_res", &o_res, NS), ("fc1_mm", &fc1_mm, NH), ("fc1_add", &fc1_add, NH), ("gelu", &ge, NH), ("fc2_mm", &fc2_mm, NS), ("fc2_add", &fc2_add, NS), ("mlp_res", &out, NS)] {
        rows.insert(k, rows32(v, n));
    }
    Nodes { rows, out, o_res }
}

/// Every block's 16 product-side nodes (+ the residuals by value) from the recorded inputs: the portable model, and
/// the fast path at 1 and 4 threads.
#[test]
#[ignore]
fn oracle_matmul_nodes_bit_exact() {
    let m = model();
    let blocks: Vec<Block> = (0..LAYERS).map(|il| Block::new(&m, il).unwrap()).collect();
    let g = Gelu::new();
    let mut report = String::new();
    let (mut failed, mut rows_total) = (false, 0usize);
    for w in wavs() {
        check_record(&w);
        let mut line = format!("  {w:<11}");
        let mut bad = 0;
        for (il, b) in blocks.iter().enumerate() {
            let (n, s) = model_block(b, &g, &w, il).compare(&w, il);
            bad += n;
            line += &s.replace("[b", "[model b");
            let (x, att) = (block_in(&w, il), fa(&w, il));
            for threads in [1, 4] {
                let (n, s) = fast_block(b, &x, &att, threads).compare(&w, il);
                bad += n;
                line += &s.replace("[b", &format!("[{threads}t b"));
            }
            rows_total += 3 * 16 * ROWS;
        }
        failed |= bad > 0;
        report += &format!("{line} | 4 blocks x 16 nodes (model, 1 and 4 threads): {bad} differ\n");
    }
    eprint!(
        "oracle_matmul_nodes ({rows_total} node rows compared by digest: 4 blocks x 16 nodes x 1500 frames x 3 paths per \
         input — k_mm k_cpy v_mm v_add v_cpy q_mm q_add o_mm o_add o_res fc1_mm fc1_add gelu fc2_mm fc2_add mlp_res; \
         o_res and mlp_res also value by value against the norm record):\n{report}"
    );
    assert!(!failed, "a product-side node differs");
}

/// Block 0 from voaice's own mel: the conv stage (0.0.7) gives the block's input, then attn_ln fused into Q, K, V (the
/// attention's inputs), and the MLP half from the recorded attention output and voaice's own block input.
#[test]
#[ignore]
fn oracle_block0_from_mel() {
    let m = model();
    let st = ConvStage::new(&m).unwrap();
    let b = Block::new(&m, 0).unwrap();
    let t = mel::Tables::new();
    let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
    let mut report = String::new();
    let mut failed = false;
    for w in wavs() {
        let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
        let mel = plan.run(&pcm, 4).unwrap();
        report += &format!("  {w:<11}");
        let att = fa(&w, 0);
        for threads in [1, 2, 4] {
            let x = st.run(&mel.data, mel.n_len, 0, N_FRAMES, threads);
            let n_in = values_differ(&x, &block_in(&w, 0));
            let (n, _) = fast_block(&b, &x, &att, threads).compare(&w, 0);
            failed |= n_in > 0 || n > 0;
            report += &format!(" | {threads}t: input {n_in} values differ, 16 nodes {n} rows/values differ");
        }
        report += "\n";
    }
    eprint!("oracle_block0_from_mel (the WAV -> mel -> conv stage -> attn_ln -> Q, K, V; out proj -> MLP from the recorded attention):\n{report}");
    assert!(!failed, "block 0 from voaice's mel differs");
}

/// from_float on NaN-bearing rows (`whisper_oracle --mm-nan`): the reference's output at 1..8 threads against the
/// model at the same split and the fast path told that split; then the converters the source does not use.
#[test]
#[ignore]
fn oracle_mm_nan_split() {
    let m = model();
    let q = Linear::new(&m, "encoder.blocks.0.attn.query.weight", Some("encoder.blocks.0.attn.query.bias")).unwrap();
    let x = read_f32(&dir().join("nan.in.f32"));
    let rows = x.len() / NS;
    let mut report = String::new();
    let mut failed = false;
    let disc = [("scalar bit trick everywhere", Variant { convert_scalar: true, ..Variant::default() }), ("row converter, split ignored", Variant { convert_unsplit: true, ..Variant::default() })];
    let mut caught = [0usize; 2];
    let one = read_f32(&dir().join("nan.add.t1.f32"));
    for th in 1..=8 {
        let mm = read_f32(&dir().join(format!("nan.mm.t{th}.f32")));
        let add = read_f32(&dir().join(format!("nan.add.t{th}.f32")));
        let n_mm = values_differ(&q.model(&x, th, Variant::default(), false), &mm);
        let n_add = values_differ(&q.model(&x, th, Variant::default(), true), &add);
        let ql = Linear::from_parts(q.k, q.n, q.w.clone(), q.b.clone()).unwrap().with_split(th);
        let mut out = vec![0.0f32; rows * NS];
        ql.run_into(&x, None, 1, matmul::Epilogue::Bias, &mut out);
        let n_fast = values_differ(&out, &add);
        let vs1 = values_differ(&add, &one);
        failed |= n_mm + n_add + n_fast > 0;
        let mut line = format!("  {th} threads: model mul_mat {n_mm}, + bias {n_add}, fast path {n_fast} differ (the reference's own output differs from its 1-thread output in {vs1} values)");
        for (i, (label, v)) in disc.iter().enumerate() {
            let n = values_differ(&q.model(&x, th, *v, true), &add);
            caught[i] += n;
            line += &format!("; {label}: {n}");
        }
        report += &format!("{line}\n");
    }
    eprint!("oracle_mm_nan_split ({rows} rows x {NS} outputs; 10 rows carry one NaN each, 2 none):\n{report}");
    assert!(!failed, "the NaN conversion differs from the reference's");
    for (i, (label, _)) in disc.iter().enumerate() {
        assert!(caught[i] > 0, "not rejected at any thread count: {label}");
    }
}

/// Every other reading of the source the oracle must reject, on every block of every input: rows of the node it
/// changes that differ from the record (of 1500 per block), and for the residual readings the values of the residual
/// (of 576,000 per block) against the norm record.
#[test]
#[ignore]
fn oracle_matmul_discriminators() {
    let m = model();
    let blocks: Vec<Block> = (0..LAYERS).map(|il| Block::new(&m, il).unwrap()).collect();
    let g = Gelu::new();
    let labels = [
        "activations not rounded to f16 (q_add rows)",
        "one f32 accumulator (q_add rows)",
        "accumulators reduced in sequence (q_add rows)",
        "bias in the first accumulator (q_add rows)",
        "residual before the bias (o_res values)",
        "GELU from x, not the f16 table (gelu rows)",
        "scalar bit trick for from_float (q_add rows)",
        "from_float split ignored (q_add rows)",
    ];
    let vq = [
        Variant { no_f16: true, ..Variant::default() },
        Variant { single_accumulator: true, ..Variant::default() },
        Variant { sequential_reduce: true, ..Variant::default() },
        Variant { bias_in_accumulator: true, ..Variant::default() },
    ];
    let ws = wavs();
    let mut counts = vec![vec![0usize; ws.len()]; labels.len()];
    for (wi, w) in ws.iter().enumerate() {
        for (il, b) in blocks.iter().enumerate() {
            let ln = norm_rec(w, &format!("attn_ln_{il}"), "add");
            let q_rec = d(w, il, "q_add");
            for (vi, v) in vq.iter().enumerate() {
                counts[vi][wi] += diff_rows(&rows32(&b.q.model(&ln, 1, *v, true), NS), &q_rec);
            }
            for (vi, v) in [(6, Variant { convert_scalar: true, ..Variant::default() }), (7, Variant { convert_unsplit: true, ..Variant::default() })] {
                counts[vi][wi] += diff_rows(&rows32(&b.q.model(&ln, 4, v, true), NS), &q_rec);
            }
            // (mm + x) + b
            let att = fa(w, il);
            let o_mm = b.o.model(&att, 1, Variant::default(), false);
            let x = block_in(w, il);
            let bias = b.o.b.as_ref().unwrap();
            let wrong: Vec<f32> = o_mm.iter().zip(&x).enumerate().map(|(i, (a, r))| (a + r) + bias[i % NS]).collect();
            counts[4][wi] += values_differ(&wrong, &norm_rec(w, &format!("mlp_ln_{il}"), "in"));
            let ml = norm_rec(w, &format!("mlp_ln_{il}"), "add");
            let fc1_add = b.fc1.model(&ml, 1, Variant::default(), true);
            let ge = matmul::gelu_model(&g, &fc1_add, Variant { gelu_unrounded: true, ..Variant::default() });
            counts[5][wi] += diff_rows(&rows32(&ge, NH), &d(w, il, "gelu"));
        }
    }
    let mut report = format!("  {:<46} {}\n", "reading", ws.iter().map(|w| format!("{w:>11}")).collect::<String>());
    let mut missed = Vec::new();
    for (vi, label) in labels.iter().enumerate() {
        report += &format!("  {label:<46} {}\n", counts[vi].iter().map(|c| format!("{c:>11}")).collect::<String>());
        if vi < 6 && counts[vi].contains(&0) {
            missed.push(*label);
        }
    }
    eprint!(
        "oracle_matmul_discriminators (summed over the 4 blocks: rows of 6,000, or o_res values of 2,304,000; the two \
         from_float readings differ from the reference only on NaN, which no input carries — oracle_mm_nan_split \
         catches them):\n{report}"
    );
    assert!(missed.is_empty(), "not rejected on every input: {missed:?}");
    assert!(counts[6].iter().chain(&counts[7]).all(|&c| c == 0), "a from_float reading changed a finite input");
}
