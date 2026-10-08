// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.1.1: cross-attention K and V against what the shipped library computed (`whisper_oracle --cross` into
//! .oracle/cross, with v0.1.0's .oracle/encoder for `embd_enc`, run by testing/release_gate.sh). `#[ignore]`d: they need
//! the model and the records, which are not in git.
//!
//!   cargo test --release --test cross -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_CROSS (default .oracle/cross),
//! VOAICE_ORACLE_ENCODER (.oracle/encoder), VOAICE_AUDIO (.audio).
//!
//! The record is compact: a 64-bit FNV-1a digest per row (frame) of each of the cross graph's 24 computed nodes, and
//! one per row of the `kv_cross` buffer itself — k then v, every layer's 1,536 rows, the 36 padding rows of each
//! included. voaice is fed (a) the reference's own `embd_enc` (the encoder record's, checked against the digest the
//! cross record took of the state's) and (b) its own `embd_enc`, computed from its own mel.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use voaice::cross::{Cross, CrossVariant, KvCross};
use voaice::encoder::{Encoder, EncoderBuffers};
use voaice::{mel, model::Model, wav};

const NS: usize = 384;
const ROWS: usize = 1500;
const N_PAD: usize = 1536;
const LAYERS: usize = 4;
const KEYS: [&str; 6] = ["k_mm", "k_scale", "k_cpy", "v_mm", "v_add", "v_cpy"];

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn cross_dir() -> PathBuf {
    env_path("VOAICE_ORACLE_CROSS", ".oracle/cross")
}
fn enc_dir() -> PathBuf {
    env_path("VOAICE_ORACLE_ENCODER", ".oracle/encoder")
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
    let mut v: Vec<String> = std::fs::read_dir(cross_dir())
        .expect("cross oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("cross.tsv").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the cross oracle dir");
    v
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
fn rows32(x: &[f32]) -> Vec<u64> {
    x.as_chunks::<NS>().0.iter().map(|r| digest32(r)).collect()
}
fn rows16(x: &[u16]) -> Vec<u64> {
    x.as_chunks::<NS>().0.iter().map(|r| digest16(r)).collect()
}
fn diff_rows(ours: &[u64], theirs: &[u64]) -> usize {
    assert_eq!(ours.len(), theirs.len(), "row counts differ");
    ours.iter().zip(theirs).filter(|(a, b)| a != b).count()
}

/// The record of one input: the self-checks, then key -> row digests of every node, and kv_cross's rows (k then v).
struct Record {
    nodes: BTreeMap<String, Vec<u64>>,
    kv: Vec<u64>,
    embd: Vec<f32>,
}
fn record(w: &str) -> Record {
    let dir = cross_dir().join(w);
    let t = dir.join("cross.tsv");
    for k in [
        "flash_attn_layout",
        "scale_b_zero",
        "scale_same_every_layer",
        "six_nodes_per_layer",
        "weights_f16_plain_cpu",
        "src1_is_embd_enc",
        "v_bias_named",
        "k_unbiased",
        "cpy_into_kv_cross_at_il_n_pad",
        "kv_cross_zero_at_init",
        "kv_cross_padding_all_zero",
        "cpy_nodes_eq_kv_cross_rows",
        "threads_1_vs_4_bit_identical",
        "kv_cross_observed_eq_unobserved",
        "kv_cross_1_2_4_threads_identical",
        "embd_enc_every_run_identical",
        "standalone_eq_sched_1_4_threads",
    ] {
        assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
    }
    assert_eq!(meta(&t, "other_computed_nodes"), "none", "{w}: the cross graph computed a node voaice does not model");
    assert_eq!(meta(&t, "n_state"), NS.to_string());
    assert_eq!(meta(&t, "n_layer"), LAYERS.to_string());
    assert_eq!(meta(&t, "n_pad"), N_PAD.to_string());
    let all = read_u64(&dir.join("cross.d64"));
    let tsv = String::from_utf8(read(&dir.join("cross_nodes.tsv"))).unwrap();
    let mut nodes = BTreeMap::new();
    for l in tsv.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        let (rows, off): (usize, usize) = (f[5].parse().unwrap(), f[6].parse().unwrap());
        nodes.insert(f[1].to_string(), all[off..off + rows].to_vec());
    }
    assert_eq!(nodes.len(), 6 * LAYERS);
    let kv = read_u64(&dir.join("kv_cross.d64"));
    assert_eq!(kv.len(), 2 * LAYERS * N_PAD);
    // the input: the encoder record's embd_enc, which must be the state's the cross graph read
    let embd = read_f32(&enc_dir().join(w).join("embd_enc.f32"));
    assert_eq!(format!("{:016x}", digest32(&embd)), meta(&t, "embd_enc_digest"), "{w}: the encoder record's embd_enc is not the cross graph's input");
    Record { nodes, kv, embd }
}

/// Every node of `got` (key -> rows) against the record: (nodes compared, rows compared, rows differing).
fn compare_nodes(got: &BTreeMap<String, Vec<u64>>, rec: &Record) -> (usize, usize, usize) {
    let (mut n, mut rows, mut bad) = (0, 0, 0);
    for (key, theirs) in &rec.nodes {
        let ours = got.get(key).unwrap_or_else(|| panic!("no voaice node for the record's {key}"));
        n += 1;
        rows += theirs.len();
        bad += diff_rows(ours, theirs);
    }
    (n, rows, bad)
}
fn cache_rows(kv: &KvCross) -> Vec<u64> {
    let mut d = rows16(&kv.k);
    d.extend(rows16(&kv.v));
    d
}
/// The fast path's nodes (from taps) and cache.
fn fast(c: &Cross, embd: &[f32], threads: usize) -> (BTreeMap<String, Vec<u64>>, KvCross) {
    let mut kv = KvCross::default();
    let mut taps = vec![0.0f32; LAYERS * 4 * ROWS * NS];
    c.run_into(embd, threads, &mut kv, Some(&mut taps));
    let mut got = BTreeMap::new();
    for il in 0..LAYERS {
        for (i, k) in ["k_mm", "k_scale", "v_mm", "v_add"].iter().enumerate() {
            got.insert(format!("b{il}.{k}"), rows32(&taps[(il * 4 + i) * ROWS * NS..(il * 4 + i + 1) * ROWS * NS]));
        }
        let at = il * N_PAD * NS;
        got.insert(format!("b{il}.k_cpy"), rows16(&kv.k[at..at + ROWS * NS]));
        got.insert(format!("b{il}.v_cpy"), rows16(&kv.v[at..at + ROWS * NS]));
    }
    (got, kv)
}

#[test]
#[ignore]
fn oracle_cross_nodes_bit_exact() {
    let m = model();
    let mut c = Cross::new(&m).unwrap();
    assert_eq!(c.n_pad, N_PAD);
    let mut report = String::new();
    let (mut failed, mut node_rows, mut cache_total) = (false, 0usize, 0usize);
    for w in wavs() {
        let rec = record(&w);
        assert_eq!(meta(&cross_dir().join(&w).join("cross.tsv"), "kscale_bits"), format!("{:08x}", c.kscale.to_bits()), "{w}: Kscale");
        // the model, every frame of every layer
        let (kv, nodes) = c.model_cache(&rec.embd, 1, CrossVariant::default());
        let mut got = BTreeMap::new();
        for (il, nd) in nodes.iter().enumerate() {
            for (k, v) in [("k_mm", &nd.k_mm), ("k_scale", &nd.k_scale), ("v_mm", &nd.v_mm), ("v_add", &nd.v_add)] {
                got.insert(format!("b{il}.{k}"), rows32(v));
            }
            got.insert(format!("b{il}.k_cpy"), rows16(&nd.k_cpy));
            got.insert(format!("b{il}.v_cpy"), rows16(&nd.v_cpy));
        }
        let (n, rows, bad) = compare_nodes(&got, &rec);
        let cb = diff_rows(&cache_rows(&kv), &rec.kv);
        report += &format!("  {w:<11} model: {n} nodes, {bad} of {rows} rows differ; kv_cross {cb} of {} rows (padding incl.)", rec.kv.len());
        failed |= bad > 0 || cb > 0;
        node_rows += rows;
        cache_total += rec.kv.len();
        for threads in [1, 4] {
            c.set_split(threads);
            let (got, kv) = fast(&c, &rec.embd, threads);
            let (_, rows, bad) = compare_nodes(&got, &rec);
            let cb = diff_rows(&cache_rows(&kv), &rec.kv);
            report += &format!("; fast {threads}t: {bad} / {rows}, kv_cross {cb}");
            failed |= bad > 0 || cb > 0;
            node_rows += rows;
            cache_total += rec.kv.len();
        }
        report += "\n";
    }
    eprint!("oracle_cross_nodes_bit_exact (fed the reference's embd_enc; every node of sched_cross by row digest, kv_cross whole):\n{report}");
    eprintln!("  total: {node_rows} node rows and {cache_total} kv_cross rows compared");
    assert!(!failed, "the cross graph differs from the reference");
}

#[test]
#[ignore]
fn oracle_cross_end_to_end() {
    let m = model();
    let enc = Encoder::new(&m).unwrap();
    let mut c = Cross::new(&m).unwrap();
    let t = mel::Tables::new();
    let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
    let mut report = String::new();
    let (mut failed, mut nodes_total, mut rows_total) = (false, 0usize, 0usize);
    for w in wavs() {
        let rec = record(&w);
        let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
        let mel = plan.run(&pcm, 4).unwrap();
        report += &format!("  {w:<11}");
        for threads in [1, 2, 4] {
            let mut buf = EncoderBuffers::default();
            let mut e = vec![0.0f32; ROWS * NS];
            enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut e);
            let e_ok = e.iter().zip(&rec.embd).all(|(a, b)| a.to_bits() == b.to_bits());
            c.set_split(threads);
            let (got, _) = fast(&c, &e, threads);
            let (n, rows, bad) = compare_nodes(&got, &rec);
            // the pipeline itself, no taps, into a cache reused across two calls
            let mut kv = KvCross { k: vec![0xFFFF; c.cache_len()], v: vec![0xFFFF; c.cache_len()] };
            c.run_into(&e, threads, &mut kv, None);
            c.run_into(&e, threads, &mut kv, None);
            let cb = diff_rows(&cache_rows(&kv), &rec.kv);
            report += &format!(" | {threads}t: embd_enc {}, {n} nodes {bad} of {rows} rows differ, kv_cross {cb} of {}", if e_ok { "=" } else { "DIFFERS" }, rec.kv.len());
            failed |= !e_ok || bad > 0 || cb > 0;
            nodes_total += n;
            rows_total += rows + rec.kv.len();
        }
        report += "\n";
    }
    eprint!("oracle_cross_end_to_end (voaice's mel -> encoder -> cross, every node and kv_cross with its padding):\n{report}");
    eprintln!("  total: {nodes_total} node comparisons, {rows_total} rows (nodes + kv_cross)");
    assert!(!failed, "the cross graph from voaice's own embd_enc differs");
}

#[test]
#[ignore]
fn oracle_cross_discriminators() {
    let m = model();
    let c = Cross::new(&m).unwrap();
    let frames: Vec<usize> = (0..ROWS).step_by(25).chain([ROWS - 1]).collect();
    let variants: [(&str, CrossVariant, bool); 9] = [
        ("scale before the product", CrossVariant { scale_first: true, ..Default::default() }, true),
        ("scale folded into the f16 weights", CrossVariant { scale_in_weights: true, ..Default::default() }, true),
        ("scale in double", CrossVariant { scale_double: true, ..Default::default() }, true),
        ("scale after the f16 copy", CrossVariant { scale_after_f16: true, ..Default::default() }, true),
        ("V scaled too", CrossVariant { v_scaled: true, ..Default::default() }, true),
        ("V without its bias", CrossVariant { v_unbiased: true, ..Default::default() }, true),
        ("V's bias on K too", CrossVariant { k_biased: true, ..Default::default() }, true),
        ("activations not rounded to f16", CrossVariant { no_f16: true, ..Default::default() }, true),
        ("the CPYs by the row converter (finite: predicted 0)", CrossVariant { cpy_row_converter: true, ..Default::default() }, false),
    ];
    let mut counts = vec![Vec::new(); variants.len() + 3];
    let mut failed = false;
    for w in wavs() {
        let rec = record(&w);
        let x: Vec<f32> = frames.iter().flat_map(|&f| rec.embd[f * NS..(f + 1) * NS].iter().copied()).collect();
        let pick = |key: &str| -> Vec<u64> { frames.iter().map(|&f| rec.nodes[key][f]).collect() };
        // the reference reading on the sample first: it must match (else the counts below mean nothing)
        for (vi, (name, v, caught)) in std::iter::once(("reference", CrossVariant::default(), false)).chain(variants.iter().copied()).enumerate() {
            let mut bad = 0;
            for il in 0..LAYERS {
                let nd = c.model(il, &x, 1, v);
                let ours = [rows32(&nd.k_mm), rows32(&nd.k_scale), rows16(&nd.k_cpy), rows32(&nd.v_mm), rows32(&nd.v_add), rows16(&nd.v_cpy)];
                for (k, o) in KEYS.iter().zip(&ours) {
                    bad += diff_rows(o, &pick(&format!("b{il}.{k}")));
                }
            }
            if vi == 0 {
                assert_eq!(bad, 0, "{w}: the reference reading differs on the sample");
                continue;
            }
            counts[vi - 1].push(bad);
            failed |= (bad > 0) != caught;
            let _ = name;
        }
        // the cache's layout: the fast cache laid out as the other readings would put it, against the buffer's rows
        let kv = c.run(&rec.embd, 2);
        let n = ROWS * NS;
        // (a) layers packed without padding (stride n_ctx), the rest of the buffer +0
        let mut packed = KvCross { k: vec![0; kv.k.len()], v: vec![0; kv.v.len()] };
        for il in 0..LAYERS {
            packed.k[il * n..(il + 1) * n].copy_from_slice(&kv.k[il * N_PAD * NS..il * N_PAD * NS + n]);
            packed.v[il * n..(il + 1) * n].copy_from_slice(&kv.v[il * N_PAD * NS..il * N_PAD * NS + n]);
        }
        // (b) the non-flash layout: K packed, V transposed per layer ([n_state][n_ctx])
        let mut nonflash = packed.clone();
        for il in 0..LAYERS {
            for f in 0..ROWS {
                for d in 0..NS {
                    nonflash.v[il * n + d * ROWS + f] = kv.v[il * N_PAD * NS + f * NS + d];
                }
            }
        }
        // (c) padding rows not +0 (as if they held a previous window's frames: here, frame 0 repeated)
        let mut dirty = kv.clone();
        for il in 0..LAYERS {
            for r in ROWS..N_PAD {
                let (dst, src) = ((il * N_PAD + r) * NS, il * N_PAD * NS);
                dirty.k.copy_within(src..src + NS, dst);
                dirty.v.copy_within(src..src + NS, dst);
            }
        }
        for (i, alt) in [packed, nonflash, dirty].iter().enumerate() {
            let bad = diff_rows(&cache_rows(alt), &rec.kv);
            counts[variants.len() + i].push(bad);
            failed |= bad == 0;
        }
    }
    let mut report = String::new();
    let names: Vec<&str> = variants.iter().map(|v| v.0).chain(["kv_cross layers without padding (stride n_ctx)", "the non-flash layout (V transposed)", "padding rows not +0"]).collect();
    for (name, c) in names.iter().zip(&counts) {
        report += &format!("  {name:<52} {c:?}\n");
    }
    eprint!(
        "oracle_cross_discriminators (rows differing per input; products: the six nodes of every 25th frame and the last, 4 layers = {} rows; layouts: the whole buffer, {} rows):\n{report}",
        frames.len() * 6 * LAYERS,
        2 * LAYERS * N_PAD
    );
    assert!(!failed, "a discriminator was not caught, or a prediction was wrong");
}
