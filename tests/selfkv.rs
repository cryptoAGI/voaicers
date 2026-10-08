// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.1.3: the self-attention products, the mask and the f16 self KV cache against what the shipped library computed on
//! every decoder call of `whisper_full` (`whisper_oracle --selfkv` into .oracle/selfkv, run by testing/release_gate.sh).
//! `#[ignore]`d: they need the model and the records, which are not in git.
//!
//!   cargo test --release --test selfkv -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_SELFKV (default .oracle/selfkv),
//! VOAICE_ORACLE_RECORD (.oracle/tiny.en, 0.0.1's transcripts).
//!
//! The record, per decoder call (config A = 0.0.1's transcript params, config B = a 300-token prompt without timestamps;
//! whisper_full at 1 and 4 threads): the batch, kv_self's head, n and cells; per decoder layer a 64-bit FNV-1a digest per
//! row of its twelve nodes (attn_ln's NORM, MUL, ADD; Q's MUL_MAT, ADD, SCALE; K's MUL_MAT, SCALE; V's MUL_MAT, ADD;
//! the CPYs into kv_self) and the layer's input whole; the KQ_mask's rows in f32 and f16; kv_self.k and .v after the
//! call, every cell of every layer.
use std::path::{Path, PathBuf};
use voaice::decoder::{Batch, DecoderInput};
use voaice::matmul;
use voaice::model::Model;
use voaice::norm::{self, Node};
use voaice::selfkv::{KvSelf, MaskVariant, SelfAttn, SelfNodes, SelfTaps, SelfVariant};

const NODES: usize = 12;

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn dir() -> PathBuf {
    env_path("VOAICE_ORACLE_SELFKV", ".oracle/selfkv")
}
fn model() -> Model {
    Model::load_pinned(&env_path("VOAICE_MODEL", "models/ggml-tiny.en.bin")).unwrap_or_else(|e| panic!("{e}"))
}
fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e} (run testing/release_gate.sh first)", p.display()))
}
fn text(p: &Path) -> String {
    String::from_utf8(read(p)).unwrap()
}
fn meta(p: &Path, key: &str) -> String {
    text(p).lines().find_map(|l| l.strip_prefix(&format!("{key}\t"))).unwrap_or_else(|| panic!("{}: no {key}", p.display())).to_string()
}
fn inputs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir())
        .expect("selfkv oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("selfkv.tsv").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert_eq!(v.len(), 8, "the 8 recorded inputs");
    v
}
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
/// the mask's rows: every element its own word (n_kv can be odd)
fn mask_digest<T: Copy + Into<u64>>(x: &[T]) -> u64 {
    x.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| (h ^ v.into()).wrapping_mul(0x0000_0100_0000_01b3))
}
fn rows32(x: &[f32], w: usize) -> Vec<u64> {
    x.chunks_exact(w).map(digest32).collect()
}
fn rows16(x: &[u16], w: usize) -> Vec<u64> {
    x.chunks_exact(w).map(digest16).collect()
}
fn differ(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// One recorded decoder call.
struct Call {
    cfg: char,
    threads: usize,
    batch: Batch,
    head: usize,
    n_kv: usize,
    cell_pos: Vec<i32>,
    /// [n_layer][12][n]
    nodes: Vec<u64>,
    m32: Vec<u64>,
    m16: Vec<u64>,
    /// [k, v][n_layer][size]
    kv: Vec<u64>,
    /// [n_layer][n][n_state]
    input: Vec<f32>,
}
impl Call {
    fn n(&self) -> usize {
        self.batch.n_tokens()
    }
    fn node(&self, il: usize, key: usize) -> &[u64] {
        &self.nodes[(il * NODES + key) * self.n()..][..self.n()]
    }
    fn input(&self, il: usize, ns: usize) -> &[f32] {
        &self.input[il * self.n() * ns..][..self.n() * ns]
    }
}
fn ints(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return vec![];
    }
    s.split(',').map(|x| x.parse().unwrap()).collect()
}
fn calls(stem: &str, n_layer: usize, ns: usize) -> Vec<Call> {
    let d = dir().join(stem);
    let d64: Vec<u64> = read(&d.join("selfkv.d64")).as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
    let f32s: Vec<f32> = read(&d.join("selfkv_in.f32")).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let mut out = Vec::new();
    for l in text(&d.join("selfkv_calls.tsv")).lines() {
        let f: Vec<&str> = l.split('\t').collect();
        assert_eq!((f[0], f.len()), ("call", 14), "{l}");
        let n: usize = f[4].parse().unwrap();
        let size: usize = f[7].parse().unwrap();
        let (o, oi): (usize, usize) = (f[12].parse().unwrap(), f[13].parse().unwrap());
        let (token, pos) = (ints(f[8]), ints(f[9]));
        let batch = Batch { n_seq_id: vec![1; n], seq_id: vec![0; n], logits: (0..n).map(|i| pos[0] != 0 || i + 1 == n).collect(), token, pos };
        assert_eq!(batch.n_tokens(), n);
        let nn = n_layer * NODES * n;
        out.push(Call {
            cfg: f[1].chars().next().unwrap(),
            threads: f[2].parse().unwrap(),
            batch,
            head: f[5].parse().unwrap(),
            n_kv: f[6].parse().unwrap(),
            cell_pos: ints(f[10]),
            nodes: d64[o..o + nn].to_vec(),
            m32: d64[o + nn..o + nn + n].to_vec(),
            m16: d64[o + nn + n..o + nn + 2 * n].to_vec(),
            kv: d64[o + nn + 2 * n..o + nn + 2 * n + 2 * n_layer * size].to_vec(),
            input: f32s[oi..oi + n_layer * n * ns].to_vec(),
        });
    }
    out
}
/// The calls of each run (config, thread count) in order.
fn runs(c: &[Call]) -> Vec<Vec<&Call>> {
    let mut v = Vec::new();
    for cfg in ['A', 'B'] {
        for th in [1, 4] {
            let r: Vec<&Call> = c.iter().filter(|c| c.cfg == cfg && c.threads == th).collect();
            if !r.is_empty() {
                v.push(r);
            }
        }
    }
    v
}
/// The model's twelve nodes as row digests.
fn node_digests(x: &SelfNodes, ns: usize) -> [Vec<u64>; NODES] {
    [
        rows32(&x.norm, ns),
        rows32(&x.ln_mul, ns),
        rows32(&x.ln_add, ns),
        rows32(&x.q_mm, ns),
        rows32(&x.q_add, ns),
        rows32(&x.q_scale, ns),
        rows32(&x.k_mm, ns),
        rows32(&x.k_scale, ns),
        rows32(&x.v_mm, ns),
        rows32(&x.v_add, ns),
        rows16(&x.k_cpy, ns),
        rows16(&x.v_cpy, ns),
    ]
}
fn cache_digests(kv: &KvSelf) -> Vec<u64> {
    let mut d = rows16(&kv.k, kv.n_state);
    d.extend(rows16(&kv.v, kv.n_state));
    d
}
/// The fast path's nodes for one layer: attn_ln's three by 0.0.8's kernel (the fused path never writes them), the
/// products' by `layer_into` with every tap, the CPYs read back from the cache's cells.
fn fast_nodes(sa: &SelfAttn, il: usize, x: &[f32], threads: usize, kv: &mut KvSelf) -> [Vec<u64>; NODES] {
    let ns = sa.n_state;
    let rows = x.len() / ns;
    let ln = &sa.layers[il].attn_ln;
    let lnd = [Node::Norm, Node::Mul, Node::Add].map(|nd| rows32(&ln.run(x, threads, nd), ns));
    let mut q = vec![0.0f32; rows * ns];
    let mut t: Vec<Vec<f32>> = (0..6).map(|_| vec![0.0f32; rows * ns]).collect();
    {
        let [a, b, c, d, e, f] = &mut t[..] else { unreachable!() };
        let taps = SelfTaps { q_mm: Some(a), q_add: Some(b), k_mm: Some(c), k_scale: Some(d), v_mm: Some(e), v_add: Some(f) };
        sa.layer_into(il, x, threads, kv, &mut q, taps);
    }
    let at = kv.row(il, kv.head) * ns;
    let [ln0, ln1, ln2] = lnd;
    [
        ln0,
        ln1,
        ln2,
        rows32(&t[0], ns),
        rows32(&t[1], ns),
        rows32(&q, ns),
        rows32(&t[2], ns),
        rows32(&t[3], ns),
        rows32(&t[4], ns),
        rows32(&t[5], ns),
        rows16(&kv.k[at..at + rows * ns], ns),
        rows16(&kv.v[at..at + rows * ns], ns),
    ]
}

#[test]
#[ignore]
fn oracle_selfkv_record_self_checks() {
    let (mut thr, mut calls_a, mut calls_b, mut rows) = (String::new(), 0, 0, 0);
    for s in inputs() {
        let m = dir().join(&s).join("selfkv.tsv");
        for (k, want) in [
            ("model_has_no_key_bias", "yes"),
            ("every_call_every_node_mask_and_cache", "yes"),
            ("checks_scale_b0_same_cpy_at_il_size_head_mm_src1_attn_ln_f16_weights_k_unbiased_mask_f32_to_f16_once", "yes"),
            ("cells_seq_id_0_used_empty_unused_none_past_n_kv", "yes"),
            ("head_at_pos_n_kv_eq_head_plus_n", "yes"),
            ("A_result_observed_eq_unobserved_1t", "yes"),
            ("A_result_observed_eq_unobserved_4t", "yes"),
            ("B_result_observed_eq_unobserved_1t", "yes"),
            ("B_result_observed_eq_unobserved_4t", "yes"),
        ] {
            assert_eq!(meta(&m, k), want, "{s}: {k}");
        }
        if meta(&m, "A_calls_1t") != "0" {
            assert_eq!(meta(&m, "kqscale_bits"), "3eb504f3", "{s}");
            assert_eq!(meta(&m, "kv_self_size"), "512", "{s}");
            assert_eq!(meta(&m, "mask_values_zero_ninf_other_1t").split('\t').nth(2), Some("0"), "{s}: mask values other than 0 and -inf");
        }
        calls_a += meta(&m, "A_calls_1t").parse::<usize>().unwrap() + meta(&m, "A_calls_4t").parse::<usize>().unwrap();
        calls_b += meta(&m, "B_calls_1t").parse::<usize>().unwrap() + meta(&m, "B_calls_4t").parse::<usize>().unwrap();
        rows += meta(&m, "rows_1t").parse::<usize>().unwrap();
        let mut line = format!("{s}: A {}/{} calls, B {}/{}; largest batch {}, last position {}, n_kv up to {}; SCALE in place {}, q ADD in place {}; mask 0/-inf/other {}; 1 vs 4 threads:",
            meta(&m, "A_calls_1t"), meta(&m, "A_calls_4t"), meta(&m, "B_calls_1t"), meta(&m, "B_calls_4t"), meta(&m, "max_n_tokens"), meta(&m, "max_pos"),
            meta(&m, "max_n_kv"), meta(&m, "scale_in_place_1t"), meta(&m, "q_add_in_place_1t"), meta(&m, "mask_values_zero_ninf_other_1t").replace('\t', "/"));
        for cfg in ['A', 'B'] {
            line += &format!(" {cfg}: {} calls same batch, mask differ {}, cache differ (all inputs same) {};", meta(&m, &format!("{cfg}_threads_1_vs_4_calls_same_batch")),
                meta(&m, &format!("{cfg}_threads_1_vs_4_mask_differ")), meta(&m, &format!("{cfg}_threads_1_vs_4_kv_differ_all_inputs_same")));
            for il in 0..4 {
                let f: Vec<String> = meta(&m, &format!("{cfg}_threads_1_vs_4_layer{il}_input_same_differ_nodes_differ")).split('\t').map(String::from).collect();
                assert!(f[2].split(',').all(|x| x == "0"), "{s} {cfg} layer {il}: the same input gave different rows at 1 and 4 threads");
                line += &format!(" L{il} input same {} / differs {}", f[0], f[1]);
            }
        }
        thr += &line;
        thr.push('\n');
    }
    eprint!("{thr}");
    eprintln!("record: {calls_a} calls of config A and {calls_b} of B (1 and 4 threads), {rows} rows at 1 thread; at 1 vs 4 threads, wherever a node's input was the same its rows were the same");
}

/// Config A's observed 1-thread result = 0.0.1's transcript record (taken unobserved).
#[test]
#[ignore]
fn oracle_selfkv_observed_run_is_the_recorded_transcript() {
    let rec = env_path("VOAICE_ORACLE_RECORD", ".oracle/tiny.en");
    let mut n = 0;
    for s in inputs() {
        let want: Vec<String> = text(&rec.join(&s).join("transcript.tsv"))
            .lines()
            .filter(|l| l.starts_with("token\t"))
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                [f[0], f[1], f[2], f[3], f[6]].join("\t")
            })
            .collect();
        let got: Vec<String> = text(&dir().join(&s).join("selfkv_result.tsv")).lines().filter_map(|l| l.strip_prefix("A\t")).map(String::from).collect();
        assert_eq!(got, want, "{s}");
        n += got.len();
    }
    eprintln!("observed run (config A, 1 thread) = 0.0.1's transcript record: {n} result tokens (id and p bits) on 8 inputs");
}

/// Every node of every layer of every call, by the model and by the fast path, fed the reference's layer inputs; the
/// cells, head and n by `KvSelf::prepare`; the mask (model f32 and f16, fast f16); the whole cache after each call, by
/// the model's CPYs and by the fast path's.
#[test]
#[ignore]
fn oracle_selfkv_nodes_bit_exact() {
    let m = model();
    let sa = SelfAttn::new(&m).unwrap();
    let (ns, nl) = (sa.n_state, sa.layers.len());
    let (mut n_calls, mut node_rows, mut cache_rows, mut mask_rows, mut bad) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut m16 = Vec::new();
    for s in inputs() {
        let all = calls(&s, nl, ns);
        let (mut sc, mut sbad) = (0, 0);
        for run in runs(&all) {
            let (mut kv_model, mut kv_fast) = (KvSelf::for_model(&m), KvSelf::for_model(&m));
            kv_fast.k.fill(0xFFFF); // the fast cache starts dirty: clear() at the window must make it the reference's
            kv_fast.v.fill(0xFFFF);
            for c in run {
                if c.batch.pos[0] == 0 {
                    kv_model.clear();
                    kv_fast.clear();
                }
                kv_model.prepare(&c.batch).unwrap();
                kv_fast.prepare(&c.batch).unwrap();
                assert_eq!((kv_model.head, kv_model.n), (c.head, c.n_kv), "{s}: head and n");
                assert_eq!(kv_model.cells[..c.n_kv].iter().map(|x| x.pos).collect::<Vec<_>>(), c.cell_pos, "{s}: cells");
                let (f32m, f16m) = kv_model.mask_model(&c.batch, MaskVariant::default());
                kv_fast.mask_into(&c.batch, &mut m16);
                let w = c.n_kv;
                let d = |x: &[f32]| x.chunks_exact(w).map(|r| mask_digest(&r.iter().map(|a| a.to_bits()).collect::<Vec<_>>())).collect::<Vec<u64>>();
                let d16 = |x: &[u16]| x.chunks_exact(w).map(mask_digest).collect::<Vec<u64>>();
                sbad += differ(&d(&f32m), &c.m32) + differ(&d16(&f16m), &c.m16) + differ(&d16(&m16), &c.m16);
                mask_rows += 3 * c.n();
                for il in 0..nl {
                    let x = c.input(il, ns);
                    let mn = sa.model(il, x, c.threads, SelfVariant::default());
                    let want = node_digests(&mn, ns);
                    let at = kv_model.row(il, kv_model.head) * ns;
                    kv_model.k[at..at + c.n() * ns].copy_from_slice(&mn.k_cpy);
                    kv_model.v[at..at + c.n() * ns].copy_from_slice(&mn.v_cpy);
                    let fast = fast_nodes(&sa, il, x, c.threads, &mut kv_fast);
                    for key in 0..NODES {
                        sbad += differ(&want[key], c.node(il, key)) + differ(&fast[key], c.node(il, key));
                        node_rows += 2 * c.n();
                    }
                }
                sbad += differ(&cache_digests(&kv_model), &c.kv) + differ(&cache_digests(&kv_fast), &c.kv);
                cache_rows += 2 * c.kv.len();
                sc += 1;
            }
        }
        eprintln!("{s}: {sc} calls: every layer's 12 nodes (model and fast), the cells, the mask (f32, f16, fast f16), the cache after each call (model and fast) — {sbad} rows differ");
        n_calls += sc;
        bad += sbad;
    }
    eprintln!("self-attention products and kv_self: {n_calls} calls; {node_rows} node rows, {mask_rows} mask rows, {cache_rows} cache rows compared; {bad} differ");
    assert_eq!(bad, 0);
}

/// Layer 0 from voaice's own decoder input (0.1.2's `DecoderInput::run_batch`) at 1, 2 and 4 threads, the cache's
/// layer-0 cells kept across each run's calls by `KvSelf` — every node and every layer-0 cell after every call.
#[test]
#[ignore]
fn oracle_selfkv_layer0_from_voaice_input() {
    let m = model();
    let sa = SelfAttn::new(&m).unwrap();
    let d = DecoderInput::new(&m).unwrap();
    let (ns, nl) = (sa.n_state, sa.layers.len());
    let (mut rows, mut bad) = (0usize, 0usize);
    let mut x = Vec::new();
    for s in inputs() {
        let all = calls(&s, nl, ns);
        let mut sbad = 0;
        for run in runs(&all) {
            for threads in [1, 2, 4] {
                let mut kv = KvSelf::for_model(&m);
                for c in &run {
                    if c.batch.pos[0] == 0 {
                        kv.clear();
                    }
                    kv.prepare(&c.batch).unwrap();
                    d.run_batch(&c.batch, &mut x).unwrap();
                    let fast = fast_nodes(&sa, 0, &x, threads, &mut kv);
                    for (key, f) in fast.iter().enumerate() {
                        sbad += differ(f, c.node(0, key));
                        rows += c.n();
                    }
                    // layer 0's cells: k rows [0, size), v rows [n_layer · size, n_layer · size + size)
                    let got = cache_digests(&kv);
                    let sz = kv.size;
                    sbad += differ(&got[..sz], &c.kv[..sz]) + differ(&got[nl * sz..nl * sz + sz], &c.kv[nl * sz..nl * sz + sz]);
                    rows += 2 * sz;
                }
            }
        }
        eprintln!("{s}: layer 0 from voaice's decoder input at 1, 2, 4 threads — {sbad} rows differ");
        bad += sbad;
    }
    eprintln!("layer 0 from voaice's own input: {rows} rows (12 nodes + layer 0's 1,024 cache rows per call, x 3 thread counts) compared, {bad} differ");
    assert_eq!(bad, 0);
}

/// Discriminators: readings the oracle must reject, counted per input over the 1-thread calls of both configs — and the
/// ones it cannot tell apart, as predicted.
#[test]
#[ignore]
fn oracle_selfkv_discriminators() {
    let m = model();
    let sa = SelfAttn::new(&m).unwrap();
    let (ns, nl) = (sa.n_state, sa.layers.len());
    let numeric: [(&str, SelfVariant, bool); 12] = [
        ("scale before the products", SelfVariant { scale_first: true, ..Default::default() }, true),
        ("Q scaled before its bias", SelfVariant { q_scale_before_bias: true, ..Default::default() }, true),
        ("Q not scaled (the encoder's way)", SelfVariant { q_unscaled: true, ..Default::default() }, true),
        ("K not scaled", SelfVariant { k_unscaled: true, ..Default::default() }, true),
        ("K given a bias (the query's)", SelfVariant { k_biased: true, ..Default::default() }, true),
        ("V scaled", SelfVariant { v_scaled: true, ..Default::default() }, true),
        ("the scale in double", SelfVariant { scale_double: true, ..Default::default() }, true),
        ("attn_ln's mul + add fused", SelfVariant { ln: norm::Variant { fma: true, ..Default::default() }, ..Default::default() }, true),
        ("no f16 rounding of the activations", SelfVariant { mm: matmul::Variant { no_f16: true, ..Default::default() }, ..Default::default() }, true),
        ("one accumulator", SelfVariant { mm: matmul::Variant { single_accumulator: true, ..Default::default() }, ..Default::default() }, true),
        ("the accumulators in sequence", SelfVariant { mm: matmul::Variant { sequential_reduce: true, ..Default::default() }, ..Default::default() }, true),
        ("the CPYs by the row converter (indistinguishable on finite values)", SelfVariant { cpy_row_converter: true, ..Default::default() }, false),
    ];
    let masks: [(&str, MaskVariant, bool); 6] = [
        ("mask off by one (a row's own cell masked)", MaskVariant { off_by_one: true, ..Default::default() }, true),
        ("mask columns padded to 32", MaskVariant { pad32: true, ..Default::default() }, true),
        ("-inf as -65504 (f32 and f16)", MaskVariant { neg_max_f16: true, ..Default::default() }, true),
        ("-inf as f32's lowest (caught in f32; its f16 cast is -inf)", MaskVariant { neg_max_f32: true, ..Default::default() }, true),
        ("the sequence ignored (indistinguishable: one sequence, no free cell below n)", MaskVariant { no_seq: true, ..Default::default() }, false),
        ("the cast by the row converter (indistinguishable: 0 and -inf convert alike)", MaskVariant { row_converter: true, ..Default::default() }, false),
    ];
    for s in inputs() {
        let all = calls(&s, nl, ns);
        let one: Vec<&Call> = all.iter().filter(|c| c.threads == 1).collect();
        if one.is_empty() {
            eprintln!("{s}: no decoder call (whisper_full decodes nothing for it)");
            continue;
        }
        let rows: usize = one.iter().map(|c| c.n()).sum();
        let mut line = format!("{s} ({} calls, {rows} rows x 4 layers x 12 nodes):", one.len());
        for (name, v, must) in numeric {
            let mut n = 0;
            for c in &one {
                for il in 0..nl {
                    let got = node_digests(&sa.model(il, c.input(il, ns), 1, v), ns);
                    n += (0..NODES).map(|k| differ(&got[k], c.node(il, k))).sum::<usize>();
                }
            }
            line += &format!(" {name} {n};");
            if must {
                assert!(n > 0, "{s}: {name} not caught");
            } else {
                assert_eq!(n, 0, "{s}: {name} told apart");
            }
        }
        // the mask: rows of f32 and f16 that differ
        for (name, v, must) in masks {
            let (mut n32, mut n16) = (0, 0);
            for run in runs(&all).into_iter().filter(|r| r[0].threads == 1) {
                let mut kv = KvSelf::for_model(&m);
                for c in run {
                    if c.batch.pos[0] == 0 {
                        kv.clear();
                    }
                    kv.prepare(&c.batch).unwrap();
                    let (f, h) = kv.mask_model(&c.batch, v);
                    let w = f.len() / c.n();
                    n32 += differ(&f.chunks_exact(w).map(|r| mask_digest(&r.iter().map(|a| a.to_bits()).collect::<Vec<_>>())).collect::<Vec<_>>(), &c.m32);
                    n16 += differ(&h.chunks_exact(w).map(mask_digest).collect::<Vec<_>>(), &c.m16);
                }
            }
            line += &format!(" {name} {n32} f32 / {n16} f16;");
            if must {
                assert!(n32 + n16 > 0, "{s}: {name} not caught");
            } else {
                assert_eq!(n32 + n16, 0, "{s}: {name} told apart");
            }
        }
        // the cache: the model's CPYs (the reference's, checked above) written elsewhere, or the buffer kept dirty
        // (a window after the first in a run is what a buffer left uncleared can show: only jfk_x3 has one)
        let later_windows = one.iter().filter(|c| c.batch.pos[0] == 0).count() - runs(&all).iter().filter(|r| r[0].threads == 1).count();
        for (name, which) in [("K/V written one cell late", 0), ("layers 448 cells apart (unpadded)", 1), ("the buffer not cleared at a window (cells reset only)", 2)] {
            let mut n = 0;
            for run in runs(&all).into_iter().filter(|r| r[0].threads == 1) {
                let mut kv = KvSelf::for_model(&m);
                for c in run {
                    if c.batch.pos[0] == 0 {
                        let keep = (which == 2).then(|| (kv.k.clone(), kv.v.clone()));
                        kv.clear();
                        if let Some((k, v)) = keep {
                            (kv.k, kv.v) = (k, v);
                        }
                    }
                    kv.prepare(&c.batch).unwrap();
                    for il in 0..nl {
                        let mn = sa.model(il, c.input(il, ns), 1, SelfVariant::default());
                        let row = match which {
                            0 => il * kv.size + kv.head + 1,
                            1 => il * 448 + kv.head,
                            _ => il * kv.size + kv.head,
                        };
                        let at = row * ns;
                        kv.k[at..at + c.n() * ns].copy_from_slice(&mn.k_cpy);
                        kv.v[at..at + c.n() * ns].copy_from_slice(&mn.v_cpy);
                    }
                    n += differ(&cache_digests(&kv), &c.kv);
                }
            }
            line += &format!(" {name} {n};");
            if which == 2 && later_windows == 0 {
                assert_eq!(n, 0, "{s}: {name}: one window per run, nothing left to clear");
            } else {
                assert!(n > 0, "{s}: {name} not caught");
            }
        }
        eprintln!("{line}");
    }
}
