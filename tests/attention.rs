// SPDX-License-Identifier: MIT OR Apache-2.0
//! v0.1.0: flash attention and the whole encoder against what the shipped library computed (`whisper_oracle --encoder`
//! into .oracle/encoder, with 0.0.9's .oracle/matmul and 0.0.8's .oracle/norm, run by testing/release_gate.sh).
//! `#[ignore]`d: they need the model and the records, which are not in git.
//!
//!   cargo test --release --test attention -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_ENCODER (default .oracle/encoder),
//! VOAICE_ORACLE_MATMUL (.oracle/matmul), VOAICE_ORACLE_NORM (.oracle/norm), VOAICE_AUDIO (.audio).
//!
//! The records are compact: a 64-bit FNV-1a digest per row (frame) of every computed encoder node, the attention's
//! output whole per block (0.0.9's record) and `embd_enc` whole. The attention is fed Q, K and V that voaice computes
//! from attn_ln's recorded output (0.0.8's record, whole), each checked against the digests of the reference's own
//! q_add and K / V CPY nodes before it is used; the whole encoder is fed only voaice's own mel.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use voaice::attention::{self, Attention, Variant};
use voaice::conv::ConvStage;
use voaice::encoder::{Encoder, EncoderBuffers};
use voaice::matmul::{Block, MlpTaps, QkvTaps};
use voaice::norm::{LayerNorm, Node};
use voaice::{mel, model::Model, wav};

const NS: usize = 384;
const NH: usize = 1536;
const ROWS: usize = 1500;
const N_KV: usize = 1536;
const LAYERS: usize = 4;
const N_FRAMES: usize = 3000;

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn enc_dir() -> PathBuf {
    env_path("VOAICE_ORACLE_ENCODER", ".oracle/encoder")
}
fn mm_dir() -> PathBuf {
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
    let mut v: Vec<String> = std::fs::read_dir(enc_dir())
        .expect("encoder oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("encoder.tsv").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no recorded inputs in the encoder oracle dir");
    v
}
fn check_record(w: &str) {
    let t = enc_dir().join(w).join("encoder.tsv");
    for k in [
        "fa_inputs_as_whisper_builds_them",
        "kv_pad_rows_1500_1535_all_zero",
        "threads_1_vs_4_bit_identical",
        "embd_enc_observed_eq_unobserved",
        "embd_enc_1_2_4_threads_identical",
        "last_node_eq_embd_enc",
        "standalone_fa_eq_sched_1_to_8_threads",
    ] {
        assert_eq!(meta(&t, k), "yes", "{w}: the reference's own record says {k} = NO");
    }
    assert_eq!(meta(&t, "n_state"), NS.to_string());
    assert_eq!(meta(&t, "n_layer"), LAYERS.to_string());
    let m = mm_dir().join(w).join("matmul.tsv");
    assert_eq!(meta(&m, "threads_1_vs_4_bit_identical"), "yes");
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
fn diff_rows(ours: &[u64], theirs: &[u64]) -> usize {
    assert_eq!(ours.len(), theirs.len(), "row counts differ");
    ours.iter().zip(theirs).filter(|(a, b)| a != b).count()
}
fn values_differ(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// The encoder record of one input: key -> row digests (enc.tsv indexes enc.d64).
fn enc_record(w: &str) -> BTreeMap<String, Vec<u64>> {
    let dir = enc_dir().join(w);
    let all = read_u64(&dir.join("enc.d64"));
    let tsv = String::from_utf8(read(&dir.join("enc.tsv"))).unwrap();
    let mut m = BTreeMap::new();
    for l in tsv.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        let (rows, off): (usize, usize) = (f[5].parse().unwrap(), f[6].parse().unwrap());
        m.insert(f[1].to_string(), all[off..off + rows].to_vec());
    }
    m
}

/// Q (q_add), K and V (the CPY nodes) of block `il` from attn_ln's recorded output, each checked against the record.
fn qkv(b: &Block, w: &str, il: usize, rec: &BTreeMap<String, Vec<u64>>) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
    let x = read_f32(&norm_dir().join(w).join(format!("attn_ln_{il}.in.f32")));
    let (mut q, mut k16, mut v16) = (vec![0.0f32; ROWS * NS], vec![0u16; ROWS * NS], vec![0u16; ROWS * NS]);
    b.qkv_into(&x, 2, &mut q, &mut k16, &mut v16, QkvTaps::default());
    let p = format!("b{il}");
    assert_eq!(diff_rows(&rows32(&q, NS), &rec[&format!("{p}.q_add")]), 0, "{w} block {il}: Q differs from the record");
    assert_eq!(diff_rows(&rows16(&k16, NS), &rec[&format!("{p}.k_cpy")]), 0, "{w} block {il}: K differs from the record");
    assert_eq!(diff_rows(&rows16(&v16, NS), &rec[&format!("{p}.v_cpy")]), 0, "{w} block {il}: V differs from the record");
    (q, k16, v16)
}

#[test]
#[ignore]
fn oracle_attention_nodes_bit_exact() {
    let m = model();
    let blocks: Vec<Block> = (0..LAYERS).map(|il| Block::new(&m, il).unwrap()).collect();
    let a = Attention::new(NS, N_KV).unwrap();
    let mut report = String::new();
    let (mut failed, mut values, mut model_rows) = (false, 0usize, 0usize);
    let sample: Vec<usize> = (0..ROWS).step_by(10).collect();
    for w in wavs() {
        check_record(&w);
        let rec = enc_record(&w);
        report += &format!("  {w:<11}");
        for (il, b) in blocks.iter().enumerate() {
            let (q, k16, v16) = qkv(b, &w, il, &rec);
            let want = read_f32(&mm_dir().join(&w).join(format!("b{il}.fa.f32")));
            assert_eq!(diff_rows(&rows32(&want, NS), &rec[&format!("b{il}.fa")]), 0, "{w}: the two records disagree");
            let mut n = [0usize; 3];
            for (t, threads) in [1, 4].into_iter().enumerate() {
                n[t] = values_differ(&a.run(&q, &k16, &v16, threads), &want);
            }
            // the model: every frame of block 0, every 10th frame of blocks 1-3 (it is scalar and slow)
            let frames: Vec<usize> = if il == 0 { (0..ROWS).collect() } else { sample.clone() };
            let got = attention::attention_model_frames(&q, &k16, &v16, NS, N_KV, Variant::default(), &frames);
            n[2] = frames.iter().enumerate().filter(|&(o, &f)| digest32(&got[o * NS..(o + 1) * NS]) != rec[&format!("b{il}.fa")][f]).count();
            failed |= n.iter().any(|&c| c > 0);
            values += 2 * ROWS * NS;
            model_rows += frames.len();
            report += &format!(" | b{il}: fast 1t {} 4t {} values, model {} of {} rows", n[0], n[1], n[2], frames.len());
        }
        report += "\n";
    }
    eprint!(
        "oracle_attention_nodes_bit_exact (Q, K, V from attn_ln's recorded output, each = the record's digests; differing values of the \
         FLASH_ATTN_EXT node, and model rows by digest):\n{report}  fast path: {values} values compared; model: {model_rows} frames x 6 heads\n"
    );
    assert!(!failed, "attention differs from the reference");
}

#[test]
#[ignore]
fn oracle_attention_discriminators() {
    let m = model();
    let blocks: Vec<Block> = (0..LAYERS).map(|il| Block::new(&m, il).unwrap()).collect();
    let frames: Vec<usize> = (0..ROWS).step_by(25).chain([ROWS - 1]).collect();
    let d = Variant::default();
    let variants: [(&str, Variant, bool); 10] = [
        ("probabilities through glibc expf", Variant { libm_probs: true, ..d }, true),
        ("rescale through ggml_v_expf", Variant { v_expf_rescale: true, ..d }, true),
        ("no running max (global max, one pass)", Variant { no_running_max: true, ..d }, true),
        ("kv_pad's 36 zero rows excluded", Variant { pad_excluded: true, ..d }, true),
        ("scores without FMA", Variant { scores_unfused: true, ..d }, true),
        ("output accumulated without FMA", Variant { output_unfused: true, ..d }, true),
        ("softmax sums in f32", Variant { sum_f32: true, ..d }, true),
        ("out / S instead of out * (1/S)", Variant { divide: true, ..d }, true),
        ("the one-chunk path (Q f16, f16 dot, V in f16)", Variant { one_chunk: true, ..d }, true),
        ("scale on Q before the dot (x1/8 exact)", Variant { scale_on_q: true, ..d }, false),
    ];
    let mut counts = vec![vec![0usize; 8]; variants.len()];
    let mut one_chunk_ref = [0usize; 2]; // the one-chunk model (S unfused, as built; fused) against the reference's use_ref record
    let mut report = String::new();
    let ws = wavs();
    for (wi, w) in ws.iter().enumerate() {
        let rec = enc_record(w);
        for (il, b) in blocks.iter().enumerate() {
            let (q, k16, v16) = qkv(b, w, il, &rec);
            let want = &rec[&format!("b{il}.fa")];
            let fa_ref = read_u64(&enc_dir().join(w).join(format!("b{il}.fa_ref.d64")));
            let rows_of = |var: Variant, theirs: &[u64]| {
                let got = attention::attention_model_frames(&q, &k16, &v16, NS, N_KV, var, &frames);
                frames.iter().enumerate().filter(|&(o, &f)| digest32(&got[o * NS..(o + 1) * NS]) != theirs[f]).count()
            };
            for (vi, (_, var, _)) in variants.iter().enumerate() {
                counts[vi][wi] += rows_of(*var, want);
            }
            one_chunk_ref[0] += rows_of(Variant { one_chunk: true, ..d }, &fa_ref);
            one_chunk_ref[1] += rows_of(Variant { one_chunk: true, one_chunk_s_fused: true, ..d }, &fa_ref);
        }
    }
    let total = frames.len() * LAYERS;
    let mut failed = false;
    for (vi, (name, _, must)) in variants.iter().enumerate() {
        let caught_all = counts[vi].iter().all(|&c| c > 0);
        let none = counts[vi].iter().all(|&c| c == 0);
        failed |= if *must { !caught_all } else { !none };
        report += &format!(
            "  {name:<48} {} (rows differing of {total} per input: {:?})\n",
            if *must { if caught_all { "caught on every input" } else { "NOT CAUGHT" } } else if none { "indistinguishable, as predicted" } else { "DIFFERS (prediction wrong)" },
            counts[vi]
        );
    }
    report += &format!(
        "  the one-chunk model against the reference's own use_ref output (b<il>.fa_ref): as read (S = S*ms + vs) {} rows differ, S = fma(S, ms, vs) {} rows differ (of {})\n",
        one_chunk_ref[0],
        one_chunk_ref[1],
        total * ws.len()
    );
    failed |= one_chunk_ref[0] != 0 || one_chunk_ref[1] == 0;
    eprint!("oracle_attention_discriminators (every 25th frame and the last, all 6 heads, 4 blocks, by row digest):\n{report}");
    assert!(!failed, "a discriminator was not caught, or a prediction was wrong");
}

#[test]
#[ignore]
fn oracle_encoder_end_to_end() {
    let m = model();
    let enc = Encoder::new(&m).unwrap();
    let conv = ConvStage::new(&m).unwrap();
    let norms = LayerNorm::encoder(&m).unwrap();
    let ln = |name: &str| &norms.iter().find(|(n, _)| n == name).unwrap().1;
    let t = mel::Tables::new();
    let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize).unwrap();
    let mut report = String::new();
    let (mut failed, mut nodes_total, mut rows_total) = (false, 0usize, 0usize);
    for w in wavs() {
        check_record(&w);
        let rec = enc_record(&w);
        let want = read_f32(&enc_dir().join(&w).join("embd_enc.f32"));
        let pcm = wav::read(&env_path("VOAICE_AUDIO", ".audio").join(format!("{w}.wav"))).unwrap();
        let mel = plan.run(&pcm, 4).unwrap();
        report += &format!("  {w:<11}");
        for threads in [1, 2, 4] {
            // every node, by the stages' own entry points with taps, from voaice's mel
            let mut got: BTreeMap<String, Vec<u64>> = BTreeMap::new();
            let mut x = conv.run(&mel.data, mel.n_len, 0, N_FRAMES, threads);
            got.insert("pe_add".into(), rows32(&x, NS));
            let a = Attention::new(NS, N_KV).unwrap();
            let put3 = |got: &mut BTreeMap<String, Vec<u64>>, name: &str, x: &[f32]| {
                for (node, k) in [(Node::Norm, "norm"), (Node::Mul, "mul"), (Node::Add, "add")] {
                    got.insert(format!("{name}.{k}"), rows32(&ln(name).run(x, threads, node), NS));
                }
            };
            for (il, b) in enc.blocks.iter().enumerate() {
                put3(&mut got, &format!("attn_ln_{il}"), &x);
                let n = ROWS * NS;
                let (mut q, mut k16, mut v16) = (vec![0.0f32; n], vec![0u16; n], vec![0u16; n]);
                let (mut k_mm, mut v_mm, mut v_add, mut q_mm) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
                b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, QkvTaps { k_mm: Some(&mut k_mm), v_mm: Some(&mut v_mm), v_add: Some(&mut v_add), q_mm: Some(&mut q_mm) });
                let p = format!("b{il}");
                for (k, v) in [("k_mm", &k_mm), ("v_mm", &v_mm), ("v_add", &v_add), ("q_mm", &q_mm), ("q_add", &q)] {
                    got.insert(format!("{p}.{k}"), rows32(v, NS));
                }
                got.insert(format!("{p}.k_cpy"), rows16(&k16, NS));
                got.insert(format!("{p}.v_cpy"), rows16(&v16, NS));
                let att = a.run(&q, &k16, &v16, threads);
                got.insert(format!("{p}.fa"), rows32(&att, NS));
                let mut t: Vec<Vec<f32>> = [NS, NS, NS, NH, NH, NH, NS, NS].iter().map(|&c| vec![0.0f32; ROWS * c]).collect();
                let mut out = vec![0.0f32; n];
                {
                    let [o_mm, o_add, o_res, fc1_mm, fc1_add, gelu, fc2_mm, fc2_add] = &mut t[..] else { unreachable!() };
                    b.mlp_into(&att, &x, threads, &mut out, MlpTaps {
                        o_mm: Some(o_mm),
                        o_add: Some(o_add),
                        o_res: Some(o_res),
                        fc1_mm: Some(fc1_mm),
                        fc1_add: Some(fc1_add),
                        gelu: Some(gelu),
                        fc2_mm: Some(fc2_mm),
                        fc2_add: Some(fc2_add),
                    });
                }
                for (k, (v, c)) in ["o_mm", "o_add", "o_res", "fc1_mm", "fc1_add", "gelu", "fc2_mm", "fc2_add"].iter().zip(t.iter().zip([NS, NS, NS, NH, NH, NH, NS, NS])) {
                    got.insert(format!("{p}.{k}"), rows32(v, c));
                }
                got.insert(format!("{p}.mlp_res"), rows32(&out, NS));
                put3(&mut got, &format!("mlp_ln_{il}"), &t[2]);
                x = out;
            }
            put3(&mut got, "ln_post", &x);
            // compare every node the record names (its CONT is the conv stage's transpose, inside pe_add here)
            let (mut nodes, mut bad_nodes, mut bad_rows) = (0, 0, 0);
            for (key, theirs) in &rec {
                if key.starts_with('n') && key[1..].chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                let ours = got.get(key).unwrap_or_else(|| panic!("{w}: no voaice node for the record's {key}"));
                let r = diff_rows(ours, theirs);
                nodes += 1;
                rows_total += theirs.len();
                bad_rows += r;
                bad_nodes += (r > 0) as usize;
            }
            nodes_total += nodes;
            // the pipeline itself, no taps: mel -> embd_enc
            let mut buf = EncoderBuffers::default();
            let mut e = vec![0.0f32; ROWS * NS];
            enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut e);
            let ev = values_differ(&e, &want);
            let again = values_differ(&{
                enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut e);
                e.clone()
            }, &want);
            failed |= bad_nodes > 0 || ev > 0 || again > 0;
            report += &format!(" | {threads}t: {nodes} nodes, {bad_nodes} differ ({bad_rows} rows); embd_enc {ev} values differ ({again} on reuse)");
        }
        report += "\n";
    }
    eprint!(
        "oracle_encoder_end_to_end (the WAV -> voaice's mel -> conv stage -> 4 blocks with attention -> ln_post; every computed node of the \
         encoder graph by row digest, then embd_enc whole from Encoder::encode_into):\n{report}  {nodes_total} node comparisons, {rows_total} rows\n"
    );
    assert!(!failed, "the encoder differs from the reference");
}
