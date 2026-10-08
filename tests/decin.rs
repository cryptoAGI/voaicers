// SPDX-License-Identifier: MIT OR Apache-2.0
//! 0.1.2: the decoder's input against what the shipped library computed on every decoder call of `whisper_full`
//! (`whisper_oracle --decin` into .oracle/decin, run by testing/release_gate.sh). `#[ignore]`d: they need the model and
//! the records, which are not in git.
//!
//!   cargo test --release --test decin -- --ignored --nocapture --test-threads=1
//!
//! Paths: VOAICE_MODEL (default models/ggml-tiny.en.bin), VOAICE_ORACLE_DECIN (default .oracle/decin),
//! VOAICE_ORACLE_RECORD (.oracle/tiny.en, 0.0.1's transcripts).
//!
//! The record: per decoder call (config A = 0.0.1's transcript params, config B = a 300-token prompt without
//! timestamps; whisper_full at 1 and 4 threads) the batch whisper read its inputs from, and a 64-bit FNV-1a digest per
//! row of the token rows (GET_ROWS d_te), the position rows (GET_ROWS d_pe) and their sum (ADD).
use std::path::{Path, PathBuf};
use voaice::decoder::{Batch, DecinVariant, DecoderInput, Prompt};
use voaice::model::Model;

fn env_path(var: &str, default: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default))
}
fn decin_dir() -> PathBuf {
    env_path("VOAICE_ORACLE_DECIN", ".oracle/decin")
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
    let mut v: Vec<String> = std::fs::read_dir(decin_dir())
        .expect("decin oracle dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("decin.tsv").exists())
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
fn rows(x: &[f32], ns: usize) -> Vec<u64> {
    x.chunks_exact(ns).map(digest32).collect()
}

/// One recorded decoder call.
struct Call {
    cfg: char,
    threads: usize,
    k: usize,
    batch: Batch,
    te: Vec<u64>,
    pe: Vec<u64>,
    add: Vec<u64>,
}
fn ints(s: &str) -> Vec<i32> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}
fn calls(stem: &str) -> Vec<Call> {
    let dir = decin_dir().join(stem);
    let d: Vec<u64> = read(&dir.join("decin.d64")).as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
    let mut out = Vec::new();
    for l in text(&dir.join("decin_calls.tsv")).lines() {
        let f: Vec<&str> = l.split('\t').collect();
        assert_eq!((f[0], f.len()), ("call", 11), "{l}");
        let n: usize = f[4].parse().unwrap();
        let off: usize = f[10].parse().unwrap();
        let batch = Batch {
            token: ints(f[5]),
            pos: ints(f[6]),
            seq_id: ints(f[7]),
            n_seq_id: ints(f[8]),
            logits: ints(f[9]).iter().map(|&x| x != 0).collect(),
        };
        assert_eq!(batch.n_tokens(), n);
        out.push(Call {
            cfg: f[1].chars().next().unwrap(),
            threads: f[2].parse().unwrap(),
            k: f[3].parse().unwrap(),
            batch,
            te: d[off..off + n].to_vec(),
            pe: d[off + n..off + 2 * n].to_vec(),
            add: d[off + 2 * n..off + 3 * n].to_vec(),
        });
    }
    out
}
/// The run's calls split into windows: a window starts at a call whose first position is 0.
fn windows(c: &[&Call]) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = (0..c.len()).filter(|&i| c[i].batch.pos[0] == 0).collect();
    starts.iter().enumerate().map(|(j, &s)| (s, starts.get(j + 1).copied().unwrap_or(c.len()))).collect()
}
/// config B's prompt_tokens, as the oracle builds them (jfk's recorded text tokens, cycled to 300)
fn prompt_b() -> Vec<i32> {
    const W: [i32; 23] = [843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, 460, 466, 329, 345, 1265, 644, 345, 460, 466, 329, 534, 1499, 13];
    (0..300).map(|i| W[i % 23]).collect()
}

#[test]
#[ignore]
fn oracle_decin_record_self_checks() {
    for s in inputs() {
        let m = decin_dir().join(&s).join("decin.tsv");
        for (k, want) in [
            ("d_te_type", "f16"),
            ("d_pe_type", "f32"),
            ("every_call_te_pe_add_observed", "yes"),
            ("checks_batch_eq_inputs_types_order_add_src0_te_no_other_get_rows", "yes"),
            ("A_result_observed_eq_unobserved_1t", "yes"),
            ("A_result_observed_eq_unobserved_4t", "yes"),
        ] {
            assert_eq!(meta(&m, k), want, "{s}: {k}");
        }
        eprintln!(
            "{s}: A {} calls (1t) / {} (4t), B {} / {}; rows {}, longest batch {}, last position {}; calls 1 vs 4: A {}, B {}; results 1 = 4 threads: A {}, B {}",
            meta(&m, "A_calls_1t"),
            meta(&m, "A_calls_4t"),
            meta(&m, "B_calls_1t"),
            meta(&m, "B_calls_4t"),
            meta(&m, "rows_1t"),
            meta(&m, "max_n_tokens"),
            meta(&m, "max_pos"),
            meta(&m, "A_calls_threads_1_vs_4_identical"),
            meta(&m, "B_calls_threads_1_vs_4_identical"),
            meta(&m, "A_result_1t_eq_4t"),
            meta(&m, "B_result_1t_eq_4t"),
        );
    }
}

/// The observed 1-thread run's result is 0.0.1's transcript record (taken unobserved): the observation changed nothing,
/// and the steps of config A are the transcripts the roadmap names.
#[test]
#[ignore]
fn oracle_decin_observed_run_is_the_recorded_transcript() {
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
        let got: Vec<String> = text(&decin_dir().join(&s).join("decin_result.tsv")).lines().map(String::from).collect();
        assert_eq!(got, want, "{s}");
        n += got.len();
    }
    eprintln!("observed run = 0.0.1's transcript record: {n} result tokens (id and p bits) on 8 inputs");
}

/// The batches: every prompt is `Prompt::window` of its past through `Batch::prep_legacy`, every step `Batch::prep_step`
/// of the token the previous call's decoder sampled — the token the record shows fed next.
#[test]
#[ignore]
fn oracle_decin_batches() {
    let m = model();
    let sp = voaice::model::Specials::for_hparams(&m.hparams);
    let mut b = Batch::with_capacity(m.hparams.n_text_ctx as usize);
    let mut pr = Vec::new();
    let (mut prompts, mut steps, mut by_cfg) = (0, 0, [0usize; 2]);
    for s in inputs() {
        let all = calls(&s);
        for cfg in ['A', 'B'] {
            for th in [1, 4] {
                let c: Vec<&Call> = all.iter().filter(|c| c.cfg == cfg && c.threads == th).collect();
                let mut p = Prompt::for_model(&m);
                p.no_timestamps = cfg == 'B';
                let mut past = if cfg == 'B' { prompt_b() } else { vec![] };
                for (w, &(a, z)) in windows(&c).iter().enumerate() {
                    // whisper empties the past for a window that starts within 500 frames of the end (whisper.cpp:7046).
                    // Every second window here is jfk_x3's (33.0 s = 3,300 frames): it starts at 2,900 (A: the last
                    // window-1 segment ends there, testing/results) or 3,000 (B: no timestamps, a full 30 s), both
                    // within 500 of 3,300 — the seek comes from the decode loop (later increments), so it is stated here
                    if w > 0 {
                        assert_eq!(s, "jfk_x3");
                        past.clear();
                    }
                    p.window(&past, &mut pr);
                    b.prep_legacy(&pr, 0, 0);
                    assert_eq!(c[a].batch, b, "{s} {cfg} {th}t: window {w}'s prompt (call {})", c[a].k);
                    prompts += 1;
                    for i in 1..z - a {
                        b.prep_step(c[a + i].batch.token[0], pr.len(), i - 1, 0);
                        assert_eq!(c[a + i].batch, b, "{s} {cfg} {th}t: step {i} of window {w} (call {})", c[a + i].k);
                        steps += 1;
                    }
                    by_cfg[(cfg == 'B') as usize] += z - a;
                    // the text of this window becomes the next one's past (cleared above for jfk_x3's second window)
                    past.extend(c[a + 1..z].iter().map(|c| c.batch.token[0]).filter(|&t| t != sp.eot));
                }
            }
        }
    }
    eprintln!("batches: {prompts} prompts and {steps} steps = whisper's ({} calls of config A, {} of B, at 1 and 4 threads)", by_cfg[0], by_cfg[1]);
}

/// The three nodes: the model (one value at a time) and the fast path, each row's digest against the record's, on
/// every call of both configs at both thread counts.
#[test]
#[ignore]
fn oracle_decin_nodes_bit_exact() {
    let m = model();
    let d = DecoderInput::new(&m).unwrap();
    let ns = d.n_state;
    let (mut n_calls, mut n_rows, mut differ) = (0, 0, 0);
    let mut out = Vec::new();
    for s in inputs() {
        let (mut sc, mut sr) = (0, 0);
        for c in calls(&s) {
            let x = d.model(&c.batch.token, &c.batch.pos, DecinVariant::default()).unwrap();
            let (te, pe, sum) = (x.te, x.pe, x.sum);
            d.run_batch(&c.batch, &mut out).unwrap();
            for (got, want) in [(rows(&te, ns), &c.te), (rows(&pe, ns), &c.pe), (rows(&sum, ns), &c.add), (rows(&out, ns), &c.add)] {
                differ += got.iter().zip(want.iter()).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
            }
            sc += 1;
            sr += c.batch.n_tokens();
        }
        eprintln!("{s}: {sc} calls, {sr} rows: token rows, position rows, sum (model and fast) — {differ} rows differ so far");
        n_calls += sc;
        n_rows += sr;
    }
    eprintln!("decoder input: {n_calls} calls, {n_rows} rows x 4 comparisons = {} row comparisons, {differ} differ", 4 * n_rows);
    assert_eq!(differ, 0);
}

/// Discriminators: readings the oracle must reject, each counted per input on the sum's rows (ADD) — and two it cannot
/// tell apart, as predicted. The batch-level ones change the batch the model is fed.
#[test]
#[ignore]
fn oracle_decin_discriminators() {
    let m = model();
    let d = DecoderInput::new(&m).unwrap();
    let ns = d.n_state;
    let sp = voaice::model::Specials::for_hparams(&m.hparams);
    type Alter = fn(&Batch, usize, i32) -> Batch;
    let numeric: [(&str, DecinVariant, bool); 6] = [
        ("f16 subnormals flushed (DAZ)", DecinVariant { te_ftz: true, ..Default::default() }, true),
        ("token rows through bf16", DecinVariant { te_bf16: true, ..Default::default() }, true),
        ("position rows rounded to f16 (indistinguishable: every d_pe value is an f16)", DecinVariant { pe_f16: true, ..Default::default() }, false),
        ("sum rounded to f16", DecinVariant { sum_f16: true, ..Default::default() }, true),
        ("position rows first (indistinguishable: f32 + commutes)", DecinVariant { pe_first: true, ..Default::default() }, false),
        ("sum in double (indistinguishable: exact for two f32)", DecinVariant { sum_double: true, ..Default::default() }, false),
    ];
    let batch: [(&str, Alter); 4] = [
        ("positions off by one", |b, _, _| Batch { pos: b.pos.iter().map(|p| p + 1).collect(), ..b.clone() }),
        ("step positions without the prompt (i, not prompt + i)", |b, plen, _| {
            let mut x = b.clone();
            if b.pos[0] != 0 {
                x.pos[0] -= plen as i32;
            }
            x
        }),
        ("position rows from row 0", |b, _, _| Batch { pos: vec![0; b.n_tokens()], ..b.clone() }),
        ("multilingual token ids (SOT + 1)", |b, _, sot| Batch { token: b.token.iter().map(|&t| if t >= sot { t + 1 } else { t }).collect(), ..b.clone() }),
    ];
    // found by the oracle: d_pe is stored f32 but every one of its values is exactly an f16 (the checkpoint was f16),
    // so rounding it to f16 changes nothing; and the f16 subnormals the DAZ reading flushes are in SOT's own row
    let pe_inexact = d.pe.iter().filter(|&&x| voaice::f16::fp16_to_fp32(voaice::f16::fp32_to_fp16(x)).to_bits() != x.to_bits()).count();
    let sub = |t: i32| d.te[t as usize * ns..][..ns].iter().filter(|&&h| h & 0x7C00 == 0 && h & 0x3FF != 0).count();
    eprintln!(
        "d_pe: {pe_inexact} of {} values not an f16; d_te: {} subnormal f16 in {} of {} rows (SOT's row: {})",
        d.pe.len(),
        d.te.iter().filter(|&&h| h & 0x7C00 == 0 && h & 0x3FF != 0).count(),
        d.te.chunks_exact(ns).filter(|r| r.iter().any(|&h| h & 0x7C00 == 0 && h & 0x3FF != 0)).count(),
        d.n_vocab,
        sub(sp.sot)
    );
    assert_eq!(pe_inexact, 0);
    for s in inputs() {
        let all = calls(&s);
        let c: Vec<&Call> = all.iter().filter(|c| c.threads == 1).collect();
        if c.is_empty() {
            eprintln!("{s}: no decoder call (whisper_full decodes nothing for it)");
            continue;
        }
        let mut line = format!("{s} ({} rows):", c.iter().map(|c| c.batch.n_tokens()).sum::<usize>());
        for (name, v, must) in numeric {
            let mut n = 0;
            for c in &c {
                let sum = d.model(&c.batch.token, &c.batch.pos, v).unwrap().sum;
                n += rows(&sum, ns).iter().zip(&c.add).filter(|(a, b)| a != b).count();
            }
            line += &format!(" {name} {n};");
            if must {
                assert!(n > 0, "{s}: {name} not caught");
            } else {
                assert_eq!(n, 0, "{s}: {name} told apart");
            }
        }
        // the window's prompt length, for the step positions
        let mut plen = 0;
        for (name, f) in batch {
            let mut n = 0;
            for c in &c {
                if c.batch.pos[0] == 0 {
                    plen = c.batch.n_tokens();
                }
                let x = f(&c.batch, plen, sp.sot);
                let x = Batch { pos: x.pos.iter().map(|&p| p.clamp(0, d.n_ctx as i32 - 1)).collect(), ..x };
                let sum = d.model(&x.token, &x.pos, DecinVariant::default()).unwrap().sum;
                n += rows(&sum, ns).iter().zip(&c.add).filter(|(a, b)| a != b).count();
            }
            line += &format!(" {name} {n};");
            assert!(n > 0, "{s}: {name} not caught");
        }
        eprintln!("{line}");
    }
}
