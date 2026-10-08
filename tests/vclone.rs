// SPDX-License-Identifier: MIT OR Apache-2.0
//! vclone against its oracles: every shipped `.voaice` identity (whose stored print vprint.py wrote), and 2,000 metric
//! sets recorded from vprint.py itself (testing/vclone/make_oracle.py). Both must match field for field.
use voaice::vclone::{self, Check};

#[test]
fn every_shipped_identity_verifies() {
    let mut verified = 0;
    for e in std::fs::read_dir("tests/fixtures/voaice").unwrap() {
        let p = e.unwrap().path();
        match vclone::check_identity(&std::fs::read_to_string(&p).unwrap()).unwrap() {
            Check::Verified(_) => verified += 1,
            Check::Unmeasured => assert!(p.ends_with("vclone.voaice"), "{p:?} is unmeasured"),
            Check::Mismatch { fields, .. } => panic!("{p:?}: {fields:?}"),
        }
    }
    assert_eq!(verified, 10);
}

/// repr() of a Python float, parsed back; Python writes nan/inf where Rust's parser wants NaN/inf
fn py_float(s: &str) -> f64 {
    match s {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap(),
    }
}

#[test]
fn byte_identical_to_vprint_py_on_2000_metric_sets() {
    let text = std::fs::read_to_string("tests/fixtures/vprint_oracle.jsonl").unwrap();
    let mut lines = text.lines();
    let head = voaice::json::parse(lines.next().unwrap()).unwrap();
    let n = head.get("cases").and_then(|v| v.as_f64()).unwrap() as usize;
    let mut checked = 0;
    for line in lines {
        let c = voaice::json::parse(line).unwrap();
        let ins: Vec<f64> = match c.get("in").unwrap() {
            voaice::json::Value::Arr(a) => a.iter().map(|v| py_float(v.as_str().unwrap())).collect(),
            _ => panic!(),
        };
        let p = vclone::vprint(&ins.try_into().unwrap()).unwrap();
        assert_eq!(p.canonical, c.get("canonical").unwrap().as_str().unwrap(), "canonical, case {checked}");
        assert_eq!(p.hash, c.get("hash").unwrap().as_str().unwrap());
        assert_eq!(p.hash512, c.get("hash512").unwrap().as_str().unwrap());
        assert_eq!(p.uint256, c.get("uint256").unwrap().as_str().unwrap());
        checked += 1;
    }
    assert_eq!(checked, n);
}
