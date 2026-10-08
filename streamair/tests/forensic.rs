// SPDX-License-Identifier: MIT OR Apache-2.0
//! The forensic voice print against voaice's Forensic.js (testing/fclone/make_forensic_oracle.mjs): real WAVs, three
//! frame budgets each, the print recomputed from the recorded feature means, sample rate and frame count.
use streamair::forensic::{forensic_print, FEATURES};
use voaice::json::{self, Value};

#[test]
fn byte_identical_to_forensic_js() {
    let o = json::parse(&std::fs::read_to_string("tests/fixtures/forensic_oracle.json").unwrap()).unwrap();
    let Value::Arr(cases) = o.get("cases").unwrap() else { panic!() };
    for c in cases {
        let mut f = [0f64; 6];
        for (i, k) in FEATURES.iter().enumerate() {
            f[i] = c.path(&["features", k]).and_then(Value::as_str).unwrap().parse().unwrap();
        }
        let sr = c.get("sampleRate").unwrap().as_f64().unwrap() as u32;
        let n = c.get("framesUsed").unwrap().as_f64().unwrap() as u64;
        let p = forensic_print(&f, sr, n).unwrap();
        assert_eq!(p.hash, c.get("hash").unwrap().as_str().unwrap(), "{:?}", c.get("wav"));
        let Value::Arr(ms) = c.get("measures").unwrap() else { panic!() };
        assert_eq!(p.measures, ms.iter().map(|m| m.as_str().unwrap().to_string()).collect::<Vec<_>>());
    }
    assert_eq!(cases.len(), 12);
}
