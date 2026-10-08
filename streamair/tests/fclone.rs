// SPDX-License-Identifier: MIT OR Apache-2.0
//! fclone against its oracles: 1,000 landmark sets recorded from faicey's own geometry.js and faceprint.js
//! (testing/fclone/make_oracle.mjs), a faice/1 file written by ollywoo's fclone.js, an unmeasured .faice, and
//! MediaPipe's tessellation.
use streamair::fclone::{self, Check, Point, Quality};
use voaice::json::{self, Value};

fn s(v: &Value) -> &str {
    v.as_str().unwrap()
}
/// String(number) from JavaScript, parsed back (it writes NaN / Infinity where Rust's parser wants NaN / inf)
fn js_num(t: &str) -> f64 {
    match t {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ => t.parse().unwrap(),
    }
}

#[test]
fn byte_identical_to_faicey_on_1000_landmark_sets() {
    let text = std::fs::read_to_string("tests/fixtures/faceprint_oracle.jsonl").unwrap();
    let mut lines = text.lines();
    let head = json::parse(lines.next().unwrap()).unwrap();
    let idx: Vec<usize> = match head.get("indices").unwrap() {
        Value::Arr(a) => a.iter().map(|v| v.as_f64().unwrap() as usize).collect(),
        _ => panic!(),
    };
    let mut n = 0;
    for line in lines {
        let c = json::parse(line).unwrap();
        let mut lms = vec![Point::default(); 478];
        if let Value::Arr(pts) = c.get("lm").unwrap() {
            for (k, p) in pts.iter().enumerate() {
                if let Value::Arr(xyz) = p {
                    lms[idx[k]] = Point { x: js_num(s(&xyz[0])), y: js_num(s(&xyz[1])), z: js_num(s(&xyz[2])) };
                }
            }
        }
        let sym = fclone::symmetry(&lms);
        assert_eq!(sym.to_bits(), js_num(s(c.get("symmetry").unwrap())).to_bits(), "symmetry, case {n}");
        let q = Quality { confidence: js_num(s(c.get("confidence").unwrap())), symmetry: sym,
                          frontality: js_num(s(c.get("frontality").unwrap())) };
        let p = fclone::faceprint(&fclone::proportions(&lms), q).unwrap();
        assert_eq!(p.canonical, s(c.get("canonical").unwrap()), "canonical, case {n}");
        assert_eq!(p.hash, s(c.get("hash").unwrap()), "hash, case {n}");
        n += 1;
    }
    assert_eq!(n, 1000);
}

#[test]
fn faice_files_check() {
    let t = std::fs::read_to_string("tests/fixtures/synthetic.faice.json").unwrap();
    match fclone::check_faice(&t).unwrap() {
        Check::Verified(p) => assert!(p.hash.starts_with("0x")),
        other => panic!("{other:?}"),
    }
    let t = std::fs::read_to_string("tests/fixtures/sAGI.faice.json").unwrap();
    assert!(matches!(fclone::check_faice(&t).unwrap(), Check::Unmeasured));
    let edited = std::fs::read_to_string("tests/fixtures/synthetic.faice.json").unwrap().replacen("\"jawRatio\": ", "\"jawRatio\": 1", 1);
    assert!(matches!(fclone::check_faice(&edited).unwrap(), Check::Mismatch { .. }), "an edited proportion is caught");
}

#[test]
fn mediapipe_tessellation_is_a_checked_mesh() {
    let v = json::parse(&std::fs::read_to_string("tests/fixtures/mediapipe_tessellation.json").unwrap()).unwrap();
    let pairs: Vec<(u32, u32)> = match v {
        Value::Arr(a) => a.iter().map(|e| match e {
            Value::Arr(p) => (p[0].as_f64().unwrap() as u32, p[1].as_f64().unwrap() as u32),
            _ => panic!(),
        }).collect(),
        _ => panic!(),
    };
    let (tris, c) = fclone::triangles(&pairs);
    assert!(c.ok(), "{c:?}");
    assert_eq!((c.vertices, c.edges, c.faces, c.euler, c.boundary), (468, 1322, 852, -2, 88));
    assert_eq!(tris.len(), 852);
}

#[test]
fn both_clones_are_here() {
    // the voice side, through the re-export: a known identity still verifies
    let t = std::fs::read_to_string("../tests/fixtures/voaice/neural.voaice").unwrap();
    assert!(matches!(streamair::vclone::check_identity(&t).unwrap(), streamair::vclone::Check::Verified(_)));
}

#[test]
fn the_oracle_can_fail_a_naive_hypot_disagrees() {
    // the same pipeline with sqrt(dx²+dy²+dz²) in place of V8's scaled, Kahan-summed Math.hypot: if no case
    // disagreed, the 1,000 recorded sets would not be evidence that js_hypot is the right port
    let text = std::fs::read_to_string("tests/fixtures/faceprint_oracle.jsonl").unwrap();
    let mut lines = text.lines();
    let head = json::parse(lines.next().unwrap()).unwrap();
    let idx: Vec<usize> = match head.get("indices").unwrap() {
        Value::Arr(a) => a.iter().map(|v| v.as_f64().unwrap() as usize).collect(),
        _ => panic!(),
    };
    let naive = |p: &Point, q: &Point| ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2)).sqrt();
    let (mut differ, mut total) = (0, 0);
    for line in lines {
        let c = json::parse(line).unwrap();
        let mut lms = vec![Point::default(); 478];
        if let Value::Arr(pts) = c.get("lm").unwrap() {
            for (k, p) in pts.iter().enumerate() {
                if let Value::Arr(xyz) = p {
                    lms[idx[k]] = Point { x: js_num(s(&xyz[0])), y: js_num(s(&xyz[1])), z: js_num(s(&xyz[2])) };
                }
            }
        }
        use fclone::lm::*;
        let (a, b) = (&lms[CHEEK_R], &lms[CHEEK_L]);
        let w = fclone::js_hypot(&[a.x - b.x, a.y - b.y, a.z - b.z]);
        if w.to_bits() != naive(a, b).to_bits() {
            differ += 1;
        }
        total += 1;
    }
    eprintln!("naive hypot differs from V8's on {differ} of {total} face widths");
    assert!(differ > 0);
}
