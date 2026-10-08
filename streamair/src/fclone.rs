// SPDX-License-Identifier: MIT OR Apache-2.0
//! fclone — fCLONE in Rust: the face, measured and triangulated, byte-identical to the JavaScript it mirrors.
//!
//! **The faceprint (`faceprint/1`, in `.faice` files of format `faice/1`).** Twelve scale-invariant ratios of
//! MediaPipe face landmarks, each encoded `toFixed18`, plus a precision score from the capture's quality, written as
//! `{"v":1,"kind":"faceprint","measureNames":[…],"measures":[…],"precisionScore":"…"}` and hashed:
//! `"0x" + sha256(that string)`. The reference is faicey's `face_clone/faceprint.js` and `geometry.js` (mindX), which
//! ollywoo vendors unchanged; the format is cryptoAGI/sagi `engine/FAICE_FORMAT.md`. The oracle is those files run
//! under node (`testing/fclone/make_oracle.mjs`), and a `.faice` file ollywoo's fclone.js wrote.
//!
//! Two traps, both handled here rather than approximated:
//! - **`toFixed18` is not voaice's encoding.** It keeps nine real decimals, `floor(v)·10¹⁸ + round(frac·10⁹)·10⁹`,
//!   so the last nine digits are always zero. vclone's vprint uses `floor(v·1e18)`. Sharing a helper would break one
//!   of the two formats.
//! - **`Math.hypot` is not libm's `hypot`.** V8 scales every argument by the largest and sums the squares with Kahan
//!   compensation (`builtins/math.tq`), then multiplies the square root back. Every landmark distance goes through it,
//!   so [`js_hypot`] reproduces that algorithm step for step.
//!
//! **The triangulation.** MediaPipe's tessellation lists the canonical face mesh's triangles as consecutive edge
//! triples. [`triangles`] reads them and checks them: every triple closes, no edge belongs to more than two
//! triangles, and V − E + F = −2 (a disk with three holes: the outline, the two eyes, the mouth). Measured: 852
//! triangles, 468 vertices, 1,322 edges, 88 on the boundary.
//!
//! Not ported yet, on purpose: pose normalisation, frame averaging and frontality. frontality calls `Math.atan2`,
//! which V8 takes from fdlibm and Rust from the system libm, and those can differ in the last bit; it needs its own
//! oracle before it is claimed (TODO in docs/FCLONE.md).

use voaice::json::{self, Value};
use voaice::sha256;

/// The order is the format (the on-chain `uint256[]` layout): never reorder, only append.
pub const MEASURES: [&str; 12] = [
    "faceAspect",
    "jawRatio",
    "cheekRatio",
    "interocularRatio",
    "eyeWidthRatio",
    "noseLengthRatio",
    "noseWidthRatio",
    "mouthWidthRatio",
    "lipHeightRatio",
    "philtrumRatio",
    "browEyeRatio",
    "chinRatio",
];

/// MediaPipe canonical-face-model landmark indices, as geometry.js names them.
pub mod lm {
    pub const FACE_TOP: usize = 10;
    pub const CHIN: usize = 152;
    pub const CHEEK_R: usize = 234;
    pub const CHEEK_L: usize = 454;
    pub const ZYGO_R: usize = 116;
    pub const ZYGO_L: usize = 345;
    pub const JAW_R: usize = 172;
    pub const JAW_L: usize = 397;
    pub const EYE_IN_R: usize = 133;
    pub const EYE_IN_L: usize = 362;
    pub const EYE_OUT_R: usize = 33;
    pub const EYE_OUT_L: usize = 263;
    pub const EYE_TOP_R: usize = 159;
    pub const BROW_R: usize = 105;
    pub const NASION: usize = 168;
    pub const SUBNASALE: usize = 2;
    pub const ALA_R: usize = 98;
    pub const ALA_L: usize = 327;
    pub const MOUTH_R: usize = 61;
    pub const MOUTH_L: usize = 291;
    pub const LIP_TOP: usize = 13;
    pub const LIP_BOT: usize = 14;
    pub const UPPER_LIP: usize = 0;
    /// every index the measures read
    pub const USED: [usize; 23] = [
        FACE_TOP, CHIN, CHEEK_R, CHEEK_L, ZYGO_R, ZYGO_L, JAW_R, JAW_L, EYE_IN_R, EYE_IN_L, EYE_OUT_R, EYE_OUT_L,
        EYE_TOP_R, BROW_R, NASION, SUBNASALE, ALA_R, ALA_L, MOUTH_R, MOUTH_L, LIP_TOP, LIP_BOT, UPPER_LIP,
    ];
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// V8's `Math.hypot` (builtins/math.tq): any ±Infinity → +Infinity; else any NaN → NaN; the largest |argument| is
/// the scale; zero if it is zero; the squares of the scaled arguments are summed with Kahan compensation; the result
/// is `sqrt(sum) * max`.
pub fn js_hypot(args: &[f64]) -> f64 {
    let mut max = 0.0f64;
    let mut nan = false;
    let mut abs = [0.0f64; 3];
    for (i, &a) in args.iter().enumerate() {
        let v = a.abs();
        if v.is_infinite() {
            return f64::INFINITY;
        }
        if v.is_nan() {
            nan = true;
        } else if v > max {
            max = v;
        }
        abs[i] = v;
    }
    if nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for &v in &abs[..args.len()] {
        let n = v / max;
        let summand = n * n - comp;
        let prelim = sum + summand;
        comp = (prelim - sum) - summand;
        sum = prelim;
    }
    sum.sqrt() * max
}

fn d(p: &Point, q: &Point) -> f64 {
    js_hypot(&[p.x - q.x, p.y - q.y, p.z - q.z])
}

/// `x || 1e-6`: zero and NaN are falsy in JavaScript, so both become 1e-6.
fn or_tiny(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() { 1e-6 } else { x }
}

/// geometry.js `proportions()`: the twelve ratios, in MEASURES order. `lms` must hold at least 468 landmarks.
pub fn proportions(lms: &[Point]) -> [f64; 12] {
    use lm::*;
    let a = |i: usize| &lms[i];
    let face_w = or_tiny(d(a(CHEEK_R), a(CHEEK_L)));
    let face_h = or_tiny(d(a(FACE_TOP), a(CHIN)));
    let eye_r = d(a(EYE_OUT_R), a(EYE_IN_R));
    let eye_l = d(a(EYE_OUT_L), a(EYE_IN_L));
    [
        face_h / face_w,
        d(a(JAW_R), a(JAW_L)) / face_w,
        d(a(ZYGO_R), a(ZYGO_L)) / face_w,
        d(a(EYE_IN_R), a(EYE_IN_L)) / face_w,
        ((eye_r + eye_l) / 2.0) / face_w,
        d(a(NASION), a(SUBNASALE)) / face_h,
        d(a(ALA_R), a(ALA_L)) / face_w,
        d(a(MOUTH_R), a(MOUTH_L)) / face_w,
        d(a(LIP_TOP), a(LIP_BOT)) / face_h,
        d(a(SUBNASALE), a(UPPER_LIP)) / face_h,
        d(a(BROW_R), a(EYE_TOP_R)) / face_h,
        d(a(LIP_BOT), a(CHIN)) / face_h,
    ]
}

/// geometry.js `symmetry()`: left-right symmetry in [0, 1] from five mirrored pairs.
pub fn symmetry(lms: &[Point]) -> f64 {
    use lm::*;
    let cx = (lms[CHEEK_R].x + lms[CHEEK_L].x) / 2.0;
    let pairs = [(EYE_IN_R, EYE_IN_L), (EYE_OUT_R, EYE_OUT_L), (ALA_R, ALA_L), (MOUTH_R, MOUTH_L), (JAW_R, JAW_L)];
    let w = or_tiny(d(&lms[CHEEK_R], &lms[CHEEK_L]));
    let mut err = 0.0;
    for (r, l) in pairs {
        let (pr, pl) = (&lms[r], &lms[l]);
        err += ((cx - pr.x) - (pl.x - cx)).abs() / w + (pr.y - pl.y).abs() / w;
    }
    (1.0 - err / (pairs.len() as f64 * 2.0)).max(0.0)
}

/// faceprint.js `toFixed18`: `floor(v)·10¹⁸ + round(frac·10⁹)·10⁹`; non-finite or negative → 0. `Math.round` rounds
/// half up and `f64::round` half away from zero: the same for the non-negative values that reach it.
pub fn to_fixed18(v: f64) -> Result<u128, String> {
    let v = if !v.is_finite() || v < 0.0 { 0.0 } else { v };
    let whole = v.floor();
    if whole >= 3.4e20 {
        return Err(format!("measure {} is too large for toFixed18 here", v));
    }
    let frac = v - whole;
    Ok(whole as u128 * 1_000_000_000_000_000_000 + (frac * 1e9).round() as u128 * 1_000_000_000)
}

/// `v < 0 ? 0 : v > 1 ? 1 : v`; `f64::clamp` agrees on every input, NaN (kept) and -0.0 (kept) included
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Quality {
    pub confidence: f64,
    pub symmetry: f64,
    pub frontality: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Faceprint {
    pub measures: Vec<String>,
    pub precision_score: String,
    pub canonical: String,
    /// "0x" + sha256 of the canonical payload
    pub hash: String,
}

/// faceprint.js `measureFaceprint(proportions, quality)`.
pub fn faceprint(props: &[f64; 12], q: Quality) -> Result<Faceprint, String> {
    let measures = props.iter().map(|&v| to_fixed18(v).map(|b| b.to_string())).collect::<Result<Vec<_>, _>>()?;
    let precision = clamp01(0.5 * clamp01(q.confidence) + 0.3 * clamp01(q.symmetry) + 0.2 * clamp01(q.frontality));
    let precision_score = to_fixed18(precision)?.to_string();
    let canonical = Value::obj(vec![
        ("v", Value::int(1)),
        ("kind", Value::str("faceprint")),
        ("measureNames", Value::Arr(MEASURES.iter().map(|m| Value::str(*m)).collect())),
        ("measures", Value::Arr(measures.iter().map(Value::str).collect())),
        ("precisionScore", Value::str(&precision_score)),
    ])
    .to_compact();
    let hash = format!("0x{}", sha256::hex(&sha256::digest(canonical.as_bytes())));
    Ok(Faceprint { measures, precision_score, canonical, hash })
}

/// What checking a `.faice` file found.
#[derive(Debug)]
pub enum Check {
    /// `measured` and `fprint` both null: the honest state of an unmeasured face
    Unmeasured,
    Verified(Faceprint),
    Mismatch { recomputed: Faceprint, fields: Vec<String> },
}

/// Recompute a `faice/1` file's print from `measured.proportions` and `measured.quality`, and compare the stored
/// hash, measures, precision score and canonical string.
pub fn check_faice(text: &str) -> Result<Check, String> {
    let v = json::parse(text)?;
    if v.get("format").and_then(Value::as_str) != Some("faice/1") {
        return Err("not a faice/1 identity".into());
    }
    let measured = v.get("measured").ok_or("no `measured` field")?;
    let stored = v.get("fprint").ok_or("no `fprint` field")?;
    if measured.is_null() && stored.is_null() {
        return Ok(Check::Unmeasured);
    }
    if measured.is_null() || stored.is_null() {
        return Err("one of `measured` and `fprint` is null and the other is not".into());
    }
    let mut props = [0f64; 12];
    for (i, k) in MEASURES.iter().enumerate() {
        props[i] = measured.path(&["proportions", k]).and_then(Value::as_f64)
            .ok_or_else(|| format!("measured.proportions.{} is missing", k))?;
    }
    let qf = |k: &str| measured.path(&["quality", k]).and_then(Value::as_f64).unwrap_or(1.0);
    let r = faceprint(&props, Quality { confidence: qf("confidence"), symmetry: qf("symmetry"), frontality: qf("frontality") })?;
    let mut fields = Vec::new();
    if stored.get("hash").and_then(Value::as_str) != Some(r.hash.as_str()) {
        fields.push("hash".to_string());
    }
    if stored.get("precisionScore").and_then(Value::as_str) != Some(r.precision_score.as_str()) {
        fields.push("precisionScore".into());
    }
    if let Some(Value::Arr(ms)) = stored.get("measures") {
        for (i, m) in r.measures.iter().enumerate() {
            if ms.get(i).and_then(Value::as_str) != Some(m.as_str()) {
                fields.push(format!("measures[{}] ({})", i, MEASURES[i]));
            }
        }
    } else {
        fields.push("measures".into());
    }
    if let Some(c) = stored.get("canonical").and_then(Value::as_str) {
        if c != r.canonical {
            fields.push("canonical".into());
        }
    }
    Ok(if fields.is_empty() { Check::Verified(r) } else { Check::Mismatch { recomputed: r, fields } })
}

#[derive(Debug, PartialEq, Eq)]
pub struct MeshCheck {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub euler: i64,
    pub boundary: usize,
    pub unclosed: usize,
    pub non_manifold: usize,
}
impl MeshCheck {
    /// a manifold disk with three holes, every triple closed
    pub fn ok(&self) -> bool {
        self.unclosed == 0 && self.non_manifold == 0 && self.euler == -2
    }
}

/// Read the triangles from a tessellation given as (start, end) pairs, three per triangle, and check them.
pub fn triangles(tess: &[(u32, u32)]) -> (Vec<[u32; 3]>, MeshCheck) {
    use std::collections::{BTreeMap, BTreeSet};
    let (mut tris, mut unclosed) = (Vec::new(), 0);
    for t in tess.as_chunks::<3>().0 {
        let (a, b, c) = (t[0], t[1], t[2]);
        if a.1 == b.0 && b.1 == c.0 && c.1 == a.0 {
            tris.push([a.0, b.0, c.0]);
        } else {
            unclosed += 1;
        }
    }
    let mut uses: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut verts = BTreeSet::new();
    for t in &tris {
        for k in 0..3 {
            let (x, y) = (t[k], t[(k + 1) % 3]);
            verts.insert(x);
            *uses.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let check = MeshCheck {
        vertices: verts.len(),
        edges: uses.len(),
        faces: tris.len(),
        euler: verts.len() as i64 - uses.len() as i64 + tris.len() as i64,
        boundary: uses.values().filter(|&&n| n == 1).count(),
        unclosed,
        non_manifold: uses.values().filter(|&&n| n > 2).count(),
    };
    (tris, check)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fixed18_keeps_nine_decimals() {
        assert_eq!(to_fixed18(1.4107142857142856).unwrap(), 1_410_714_286_000_000_000);
        assert_eq!(to_fixed18(0.9999999999).unwrap(), 1_000_000_000_000_000_000); // the fraction rounds up to 1
        assert_eq!(to_fixed18(-3.0).unwrap(), 0);
        assert_eq!(to_fixed18(f64::NAN).unwrap(), 0);
    }

    #[test]
    fn js_hypot_edges() {
        assert_eq!(js_hypot(&[3.0, 4.0, 0.0]), 5.0);
        assert_eq!(js_hypot(&[0.0, -0.0, 0.0]), 0.0);
        assert!(js_hypot(&[f64::NAN, 1.0, 1.0]).is_nan());
        assert_eq!(js_hypot(&[f64::NAN, f64::NEG_INFINITY, 1.0]), f64::INFINITY);
    }

    #[test]
    fn a_broken_triple_is_caught() {
        let ok = [(0, 1), (1, 2), (2, 0)];
        assert_eq!(triangles(&ok).1.unclosed, 0);
        let bad = [(0, 1), (1, 2), (2, 3)];
        assert_eq!(triangles(&bad).1.unclosed, 1);
    }
}
