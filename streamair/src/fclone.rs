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
//! **The capture pipeline.** pose normalisation, frame averaging, frontality and the whole `fclone(frames)` path are
//! ported too. frontality calls `Math.atan2`, which V8 takes from fdlibm and Rust's `f64::atan2` from the system
//! libm, so [`js_atan2`] is fdlibm's `atan2`/`atan`, step for step. `Math.max` keeps NaN where `f64::max` drops it,
//! so [`js_max`] does too. The oracle regenerates the same inputs from an exact LCG and compares digests of raw f64
//! bits (testing/fclone/make_pose_oracle.mjs).

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

// ── V8's Math.atan2: fdlibm (Sun, freely redistributable), as ported in V8's src/base/ieee754.cc ──────────────

// the constants are fdlibm's, digit for digit (the doubles they parse to are what matter, and they are the same);
// written verbatim so they can be checked against the source
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const ATANHI: [f64; 4] = [4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00];
// the constants are fdlibm's, digit for digit (the doubles they parse to are what matter, and they are the same);
// written verbatim so they can be checked against the source
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const ATANLO: [f64; 4] = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17];
// the constants are fdlibm's, digit for digit (the doubles they parse to are what matter, and they are the same);
// written verbatim so they can be checked against the source
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const AT: [f64; 11] = [
    3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01, -1.11111104054623557880e-01,
    9.09088713343650656196e-02, -7.69187620504482999495e-02, 6.66107313738753120669e-02, -5.83357013379057348645e-02,
    4.97687799461593236017e-02, -3.65315727442169155270e-02, 1.62858201153657823623e-02,
];

fn words(x: f64) -> (i32, u32) {
    let b = x.to_bits();
    ((b >> 32) as u32 as i32, b as u32)
}

/// fdlibm `atan` (s_atan.c), step for step.
pub fn fdlibm_atan(x0: f64) -> f64 {
    let mut x = x0;
    let (hx, lo) = words(x);
    let ix = hx & 0x7fff_ffff;
    let id: i32;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        if ix > 0x7ff0_0000 || (ix == 0x7ff0_0000 && lo != 0) {
            return x + x; // NaN
        }
        return if hx > 0 { ATANHI[3] + ATANLO[3] } else { -ATANHI[3] - ATANLO[3] };
    }
    if ix < 0x3fdc_0000 {
        // |x| < 0.4375
        if ix < 0x3e40_0000 && 1.0e300 + x > 1.0 {
            return x; // |x| < 2^-27
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3ff3_0000 {
            if ix < 0x3fe6_0000 {
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x4003_8000 {
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            id = 3;
            x = -1.0 / x;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return x - x * (s1 + s2);
    }
    let i = id as usize;
    let z = ATANHI[i] - ((x * (s1 + s2) - ATANLO[i]) - x);
    if hx < 0 { -z } else { z }
}

// the constants are fdlibm's, digit for digit (the doubles they parse to are what matter, and they are the same);
// written verbatim so they can be checked against the source
#[allow(clippy::excessive_precision, clippy::approx_constant)]
/// fdlibm `atan2` (e_atan2.c) as V8 runs it for `Math.atan2(y, x)`.
pub fn js_atan2(y: f64, x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    const PI_O_4: f64 = 7.8539816339744827900E-01;
    const PI_O_2: f64 = 1.5707963267948965580E+00;
    const PI: f64 = 3.1415926535897931160E+00;
    const PI_LO: f64 = 1.2246467991473531772E-16;
    let (hx, lx) = words(x);
    let (hy, ly) = words(y);
    let ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;
    let nz = |l: u32| (l | l.wrapping_neg()) >> 31;
    if (ix as u32 | nz(lx)) > 0x7ff0_0000 || (iy as u32 | nz(ly)) > 0x7ff0_0000 {
        return x + y; // NaN
    }
    if (hx.wrapping_sub(0x3ff0_0000) as u32 | lx) == 0 {
        return fdlibm_atan(y); // x = 1.0
    }
    let mut m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy as u32 | ly) == 0 {
        return match m {
            0 | 1 => y,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if (ix as u32 | lx) == 0 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    if ix == 0x7ff0_0000 {
        return if iy == 0x7ff0_0000 {
            match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => 3.0 * PI_O_4 + TINY,
                _ => -3.0 * PI_O_4 - TINY,
            }
        } else {
            match m {
                0 => 0.0,
                1 => -0.0,
                2 => PI + TINY,
                _ => -PI - TINY,
            }
        };
    }
    if iy == 0x7ff0_0000 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        m &= 1;
        PI_O_2 + 0.5 * PI_LO
    } else if hx < 0 && k < -60 {
        0.0
    } else {
        fdlibm_atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

/// `Math.max(a, b)` for two numbers: NaN if either is NaN (f64::max would return the other).
fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a > b { a } else if b > a { b } else if a == 0.0 && b == 0.0 && (a.is_sign_negative() && b.is_sign_negative()) { -0.0 } else if a == 0.0 && b == 0.0 { 0.0 } else { a }
}

/// geometry.js `frontality(matrix)`: 1 for no matrix (or not 16 long), else
/// `max(0, 1 − (|yaw| + |pitch|) / π)` with yaw = atan2(m8, m10), pitch = atan2(−m9, hypot(m8, m10)).
pub fn frontality(m: Option<&[f64]>) -> f64 {
    let Some(m) = m.filter(|m| m.len() == 16) else { return 1.0 };
    let yaw = js_atan2(m[8], m[10]);
    let pitch = js_atan2(-m[9], js_hypot(&[m[8], m[10]]));
    js_max(0.0, 1.0 - (yaw.abs() + pitch.abs()) / std::f64::consts::PI)
}

fn mean(pts: &[Point], k: usize) -> f64 {
    let mut s = 0.0;
    for p in pts {
        s += [p.x, p.y, p.z][k];
    }
    s / pts.len() as f64
}

/// geometry.js `poseNormalize(lms, matrix)`: the inverse rotation (the 3×3 block transposed, column-major) when a
/// 16-entry matrix is given, then centred on the mean and scaled by the inner-eye distance (`|| 1e-6`).
pub fn pose_normalize(lms: &[Point], m: Option<&[f64]>) -> Vec<Point> {
    let mut pts: Vec<Point> = lms.to_vec();
    if let Some(m) = m.filter(|m| m.len() == 16) {
        for p in &mut pts {
            let (x, y, z) = (p.x, p.y, p.z);
            *p = Point {
                x: m[0] * x + m[1] * y + m[2] * z,
                y: m[4] * x + m[5] * y + m[6] * z,
                z: m[8] * x + m[9] * y + m[10] * z,
            };
        }
    }
    let (cx, cy, cz) = (mean(&pts, 0), mean(&pts, 1), mean(&pts, 2));
    let scale = or_tiny(d(&pts[lm::EYE_IN_R], &pts[lm::EYE_IN_L]));
    pts.iter().map(|p| Point { x: (p.x - cx) / scale, y: (p.y - cy) / scale, z: (p.z - cz) / scale }).collect()
}

/// One captured frame: landmarks, and the facial transformation matrix when the landmarker gave one.
#[derive(Clone, Debug)]
pub struct Frame {
    pub landmarks: Vec<Point>,
    pub matrix: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct Aggregate {
    pub landmarks: Vec<Point>,
    pub frontal_index: usize,
    pub frames: usize,
    pub mean_frontality: f64,
}

/// geometry.js `aggregate(frames)`: one frame is returned as it is; several are pose-normalised, weighted by
/// `max(0.05, frontality)` and averaged per vertex. (Frames with no landmarks are dropped first, as there.)
pub fn aggregate(frames: &[Frame]) -> Result<Aggregate, String> {
    let valid: Vec<&Frame> = frames.iter().filter(|f| !f.landmarks.is_empty()).collect();
    if valid.is_empty() {
        return Err("face_clone: no valid frames to aggregate".into());
    }
    if valid.len() == 1 {
        return Ok(Aggregate { landmarks: valid[0].landmarks.clone(), frontal_index: 0, frames: 1,
                              mean_frontality: frontality(valid[0].matrix.as_deref()) });
    }
    let normed: Vec<(Vec<Point>, f64)> = valid
        .iter()
        .map(|f| (pose_normalize(&f.landmarks, f.matrix.as_deref()), js_max(0.05, frontality(f.matrix.as_deref()))))
        .collect();
    let (mut frontal_index, mut best, mut fsum) = (0, -1.0f64, 0.0f64);
    for (i, (_, w)) in normed.iter().enumerate() {
        if *w > best {
            best = *w;
            frontal_index = i;
        }
        fsum += *w;
    }
    let mut wsum = 0.0;
    for (_, w) in &normed {
        wsum += *w;
    }
    let n = normed[0].0.len();
    let mut out = Vec::with_capacity(n);
    for v in 0..n {
        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
        for (pts, w) in &normed {
            let p = pts[v];
            x += p.x * w;
            y += p.y * w;
            z += p.z * w;
        }
        out.push(Point { x: x / wsum, y: y / wsum, z: z / wsum });
    }
    Ok(Aggregate { landmarks: out, frontal_index, frames: valid.len(), mean_frontality: fsum / normed.len() as f64 })
}

/// ollywoo fclone.js `fclone(frames)` up to the print: aggregate, the twelve proportions, symmetry, quality
/// (`confidence = min(1, frames / 20)`, the frontality of the chosen frame), the faceprint. Refuses fewer than
/// `min_frames` frames, as fclone.js does (it defaults to 5).
pub fn fclone_frames(frames: &[Frame], min_frames: usize) -> Result<(Faceprint, Aggregate, Quality), String> {
    if frames.len() < min_frames {
        return Err(format!("only {} frames had a face; need {}", frames.len(), min_frames));
    }
    let agg = aggregate(frames)?;
    let props = proportions(&agg.landmarks);
    let sym = symmetry(&agg.landmarks);
    let best = frames.get(agg.frontal_index).unwrap_or(&frames[0]);
    let q = Quality { confidence: (frames.len() as f64 / 20.0).min(1.0), symmetry: sym,
                      frontality: frontality(best.matrix.as_deref()) };
    Ok((faceprint(&props, q)?, agg, q))
}

// ── the persona print: face and voice bound into one (faicey persona.js) ─────────────────────────────────────────

/// A print as persona.js accepts it: a hash, its measures as decimal strings, and a precision score (persona.js reads
/// `precisionScore ?? precision ?? "0"`, so a print without one, like a dvscope/1 vprint, counts as precision 0).
#[derive(Clone, Debug)]
pub struct PrintRef {
    pub hash: String,
    pub measures: Vec<String>,
    pub precision_score: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonaPrint {
    pub hash: String,
    pub modalities: Vec<&'static str>,
    pub measures: Vec<String>,
    pub precision_score: String,
    pub canonical: String,
}

/// persona.js `personaPrint({face, voice})`: the payload `{"v":1,"kind":"persona","modalities":[…],"faceHash":…,
/// "voiceHash":…,"measures":[face…, voice…]}`, hashed `"0x" + sha256`; the precision is the product of the present
/// modalities' precisions (each `Number(score) / 1e18`), clamped, kept to nine decimals. Neither modality is an error,
/// so a persona print cannot exist without something measured.
pub fn persona_print(face: Option<&PrintRef>, voice: Option<&PrintRef>) -> Result<PersonaPrint, String> {
    if face.is_none() && voice.is_none() {
        return Err("persona: need a faceprint and/or a voiceprint".into());
    }
    let modalities: Vec<&'static str> = [face.map(|_| "face"), voice.map(|_| "voice")].into_iter().flatten().collect();
    let measures: Vec<String> = face.iter().chain(voice.iter()).flat_map(|p| p.measures.iter().cloned()).collect();
    let canonical = Value::obj(vec![
        ("v", Value::int(1)),
        ("kind", Value::str("persona")),
        ("modalities", Value::Arr(modalities.iter().map(|m| Value::str(*m)).collect())),
        ("faceHash", Value::str(face.map(|p| p.hash.as_str()).unwrap_or(""))),
        ("voiceHash", Value::str(voice.map(|p| p.hash.as_str()).unwrap_or(""))),
        ("measures", Value::Arr(measures.iter().map(Value::str).collect())),
    ])
    .to_compact();
    let prec = |p: Option<&PrintRef>| -> f64 {
        match p {
            None => 1.0,
            Some(p) => p.precision_score.as_deref().unwrap_or("0").trim().parse::<f64>().unwrap_or(f64::NAN) / 1e18,
        }
    };
    let precision = clamp01(prec(face) * prec(voice));
    // BigInt(Math.round(precision * 1e9)) * 10n ** 9n; NaN would throw in JavaScript, so it is refused here
    if precision.is_nan() {
        return Err("persona: a precision score is not a number".into());
    }
    let precision_score = ((precision * 1e9).round() as u128 * 1_000_000_000).to_string();
    let hash = format!("0x{}", sha256::hex(&sha256::digest(canonical.as_bytes())));
    Ok(PersonaPrint { hash, modalities, measures, precision_score, canonical })
}

// ── the face mint rule ────────────────────────────────────────────────────────────────────────────────────────

/// Whether the face a forge log describes may be minted, and every reason when it may not. The voice rule
/// (`vclone::mintable`) asks for a measured voice, consent for the ref that was cloned and a cleared engine; a face
/// asks for the analogue, against the faceprint the token would commit to:
/// - a `measure` event with `modality: "face"` whose `hash` is that faceprint;
/// - its capture recorded `image_kept: false` (fCLONE keeps landmarks, never a photograph);
/// - the person's latest face consent (`consent` with `modality: "face"`) has scope `mint`, names that faceprint, and
///   is not a revocation.
pub fn mintable_face(log: &crate::vclone::Log, fprint: &str) -> Result<(), Vec<String>> {
    use crate::vclone::Kind;
    let modality = |e: &&crate::vclone::Event| e.body.get("modality").and_then(Value::as_str) == Some("face");
    let mut why = Vec::new();
    let measured = log.events.iter().filter(modality).any(|e| {
        e.kind == Kind::Measure && e.body.get("hash").and_then(Value::as_str) == Some(fprint)
    });
    if !measured {
        why.push("no face measurement with this faceprint".to_string());
    }
    let capture = log.events.iter().filter(modality).rev().find(|e| e.kind == Kind::Capture);
    match capture.and_then(|c| c.body.get("image_kept")) {
        Some(Value::Bool(false)) => {}
        Some(_) => why.push("the face capture kept an image".into()),
        None => why.push("the face capture does not record that no image was kept".into()),
    }
    match log.events.iter().filter(modality).rev().find(|e| e.kind == Kind::Consent) {
        None => why.push("no face consent from the person".into()),
        Some(c) => {
            if c.body.get("revoked") == Some(&Value::Bool(true)) {
                why.push("the face consent was revoked".into());
            } else if c.body.get("scope").and_then(Value::as_str) != Some("mint") {
                why.push("the latest face consent is not for `mint`".into());
            }
            if c.body.get("fprint").and_then(Value::as_str) != Some(fprint) {
                why.push("the face consent names a different faceprint".into());
            }
        }
    }
    if why.is_empty() { Ok(()) } else { Err(why) }
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
    fn the_face_mint_rule() {
        use crate::vclone::{Kind, Log};
        let fp = format!("0x{}", "ab".repeat(32));
        let face = |extra: Vec<(&str, Value)>| {
            let mut kv = vec![("modality", Value::str("face"))];
            kv.extend(extra);
            Value::obj(kv)
        };
        let mut log = Log::default();
        assert_eq!(mintable_face(&log, &fp).unwrap_err().len(), 3);
        log.append(Kind::Capture, "t", "p", face(vec![("image_kept", Value::Bool(false))]));
        log.append(Kind::Measure, "t", "p", face(vec![("hash", Value::str(&fp))]));
        assert_eq!(mintable_face(&log, &fp).unwrap_err(), vec!["no face consent from the person"]);
        // a voice consent does not cover the face
        log.append(Kind::Consent, "t", "p", Value::obj(vec![("scope", Value::str("mint"))]));
        assert!(mintable_face(&log, &fp).is_err());
        log.append(Kind::Consent, "t", "p", face(vec![("scope", Value::str("mint")), ("fprint", Value::str(&fp))]));
        assert!(mintable_face(&log, &fp).is_ok());
        assert!(mintable_face(&log, "0x00").is_err(), "a different faceprint");
        log.append(Kind::Consent, "t", "p", face(vec![("scope", Value::str("mint")), ("fprint", Value::str(&fp)),
                                                      ("revoked", Value::Bool(true))]));
        assert_eq!(mintable_face(&log, &fp).unwrap_err(), vec!["the face consent was revoked"]);
    }

    #[test]
    fn a_broken_triple_is_caught() {
        let ok = [(0, 1), (1, 2), (2, 0)];
        assert_eq!(triangles(&ok).1.unclosed, 0);
        let bad = [(0, 1), (1, 2), (2, 3)];
        assert_eq!(triangles(&bad).1.unclosed, 1);
    }
}
