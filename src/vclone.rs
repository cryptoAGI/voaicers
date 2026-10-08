// SPDX-License-Identifier: MIT OR Apache-2.0
//! vclone — voaice's voice identities in Rust: the vprint, the `.voaice` file, and the forge's event log.
//!
//! **The vprint** (`dvscope/1`) is a fingerprint of a *measurement*, not of a speaker: eight acoustic metrics, each
//! `floor(v * 1e18)` with 1e18 as an f64 (so the multiply's rounding is part of the value), written as
//! `{"v":"dvscope/1","precision":18,"metrics":{…}}` in a fixed metric order with no whitespace, then hashed with
//! SHA-256 and SHA-512, the 256-bit digest also read as a uint256. Its definition is the browser's `DVScope`
//! (DeltaVerse `engine/oscilloscope.js`) and its server twin is cryptoAGI/voaice `tools/vprint.py`; this is the third
//! implementation, and it is held to the same rule: byte-identical canonical string, identical digests. The oracle
//! is vprint.py itself (`testing/vclone/make_oracle.py` records its outputs) and every shipped `.voaice` file, whose
//! stored print vprint.py wrote.
//!
//! **The forge log** records how a voice identity came to be: capture, measure, ref, consent, then the persona's
//! other facets (model, actor, prompt, skill, tool, language), then forge. Each event is canonical JSON chained to
//! the previous one by SHA-256, so the log is tamper-evident and replayable: the `.persona` that ollywoo's
//! `forgePersona()` downloads becomes the *projection* of a log, not a file that asserts its own history.
//! [`mintable`] reads a log and says whether the voice in it may become an iNFT, and why not when it may not.

use crate::json::{self, Value};
use crate::{sha256, sha512};

/// The metric order is part of the format: a different order is a different string and a different hash.
pub const METRICS: [&str; 8] = [
    "rms",
    "dominantFrequency",
    "spectralCentroid",
    "spectralRolloff",
    "zeroCrossingRate",
    "spectralBandwidth",
    "spectralFlux",
    "harmonicNoiseRatio",
];
pub const VERSION: &str = "dvscope/1";
pub const PRECISION: u32 = 18;
const MUL: f64 = 1e18;

/// `BigInt(Math.floor(v * 1e18))` / `math.floor(float(v) * 1e18)`: the f64 multiply rounds, then the floor is exact
/// (an f64 that large is an integer). Non-finite → 0, as both twins do. Values whose scaled magnitude exceeds i128
/// (|v| > 1.7e20) are refused rather than silently clamped.
pub fn to_precision18(v: f64) -> Result<i128, String> {
    if !v.is_finite() {
        return Ok(0);
    }
    let p = (v * MUL).floor();
    if p.abs() >= 1.7e38 {
        return Err(format!("metric {} is too large for the 18-decimal encoding here", v));
    }
    Ok(p as i128)
}

/// `fromPrecision18`: sign, integer part, '.', 18 zero-padded digits.
pub fn from_precision18(b: i128) -> String {
    let neg = b < 0;
    let u = b.unsigned_abs();
    let q = u / 10u128.pow(PRECISION);
    let r = u % 10u128.pow(PRECISION);
    format!("{}{}.{:018}", if neg { "-" } else { "" }, q, r)
}

/// The decimal value of a big-endian 256-bit number (the digest read as a uint256).
pub fn uint256_decimal(be: &[u8; 32]) -> String {
    let mut n = be.to_vec();
    let mut digits = Vec::new();
    while n.iter().any(|&b| b != 0) {
        let mut rem = 0u32;
        for b in n.iter_mut() {
            let cur = (rem << 8) | *b as u32;
            *b = (cur / 10) as u8;
            rem = cur % 10;
        }
        digits.push(b'0' + rem as u8);
    }
    if digits.is_empty() {
        return "0".into();
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vprint {
    pub hash: String,
    pub hash512: String,
    pub short: String,
    pub uint256: String,
    /// (metric, wei, decimal), in METRICS order
    pub precision18: Vec<(&'static str, String, String)>,
    pub canonical: String,
}

/// The print of eight metric values given in METRICS order.
pub fn vprint(values: &[f64; 8]) -> Result<Vprint, String> {
    let mut p18 = Vec::with_capacity(8);
    let mut metrics = Vec::with_capacity(8);
    for (k, &v) in METRICS.iter().zip(values) {
        let b = to_precision18(v)?;
        let (wei, dec) = (b.to_string(), from_precision18(b));
        metrics.push((k.to_string(), Value::obj(vec![("wei", Value::str(&wei)), ("decimal", Value::str(&dec))])));
        p18.push((*k, wei, dec));
    }
    let canonical = Value::obj(vec![
        ("v", Value::str(VERSION)),
        ("precision", Value::int(PRECISION as i128)),
        ("metrics", Value::Obj(metrics)),
    ])
    .to_compact();
    let h = sha256::digest(canonical.as_bytes());
    let hash = sha256::hex(&h);
    Ok(Vprint {
        short: hash[..16].to_string(),
        uint256: uint256_decimal(&h),
        hash,
        hash512: sha256::hex(&sha512::digest(canonical.as_bytes())),
        precision18: p18,
        canonical,
    })
}

/// What checking a `.voaice` file found.
#[derive(Debug)]
pub enum Check {
    /// `measured` and `vprint` are both null: the honest state of an unmeasured identity (vclone.voaice)
    Unmeasured,
    /// the print recomputed from `measured.metrics` equals the stored one, field by field
    Verified(Vprint),
    /// they differ; the fields that differ are named
    Mismatch { recomputed: Vprint, fields: Vec<String> },
}

/// Check a `.voaice` identity (`format: "voaice/1"`): recompute its vprint from `measured.metrics` and compare
/// every stored field (hash, hash512, short, uint256, version, each metric's wei and decimal).
pub fn check_identity(text: &str) -> Result<Check, String> {
    let v = json::parse(text)?;
    if v.get("format").and_then(Value::as_str) != Some("voaice/1") {
        return Err("not a voaice/1 identity".into());
    }
    let measured = v.get("measured").ok_or("no `measured` field")?;
    let stored = v.get("vprint").ok_or("no `vprint` field")?;
    if measured.is_null() && stored.is_null() {
        return Ok(Check::Unmeasured);
    }
    if measured.is_null() || stored.is_null() {
        return Err("one of `measured` and `vprint` is null and the other is not".into());
    }
    let mut vals = [0f64; 8];
    for (i, k) in METRICS.iter().enumerate() {
        vals[i] = measured
            .path(&["metrics", k])
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("measured.metrics.{} is missing or not a number", k))?;
    }
    let r = vprint(&vals)?;
    let mut fields = Vec::new();
    for (name, mine) in [("hash", &r.hash), ("hash512", &r.hash512), ("short", &r.short), ("uint256", &r.uint256)] {
        if stored.get(name).and_then(Value::as_str) != Some(mine.as_str()) {
            fields.push(name.to_string());
        }
    }
    if stored.get("version").and_then(Value::as_str) != Some(VERSION) {
        fields.push("version".into());
    }
    for (k, wei, dec) in &r.precision18 {
        if stored.path(&["precision18", k, "wei"]).and_then(Value::as_str) != Some(wei.as_str()) {
            fields.push(format!("precision18.{}.wei", k));
        }
        if stored.path(&["precision18", k, "decimal"]).and_then(Value::as_str) != Some(dec.as_str()) {
            fields.push(format!("precision18.{}.decimal", k));
        }
    }
    if let Some(c) = stored.get("canonical").and_then(Value::as_str) {
        if c != r.canonical {
            fields.push("canonical".into());
        }
    }
    Ok(if fields.is_empty() { Check::Verified(r) } else { Check::Mismatch { recomputed: r, fields } })
}

// ── the forge log ──────────────────────────────────────────────────────────────────────────────────────────────

pub const EVENT_FORMAT: &str = "vclone-event/1";
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// What happened. The first four make the voice; the next seven are the rest of a .persona; forge seals it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// a reference recording was taken (body: seconds, sampleRate, bytes, sha256 of the WAV)
    Capture,
    /// a measurement of it (body: the vprint, with its params)
    Measure,
    /// the recording was deposited as a cloning reference (body: ref id, host)
    Ref,
    /// the speaker's consent (body: who, scope: render|clone|mint, ref, revocable, signature when there is one)
    Consent,
    /// the persona's model facet (body: base, imprint status; a browser cannot make one)
    Model,
    /// the actor: face, theme, the embodiment
    Actor,
    /// the instruction the actor speaks from
    Prompt,
    /// a skill granted
    Skill,
    /// a tool granted
    Tool,
    /// a language the voice speaks (body: tag, pronunciation table version)
    Language,
    /// the persona sealed from everything above (body: persona id, the bundle's sha256)
    Forge,
}

impl Kind {
    pub const ALL: [Kind; 11] = [
        Kind::Capture,
        Kind::Measure,
        Kind::Ref,
        Kind::Consent,
        Kind::Model,
        Kind::Actor,
        Kind::Prompt,
        Kind::Skill,
        Kind::Tool,
        Kind::Language,
        Kind::Forge,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Kind::Capture => "capture",
            Kind::Measure => "measure",
            Kind::Ref => "ref",
            Kind::Consent => "consent",
            Kind::Model => "model",
            Kind::Actor => "actor",
            Kind::Prompt => "prompt",
            Kind::Skill => "skill",
            Kind::Tool => "tool",
            Kind::Language => "language",
            Kind::Forge => "forge",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.name() == s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub seq: u64,
    pub prev: String,
    pub kind: Kind,
    /// the time as the producer stated it (ISO 8601); recorded, not trusted (kairos/chronos verify time elsewhere)
    pub at: String,
    pub persona: String,
    pub body: Value,
    /// sha256 of the canonical form
    pub id: String,
}

impl Event {
    /// `{"v":"vclone-event/1","seq":n,"prev":…,"kind":…,"at":…,"persona":…,"body":{…}}`, compact, in this order.
    pub fn canonical(&self) -> String {
        Value::obj(vec![
            ("v", Value::str(EVENT_FORMAT)),
            ("seq", Value::int(self.seq as i128)),
            ("prev", Value::str(&self.prev)),
            ("kind", Value::str(self.kind.name())),
            ("at", Value::str(&self.at)),
            ("persona", Value::str(&self.persona)),
            ("body", self.body.clone()),
        ])
        .to_compact()
    }
    /// One JSONL line: the canonical object with its id appended.
    pub fn line(&self) -> String {
        let c = self.canonical();
        format!("{},\"id\":\"{}\"}}", &c[..c.len() - 1], self.id)
    }
}

/// An append-only, hash-chained log of one persona's forge.
#[derive(Default, Debug)]
pub struct Log {
    pub events: Vec<Event>,
}

impl Log {
    pub fn append(&mut self, kind: Kind, at: &str, persona: &str, body: Value) -> &Event {
        let (seq, prev) = match self.events.last() {
            Some(e) => (e.seq + 1, e.id.clone()),
            None => (0, GENESIS.to_string()),
        };
        let mut e = Event { seq, prev, kind, at: at.into(), persona: persona.into(), body, id: String::new() };
        e.id = sha256::hex(&sha256::digest(e.canonical().as_bytes()));
        self.events.push(e);
        self.events.last().unwrap()
    }

    pub fn to_jsonl(&self) -> String {
        self.events.iter().map(|e| e.line() + "\n").collect()
    }

    /// Read a JSONL log and verify it: format, sequence, the chain, and each id against its canonical form.
    pub fn from_jsonl(text: &str) -> Result<Log, String> {
        let mut log = Log::default();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v = json::parse(line).map_err(|e| format!("line {}: {}", n + 1, e))?;
            let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string).ok_or(format!("line {}: no `{}`", n + 1, k));
            if s("v")? != EVENT_FORMAT {
                return Err(format!("line {}: not a {}", n + 1, EVENT_FORMAT));
            }
            let kind = Kind::parse(&s("kind")?).ok_or(format!("line {}: unknown kind", n + 1))?;
            let seq = v.get("seq").and_then(Value::as_f64).ok_or(format!("line {}: no seq", n + 1))? as u64;
            let e = Event {
                seq,
                prev: s("prev")?,
                kind,
                at: s("at")?,
                persona: s("persona")?,
                body: v.get("body").cloned().unwrap_or(Value::Null),
                id: s("id")?,
            };
            let want_prev = log.events.last().map(|p| p.id.clone()).unwrap_or_else(|| GENESIS.to_string());
            if e.seq != log.events.len() as u64 {
                return Err(format!("line {}: seq {} out of order", n + 1, e.seq));
            }
            if e.prev != want_prev {
                return Err(format!("line {}: the chain is broken (prev does not match)", n + 1));
            }
            if sha256::hex(&sha256::digest(e.canonical().as_bytes())) != e.id {
                return Err(format!("line {}: the id is not the hash of the event (edited?)", n + 1));
            }
            log.events.push(e);
        }
        Ok(log)
    }

    pub fn last_of(&self, kind: Kind) -> Option<&Event> {
        self.events.iter().rev().find(|e| e.kind == kind)
    }
}

/// Whether the voice a forge log describes may be minted, and the reasons when it may not. The rules are the voice
/// cards' (PYTHAI/voaice cards/), applied to a voice that belongs to a person:
/// - its voice was measured (a `measure` event with no modality, or `modality: "voice"`; a face print does not count);
/// - the person consented, with scope `mint`, to this reference (when the voice was cloned from one);
/// - the engine that speaks it is cleared for that use (`licence_cleared: true` on the ref event; until the cloning
///   model's licence is recorded, a cloned voice is not mintable);
/// - the consent was not revoked later in the log.
pub fn mintable(log: &Log) -> Result<(), Vec<String>> {
    let mut why = Vec::new();
    // the voice rule: a face measurement (fCLONE, `modality: "face"`) does not stand in for a voiceprint
    let voice_measured = log.events.iter().any(|e| {
        e.kind == Kind::Measure && matches!(e.body.get("modality").and_then(Value::as_str), None | Some("voice"))
    });
    if !voice_measured {
        why.push("never measured: there is no vprint to commit to".to_string());
    }
    let r = log.last_of(Kind::Ref);
    let consent = log.events.iter().rev().find(|e| {
        e.kind == Kind::Consent && e.body.get("scope").and_then(Value::as_str) == Some("mint")
    });
    match consent {
        None => why.push("no consent with scope `mint` from the speaker".into()),
        Some(c) => {
            if c.body.get("revoked").map(|v| *v == Value::Bool(true)).unwrap_or(false) {
                why.push("the speaker's consent was revoked".into());
            }
            if let (Some(r), Some(cr)) = (r, c.body.get("ref").and_then(Value::as_str)) {
                if r.body.get("ref").and_then(Value::as_str) != Some(cr) {
                    why.push("the consent names a different reference than the one cloned".into());
                }
            } else if r.is_some() {
                why.push("the consent does not name the reference it covers".into());
            }
        }
    }
    if let Some(r) = r {
        if r.body.get("licence_cleared") != Some(&Value::Bool(true)) {
            why.push("the cloning engine's licence is not recorded as cleared for this use".into());
        }
    }
    if why.is_empty() { Ok(()) } else { Err(why) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision18_matches_the_float_multiply_not_the_exact_one() {
        // vprint.py's math.floor(v * 1e18): 0.7 gives 700000000000000000, where the exact product 0.7·10^18 of the
        // stored double would floor to 699999999999999955; 1.1 gives …128 against the exact …088
        assert_eq!(to_precision18(0.7).unwrap(), 700000000000000000);
        assert_eq!(to_precision18(1.1).unwrap(), 1100000000000000128);
        assert_eq!(to_precision18(96.8048780487805).unwrap(), 96804878048780500992);
        assert_eq!(from_precision18(1100000000000000128), "1.100000000000000128");
        assert_eq!(from_precision18(-5), "-0.000000000000000005");
        assert_eq!(to_precision18(f64::NAN).unwrap(), 0);
    }

    #[test]
    fn uint256_of_small_and_max() {
        let mut b = [0u8; 32];
        assert_eq!(uint256_decimal(&b), "0");
        b[31] = 255;
        assert_eq!(uint256_decimal(&b), "255");
        let m = [255u8; 32];
        assert_eq!(uint256_decimal(&m),
                   "115792089237316195423570985008687907853269984665640564039457584007913129639935");
    }

    #[test]
    fn the_log_chains_and_refuses_an_edit() {
        let mut log = Log::default();
        log.append(Kind::Capture, "2026-10-08T00:00:00Z", "p1", Value::obj(vec![("seconds", Value::int(12))]));
        log.append(Kind::Measure, "2026-10-08T00:00:01Z", "p1", Value::obj(vec![("hash", Value::str("ab"))]));
        log.append(Kind::Language, "2026-10-08T00:00:02Z", "p1", Value::obj(vec![("tag", Value::str("en-GB"))]));
        let text = log.to_jsonl();
        let back = Log::from_jsonl(&text).unwrap();
        assert_eq!(back.events, log.events);
        let edited = text.replacen("en-GB", "en-US", 1);
        assert!(Log::from_jsonl(&edited).unwrap_err().contains("not the hash"));
        let dropped: String = text.lines().enumerate().filter(|(i, _)| *i != 1).map(|(_, l)| format!("{}\n", l)).collect();
        assert!(Log::from_jsonl(&dropped).is_err());
    }

    #[test]
    fn mintable_needs_measure_consent_and_a_cleared_engine() {
        let mut log = Log::default();
        log.append(Kind::Capture, "t", "p", Value::Null);
        assert_eq!(mintable(&log).unwrap_err().len(), 2);
        log.append(Kind::Measure, "t", "p", Value::obj(vec![("modality", Value::str("face"))]));
        assert_eq!(mintable(&log).unwrap_err().len(), 2, "a face print is not a voiceprint");
        log.append(Kind::Measure, "t", "p", Value::Null);
        log.append(Kind::Ref, "t", "p", Value::obj(vec![("ref", Value::str("r1"))]));
        log.append(Kind::Consent, "t", "p", Value::obj(vec![("scope", Value::str("mint")), ("ref", Value::str("r1"))]));
        assert_eq!(mintable(&log).unwrap_err(), vec!["the cloning engine's licence is not recorded as cleared for this use"]);
        log.append(Kind::Ref, "t", "p", Value::obj(vec![("ref", Value::str("r1")), ("licence_cleared", Value::Bool(true))]));
        assert!(mintable(&log).is_ok());
        log.append(Kind::Consent, "t", "p", Value::obj(vec![("scope", Value::str("mint")), ("ref", Value::str("r1")),
                                                           ("revoked", Value::Bool(true))]));
        assert!(mintable(&log).is_err());
    }
}
