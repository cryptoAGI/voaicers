// SPDX-License-Identifier: MIT OR Apache-2.0
//! forensic — the forensic voice print ollywoo's `/voicey/measure` returns (voaice `Forensic.voiceprint()`), the second
//! voice print beside vclone's `dvscope/1`.
//!
//! Six features, averaged over a clip's voiced frames (`dominantFrequency`, `amplitude`, `spectralCentroid`,
//! `spectralRolloff`, `zeroCrossingRate`, `harmonicNoiseRatio`, in that order, SoundWaveToken's field order), each
//! encoded `toFixed18` (nine real decimals, the face print's encoding, not the vprint's), then
//! `{"v":1,"kind":"forensic-voiceprint","sampleRate":…,"framesUsed":…,"measures":[…]}` hashed `"0x" + sha256`.
//!
//! What is ported is the print over the features: given the six means, the sample rate and the frame count, the hash
//! is reproduced byte for byte (`tests/forensic.rs`, against Forensic.js on real WAVs at several frame budgets). The
//! DSP that produces the features (voice-activity detection, the analyser's FFT, pitch, flatness) is not ported yet:
//! V8 takes `Math.log`/`Math.exp` from fdlibm, so it needs the same treatment `atan2` got (docs/FCLONE.md).
//!
//! `/voicey/measure` returns `framesUsed` and the measures with the hash (since 2026-10-08), so a print it hands out
//! can be recomputed from its own response with [`forensic_print`].

use crate::fclone::to_fixed18;
use voaice::json::Value;
use voaice::sha256;

pub const FEATURES: [&str; 6] =
    ["dominantFrequency", "amplitude", "spectralCentroid", "spectralRolloff", "zeroCrossingRate", "harmonicNoiseRatio"];

#[derive(Clone, Debug, PartialEq)]
pub struct ForensicPrint {
    pub measures: Vec<String>,
    pub canonical: String,
    pub hash: String,
}

/// Forensic.js `voiceprint()`'s print, from its six feature means (in FEATURES order).
pub fn forensic_print(features: &[f64; 6], sample_rate: u32, frames_used: u64) -> Result<ForensicPrint, String> {
    let measures = features.iter().map(|&v| to_fixed18(v).map(|b| b.to_string())).collect::<Result<Vec<_>, _>>()?;
    let canonical = Value::obj(vec![
        ("v", Value::int(1)),
        ("kind", Value::str("forensic-voiceprint")),
        ("sampleRate", Value::int(sample_rate as i128)),
        ("framesUsed", Value::int(frames_used as i128)),
        ("measures", Value::Arr(measures.iter().map(Value::str).collect())),
    ])
    .to_compact();
    let hash = format!("0x{}", sha256::hex(&sha256::digest(canonical.as_bytes())));
    Ok(ForensicPrint { measures, canonical, hash })
}
