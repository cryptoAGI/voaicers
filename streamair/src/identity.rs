// SPDX-License-Identifier: MIT OR Apache-2.0
//! identity — the identities a stream was made from, written into its OpusTags, and checked on read.
//!
//! A `.opus` streamair writes can name the voice, the face and the forge it came from, as Vorbis comments in its
//! OpusTags header (RFC 7845 §5.2):
//!
//! ```text
//! VOAICE_VPRINT=dvscope/1:<64 hex>        the voice's vprint (vclone)
//! FAICE_FPRINT=faceprint/1:0x<64 hex>     the face's faceprint (fclone)
//! PERSONA_PRINT=0x<64 hex>                the two bound together (fclone::persona_print)
//! VCLONE_FORGE_HEAD=<64 hex>              the forge log's head event id (vclone-event/1)
//! ```
//!
//! Each value carries its format's version, so a reader never compares a dvscope/1 print with something else.
//! Writing refuses a malformed value; reading ignores unknown comments, takes field names case-insensitively (as the
//! Vorbis comment rules require), and refuses a recognised field whose value is malformed or repeated with another
//! value. [`Identity::verify`] recomputes the prints from `.voaice` / `.faice` files and says which agree.
//!
//! These are prints of measurements, not biometrics, and a tag is a claim: it proves nothing until it is checked
//! against the files it names.

use crate::{fclone, vclone};

pub const VPRINT: &str = "VOAICE_VPRINT";
pub const FPRINT: &str = "FAICE_FPRINT";
pub const PERSONA: &str = "PERSONA_PRINT";
pub const FORGE_HEAD: &str = "VCLONE_FORGE_HEAD";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Identity {
    /// the dvscope/1 vprint, 64 lowercase hex
    pub vprint: Option<String>,
    /// the faceprint/1 hash, "0x" + 64 lowercase hex
    pub fprint: Option<String>,
    /// the persona print, "0x" + 64 lowercase hex
    pub persona: Option<String>,
    /// the forge log's head event id, 64 lowercase hex
    pub forge_head: Option<String>,
}

fn hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hex0x(s: &str) -> bool {
    s.strip_prefix("0x").is_some_and(hex64)
}

impl Identity {
    /// The comments to put in OpusTags (in a fixed order), or the first malformed value.
    pub fn to_comments(&self) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        if let Some(v) = &self.vprint {
            if !hex64(v) {
                return Err(format!("{VPRINT}: not 64 lowercase hex: {v}"));
            }
            out.push(format!("{VPRINT}={}:{v}", vclone::VERSION));
        }
        if let Some(v) = &self.fprint {
            if !hex0x(v) {
                return Err(format!("{FPRINT}: not 0x + 64 lowercase hex: {v}"));
            }
            out.push(format!("{FPRINT}=faceprint/1:{v}"));
        }
        if let Some(v) = &self.persona {
            if !hex0x(v) {
                return Err(format!("{PERSONA}: not 0x + 64 lowercase hex: {v}"));
            }
            out.push(format!("{PERSONA}={v}"));
        }
        if let Some(v) = &self.forge_head {
            if !hex64(v) {
                return Err(format!("{FORGE_HEAD}: not 64 lowercase hex: {v}"));
            }
            out.push(format!("{FORGE_HEAD}={v}"));
        }
        Ok(out)
    }

    /// Read the identity from OpusTags comments. Unknown comments are ignored; a known field that is malformed, or
    /// that appears twice with different values, is refused.
    pub fn from_comments<S: AsRef<str>>(comments: &[S]) -> Result<Identity, String> {
        let mut id = Identity::default();
        for c in comments {
            let c = c.as_ref();
            let Some((k, v)) = c.split_once('=') else { continue };
            let slot = match k.to_ascii_uppercase().as_str() {
                VPRINT => {
                    let h = v.strip_prefix("dvscope/1:").filter(|h| hex64(h)).ok_or(format!("{VPRINT}: malformed: {v}"))?;
                    (&mut id.vprint, h.to_string())
                }
                FPRINT => {
                    let h = v.strip_prefix("faceprint/1:").filter(|h| hex0x(h)).ok_or(format!("{FPRINT}: malformed: {v}"))?;
                    (&mut id.fprint, h.to_string())
                }
                PERSONA if hex0x(v) => (&mut id.persona, v.to_string()),
                FORGE_HEAD if hex64(v) => (&mut id.forge_head, v.to_string()),
                PERSONA | FORGE_HEAD => return Err(format!("{k}: malformed: {v}")),
                _ => continue,
            };
            match slot.0 {
                Some(prev) if *prev != slot.1 => return Err(format!("{k} appears twice with different values")),
                _ => *slot.0 = Some(slot.1),
            }
        }
        Ok(id)
    }

    /// Check the tagged prints against the identity files they name: recompute the vprint from a `.voaice` and the
    /// faceprint from a `.faice`, and the persona print from the two. Each line says agrees / differs / not checked.
    pub fn verify(&self, voaice: Option<&str>, faice: Option<&str>) -> Result<Vec<(String, bool)>, String> {
        let mut out = Vec::new();
        let mut vref = None;
        let mut fref = None;
        if let (Some(tag), Some(text)) = (&self.vprint, voaice) {
            match vclone::check_identity(text)? {
                vclone::Check::Verified(p) => {
                    out.push((format!("{VPRINT} vs the .voaice"), p.hash == *tag));
                    vref = Some(fclone::PrintRef { hash: p.hash.clone(),
                        measures: p.precision18.iter().map(|(_, wei, _)| wei.clone()).collect(), precision_score: None });
                }
                _ => return Err("the .voaice does not verify on its own".into()),
            }
        }
        if let (Some(tag), Some(text)) = (&self.fprint, faice) {
            match fclone::check_faice(text)? {
                fclone::Check::Verified(p) => {
                    out.push((format!("{FPRINT} vs the .faice"), p.hash == *tag));
                    fref = Some(fclone::PrintRef { hash: p.hash.clone(), measures: p.measures.clone(),
                                                   precision_score: Some(p.precision_score.clone()) });
                }
                _ => return Err("the .faice does not verify on its own".into()),
            }
        }
        if let Some(tag) = &self.persona {
            if fref.is_some() || vref.is_some() {
                let p = fclone::persona_print(fref.as_ref(), vref.as_ref())?;
                out.push((format!("{PERSONA} vs the two prints"), p.hash == *tag));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_refusals() {
        let id = Identity { vprint: Some("a".repeat(64)), fprint: Some(format!("0x{}", "b".repeat(64))),
                            persona: Some(format!("0x{}", "c".repeat(64))), forge_head: Some("d".repeat(64)) };
        let c = id.to_comments().unwrap();
        assert_eq!(c[0], format!("VOAICE_VPRINT=dvscope/1:{}", "a".repeat(64)));
        let mut with_noise = vec!["ENCODER=streamair".to_string(), "voaice_vprint=dvscope/1:".to_string() + &"a".repeat(64)];
        with_noise.extend(c);
        assert_eq!(Identity::from_comments(&with_noise).unwrap(), id);
        assert!(Identity { vprint: Some("A".repeat(64)), ..Default::default() }.to_comments().is_err());
        assert!(Identity::from_comments(&["FAICE_FPRINT=faceprint/2:0x".to_string() + &"b".repeat(64)]).is_err());
        let twice = [format!("PERSONA_PRINT=0x{}", "c".repeat(64)), format!("PERSONA_PRINT=0x{}", "e".repeat(64))];
        assert!(Identity::from_comments(&twice).is_err());
    }
}
