// SPDX-License-Identifier: MIT OR Apache-2.0
//! The identities a stream carries, end to end: prints recomputed from a real .voaice and an ollywoo .faice, bound into
//! a persona print, written into a .opus's OpusTags by streamair's writer, read back from the bytes, and verified
//! against the files. STREAMAIR_TAGGED_OUT=<path> also saves the .opus (for opusinfo, the external check).
use streamair::fclone::{self, PrintRef};
use streamair::identity::Identity;
use streamair::ogg::{mux, Stream};
use streamair::vclone;

/// The OpusTags packet's comments, from the second page streamair writes (a single page here).
fn tags_of(f: &[u8]) -> Vec<String> {
    let n0 = f[26] as usize;
    let p1 = 27 + n0 + f[27..27 + n0].iter().map(|&x| x as usize).sum::<usize>();
    let n1 = f[p1 + 26] as usize;
    let body = &f[p1 + 27 + n1..];
    assert_eq!(&body[..8], b"OpusTags");
    let rd = |o: usize| u32::from_le_bytes(body[o..o + 4].try_into().unwrap()) as usize;
    let vlen = rd(8);
    let mut o = 12 + vlen;
    let count = rd(o);
    o += 4;
    (0..count).map(|_| { let l = rd(o); let c = String::from_utf8(body[o + 4..o + 4 + l].to_vec()).unwrap(); o += 4 + l; c }).collect()
}

#[test]
fn identities_round_trip_through_opustags_and_verify() {
    let voaice = std::fs::read_to_string("../tests/fixtures/voaice/neural.voaice").unwrap();
    let faice = std::fs::read_to_string("tests/fixtures/synthetic.faice.json").unwrap();
    let vclone::Check::Verified(v) = vclone::check_identity(&voaice).unwrap() else { panic!() };
    let fclone::Check::Verified(f) = fclone::check_faice(&faice).unwrap() else { panic!() };
    let persona = fclone::persona_print(
        Some(&PrintRef { hash: f.hash.clone(), measures: f.measures.clone(), precision_score: Some(f.precision_score.clone()) }),
        Some(&PrintRef { hash: v.hash.clone(), measures: v.precision18.iter().map(|(_, w, _)| w.clone()).collect(), precision_score: None }),
    ).unwrap();
    let id = Identity { vprint: Some(v.hash.clone()), fprint: Some(f.hash.clone()), persona: Some(persona.hash.clone()),
                        forge_head: Some("09b3f74131da".to_string() + &"0".repeat(52)) };
    let comments = id.to_comments().unwrap();
    let mut all: Vec<&str> = vec!["ENCODER=streamair"];
    all.extend(comments.iter().map(String::as_str));
    let packets: Vec<Vec<u8>> = (0..51).map(|_| vec![0xF8]).collect();
    let file = mux(&Stream { serial: 7, channels: 1, pre_skip: 312, input_rate: 48000, vendor: "streamair", comments: &all,
                             samples_48k: 48000, packets_per_page: 50 }, &packets).unwrap();
    let back = Identity::from_comments(&tags_of(&file)).unwrap();
    assert_eq!(back, id);
    let checks = back.verify(Some(&voaice), Some(&faice)).unwrap();
    assert_eq!(checks.len(), 3);
    assert!(checks.iter().all(|(_, ok)| *ok), "{checks:?}");
    // a tag that names another voice is caught
    let wrong = Identity { vprint: Some("0".repeat(64)), ..back.clone() };
    assert!(wrong.verify(Some(&voaice), None).unwrap().iter().any(|(_, ok)| !ok));
    if let Ok(p) = std::env::var("STREAMAIR_TAGGED_OUT") {
        std::fs::write(p, &file).unwrap();
    }
}
