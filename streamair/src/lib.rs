//! streamair — CPU → .opus in zero-dependency Rust.
//!
//! 0.0.1 is the container: Ogg pages and Opus encapsulation, checked against opusinfo/opusdec. The encoder
//! (CELT first, then SILK) follows stage by stage; see docs/ROADMAP.md.
//!
//! streamair also carries the identities a stream is made from: [`vclone`] (the voice, from voaice.rs) and [`fclone`]
//! (the face, the faice/1 faceprint and the face mesh), both byte-identical to the JavaScript ollywoo runs.
pub mod fclone;
pub mod ogg;

/// vCLONE — the voice identity (the dvscope/1 vprint) and the hash-chained forge log, from voaice.rs. Re-exported so
/// streamair is the one place that carries both clones: [`vclone`] for the voice, [`fclone`] for the face.
pub use voaice::vclone;
