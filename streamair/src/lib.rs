//! streamair — CPU → .opus in zero-dependency Rust.
//!
//! 0.0.1 is the container: Ogg pages and Opus encapsulation, checked against opusinfo/opusdec. The encoder
//! (CELT first, then SILK) follows stage by stage; see docs/ROADMAP.md.
pub mod ogg;
