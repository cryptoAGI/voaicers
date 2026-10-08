// SPDX-License-Identifier: MIT OR Apache-2.0
//! `voaice` — the command line of voaice.rs 0.0.1.
//!
//!   voaice info  <model.bin>                  verify the pin, print the model summary
//!   voaice mel   <model.bin> <in.wav> [out]   the log-mel spectrogram: shape and sha256 (and raw f32 to `out`)
//!   voaice version
use std::path::Path;
use std::process::ExitCode;
use voaice::{mel, model::Model, sha256, wav};

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("info") if args.len() == 2 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            print!("{}", m.summary());
            Ok(())
        }
        Some("mel") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let start = std::time::Instant::now();
            let mel = mel::log_mel_spectrogram(&t, &pcm, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?;
            let took = start.elapsed();
            let bytes: Vec<u8> = mel.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "mel {} x {} (n_len_org {}), {} samples, sha256 {}, {:.1} ms",
                mel.n_mel,
                mel.n_len,
                mel.n_len_org,
                pcm.len(),
                sha256::hex(&sha256::digest(&bytes)),
                took.as_secs_f64() * 1e3
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("version") => {
            println!("voaice {} (reference: whisper.cpp 080bbbe8, ggml 0.16.0)", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err("usage: voaice info <model.bin> | voaice mel <model.bin> <in.wav> [out.f32] | voaice version".into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("voaice: {e}");
            ExitCode::FAILURE
        }
    }
}
