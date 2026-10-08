// SPDX-License-Identifier: MIT OR Apache-2.0
//! `voaice` — the command line of voaice.rs.
//!
//!   voaice info  <model.bin>                                verify the pin, print the model summary
//!   voaice mel   <model.bin> <in.wav> [out] [--threads N]   the log-mel spectrogram: shape and sha256 (and raw f32 to `out`)
//!   voaice bench-mel <model.bin> <in.wav> [--threads N]     the mel's heap peak, wall (best of 10), CPU per call, peak RSS
//!   voaice vclone check <file.voaice>...                   recompute each identity's vprint and compare every field
//!   voaice vclone print <8 metrics>                         the dvscope/1 print of eight values (vprint.py's twin)
//!   voaice vclone log <events.jsonl>                        verify a forge log's chain and say whether it is mintable
//!   voaice version
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::process::ExitCode;
use voaice::{measure, mel, model::Model, sha256, vclone, wav};

/// The system allocator, counting live heap bytes and their peak, so `bench-mel` can report the heap a call needs
/// (std only: a `GlobalAlloc` wrapper, no crate). Thread stacks are mapped, not allocated, and are not counted.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                PEAK.fetch_max(LIVE.fetch_add(new - l.size(), Relaxed) + new - l.size(), Relaxed);
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Remove `--threads N` from the arguments; 1 when absent.
fn take_threads(args: &mut Vec<String>) -> Result<usize, String> {
    match args.iter().position(|a| a == "--threads") {
        None => Ok(1),
        Some(i) => {
            let v = args.get(i + 1).ok_or("--threads needs a number")?.parse::<usize>().map_err(|e| format!("--threads: {e}"))?;
            args.drain(i..i + 2);
            Ok(v.max(1))
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut args = args.to_vec();
    let threads = take_threads(&mut args)?;
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
            let plan = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?;
            let start = std::time::Instant::now();
            let mel = plan.run(&pcm, threads)?;
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
        Some("bench-mel") if args.len() == 3 => {
            let (filters, n_mel, n_fft) = {
                let m = Model::load_pinned(Path::new(&args[1]))?;
                (m.filters.clone(), m.filters_n_mel as usize, m.filters_n_fft as usize)
            }; // the model's 78 MB are dropped here, before the peak is reset: what is measured is the mel's
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let plan = mel::MelPlan::new(&t, &filters, n_mel, n_fft)?;
            let mut err = None;
            let mut heap_peak = None;
            let b = measure::bench(
                || {
                    let live = LIVE.load(Relaxed);
                    PEAK.store(live, Relaxed);
                    match plan.run(&pcm, threads) {
                        Err(e) => err = Some(e),
                        Ok(mel) => {
                            // the first call's: the bytes the call had live at its peak, its output included
                            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
                            drop(mel);
                        }
                    }
                },
                10,
                1.0,
            );
            if let Some(e) = err {
                return Err(e);
            }
            let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
            println!(
                "bench-mel threads {threads} samples {} heap_peak_kb {} wall_best_ms {:.3} cpu_ms_per_call {} cpu_reps {} rss_peak_delta_kb {} rss_peak_kb {}",
                pcm.len(),
                heap_peak.unwrap_or(0).div_ceil(1024),
                b.wall_best_ms,
                opt(b.cpu_ms_per_call.map(|c| format!("{c:.3}"))),
                b.cpu_reps,
                opt(b.rss_peak_delta_kb.map(|v| v.to_string())),
                opt(b.rss_peak_kb.map(|v| v.to_string()))
            );
            Ok(())
        }
        Some("vclone") => vclone_cmd(&args[1..]),
        Some("version") => {
            println!("voaice {} (reference: whisper.cpp 080bbbe8, ggml 0.16.0)", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err("usage: voaice info <model.bin> | voaice mel <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-mel <model.bin> <in.wav> [--threads N] | voaice vclone check <file.voaice>... | voaice vclone print <8 metrics> | voaice vclone log <events.jsonl> | voaice version".into()),
    }
}

/// `voaice vclone …` — voice identities (src/vclone.rs).
fn vclone_cmd(a: &[String]) -> Result<(), String> {
    match a.first().map(String::as_str) {
        Some("check") if a.len() >= 2 => {
            let mut bad = 0;
            for f in &a[1..] {
                let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
                match vclone::check_identity(&text) {
                    Ok(vclone::Check::Verified(p)) => println!("verified    {}  {f}", p.short),
                    Ok(vclone::Check::Unmeasured) => println!("unmeasured  {:16}  {f}", "-"),
                    Ok(vclone::Check::Mismatch { recomputed, fields }) => {
                        bad += 1;
                        println!("MISMATCH    {}  {f}: {}", recomputed.short, fields.join(", "))
                    }
                    Err(e) => {
                        bad += 1;
                        println!("ERROR       {:16}  {f}: {e}", "-")
                    }
                }
            }
            if bad > 0 { Err(format!("{bad} file(s) did not verify")) } else { Ok(()) }
        }
        Some("print") if a.len() == 9 => {
            let mut v = [0f64; 8];
            for (i, s) in a[1..].iter().enumerate() {
                v[i] = s.parse().map_err(|_| format!("not a number: {s}"))?;
            }
            let p = vclone::vprint(&v)?;
            println!("{}\n{}\n{}\n{}", p.hash, p.hash512, p.uint256, p.canonical);
            Ok(())
        }
        Some("log") if a.len() == 2 => {
            let text = std::fs::read_to_string(&a[1]).map_err(|e| format!("{}: {e}", a[1]))?;
            let log = vclone::Log::from_jsonl(&text)?;
            println!("chain verified: {} events", log.events.len());
            match vclone::mintable(&log) {
                Ok(()) => println!("mintable: yes"),
                Err(why) => println!("mintable: no\n  - {}", why.join("\n  - ")),
            }
            Ok(())
        }
        _ => Err("usage: voaice vclone check <file.voaice>... | vclone print <rms dominantFrequency spectralCentroid spectralRolloff zeroCrossingRate spectralBandwidth spectralFlux harmonicNoiseRatio> | vclone log <events.jsonl>".into()),
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
