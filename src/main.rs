// SPDX-License-Identifier: MIT OR Apache-2.0
//! `voaice` — the command line of voaice.rs.
//!
//!   voaice info  <model.bin>                                verify the pin, print the model summary
//!   voaice mel   <model.bin> <in.wav> [out] [--threads N]   the log-mel spectrogram: shape and sha256 (and raw f32 to `out`)
//!   voaice bench-mel <model.bin> <in.wav> [--threads N]     the mel's heap peak, wall (best of 10), CPU per call, peak RSS
//!   voaice bench-f16 init|rows                             (0.0.3) the GELU table's build; f32<->f16 rows and the GELU op
//!   voaice opus info <file.opus>                           (0.0.4) read the Ogg/Opus stream page by page: headers, counts, exact duration
//!   voaice bench-opus <file.opus>                           (0.0.4) the CRC (sliced vs one byte at a time), the reader's throughput, heap peak
//!   voaice resample <in.wav> [out.f32]                     (0.0.5) any WAV -> whisper's 16 kHz mono f32, as whisper-cli reads it
//!   voaice bench-resample <in.wav>                          (0.0.5) that read's heap peak, wall (best of 10), CPU per call, peak RSS
//!   voaice conv1 <model.bin> <in.wav> [out.f32] [--threads N]   (0.0.6) encoder conv1 + bias + GELU of the first 30-s window
//!   voaice bench-conv1 <model.bin> <in.wav> conv1|gelu [--threads N]   (0.0.6) its heap peak, wall (best of 10), CPU per call, RSS
//!   voaice conv <model.bin> <in.wav> [out.f32] [--threads N]    (0.0.7) the conv stage: mel -> conv1 -> conv2 -> + positions (the encoder's input)
//!   voaice bench-conv <model.bin> <in.wav> conv2|stage [--threads N]   (0.0.7) conv2 (+ bias + GELU), or the whole stage: heap, wall, CPU, RSS
//!   voaice norm <model.bin> <in.wav> [out.f32] [--threads N]    (0.0.8) the encoder input through block 0's attn_ln (norm, * w, + b)
//!   voaice bench-norm <model.bin> <in.wav> norm|chain [--threads N]   (0.0.8) the NORM node, or norm -> * w -> + b: heap, wall, CPU, RSS
//!   voaice qkv  <model.bin> <in.wav> [q.f32] [--threads N]     (0.0.9) block 0's attention inputs from the WAV: attn_ln -> Q + b, K and V + b as f16
//!   voaice bench-mm <model.bin> <in.wav> q|fc1|fc2|qkv|mlp|block [--threads N]   (0.0.9) block 0's products: heap, wall, CPU, RSS
//!   voaice encode <model.bin> <in.wav> [out.f32] [--threads N]  (v0.1.0) the whole encoder: the WAV -> mel -> embd_enc (sha256, digest)
//!   voaice bench-attn <model.bin> <in.wav> [--threads N]        (v0.1.0) block 0's flash attention: heap, wall, CPU, RSS
//!   voaice bench-encode <model.bin> <in.wav> [--threads N]      (v0.1.0) the whole encoder, mel -> embd_enc: heap, wall, CPU, RSS
//!   voaice cross <model.bin> <in.wav> [out.f16] [--threads N]   (0.1.1) the cross-attention cache: the WAV -> embd_enc -> kv_cross (k then v)
//!   voaice bench-cross <model.bin> <in.wav> cross|whole [--threads N]   (0.1.1) the cross K/V from embd_enc, or mel -> embd_enc -> kv_cross
//!   voaice decin <model.bin> <token,...> [--pos N]               (0.1.2) the decoder's input for tokens at positions N..: digest per row
//!   voaice bench-decin <model.bin> <n_tokens>                 (0.1.2) the decoder's input, batch prep + rows: wall and CPU per call
//!   voaice bench-selfkv <model.bin> <n_tokens> block|call [--threads N]   (0.1.3) attn_ln -> Q, K, V -> the f16 self cache (+ the mask)
//!   voaice vclone check <file.voaice>...                   recompute each identity's vprint and compare every field
//!   voaice vclone print <8 metrics>                         the dvscope/1 print of eight values (vprint.py's twin)
//!   voaice vclone log <events.jsonl>                        verify a forge log's chain and say whether it is mintable
//!   voaice version
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::process::ExitCode;
use voaice::{attention, conv, cross, decoder, encoder, f16, gelu, matmul, measure, mel, model::Model, norm, ogg, resample, selfkv, sha256, vclone, wav};

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
        Some("bench-f16") if args.len() == 2 => bench_f16(&args[1]),
        Some("conv1") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let c = conv::Conv1::new(&m)?;
            let n_frames = 2 * m.hparams.n_audio_ctx as usize;
            let start = std::time::Instant::now();
            let y = c.run_gelu(&mel.data, mel.n_len, 0, n_frames, threads);
            let took = start.elapsed();
            let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!("conv1+bias+gelu {} x {n_frames}, sha256 {}, {:.1} ms", c.n_out, sha256::hex(&sha256::digest(&bytes)), took.as_secs_f64() * 1e3);
            if let Some(out) = args.get(3) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-conv1") if args.len() == 4 && (args[3] == "conv1" || args[3] == "gelu") => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
            let n_frames = 2 * m.hparams.n_audio_ctx as usize;
            let c = conv::Conv1::new(&m)?;
            drop(m); // the model's 78 MB go before the peak is reset; the plan keeps the widened weights (368 KiB)
            let gelu = args[3] == "gelu";
            let mut heap_peak = None;
            // the output is allocated by the first call and kept, as a caller keeps it (the reference's graph keeps its
            // tensors too): the first call's peak includes it, the timed calls reuse it
            let mut out: Vec<f32> = Vec::new();
            let b = measure::bench(
                || {
                    let live = LIVE.load(Relaxed);
                    PEAK.store(live, Relaxed);
                    if out.is_empty() {
                        out = vec![0.0; c.n_out * n_frames];
                    }
                    c.run_into(&mel.data, mel.n_len, 0, n_frames, threads, gelu, &mut out);
                    heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
                },
                10,
                1.0,
            );
            let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
            println!(
                "bench-conv1 what {} threads {threads} wall_best_ms {:.3} cpu_ms_per_call {} cpu_reps {} heap_peak_kb {} rss_peak_delta_kb {}",
                args[3],
                b.wall_best_ms,
                opt(b.cpu_ms_per_call.map(|c| format!("{c:.3}"))),
                b.cpu_reps,
                heap_peak.unwrap_or(0).div_ceil(1024),
                opt(b.rss_peak_delta_kb.map(|v| v.to_string()))
            );
            Ok(())
        }
        Some("conv") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let st = conv::ConvStage::new(&m)?;
            let n_frames = 2 * m.hparams.n_audio_ctx as usize;
            let start = std::time::Instant::now();
            let y = st.run(&mel.data, mel.n_len, 0, n_frames, threads);
            let took = start.elapsed();
            let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "encoder input (conv1, conv2, + positions) {} x {} frame-major, sha256 {}, {:.1} ms",
                st.conv2.frames_out(n_frames),
                st.conv2.n_out,
                sha256::hex(&sha256::digest(&bytes)),
                took.as_secs_f64() * 1e3
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-conv") if args.len() == 4 && (args[3] == "conv2" || args[3] == "stage") => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
            let n_frames = 2 * m.hparams.n_audio_ctx as usize;
            let st = conv::ConvStage::new(&m)?;
            drop(m); // the model goes before the peak is reset; the plan keeps the widened weights and the positions
            let stage = args[3] == "stage";
            // conv2 alone reads conv1's GELU output, computed here, outside the measurement (the reference's bench
            // computes it with its own conv1 graph beforehand too)
            let x = if stage { Vec::new() } else { st.conv1.run_gelu(&mel.data, mel.n_len, 0, n_frames, threads) };
            let n_out = st.conv2.n_out * st.conv2.frames_out(n_frames);
            let mut heap_peak = None;
            // the buffers are allocated by the first call and kept, as a caller keeps them: conv2 = its output
            // (embd_conv, [n_state][n_ctx]); stage = conv1's output as f16 (scratch) and the encoder input
            let (mut out, mut scratch): (Vec<f32>, Vec<u16>) = (Vec::new(), Vec::new());
            let b = measure::bench(
                || {
                    let live = LIVE.load(Relaxed);
                    PEAK.store(live, Relaxed);
                    if out.is_empty() {
                        out = vec![0.0; n_out];
                    }
                    if stage {
                        st.run_into(&mel.data, mel.n_len, 0, n_frames, threads, &mut scratch, &mut out);
                    } else {
                        st.conv2.run_into(&x, n_frames, threads, conv::Epilogue::BiasGelu, &mut out);
                    }
                    heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
                },
                10,
                1.0,
            );
            let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
            println!(
                "bench-conv what {} threads {threads} wall_best_ms {:.3} cpu_ms_per_call {} cpu_reps {} heap_peak_kb {} rss_peak_delta_kb {}",
                args[3],
                b.wall_best_ms,
                opt(b.cpu_ms_per_call.map(|c| format!("{c:.3}"))),
                b.cpu_reps,
                heap_peak.unwrap_or(0).div_ceil(1024),
                opt(b.rss_peak_delta_kb.map(|v| v.to_string()))
            );
            Ok(())
        }
        Some("norm") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let st = conv::ConvStage::new(&m)?;
            let ln = norm::LayerNorm::new(&m, "encoder.blocks.0.attn_ln")?;
            let x = st.run(&mel.data, mel.n_len, 0, 2 * m.hparams.n_audio_ctx as usize, threads);
            let start = std::time::Instant::now();
            let y = ln.run(&x, threads, norm::Node::Add);
            let took = start.elapsed();
            let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "block 0 attn_ln (norm eps 1e-5, * w, + b) {} x {} frame-major, sha256 {}, {:.3} ms",
                y.len() / ln.n,
                ln.n,
                sha256::hex(&sha256::digest(&bytes)),
                took.as_secs_f64() * 1e3
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-norm") if args.len() == 4 && (args[3] == "norm" || args[3] == "chain") => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
            // the input: the encoder's input (the conv stage), computed here, outside the measurement, as the
            // reference's bench computes it with its own graph beforehand
            let x = conv::ConvStage::new(&m)?.run(&mel.data, mel.n_len, 0, 2 * m.hparams.n_audio_ctx as usize, threads);
            let ln = norm::LayerNorm::new(&m, "encoder.blocks.0.attn_ln")?;
            drop(m);
            let node = if args[3] == "chain" { norm::Node::Add } else { norm::Node::Norm };
            let mut heap_peak = None;
            let mut out: Vec<f32> = Vec::new(); // allocated by the first call and kept, as a caller keeps it
            let b = measure::bench(
                || {
                    let live = LIVE.load(Relaxed);
                    PEAK.store(live, Relaxed);
                    if out.is_empty() {
                        out = vec![0.0; x.len()];
                    }
                    ln.run_into(&x, threads, node, &mut out);
                    heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
                },
                10,
                1.0,
            );
            let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
            println!(
                "bench-norm what {} threads {threads} wall_best_ms {:.4} cpu_ms_per_call {} cpu_reps {} heap_peak_kb {} rss_peak_delta_kb {}",
                args[3],
                b.wall_best_ms,
                opt(b.cpu_ms_per_call.map(|c| format!("{c:.4}"))),
                b.cpu_reps,
                heap_peak.unwrap_or(0).div_ceil(1024),
                opt(b.rss_peak_delta_kb.map(|v| v.to_string()))
            );
            Ok(())
        }
        Some("qkv") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let st = conv::ConvStage::new(&m)?;
            let b = matmul::Block::new(&m, 0)?;
            let x = st.run(&mel.data, mel.n_len, 0, 2 * m.hparams.n_audio_ctx as usize, threads);
            let n = b.q.n;
            let (mut q, mut k16, mut v16) = (vec![0.0f32; x.len()], vec![0u16; x.len()], vec![0u16; x.len()]);
            let start = std::time::Instant::now();
            b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, matmul::QkvTaps::default());
            let took = start.elapsed();
            let qb: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
            let hex16 = |h: &[u16]| sha256::hex(&sha256::digest(&h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
            println!(
                "block 0 attention inputs (attn_ln -> Q + b f32, K f16, V + b f16) {} x {n} frame-major, sha256 Q {} K {} V {}, {:.3} ms",
                q.len() / n,
                sha256::hex(&sha256::digest(&qb)),
                hex16(&k16),
                hex16(&v16),
                took.as_secs_f64() * 1e3
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &qb).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-mm") if args.len() == 4 && ["q", "fc1", "fc2", "qkv", "mlp", "block"].contains(&args[3].as_str()) => bench_mm(&args[1], &args[2], &args[3], threads),
        Some("encode") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let enc = encoder::Encoder::new(&m)?;
            drop(m);
            let mut buf = encoder::EncoderBuffers::default();
            let mut e = vec![0.0f32; enc.n_ctx * enc.n_state];
            let start = std::time::Instant::now();
            enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut e);
            let took = start.elapsed();
            let eb: Vec<u8> = e.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "embd_enc {} x {} frame-major f32, sha256 {}, digest {:016x}, {:.3} ms at {threads} thread(s) (mel -> conv -> {} blocks -> ln_post)",
                enc.n_ctx,
                enc.n_state,
                sha256::hex(&sha256::digest(&eb)),
                digest32(&e),
                took.as_secs_f64() * 1e3,
                enc.blocks.len()
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &eb).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("cross") if args.len() == 3 || args.len() == 4 => {
            let m = Model::load_pinned(Path::new(&args[1]))?;
            let pcm = wav::read(Path::new(&args[2]))?;
            let t = mel::Tables::new();
            let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, threads)?;
            let enc = encoder::Encoder::new(&m)?;
            let cr = cross::Cross::new(&m)?;
            drop(m);
            let e = enc.encode(&mel.data, mel.n_len, 0, threads);
            let mut kv = cross::KvCross::default();
            let start = std::time::Instant::now();
            cr.run_into(&e, threads, &mut kv, None);
            let took = start.elapsed();
            let bytes: Vec<u8> = kv.k.iter().chain(&kv.v).flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "kv_cross {} layers x {} rows ({} frames + {} rows of +0) x {} f16, k then v: sha256 {}, digest {:016x}; {:.3} ms at {threads} thread(s) (embd_enc -> K x {:08x}, V + b)",
                cr.layers.len(),
                cr.n_pad,
                cr.n_ctx,
                cr.n_pad - cr.n_ctx,
                cr.n_state,
                sha256::hex(&sha256::digest(&bytes)),
                kv_digest(&kv),
                took.as_secs_f64() * 1e3,
                cr.kscale.to_bits()
            );
            if let Some(out) = args.get(3) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-cross") if args.len() == 4 && (args[3] == "cross" || args[3] == "whole") => bench_cross(&args[1], &args[2], args[3] == "whole", threads),
        Some("decin") if args.len() == 3 || (args.len() == 5 && args[3] == "--pos") => decin_cmd(&args[1], &args[2], args.get(4)),
        Some("bench-decin") if args.len() == 3 => bench_decin(&args[1], &args[2]),
        Some("bench-selfkv") if args.len() == 4 && (args[3] == "block" || args[3] == "call") => bench_selfkv(&args[1], &args[2], args[3] == "call", threads),
        Some("bench-attn") if args.len() == 3 => bench_attn(&args[1], &args[2], threads),
        Some("bench-encode") if args.len() == 3 => bench_encode(&args[1], &args[2], threads),
        Some("opus") if args.len() == 3 && args[1] == "info" => opus_info(&args[2]),
        Some("bench-opus") if args.len() == 2 => bench_opus(&args[1]),
        Some("resample") if args.len() == 2 || args.len() == 3 => {
            let start = std::time::Instant::now();
            let (fmt, pcm) = resample::read(Path::new(&args[1]))?;
            let took = start.elapsed();
            let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            println!(
                "resample {} Hz {} ch {} {} frames -> 16000 Hz mono f32 {} samples, sha256 {}, {:.2} ms",
                fmt.rate,
                fmt.channels,
                fmt.sample.name(),
                fmt.frames(),
                pcm.len(),
                sha256::hex(&sha256::digest(&bytes)),
                took.as_secs_f64() * 1e3
            );
            if let Some(out) = args.get(2) {
                std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            }
            Ok(())
        }
        Some("bench-resample") if args.len() == 2 => {
            // each call is the whole read, as the reference's: open, parse, convert, mix, resample, the vector out
            let path = Path::new(&args[1]);
            let mut err = None;
            let mut heap_peak = None;
            let mut samples = 0;
            let b = measure::bench(
                || {
                    let live = LIVE.load(Relaxed);
                    PEAK.store(live, Relaxed);
                    match resample::read(path) {
                        Err(e) => err = Some(e),
                        Ok((_, pcm)) => {
                            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
                            samples = pcm.len();
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
                "bench-resample samples {samples} heap_peak_kb {} wall_best_ms {:.3} cpu_ms_per_call {} cpu_reps {} rss_peak_delta_kb {} rss_peak_kb {}",
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
        _ => Err("usage: voaice info <model.bin> | voaice mel <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-mel <model.bin> <in.wav> [--threads N] | voaice bench-f16 init|rows | voaice conv1 <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-conv1 <model.bin> <in.wav> conv1|gelu [--threads N] | voaice conv <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-conv <model.bin> <in.wav> conv2|stage [--threads N] | voaice norm <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-norm <model.bin> <in.wav> norm|chain [--threads N] | voaice qkv <model.bin> <in.wav> [q.f32] [--threads N] | voaice bench-mm <model.bin> <in.wav> q|fc1|fc2|qkv|mlp|block [--threads N] | voaice encode <model.bin> <in.wav> [out.f32] [--threads N] | voaice bench-attn <model.bin> <in.wav> [--threads N] | voaice bench-encode <model.bin> <in.wav> [--threads N] | voaice cross <model.bin> <in.wav> [out.f16] [--threads N] | voaice bench-cross <model.bin> <in.wav> cross|whole [--threads N] | voaice decin <model.bin> <token,...> [--pos N] | voaice bench-decin <model.bin> <n_tokens> | voaice bench-selfkv <model.bin> <n_tokens> block|call [--threads N] | voaice opus info <file.opus> | voaice bench-opus <file.opus> | voaice resample <in.wav> [out.f32] | voaice bench-resample <in.wav> | voaice vclone check <file.voaice>... | voaice vclone print <8 metrics> | voaice vclone log <events.jsonl> | voaice version".into()),
    }
}

/// `voaice bench-mm <model> <wav> <what>` — the same measurements as `whisper_oracle --bench-mm`, on the same inputs,
/// computed beforehand from the WAV (X = the encoder's input): q = Q + bias and fc1 = fc1 + bias + GELU on attn_ln_0(X)
/// (fc1's stand-in for mlp_ln's output); fc2 = fc2 + bias on that GELU; qkv = attn_ln -> Q + b, K, V + b with K and V as
/// f16 (attn_ln fused into the conversion); mlp = out proj + b + X -> mlp_ln -> fc1 + b -> GELU -> fc2 + b + residual,
/// with V's output standing in for the attention; block = qkv then mlp. Outputs allocated by the first call and kept.
fn bench_mm(model: &str, wavp: &str, what: &str, threads: usize) -> Result<(), String> {
    let m = Model::load_pinned(Path::new(model))?;
    let pcm = wav::read(Path::new(wavp))?;
    let t = mel::Tables::new();
    let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
    let x = conv::ConvStage::new(&m)?.run(&mel.data, mel.n_len, 0, 2 * m.hparams.n_audio_ctx as usize, threads);
    let b = matmul::Block::new(&m, 0)?;
    drop(m);
    let (ns, nh) = (b.q.n, b.fc1.n);
    let rows = x.len() / ns;
    let ln = b.attn_ln.run(&x, threads, norm::Node::Add);
    let mut ge = Vec::new();
    if what == "fc2" {
        ge = vec![0.0f32; rows * nh];
        b.fc1.run_into(&ln, None, threads, matmul::Epilogue::BiasGelu(&b.gelu), &mut ge);
    }
    let mut vstand = Vec::new();
    if what == "mlp" {
        let (mut q, mut k16, mut v16) = (vec![0.0f32; x.len()], vec![0u16; x.len()], vec![0u16; x.len()]);
        vstand = vec![0.0f32; x.len()];
        b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, matmul::QkvTaps { v_add: Some(&mut vstand), ..Default::default() });
    }
    let mut heap_peak = None;
    // the outputs, allocated by the first call and kept (as a caller keeps them)
    let (mut out, mut q, mut va): (Vec<f32>, Vec<f32>, Vec<f32>) = Default::default();
    let (mut k16, mut v16): (Vec<u16>, Vec<u16>) = Default::default();
    let bm = measure::bench(
        || {
            let live = LIVE.load(Relaxed);
            PEAK.store(live, Relaxed);
            match what {
                "q" | "fc1" | "fc2" => {
                    if out.is_empty() {
                        out = vec![0.0; rows * if what == "fc1" { nh } else { ns }];
                    }
                    match what {
                        "q" => b.q.run_into(&ln, None, threads, matmul::Epilogue::Bias, &mut out),
                        "fc1" => b.fc1.run_into(&ln, None, threads, matmul::Epilogue::BiasGelu(&b.gelu), &mut out),
                        _ => b.fc2.run_into(&ge, None, threads, matmul::Epilogue::Bias, &mut out),
                    }
                }
                "mlp" => {
                    if out.is_empty() {
                        out = vec![0.0; x.len()];
                    }
                    b.mlp_into(&vstand, &x, threads, &mut out, matmul::MlpTaps::default());
                }
                _ => {
                    if q.is_empty() {
                        (q, k16, v16) = (vec![0.0; x.len()], vec![0; x.len()], vec![0; x.len()]);
                        if what == "block" {
                            (va, out) = (vec![0.0; x.len()], vec![0.0; x.len()]);
                        }
                    }
                    if what == "block" {
                        b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, matmul::QkvTaps { v_add: Some(&mut va), ..Default::default() });
                        b.mlp_into(&va, &x, threads, &mut out, matmul::MlpTaps::default());
                    } else {
                        b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, matmul::QkvTaps::default());
                    }
                }
            }
            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
        },
        10,
        1.0,
    );
    let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
    println!(
        "bench-mm what {what} threads {threads} wall_best_ms {:.4} cpu_ms_per_call {} cpu_reps {} heap_peak_kb {} rss_peak_delta_kb {}",
        bm.wall_best_ms,
        opt(bm.cpu_ms_per_call.map(|c| format!("{c:.4}"))),
        bm.cpu_reps,
        heap_peak.unwrap_or(0).div_ceil(1024),
        opt(bm.rss_peak_delta_kb.map(|v| v.to_string()))
    );
    Ok(())
}

/// The oracle's digest32: 64-bit FNV-1a over the f32 bits packed two to a u64 word (little-endian), so `voaice encode`
/// and `whisper_oracle --bench-encode` print comparable digests of `embd_enc`.
fn digest32(x: &[f32]) -> u64 {
    x.as_chunks::<2>().0.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
        (h ^ (p[0].to_bits() as u64 | (p[1].to_bits() as u64) << 32)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// `voaice bench-selfkv <model> <n_tokens> block|call` — the same measurement as `whisper_oracle --bench-selfkv`: the
/// same input (x[i] = ((i·7919) mod 2001 − 1000) / 256) for n_tokens rows; head 0 for a prompt, 226 for a step (n = 1),
/// the cells below the head used by sequence 0 at their own positions, `KvSelf::prepare` run once before timing (the
/// reference finds its slot outside the graph too). block = layer 0's attn_ln -> Q, K, V -> the f16 cells; call =
/// every layer's, each reading the same input, and the f16 mask built from the cells (the reference times only the
/// mask's cast: whisper fills its f32 mask on the host, outside the graph — voaice's measured call does both). One call
/// is microseconds to milliseconds: wall = the mean over >= 1 s (>= 20 calls), cpu = CPU per call over the same loop.
fn bench_selfkv(model: &str, n: &str, call: bool, threads: usize) -> Result<(), String> {
    let n: usize = n.parse().map_err(|e| format!("n_tokens: {e}"))?;
    let m = Model::load_pinned(Path::new(model))?;
    let sa = selfkv::SelfAttn::new(&m)?;
    let mut kv = selfkv::KvSelf::for_model(&m);
    drop(m);
    let ns = sa.n_state;
    let n_layer = if call { sa.layers.len() } else { 1 };
    let head = if n == 1 { 226 } else { 0 };
    if head + n > kv.size {
        return Err(format!("bench-selfkv: {n} tokens do not fit the cache's {} cells", kv.size));
    }
    let xs: Vec<f32> = (0..n * ns).map(|i| ((i * 7919 % 2001) as i32 - 1000) as f32 / 256.0).collect();
    for (i, c) in kv.cells.iter_mut().enumerate().take(head) {
        *c = selfkv::Cell { pos: i as i32, seq: 1 };
    }
    let mut b = decoder::Batch::with_capacity(n);
    b.prep_legacy(&vec![0; n], head as i32, 0);
    kv.prepare(&b)?;
    if kv.head != head || kv.n != head + n {
        return Err("bench-selfkv: the slot is not where the reference's graph writes".into());
    }
    let (mut q, mut mask) = (vec![0.0f32; n * ns], Vec::with_capacity((head + n) * n));
    let mut run = |kv: &mut selfkv::KvSelf| {
        if call {
            kv.mask_into(&b, &mut mask);
        }
        for il in 0..n_layer {
            sa.layer_into(il, &xs, threads, kv, &mut q, selfkv::SelfTaps::default());
        }
    };
    run(&mut kv);
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    run(&mut kv); // a call after the first: what it allocates (scratch per call; the cache and q are the caller's)
    let heap = PEAK.load(Relaxed) - live;
    let (c0, w0) = (measure::cpu_seconds(), std::time::Instant::now());
    let mut reps = 0u64;
    while reps < 20 || w0.elapsed().as_secs_f64() < 1.0 {
        run(&mut kv);
        reps += 1;
    }
    let (w1, c1) = (w0.elapsed().as_secs_f64(), measure::cpu_seconds());
    let cpu = match (c0, c1) {
        (Some(a), Some(b)) => format!("{:.3}", (b - a) * 1e6 / reps as f64),
        _ => "n/a".into(),
    };
    let fold = |h: u64, d: u64| (h ^ d).wrapping_mul(0x0000_0100_0000_01b3);
    let mut h = digest32(&q);
    for c in [&kv.k, &kv.v] {
        for il in 0..n_layer {
            for r in 0..n {
                let at = kv.row(il, head + r) * ns;
                h = fold(h, digest16(&c[at..at + ns]));
            }
        }
    }
    if call {
        for row in mask.chunks_exact(kv.n) {
            h = fold(h, row.iter().fold(0xcbf2_9ce4_8422_2325u64, |g, &v| fold(g, v as u64)));
        }
    }
    let held: usize = sa.layers[..n_layer].iter().map(|l| l.q.bytes() + l.k.bytes() + l.v.bytes() + 8 * l.attn_ln.n).sum::<usize>() + kv.bytes();
    println!(
        "bench-selfkv threads {threads} n_tokens {n} what {} wall_us_per_call {:.3} cpu_us_per_call {cpu} reps {reps} heap_per_call_bytes {heap} held_kb {} out_digest {h:016x}",
        if call { "call" } else { "block" },
        w1 * 1e6 / reps as f64,
        held.div_ceil(1024)
    );
    Ok(())
}

/// The oracle's digest16: 64-bit FNV-1a over the f16 bits packed four to a u64 word (little-endian).
fn digest16(x: &[u16]) -> u64 {
    x.as_chunks::<4>().0.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
        (h ^ (p[0] as u64 | (p[1] as u64) << 16 | (p[2] as u64) << 32 | (p[3] as u64) << 48)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One line in `bench-*`'s shape.
fn bench_line(name: &str, threads: usize, bm: &measure::Bench, heap: usize, extra: &str) {
    let opt = |v: Option<String>| v.unwrap_or_else(|| "n/a".into());
    println!(
        "{name} threads {threads} wall_best_ms {:.4} cpu_ms_per_call {} cpu_reps {} heap_peak_kb {} rss_peak_delta_kb {}{extra}",
        bm.wall_best_ms,
        opt(bm.cpu_ms_per_call.map(|c| format!("{c:.4}"))),
        bm.cpu_reps,
        heap.div_ceil(1024),
        opt(bm.rss_peak_delta_kb.map(|v| v.to_string()))
    );
}

/// `voaice bench-attn <model> <wav>` — the same measurement as `whisper_oracle --bench-attn`: block 0's flash
/// attention on Q, K, V computed beforehand from the WAV; its output and the widened K/V scratch allocated by the
/// first call and kept (as the reference's graph keeps its tensors and work buffer).
fn bench_attn(model: &str, wavp: &str, threads: usize) -> Result<(), String> {
    let m = Model::load_pinned(Path::new(model))?;
    let pcm = wav::read(Path::new(wavp))?;
    let t = mel::Tables::new();
    let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
    let n_ctx = m.hparams.n_audio_ctx as usize;
    let x = conv::ConvStage::new(&m)?.run(&mel.data, mel.n_len, 0, 2 * n_ctx, threads);
    let b = matmul::Block::new(&m, 0)?;
    drop(m);
    let (mut q, mut k16, mut v16) = (vec![0.0f32; x.len()], vec![0u16; x.len()], vec![0u16; x.len()]);
    b.qkv_into(&x, threads, &mut q, &mut k16, &mut v16, matmul::QkvTaps::default());
    let a = attention::Attention::new(b.q.n, attention::n_kv_pad(n_ctx))?;
    let (mut out, mut scratch): (Vec<f32>, Vec<f32>) = Default::default();
    let mut heap_peak = None;
    let bm = measure::bench(
        || {
            let live = LIVE.load(Relaxed);
            PEAK.store(live, Relaxed);
            if out.is_empty() {
                out = vec![0.0; q.len()];
            }
            a.run_into(&q, &k16, &v16, threads, &mut scratch, &mut out);
            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
        },
        10,
        1.0,
    );
    bench_line("bench-attn", threads, &bm, heap_peak.unwrap_or(0), "");
    Ok(())
}

/// `voaice bench-encode <model> <wav>` — the same measurement as `whisper_oracle --bench-encode`: the whole encoder
/// from the mel (computed beforehand, as the reference's state holds it) to `embd_enc`. The buffers between stages are
/// allocated by the first call and kept (an [`encoder::EncoderBuffers`] the caller owns); the widened weights are held
/// with the model, outside the call, as the reference's f16 weights are.
fn bench_encode(model: &str, wavp: &str, threads: usize) -> Result<(), String> {
    let m = Model::load_pinned(Path::new(model))?;
    let pcm = wav::read(Path::new(wavp))?;
    let t = mel::Tables::new();
    let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
    let enc = encoder::Encoder::new(&m)?;
    drop(m);
    let mut buf = encoder::EncoderBuffers::default();
    let mut out: Vec<f32> = Vec::new();
    let mut heap_peak = None;
    let bm = measure::bench(
        || {
            let live = LIVE.load(Relaxed);
            PEAK.store(live, Relaxed);
            if out.is_empty() {
                out = vec![0.0; enc.n_ctx * enc.n_state];
            }
            enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut out);
            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
        },
        10,
        2.0,
    );
    bench_line("bench-encode", threads, &bm, heap_peak.unwrap_or(0), &format!(" embd_enc_digest {:016x}", digest32(&out)));
    Ok(())
}

/// `voaice decin <model> <token,...> [--pos N]` — the decoder's input for those tokens at positions N, N+1, ... (a
/// prompt batch when N is 0): per row the token, its position, whether whisper copies its logits out, and the oracle's
/// digest32 of the row.
fn decin_cmd(model: &str, toks: &str, pos: Option<&String>) -> Result<(), String> {
    let m = Model::load_pinned(Path::new(model))?;
    let d = decoder::DecoderInput::new(&m)?;
    let tokens = toks.split(',').map(|t| t.trim().parse::<i32>().map_err(|e| format!("token {t}: {e}"))).collect::<Result<Vec<_>, _>>()?;
    let n_past = pos.map(|p| p.parse::<i32>().map_err(|e| format!("--pos: {e}"))).transpose()?.unwrap_or(0);
    let mut b = decoder::Batch::with_capacity(tokens.len());
    b.prep_legacy(&tokens, n_past, 0);
    let mut out = Vec::new();
    d.run_batch(&b, &mut out)?;
    for (i, row) in out.chunks_exact(d.n_state).enumerate() {
        println!("row {i} token {} pos {} logits {} digest {:016x}", b.token[i], b.pos[i], b.logits[i] as u8, digest32(row));
    }
    Ok(())
}

/// `voaice bench-decin <model> <n_tokens>` — the same measurement as `whisper_oracle --bench-decin`: the same tokens
/// (token i = (i · 7919 + 50257) mod n_vocab at position i), one call = the batch prepared (whisper_batch_prep_legacy)
/// and the rows computed into the caller's buffer, kept between calls. A call is microseconds, so wall is the mean over
/// at least a second of calls, not a best of 10; CPU per call over the same loop.
fn bench_decin(model: &str, n: &str) -> Result<(), String> {
    let n: usize = n.parse().map_err(|e| format!("n_tokens: {e}"))?;
    let m = Model::load_pinned(Path::new(model))?;
    let d = decoder::DecoderInput::new(&m)?;
    let n_vocab = m.hparams.n_vocab as i64;
    drop(m);
    let tokens: Vec<i32> = (0..n as i64).map(|i| ((i * 7919 + 50257) % n_vocab) as i32).collect();
    let mut b = decoder::Batch::with_capacity(n);
    let mut out = Vec::with_capacity(n * d.n_state);
    let mut call = || -> Result<(), String> {
        b.prep_legacy(&tokens, 0, 0);
        d.run_batch(&b, &mut out)
    };
    call()?;
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    call()?; // a call after the first: what it allocates (the buffers are the caller's, sized by the first)
    let heap = PEAK.load(Relaxed) - live;
    let (c0, w0) = (measure::cpu_seconds(), std::time::Instant::now());
    let mut reps = 0u64;
    while reps < 1000 || w0.elapsed().as_secs_f64() < 1.0 {
        call()?;
        reps += 1;
    }
    let (w1, c1) = (w0.elapsed().as_secs_f64(), measure::cpu_seconds());
    let cpu = match (c0, c1) {
        (Some(a), Some(b)) => format!("{:.4}", (b - a) * 1e6 / reps as f64),
        _ => "n/a".into(),
    };
    println!(
        "bench-decin threads 1 n_tokens {n} wall_us_per_call {:.4} cpu_us_per_call {cpu} reps {reps} heap_per_call_bytes {heap} held_kb {} out_digest {:016x}",
        w1 * 1e6 / reps as f64,
        d.bytes().div_ceil(1024),
        digest32(&out)
    );
    Ok(())
}

/// The oracle's digest16 over the whole cache, k then v (padding rows included): `whisper_oracle --bench-cross` and
/// `--bench-encode` print the same of the state's kv_cross.
fn kv_digest(kv: &cross::KvCross) -> u64 {
    let all: Vec<u16> = kv.k.iter().chain(&kv.v).copied().collect();
    all.as_chunks::<4>().0.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, p| {
        (h ^ (p[0] as u64 | (p[1] as u64) << 16 | (p[2] as u64) << 32 | (p[3] as u64) << 48)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// `voaice bench-cross <model> <wav> cross|whole` — `cross`: the same measurement as `whisper_oracle --bench-cross`,
/// every decoder layer's K and V from `embd_enc` (computed beforehand) into the cache; `whole`: the same as
/// `whisper_oracle --bench-encode` — whisper_encode_with_state runs the conv graph, the encoder graph AND the cross
/// graph — so mel -> embd_enc -> kv_cross. The cache and the encoder's buffers are allocated by the first call and
/// kept (the reference's are allocated by whisper_init_state); the widened weights are held outside the call.
fn bench_cross(model: &str, wavp: &str, whole: bool, threads: usize) -> Result<(), String> {
    let m = Model::load_pinned(Path::new(model))?;
    let pcm = wav::read(Path::new(wavp))?;
    let t = mel::Tables::new();
    let mel = mel::MelPlan::new(&t, &m.filters, m.filters_n_mel as usize, m.filters_n_fft as usize)?.run(&pcm, 1)?;
    let enc = encoder::Encoder::new(&m)?;
    let cr = cross::Cross::new(&m)?;
    drop(m);
    let mut buf = encoder::EncoderBuffers::default();
    let mut e: Vec<f32> = if whole { Vec::new() } else { enc.encode(&mel.data, mel.n_len, 0, threads) };
    let mut kv = cross::KvCross::default();
    let mut heap_peak = None;
    let bm = measure::bench(
        || {
            let live = LIVE.load(Relaxed);
            PEAK.store(live, Relaxed);
            if whole {
                if e.is_empty() {
                    e = vec![0.0; enc.n_ctx * enc.n_state];
                }
                enc.encode_into(&mel.data, mel.n_len, 0, threads, &mut buf, &mut e);
            }
            cr.run_into(&e, threads, &mut kv, None);
            heap_peak.get_or_insert(PEAK.load(Relaxed) - live);
        },
        10,
        if whole { 2.0 } else { 1.0 },
    );
    let extra = format!("{} kv_cross_digest {:016x}", if whole { format!(" embd_enc_digest {:016x}", digest32(&e)) } else { String::new() }, kv_digest(&kv));
    bench_line(if whole { "bench-cross what whole" } else { "bench-cross what cross" }, threads, &bm, heap_peak.unwrap_or(0), &extra);
    Ok(())
}

/// `voaice bench-f16 init|rows` — the same measurements as `whisper_oracle --bench-f16`, in a fresh process each:
/// `init` times the first build of the GELU table (the reference's `ggml_cpu_init` also builds its other tables),
/// `rows` the f32↔f16 rows on 384 × 1500 values and the GELU op on 1536 × 1500, best of 10, on the same inputs.
fn bench_f16(what: &str) -> Result<(), String> {
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    if what == "init" {
        let t = std::time::Instant::now();
        let g = gelu::Gelu::new();
        let first = ms(t);
        std::hint::black_box(&g);
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            std::hint::black_box(gelu::Gelu::new());
            best = best.min(ms(t));
        }
        println!("bench-f16 init_ms {first:.3} init_best_of_10_ms {best:.3}");
        return Ok(());
    }
    if what != "rows" {
        return Err("bench-f16 init | rows".into());
    }
    let (n, ng) = (384 * 1500, 1536 * 1500);
    let mut s: u64 = 1;
    let x: Vec<f32> = (0..ng)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as i64 % 2000001 - 1000000) as f32 * 1e-5
        })
        .collect();
    let mut y = vec![0f32; ng];
    let mut h = vec![0u16; n];
    let best = |f: &mut dyn FnMut()| {
        f();
        (0..10).map(|_| {
            let t = std::time::Instant::now();
            f();
            ms(t)
        }).fold(f64::INFINITY, f64::min)
    };
    let to16 = best(&mut || f16::fp32_to_fp16_row(&x[..n], &mut h));
    let to32 = best(&mut || f16::fp16_to_fp32_row(&h, &mut y[..n]));
    let g = gelu::Gelu::new();
    let ge = best(&mut || g.row(&x, &mut y));
    let ges = best(&mut || g.row_scalar(&x, &mut y));
    println!("bench-f16 fp32_to_fp16_row_ms {to16:.3} fp16_to_fp32_row_ms {to32:.3} gelu_ms {ge:.3} gelu_scalar_ms {ges:.3} n_row {n} n_gelu {ng}");
    Ok(())
}

/// opusinfo's "Playback length" format: minutes, seconds and milliseconds, each truncated.
fn playback_length(samples: u64) -> String {
    let t = samples as f64 / 48000.0;
    let m = (t as u64) / 60;
    let s = t as u64 - m * 60;
    let ms = ((t - (m * 60) as f64 - s as f64) * 1000.0) as u64;
    format!("{m}m:{s:02}.{ms:03}s")
}

/// `voaice opus info <file.opus>` — every field the oracle compares, read in one pass with bounded memory.
fn opus_info(path: &str) -> Result<(), String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut r = ogg::Reader::new(f).map_err(|e| format!("{path}: {e}"))?;
    let (head, tags) = (r.head().clone(), r.tags().clone());
    let mut packet_bytes = 0u64;
    while let Some(p) = r.next_packet().map_err(|e| format!("{path}: {e}"))? {
        packet_bytes += p.data.len() as u64;
    }
    let s = r.summary();
    println!("file            {path}");
    println!("bytes           {}", s.bytes);
    println!("pages           {}", s.pages);
    println!("packets         {} audio ({} bytes; largest {} bytes)", s.packets, packet_bytes, s.max_packet_bytes);
    println!("version         {}", head.version);
    println!("channels        {}", head.channels);
    println!("pre-skip        {}", head.pre_skip);
    println!("input rate      {} Hz", head.input_rate);
    println!("output gain     {} (Q7.8) = {} dB", head.gain_q8, head.gain_q8 as f64 / 256.0);
    if head.mapping_family == 0 {
        println!("mapping family  0");
    } else {
        println!("mapping family  {} ({} streams, {} coupled; mapping {:?})", head.mapping_family, head.streams, head.coupled, head.mapping);
    }
    println!("vendor          {}", tags.vendor);
    println!("comments        {}{}", tags.declared, if tags.truncated { " (truncated at the packet bound)" } else { "" });
    for c in &tags.comments {
        let shown: String = c.chars().take(120).collect();
        println!("  {shown}{}", if shown.len() < c.len() { " …" } else { "" });
    }
    println!("packet samples  {} min, {} max (48 kHz)", s.min_packet_samples, s.max_packet_samples);
    println!("start granule   {}", s.start_granule);
    println!("last granule    {}", s.last_granule);
    println!("decoded         {} samples at 48 kHz", s.decoded);
    println!("end trim        {} samples", s.end_trim);
    println!("duration        {} samples at 48 kHz ({})", s.duration, playback_length(s.duration));
    Ok(())
}

/// `voaice bench-opus <file.opus>` — measured after the oracle (the gate's step 6): the CRC on 16 MiB, sliced against
/// one byte at a time; the whole reader over the file from memory and from the file system; the heap one read needs.
fn bench_opus(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut s: u64 = 7;
    let buf: Vec<u8> = (0..16 << 20).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 33) as u8 }).collect();
    let best = |f: &mut dyn FnMut()| {
        f();
        (0..7).map(|_| { let t = std::time::Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min)
    };
    let mut c = (0, 0);
    let t8 = best(&mut || c.0 = std::hint::black_box(ogg::crc32(std::hint::black_box(&buf))));
    let t1 = best(&mut || c.1 = std::hint::black_box(ogg::crc32_bytewise(std::hint::black_box(&buf))));
    if c.0 != c.1 {
        return Err("sliced and bytewise CRCs differ".into());
    }
    let mib = buf.len() as f64 / (1 << 20) as f64;
    drop(buf);
    let read = |src: &mut dyn std::io::Read| -> Result<ogg::Summary, String> {
        ogg::Reader::new(src).and_then(|r| r.finish()).map_err(|e| e.to_string())
    };
    // the heap of one read from the file: the page buffer, the carry and the tags, nothing proportional to the file
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    let mut file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let sum = read(&mut file)?;
    drop(file);
    let heap_peak = PEAK.load(Relaxed) - live;
    let mut err = None;
    let mem = measure::bench(|| if let Err(e) = read(&mut &bytes[..]) { err = Some(e) }, 10, 1.0);
    let disk = measure::bench(|| {
        match std::fs::File::open(path) {
            Ok(mut f) => if let Err(e) = read(&mut f) { err = Some(e) },
            Err(e) => err = Some(e.to_string()),
        }
    }, 10, 1.0);
    if let Some(e) = err {
        return Err(e);
    }
    let mbs = |ms: f64| bytes.len() as f64 / (1 << 20) as f64 / (ms / 1e3);
    let opt = |v: Option<f64>| v.map(|c| format!("{c:.4}")).unwrap_or_else(|| "n/a".into());
    println!(
        "bench-opus bytes {} pages {} packets {} duration {} crc_sliced_mib_s {:.0} crc_bytewise_mib_s {:.0} crc_speedup {:.2} read_mem_ms {:.4} read_mem_mib_s {:.0} read_mem_cpu_ms {} read_file_ms {:.4} read_file_mib_s {:.0} read_file_cpu_ms {} heap_peak_bytes {} audio_s_per_cpu_s {:.0}",
        bytes.len(), sum.pages, sum.packets, sum.duration,
        mib / t8, mib / t1, t1 / t8,
        mem.wall_best_ms, mbs(mem.wall_best_ms), opt(mem.cpu_ms_per_call),
        disk.wall_best_ms, mbs(disk.wall_best_ms), opt(disk.cpu_ms_per_call),
        heap_peak,
        sum.duration as f64 / 48000.0 / (mem.cpu_ms_per_call.unwrap_or(mem.wall_best_ms) / 1e3)
    );
    Ok(())
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
