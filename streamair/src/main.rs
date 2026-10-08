//! `streamair silence <seconds> <out.opus>` — an Ogg Opus file of exact length made of 20 ms CELT DTX frames.
//! `streamair fclone check <file.faice>...` / `streamair vclone check <file.voaice>...` — recompute each identity's
//! print and compare every stored field (the face from src/fclone.rs, the voice from voaice.rs's vclone).
//! It exercises every container rule (headers, lacing, granules, pre-skip, end trim) with no encoder at all,
//! which is what the 0.0.1 oracle feeds to opusinfo and opusdec.
use streamair::fclone;
use streamair::ogg::{mux, Stream};
use streamair::vclone;

fn check(kind: &str, files: &[String]) -> ! {
    let mut bad = 0;
    for f in files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| { eprintln!("{f}: {e}"); std::process::exit(1) });
        let line = if kind == "fclone" {
            match fclone::check_faice(&text) {
                Ok(fclone::Check::Verified(p)) => format!("verified    {}", &p.hash[..18]),
                Ok(fclone::Check::Unmeasured) => "unmeasured  -".into(),
                Ok(fclone::Check::Mismatch { fields, .. }) => { bad += 1; format!("MISMATCH    {}", fields.join(", ")) }
                Err(e) => { bad += 1; format!("ERROR       {e}") }
            }
        } else {
            match vclone::check_identity(&text) {
                Ok(vclone::Check::Verified(p)) => format!("verified    {}", p.short),
                Ok(vclone::Check::Unmeasured) => "unmeasured  -".into(),
                Ok(vclone::Check::Mismatch { fields, .. }) => { bad += 1; format!("MISMATCH    {}", fields.join(", ")) }
                Err(e) => { bad += 1; format!("ERROR       {e}") }
            }
        };
        println!("{line}  {f}");
    }
    std::process::exit(if bad > 0 { 1 } else { 0 })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() >= 4 && (a[1] == "fclone" || a[1] == "vclone") && a[2] == "check" {
        check(&a[1], &a[3..]);
    }
    if a.len() != 4 || a[1] != "silence" {
        eprintln!("usage: streamair silence <seconds> <out.opus> | streamair fclone check <file.faice>... | streamair vclone check <file.voaice>...");
        std::process::exit(2);
    }
    let secs: f64 = a[2].parse().unwrap_or_else(|_| { eprintln!("seconds must be a number"); std::process::exit(2) });
    let samples = (secs * 48000.0).round() as u64;
    let pre_skip: u16 = 312;
    let need = pre_skip as u64 + samples;
    let n = need.div_ceil(960) as usize;
    // TOC 0xF8: config 31 (CELT fullband 20 ms), mono, one frame, no payload = DTX (RFC 6716 §3.2.1)
    let packets: Vec<Vec<u8>> = (0..n).map(|_| vec![0xF8]).collect();
    let s = Stream { serial: 0x5354_4D41, channels: 1, pre_skip, input_rate: 48000, vendor: "streamair 0.0.1",
                     comments: &["ENCODER=streamair 0.0.1"], samples_48k: samples, packets_per_page: 50 };
    let f = mux(&s, &packets).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
    std::fs::write(&a[3], &f).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
    println!("{} bytes, {} packets, {} samples at 48 kHz", f.len(), n, samples);
}
