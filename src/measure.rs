// SPDX-License-Identifier: MIT OR Apache-2.0
//! Efficiency, measured from the kernel's own accounting with no crates and no libc bindings (Linux `/proc`):
//! - CPU seconds: `/proc/self/stat` fields 14 and 15 (utime + stime, every thread of the process including the ones
//!   already joined), in clock ticks of 1/100 s — Linux's USER_HZ, fixed at 100 for the `/proc` ABI on x86-64 and
//!   arm64. Ten-millisecond ticks are coarse, so a caller measures a loop long enough to make them small (`bench`).
//! - peak RSS: `VmHWM` in `/proc/self/status`, after resetting the peak by writing `5` to `/proc/self/clear_refs`
//!   (Linux ≥ 4.0), so the peak measured is the call's, not the model load's before it.
//!
//! On a host without `/proc` these return `None`, and the caller says so instead of printing a number.

/// utime + stime of the whole process, in seconds.
pub fn cpu_seconds() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // field 2 (comm) is in parentheses and may hold spaces: count from the last ')'
    let rest = &s[s.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split(' ').collect();
    // rest starts at field 3 (state): utime is field 14, stime field 15
    let ticks = f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?;
    Some(ticks as f64 / 100.0)
}

fn status_kb(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with(key))?;
    line[key.len()..].trim().trim_end_matches("kB").trim().parse().ok()
}

/// Resident set size now, in KiB.
pub fn rss_kb() -> Option<u64> {
    status_kb("VmRSS:")
}

/// Peak resident set size since the last [`reset_peak_rss`], in KiB.
pub fn peak_rss_kb() -> Option<u64> {
    status_kb("VmHWM:")
}

/// Reset the peak RSS to the current RSS. Returns false if the kernel would not.
pub fn reset_peak_rss() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// What [`bench`] measured.
pub struct Bench {
    /// the fastest of `best_of` timed calls, milliseconds
    pub wall_best_ms: f64,
    /// CPU time per call (all threads), milliseconds, over `cpu_reps` calls; None without /proc
    pub cpu_ms_per_call: Option<f64>,
    pub cpu_reps: usize,
    /// peak RSS during the first call minus RSS before it, KiB; None without /proc
    pub rss_peak_delta_kb: Option<u64>,
    /// peak RSS of the process during the first call, KiB
    pub rss_peak_kb: Option<u64>,
}

/// Measure `f`: the first call for memory (peak reset just before it), then `best_of` timed calls for wall time,
/// then a loop of at least `min_cpu_s` seconds (and at least `best_of` calls) for CPU time per call.
pub fn bench<F: FnMut()>(mut f: F, best_of: usize, min_cpu_s: f64) -> Bench {
    let before = rss_kb();
    let reset = reset_peak_rss();
    f();
    let peak = if reset { peak_rss_kb() } else { None };
    let rss_peak_delta_kb = match (peak, before) {
        (Some(p), Some(b)) => Some(p.saturating_sub(b)),
        _ => None,
    };
    let mut best = f64::INFINITY;
    for _ in 0..best_of.max(1) {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let c0 = cpu_seconds();
    let t0 = std::time::Instant::now();
    let mut reps = 0;
    while reps < best_of.max(1) || t0.elapsed().as_secs_f64() < min_cpu_s {
        f();
        reps += 1;
    }
    let cpu_ms_per_call = match (c0, cpu_seconds()) {
        (Some(a), Some(b)) => Some((b - a) * 1e3 / reps as f64),
        _ => None,
    };
    Bench { wall_best_ms: best, cpu_ms_per_call, cpu_reps: reps, rss_peak_delta_kb, rss_peak_kb: peak }
}

#[cfg(test)]
mod tests {
    #[test]
    fn proc_reads() {
        if std::path::Path::new("/proc/self/stat").exists() {
            assert!(super::cpu_seconds().is_some());
            assert!(super::rss_kb().unwrap() > 0);
            assert!(super::peak_rss_kb().unwrap() >= super::rss_kb().unwrap() / 2);
        }
    }
}
