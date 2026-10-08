// SPDX-License-Identifier: MIT OR Apache-2.0
//! The log-mel front end: whisper.cpp's `log_mel_spectrogram` (src/whisper.cpp at the pinned commit), reproduced
//! operation for operation so the f32 output has the same bits.
//!
//! What "the same arithmetic" means here, established from the shipped `libwhisper.so` (testing/oracle):
//! - libwhisper is compiled `-O3` for baseline x86-64 (no `-march`), so there are no FMA instructions in it
//!   (`objdump -d libwhisper.so | grep -c vfmadd` = 0) and no vector float reductions: every `a*b + c` is two
//!   roundings, evaluated left to right. Rust does not contract either, so plain `*` and `+` are the reference.
//! - Its libm imports for this path are `sincosf` (GCC fused the table's `sinf`/`cosf` pair), `cosf` (the Hann window)
//!   and `log10` (double). This module calls those same glibc symbols, declared by hand, so the tables and the
//!   logarithm are the reference's own. (In-crate, correctly-matched versions are on the roadmap, for portability.)
//! - The FFT is whisper.cpp's: a radix-2 recursion 400 → 200 → 100 → 50 → 25, and a naive DFT at the odd size 25
//!   whose twiddle is `table[(k*n*step) % 400]`; f32 accumulators, the butterfly's terms added in source order.
//! - Power `re*re + im*im` in f32; each mel band sums four f32 products in f32, `((p0+p1)+p2)+p3`, then adds that to an
//!   f64 accumulator; the 201st bin is added alone. `log10(max(sum, 1e-10))` in f64, stored as f32.
//! - Clamp and scale in f64 against the f32 values: `max = max(mel) - 8`, `x = max(x, max)`, `x = (x + 4) / 4`.
//! - Padding: 200 samples reflected at the start (samples[200..=1]), the audio, then 30 s + 200 samples of zeros.
//!   Frames past the audio are computed only up to `(n + 200) / 160`; the rest are `log10(1e-10)` directly.
//! - Threads: whisper gives frame i to thread i % n, each frame computed whole by one thread, so the count does not
//!   change the bits (the oracle checks 1 against 4 threads). This port splits frames into contiguous runs, one per
//!   thread, each frame still computed whole by one thread: the bits do not depend on `threads` (tested on all 8 inputs).
//!
//! 0.0.2 — the same arithmetic, without the waste (every float operation, and its order, is the reference's; only
//! integer and memory work changed):
//! - no heap allocation per frame or per FFT level: the recursion is unrolled bottom-up — the 16 leaf DFTs read their
//!   input by stride from one 400-float frame, and every butterfly runs in place (the recursion lays a sub-problem out
//!   as [even half's spectrum | odd half's spectrum], and butterfly k reads and writes exactly those two slots), so
//!   0.0.1's four Vecs per level per frame are gone; the spectrum is held split (re | im) so butterfly k is lane k of
//!   a vector, with each level's twiddles `cos[k*step]`, `-sin[k*step]` gathered once into contiguous tables;
//! - no 30-s padded copy of the audio: a frame only ever reads the reflected head and the audio itself;
//! - the 25-point DFT's twiddles `table[(k*n*16) % 400]` are gathered once into [n][k] tables, and the 25 outputs are
//!   computed side by side: each lane is one k, accumulated over n in the reference's order (vector lanes are
//!   independent, nothing is reassociated); on x86-64 with AVX2 the same loop is compiled 8 lanes wide (mul and add
//!   only — FMA is a separate target feature, not enabled, and Rust does not contract `a * b + c` regardless);
//!   (lanes across the 16 leaves instead — their inputs x[s + 16n] are contiguous — was tried and measured slower);
//! - a mel band skips the groups of four bins, before its first and after its last non-zero filter value, whose
//!   filter is all zero: such a group sums to +0 (`p >= 0`, `p * 0 = +0`), and adding +0 to the f64 sum is exact, so the bits are the same — unless a power is not finite
//!   (`inf * 0 = NaN`), in which case that frame takes the full loop;
//! - `log10(std::max(sum, 1e-10))` keeps C++'s `max` (a NaN sum stays NaN; Rust's `f64::max` would return 1e-10).

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
const SIN_COS_N: usize = N_FFT;

// glibc libm, the same symbols libwhisper.so imports (`objdump -T`)
extern "C" {
    fn sincosf(x: f32, s: *mut f32, c: *mut f32);
    fn cosf(x: f32) -> f32;
    fn log10(x: f64) -> f64;
}

/// whisper_global_cache: the sin/cos table and the periodic Hann window.
pub struct Tables {
    pub sin: [f32; SIN_COS_N],
    pub cos: [f32; SIN_COS_N],
    pub hann: [f32; N_FFT],
}

impl Tables {
    pub fn new() -> Tables {
        let mut t = Tables { sin: [0.0; SIN_COS_N], cos: [0.0; SIN_COS_N], hann: [0.0; N_FFT] };
        for i in 0..SIN_COS_N {
            // double theta = (2 * M_PI * i) / SIN_COS_N_COUNT; sinf(theta), cosf(theta): theta narrowed to float
            let theta = (2.0 * std::f64::consts::PI * i as f64) / SIN_COS_N as f64;
            let (mut s, mut c) = (0.0f32, 0.0f32);
            unsafe { sincosf(theta as f32, &mut s, &mut c) };
            t.sin[i] = s;
            t.cos[i] = c;
        }
        for i in 0..N_FFT {
            // output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));  periodic: offset = 0
            let arg = (2.0 * std::f64::consts::PI * i as f64) / N_FFT as f64;
            let c = unsafe { cosf(arg as f32) } as f64;
            t.hann[i] = (0.5 * (1.0 - c)) as f32;
        }
        t
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}


/// Below this many audio frames per thread, extra threads cost more to spawn than they save (measured: 30 frames
/// on 4 threads took longer than on 1).
const MIN_FRAMES_PER_THREAD: usize = 128;

/// The leaf of the recursion: 400 / 2^4.
const LEAF: usize = 25;
/// The leaf's outputs, padded to a multiple of 8 lanes.
const LANES: usize = 32;

/// What is invariant across frames and calls: the window, the butterfly twiddles, the leaf DFT's gathered twiddles,
/// and each mel band's non-zero span of the filterbank. Build once per model; [`MelPlan::run`] then allocates only
/// its output.
pub struct MelPlan<'f> {
    hann: [f32; N_FFT],
    /// dft_cos[n][k] = cos[(k*n*16) % 400], zero past k = 24
    dft_cos: [[f32; LANES]; LEAF],
    dft_sin: [[f32; LANES]; LEAF],
    /// per butterfly level (n = 50, 100, 200, 400): cos[k*400/n] and -sin[k*400/n], k < n/2
    tw_re: [[f32; N_FFT / 2]; 4],
    tw_im: [[f32; N_FFT / 2]; 4],
    filters: &'f [f32],
    n_mel: usize,
    n_fft: usize,
    /// per band: the groups of four bins [g0, g1) that hold a non-zero filter value, and whether bin 200 does
    spans: Vec<(usize, usize, bool)>,
}

impl<'f> MelPlan<'f> {
    pub fn new(t: &Tables, filters: &'f [f32], n_mel: usize, n_fft: usize) -> Result<MelPlan<'f>, String> {
        if n_fft != 1 + N_FFT / 2 {
            return Err(format!("filterbank has {n_fft} bins; whisper's FFT gives {}", 1 + N_FFT / 2));
        }
        if filters.len() != n_mel * n_fft {
            return Err("filterbank size does not match n_mel * n_fft".into());
        }
        let step = SIN_COS_N / LEAF;
        let mut dft_cos = [[0.0f32; LANES]; LEAF];
        let mut dft_sin = [[0.0f32; LANES]; LEAF];
        for n in 0..LEAF {
            for k in 0..LEAF {
                let idx = (k * n * step) % SIN_COS_N;
                dft_cos[n][k] = t.cos[idx];
                dft_sin[n][k] = t.sin[idx];
            }
        }
        let mut tw_re = [[0.0f32; N_FFT / 2]; 4];
        let mut tw_im = [[0.0f32; N_FFT / 2]; 4];
        for (level, n) in [2 * LEAF, 4 * LEAF, 8 * LEAF, 16 * LEAF].into_iter().enumerate() {
            for k in 0..n / 2 {
                tw_re[level][k] = t.cos[k * (SIN_COS_N / n)];
                tw_im[level][k] = -t.sin[k * (SIN_COS_N / n)];
            }
        }
        let groups = n_fft / 4; // the reference's `k < n_fft - 3` loop: 50 groups, then bin 200 alone
        let spans = (0..n_mel)
            .map(|j| {
                let f = &filters[j * n_fft..(j + 1) * n_fft];
                let nz = |g: usize| f[4 * g..4 * g + 4].iter().any(|&v| v != 0.0);
                let g0 = (0..groups).find(|&g| nz(g)).unwrap_or(groups);
                let g1 = (0..groups).rev().find(|&g| nz(g)).map_or(g0, |g| g + 1);
                (g0, g1, f[4 * groups..].iter().any(|&v| v != 0.0))
            })
            .collect();
        Ok(MelPlan { hann: t.hann, dft_cos, dft_sin, tw_re, tw_im, filters, n_mel, n_fft, spans })
    }

    /// The log-mel spectrogram of 16 kHz mono f32 samples, frames split across `threads` (0 is taken as 1).
    pub fn run(&self, samples: &[f32], threads: usize) -> Result<Mel, String> {
        self.run_impl::<false>(samples, threads)
    }

    fn run_impl<const FMA: bool>(&self, samples: &[f32], threads: usize) -> Result<Mel, String> {
        let n = samples.len();
        let pad2 = N_FFT / 2; // stage_2_pad
        // whisper.cpp reads samples[1..=200] for the reflective pad and does not check the length: fewer than 201
        // samples is undefined behaviour there, so it is refused here
        if n < pad2 + 1 {
            return Err(format!("need at least {} samples (whisper.cpp reads samples[1..=200] for its pad); got {n}", pad2 + 1));
        }
        let pad1 = SAMPLE_RATE * 30; // stage_1_pad
        let n_len = (n + pad1 + 2 * pad2 - N_FFT) / HOP;
        let n_len_org = 1 + (n + pad2 - N_FFT) / HOP; // n >= 201: no negative division
        let n_mel = self.n_mel;
        let mut data = vec![0.0f32; n_mel * n_len];
        let n_audio = ((n + pad2) / HOP + 1).min(n_len);

        // a thread is worth its spawn only with work to do: at least MIN_FRAMES_PER_THREAD frames each (which thread
        // computes a frame never changes its bits, so this is only about time)
        let threads = threads.clamp(1, n_audio.div_ceil(MIN_FRAMES_PER_THREAD).max(1));
        let out = SharedOut(data.as_mut_ptr());
        let per = n_audio.div_ceil(threads);
        let mut mmax = -1e20f64;
        if threads == 1 {
            mmax = self.frames::<FMA>(samples, 0, n_audio, n_len, &out);
        } else {
            let out = &out;
            let maxes: Vec<f64> = std::thread::scope(|s| {
                let hs: Vec<_> = (1..threads)
                    .map(|w| {
                        let (a, b) = ((w * per).min(n_audio), ((w + 1) * per).min(n_audio));
                        s.spawn(move || self.frames::<FMA>(samples, a, b, n_len, out))
                    })
                    .collect();
                let mut v = vec![self.frames::<FMA>(samples, 0, per.min(n_audio), n_len, out)];
                v.extend(hs.into_iter().map(|h| h.join().expect("mel worker panicked")));
                v
            });
            for m in maxes {
                if m > mmax {
                    mmax = m;
                }
            }
        }
        // frames past the audio: the reference's `log10(1e-10)` without an FFT
        if n_audio < n_len {
            let floor = unsafe { log10(1e-10) } as f32;
            for row in data.chunks_exact_mut(n_len) {
                row[n_audio..].fill(floor);
            }
            if (floor as f64) > mmax {
                mmax = floor as f64;
            }
        }
        // clamping and normalization, in double against the float values. The max above is the reference's
        // `if (x > mmax) mmax = x` over every value: a max does not depend on the order it is taken in.
        mmax -= 8.0;
        let clamp = |part: &mut [f32]| {
            for v in part.iter_mut() {
                if (*v as f64) < mmax {
                    *v = mmax as f32;
                }
                *v = ((*v as f64 + 4.0) / 4.0) as f32;
            }
        };
        if threads == 1 {
            clamp(&mut data);
        } else {
            let chunk = data.len().div_ceil(threads);
            std::thread::scope(|s| {
                let mut parts = data.chunks_mut(chunk);
                let first = parts.next();
                for p in parts {
                    s.spawn(move || clamp(p));
                }
                if let Some(p) = first {
                    clamp(p);
                }
            });
        }
        Ok(Mel { n_mel, n_len, n_len_org, data })
    }

    /// Frames [a, b): FFT, power, mel bands, log10, written into column i of `out`. Returns the max written value
    /// as the reference compares it (f32 widened, NaN never taken).
    fn frames<const FMA: bool>(&self, samples: &[f32], a: usize, b: usize, n_len: usize, out: &SharedOut) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if !FMA && std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2 (checked above)
            return unsafe { self.frames_avx2(samples, a, b, n_len, out) };
        }
        self.frames_body::<FMA>(samples, a, b, n_len, out)
    }

    /// The same loop compiled with AVX2 enabled: wider vectors of the same multiplies and adds (no FMA: that is a
    /// separate target feature, not enabled, and Rust does not contract `a * b + c` regardless).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn frames_avx2(&self, samples: &[f32], a: usize, b: usize, n_len: usize, out: &SharedOut) -> f64 {
        self.frames_body::<false>(samples, a, b, n_len, out)
    }

    #[inline(always)]
    fn frames_body<const FMA: bool>(&self, samples: &[f32], a: usize, b: usize, n_len: usize, out: &SharedOut) -> f64 {
        let n_ext = samples.len() + N_FFT / 2; // the worker's n_samples
        let (n_mel, n_fft) = (self.n_mel, self.n_fft);
        let groups = n_fft / 4;
        let mut frame = [0.0f32; N_FFT];
        let mut sre = [0.0f32; N_FFT];
        let mut sim = [0.0f32; N_FFT];
        let mut power = [0.0f32; 1 + N_FFT / 2];
        let mut mmax = -1e20f64;
        for i in a..b {
            let offset = i * HOP;
            let m = N_FFT.min(n_ext - offset);
            // padded[offset + j]: the reflected head samples[200 - p] for p < 200, else samples[p - 200]
            if offset >= N_FFT / 2 {
                let src = &samples[offset - N_FFT / 2..offset - N_FFT / 2 + m];
                for ((x, &h), &s) in frame[..m].iter_mut().zip(&self.hann[..m]).zip(src) {
                    *x = h * s;
                }
            } else {
                for (j, x) in frame[..m].iter_mut().enumerate() {
                    let p = offset + j;
                    let s = if p < N_FFT / 2 { samples[N_FFT / 2 - p] } else { samples[p - N_FFT / 2] };
                    *x = self.hann[j] * s;
                }
            }
            frame[m..].fill(0.0);
            self.fft::<FMA>(&frame, &mut sre, &mut sim);
            for ((p, &re), &im) in power.iter_mut().zip(&sre).zip(&sim) {
                *p = re * re + im * im;
            }
            let finite = power.iter().all(|p| p.is_finite());
            for j in 0..n_mel {
                let f = &self.filters[j * n_fft..(j + 1) * n_fft];
                let (g0, g1, tail) = if finite { self.spans[j] } else { (0, groups, true) };
                let mut sum = 0.0f64;
                for g in g0..g1 {
                    let k = 4 * g;
                    // four products summed in float, left to right, then widened
                    let s4: f32 = power[k] * f[k] + power[k + 1] * f[k + 1] + power[k + 2] * f[k + 2] + power[k + 3] * f[k + 3];
                    sum += s4 as f64;
                }
                if tail {
                    for k in 4 * groups..n_fft {
                        sum += (power[k] * f[k]) as f64;
                    }
                }
                // std::max(sum, 1e-10): `sum < 1e-10 ? 1e-10 : sum`
                let v = unsafe { log10(if sum < 1e-10 { 1e-10 } else { sum }) } as f32;
                // SAFETY: (j, i) is inside data, and frame i belongs to this call alone
                unsafe { *out.0.add(j * n_len + i) = v };
                if (v as f64) > mmax {
                    mmax = v as f64;
                }
            }
        }
        mmax
    }

    /// whisper.cpp's recursive FFT, unrolled bottom-up with the same arithmetic. The recursion lays each sub-problem
    /// out as [its even half's spectrum | its odd half's spectrum], so the 16 leaves land in 25-bin blocks (block r
    /// holds the leaf that starts at sample bitrev4(r), stride 16), and each level then runs the reference's
    /// butterflies in place on consecutive blocks of n bins: sizes 50, 100, 200, 400. The spectrum is held split
    /// (`re`, `im`) rather than interleaved, so butterfly k of a level is lane k of a vector: lanes never mix.
    #[inline(always)]
    fn fft<const FMA: bool>(&self, x: &[f32; N_FFT], re: &mut [f32; N_FFT], im: &mut [f32; N_FFT]) {
        const LEAVES: usize = N_FFT / LEAF; // 16
        for r in 0..LEAVES {
            let start = r.reverse_bits() >> (usize::BITS - 4);
            let o = LEAF * r;
            self.dft::<FMA>(x, start, LEAVES, &mut re[o..o + LEAF], &mut im[o..o + LEAF]);
        }
        let mut n = 2 * LEAF;
        let mut level = 0;
        while n <= N_FFT {
            let (tc, ts) = (&self.tw_re[level], &self.tw_im[level]);
            for (br, bi) in re.chunks_exact_mut(n).zip(im.chunks_exact_mut(n)) {
                Self::butterflies::<FMA>(n / 2, br, bi, tc, ts);
            }
            n *= 2;
            level += 1;
        }
    }

    /// One recursion step on [even spectrum (half bins) | odd spectrum (half bins)], in place: butterfly k reads
    /// bins k and k + half and writes out[k] and out[k + half] — the same two bins. `tc[k] = cos[k*step]`,
    /// `ts[k] = -sin[k*step]` (the negation is exact).
    #[inline(always)]
    fn butterflies<const FMA: bool>(half: usize, br: &mut [f32], bi: &mut [f32], tc: &[f32], ts: &[f32]) {
        let (lr, hr) = br.split_at_mut(half);
        let (li, hi) = bi.split_at_mut(half);
        let (tc, ts) = (&tc[..half], &ts[..half]);
        let (hr, hi) = (&mut hr[..half], &mut hi[..half]);
        for k in 0..half {
            let (re, im) = (tc[k], ts[k]);
            let (er, ei) = (lr[k], li[k]);
            let (re_odd, im_odd) = (hr[k], hi[k]);
            if FMA {
                lr[k] = (-im).mul_add(im_odd, re.mul_add(re_odd, er));
                li[k] = im.mul_add(re_odd, re.mul_add(im_odd, ei));
                hr[k] = im.mul_add(im_odd, (-re).mul_add(re_odd, er));
                hi[k] = (-im).mul_add(re_odd, (-re).mul_add(im_odd, ei));
            } else {
                // source order: a + b*c - d*e == (a + b*c) - d*e, each product rounded, no fusion
                lr[k] = er + re * re_odd - im * im_odd;
                li[k] = ei + re * im_odd + im * re_odd;
                hr[k] = er - re * re_odd + im * im_odd;
                hi[k] = ei - re * im_odd - im * re_odd;
            }
        }
    }

    /// The naive 25-point DFT, every k at once: lane k accumulates over n in order, `re += x*cos; im -= x*sin`.
    #[inline(always)]
    fn dft<const FMA: bool>(&self, x: &[f32; N_FFT], start: usize, stride: usize, out_re: &mut [f32], out_im: &mut [f32]) {
        let mut re = [0.0f32; LANES];
        let mut im = [0.0f32; LANES];
        for n in 0..LEAF {
            let v = x[start + n * stride];
            let (c, s) = (&self.dft_cos[n], &self.dft_sin[n]);
            for k in 0..LANES {
                if FMA {
                    re[k] = v.mul_add(c[k], re[k]);
                    im[k] = (-v).mul_add(s[k], im[k]);
                } else {
                    re[k] += v * c[k];
                    im[k] -= v * s[k];
                }
            }
        }
        out_re.copy_from_slice(&re[..LEAF]);
        out_im.copy_from_slice(&im[..LEAF]);
    }
}

/// The output pointer the workers share; each writes only its own frames' columns.
struct SharedOut(*mut f32);
// SAFETY: workers write disjoint (band, frame) cells, and the scope joins them before `data` is read again
unsafe impl Sync for SharedOut {}

/// A log-mel spectrogram as whisper.cpp holds it (`whisper_mel`): `data` is [n_mel][n_len], row-major.
pub struct Mel {
    pub n_mel: usize,
    pub n_len: usize,
    /// whisper's "semi-padded" length: frames that hold audio (what whisper_n_len reports)
    pub n_len_org: usize,
    pub data: Vec<f32>,
}

/// whisper.cpp's `log_mel_spectrogram` for 16 kHz mono f32 samples and a model's filterbank ([n_mel][n_fft]), one thread.
pub fn log_mel_spectrogram(t: &Tables, samples: &[f32], filters: &[f32], n_mel: usize, n_fft: usize) -> Result<Mel, String> {
    MelPlan::new(t, filters, n_mel, n_fft)?.run(samples, 1)
}

/// The same, frames split across `threads`; the bits do not depend on the count.
pub fn log_mel_spectrogram_threads(
    t: &Tables,
    samples: &[f32],
    filters: &[f32],
    n_mel: usize,
    n_fft: usize,
    threads: usize,
) -> Result<Mel, String> {
    MelPlan::new(t, filters, n_mel, n_fft)?.run(samples, threads)
}

/// The same pipeline with the FFT's multiply-adds fused, as a build of whisper.cpp with FMA contraction would run
/// it. NOT the reference: it exists so the oracle can show it tells the two float orders apart.
pub fn log_mel_spectrogram_fused_fft(t: &Tables, samples: &[f32], filters: &[f32], n_mel: usize, n_fft: usize) -> Result<Mel, String> {
    MelPlan::new(t, filters, n_mel, n_fft)?.run_impl::<true>(samples, 1)
}

#[cfg(test)]
mod port_0_0_1 {
    //! 0.0.1's port, kept verbatim as the unit tests' second witness (the oracle is the first): allocating,
    //! single-threaded, the reference's arithmetic written the obvious way.
    use super::*;
/// whisper.cpp's naive DFT (used at the odd size 25). `out` is interleaved re, im.
pub fn dft<const FMA: bool>(t: &Tables, input: &[f32], out: &mut [f32]) {
    let n_len = input.len();
    let step = SIN_COS_N / n_len;
    for k in 0..n_len {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (n, &x) in input.iter().enumerate() {
            let idx = (k * n * step) % SIN_COS_N;
            if FMA {
                re = x.mul_add(t.cos[idx], re);
                im = (-x).mul_add(t.sin[idx], im);
            } else {
                re += x * t.cos[idx];
                im -= x * t.sin[idx];
            }
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// whisper.cpp's recursive Cooley-Tukey FFT on real input. `out` is interleaved, 2*N floats.
pub fn fft<const FMA: bool>(t: &Tables, input: &[f32], out: &mut [f32]) {
    let n = input.len();
    if n == 1 {
        out[0] = input[0];
        out[1] = 0.0;
        return;
    }
    let half = n / 2;
    if n - half * 2 == 1 {
        dft::<FMA>(t, input, out);
        return;
    }
    let even: Vec<f32> = (0..half).map(|i| input[2 * i]).collect();
    let odd: Vec<f32> = (0..half).map(|i| input[2 * i + 1]).collect();
    let mut even_fft = vec![0.0f32; 2 * half];
    let mut odd_fft = vec![0.0f32; 2 * half];
    fft::<FMA>(t, &even, &mut even_fft);
    fft::<FMA>(t, &odd, &mut odd_fft);
    let step = SIN_COS_N / n;
    for k in 0..half {
        let idx = k * step;
        let re = t.cos[idx];
        let im = -t.sin[idx];
        let (re_odd, im_odd) = (odd_fft[2 * k], odd_fft[2 * k + 1]);
        let (er, ei) = (even_fft[2 * k], even_fft[2 * k + 1]);
        // source order: a + b*c - d*e  ==  (a + b*c) - d*e, each product rounded, no fusion
        if FMA {
            // what an -march=haswell build of the same source would compute (GCC contracts a + b*c into one fma)
            out[2 * k] = (-im).mul_add(im_odd, re.mul_add(re_odd, er));
            out[2 * k + 1] = im.mul_add(re_odd, re.mul_add(im_odd, ei));
            out[2 * (k + half)] = im.mul_add(im_odd, (-re).mul_add(re_odd, er));
            out[2 * (k + half) + 1] = (-im).mul_add(re_odd, (-re).mul_add(im_odd, ei));
        } else {
            out[2 * k] = er + re * re_odd - im * im_odd;
            out[2 * k + 1] = ei + re * im_odd + im * re_odd;
            out[2 * (k + half)] = er - re * re_odd + im * im_odd;
            out[2 * (k + half) + 1] = ei - re * im_odd - im * re_odd;
        }
    }
}

pub fn log_mel_spectrogram_impl<const FMA: bool>(
    t: &Tables,
    samples: &[f32],
    filters: &[f32],
    n_mel: usize,
    n_fft: usize,
) -> Result<Mel, String> {
    if n_fft != 1 + N_FFT / 2 {
        return Err(format!("filterbank has {n_fft} bins; whisper's FFT gives {}", 1 + N_FFT / 2));
    }
    if filters.len() != n_mel * n_fft {
        return Err("filterbank size does not match n_mel * n_fft".into());
    }
    let n = samples.len();
    let pad2 = N_FFT / 2; // stage_2_pad
    // whisper.cpp reads samples[1..=200] for the reflective pad and does not check the length: fewer than 201
    // samples is undefined behaviour there, so it is refused here
    if n < pad2 + 1 {
        return Err(format!("need at least {} samples (whisper.cpp reads samples[1..=200] for its pad); got {n}", pad2 + 1));
    }
    let pad1 = SAMPLE_RATE * 30; // stage_1_pad
    let mut padded = vec![0.0f32; n + pad1 + 2 * pad2];
    padded[pad2..pad2 + n].copy_from_slice(samples);
    // std::reverse_copy(samples + 1, samples + 1 + 200, padded.begin()): padded[j] = samples[200 - j]
    for j in 0..pad2 {
        padded[j] = samples[pad2 - j];
    }
    let n_len = (padded.len() - N_FFT) / HOP;
    let n_len_org = 1 + (n + pad2 - N_FFT) / HOP; // n >= 201, so n + 200 - 400 >= 1: no negative division
    let mut data = vec![0.0f32; n_mel * n_len];

    // the worker, single-threaded: frames with audio, then the all-zero tail
    let n_ext = n + pad2; // the worker's n_samples
    let mut fft_in = vec![0.0f32; N_FFT];
    let mut fft_out = vec![0.0f32; 2 * N_FFT];
    let n_audio_frames = (n_ext / HOP + 1).min(n_len);
    for i in 0..n_audio_frames {
        let offset = i * HOP;
        let m = N_FFT.min(n_ext - offset);
        for j in 0..m {
            fft_in[j] = t.hann[j] * padded[offset + j];
        }
        for v in fft_in.iter_mut().skip(m) {
            *v = 0.0;
        }
        fft::<FMA>(t, &fft_in, &mut fft_out);
        let mut power = [0.0f32; 1 + N_FFT / 2];
        for (j, p) in power.iter_mut().enumerate() {
            let (re, im) = (fft_out[2 * j], fft_out[2 * j + 1]);
            *p = re * re + im * im;
        }
        for j in 0..n_mel {
            let f = &filters[j * n_fft..(j + 1) * n_fft];
            let mut sum = 0.0f64;
            let mut k = 0;
            while k + 3 < n_fft {
                // for (k = 0; k < n_fft - 3; k += 4): four products summed in float, then widened
                let s4: f32 = power[k] * f[k] + power[k + 1] * f[k + 1] + power[k + 2] * f[k + 2] + power[k + 3] * f[k + 3];
                sum += s4 as f64;
                k += 4;
            }
            while k < n_fft {
                sum += (power[k] * f[k]) as f64;
                k += 1;
            }
            let v = unsafe { log10(sum.max(1e-10)) };
            data[j * n_len + i] = v as f32;
        }
    }
    let floor = unsafe { log10(1e-10) } as f32;
    for i in n_audio_frames..n_len {
        for j in 0..n_mel {
            data[j * n_len + i] = floor;
        }
    }

    // clamping and normalization, in double against the float values
    let mut mmax = -1e20f64;
    for &v in &data {
        if (v as f64) > mmax {
            mmax = v as f64;
        }
    }
    mmax -= 8.0;
    for v in data.iter_mut() {
        if (*v as f64) < mmax {
            *v = mmax as f32;
        }
        *v = ((*v as f64 + 4.0) / 4.0) as f32;
    }
    Ok(Mel { n_mel, n_len, n_len_org, data })
}

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft_closely() {
        // not the oracle (that is bit-exactness against libwhisper); a sanity check that the recursion is an FFT
        let t = Tables::new();
        let x: Vec<f32> = (0..N_FFT).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let mut a = vec![0.0; 2 * N_FFT];
        let mut b = vec![0.0; 2 * N_FFT];
        port_0_0_1::fft::<false>(&t, &x, &mut a);
        port_0_0_1::dft::<false>(&t, &x, &mut b);
        let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "fft vs dft differ by {worst}");
        assert_ne!(a, b, "the two orders should differ in the last bits (else this test proves nothing about order)");
    }

    #[test]
    fn hann_is_periodic() {
        let t = Tables::new();
        assert_eq!(t.hann[0], 0.0);
        assert_eq!(t.hann[200], 1.0);
        assert_eq!(t.hann[100].to_bits(), t.hann[300].to_bits());
    }

    #[test]
    fn refuses_too_short() {
        let t = Tables::new();
        let f = vec![0.0; 80 * 201];
        assert!(log_mel_spectrogram(&t, &[0.0; 200], &f, 80, 201).is_err());
        let m = log_mel_spectrogram(&t, &[0.0; 201], &f, 80, 201).unwrap();
        assert_eq!((m.n_len, m.n_len_org), (3001, 1));
    }

    /// a deterministic triangular filterbank with zero runs, like the model's (the real one is the oracle's job)
    fn bank(n_mel: usize) -> Vec<f32> {
        let mut f = vec![0.0f32; n_mel * 201];
        for j in 0..n_mel {
            let c = 2 + j * 196 / n_mel;
            for k in c.saturating_sub(3)..(c + 4).min(201) {
                f[j * 201 + k] = 0.01 * (4.0 - (k as f32 - c as f32).abs()) + j as f32 * 1e-4;
            }
        }
        f[201 * (n_mel - 1) + 200] = 0.5; // a band that uses the lone 201st bin
        f
    }

    fn audio(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761) | 1;
        (0..n)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 0.4 + 0.5 * ((i as f32) * 0.031).sin()
            })
            .collect()
    }

    #[test]
    fn same_bits_as_the_0_0_1_port_at_every_thread_count() {
        let t = Tables::new();
        let f = bank(80);
        let plan = MelPlan::new(&t, &f, 80, 201).unwrap();
        for (n, seed) in [(201, 1), (4800, 2), (12345, 3), (40000, 4)] {
            let x = audio(n, seed);
            let old = port_0_0_1::log_mel_spectrogram_impl::<false>(&t, &x, &f, 80, 201).unwrap();
            for threads in [1, 2, 3, 4, 7] {
                let new = plan.run(&x, threads).unwrap();
                assert_eq!((new.n_len, new.n_len_org), (old.n_len, old.n_len_org));
                let same = new.data.iter().zip(&old.data).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "n={n} threads={threads}: bits differ from 0.0.1's port");
            }
            let fused_old = port_0_0_1::log_mel_spectrogram_impl::<true>(&t, &x, &f, 80, 201).unwrap();
            let fused_new = log_mel_spectrogram_fused_fft(&t, &x, &f, 80, 201).unwrap();
            assert!(fused_new.data.iter().zip(&fused_old.data).all(|(a, b)| a.to_bits() == b.to_bits()), "fused variant moved");
        }
    }

    #[test]
    fn nan_band_sum_stays_nan_like_std_max() {
        // std::max(NaN, 1e-10) is NaN in C++; f64::max would give 1e-10. An infinite sample makes NaN powers.
        let t = Tables::new();
        let f = bank(80);
        let mut x = audio(4800, 9);
        x[1000] = f32::INFINITY;
        let m = log_mel_spectrogram(&t, &x, &f, 80, 201).unwrap();
        assert!(m.data.iter().any(|v| v.is_nan()));
    }
}
