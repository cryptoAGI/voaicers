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
//!   change the bits (the oracle checks 1 against 4 threads). This port is single-threaded for now.

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

/// whisper.cpp's naive DFT (used at the odd size 25). `out` is interleaved re, im.
fn dft<const FMA: bool>(t: &Tables, input: &[f32], out: &mut [f32]) {
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
fn fft<const FMA: bool>(t: &Tables, input: &[f32], out: &mut [f32]) {
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

/// A log-mel spectrogram as whisper.cpp holds it (`whisper_mel`): `data` is [n_mel][n_len], row-major.
pub struct Mel {
    pub n_mel: usize,
    pub n_len: usize,
    /// whisper's "semi-padded" length: frames that hold audio (what whisper_n_len reports)
    pub n_len_org: usize,
    pub data: Vec<f32>,
}

/// whisper.cpp's `log_mel_spectrogram` for 16 kHz mono f32 samples and a model's filterbank ([n_mel][n_fft]).
pub fn log_mel_spectrogram(t: &Tables, samples: &[f32], filters: &[f32], n_mel: usize, n_fft: usize) -> Result<Mel, String> {
    log_mel_spectrogram_impl::<false>(t, samples, filters, n_mel, n_fft)
}

/// The same pipeline with the FFT's multiply-adds fused, as a build of whisper.cpp with FMA contraction would run
/// it. NOT the reference: it exists so the oracle can show it tells the two float orders apart.
pub fn log_mel_spectrogram_fused_fft(t: &Tables, samples: &[f32], filters: &[f32], n_mel: usize, n_fft: usize) -> Result<Mel, String> {
    log_mel_spectrogram_impl::<true>(t, samples, filters, n_mel, n_fft)
}

fn log_mel_spectrogram_impl<const FMA: bool>(
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
        fft::<false>(&t, &x, &mut a);
        dft::<false>(&t, &x, &mut b);
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
}
