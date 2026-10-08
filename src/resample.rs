// SPDX-License-Identifier: MIT OR Apache-2.0
//! Any WAV in, whisper's 16 kHz mono f32 out, bit for bit as whisper-cli reads it (0.0.5).
//!
//! whisper-cli reads audio through `read_audio_data` (examples/common-whisper.cpp), which opens the file with
//! miniaudio 0.11.24's decoder configured `ma_decoder_config_init(ma_format_f32, 1, 16000)` and compiled into
//! libcommon `-O3 -DNDEBUG` with no `-march` (SSE2 only: no FMA can exist in it). What that decoder does to a WAV,
//! read from the pinned source and confirmed by the oracle (testing/oracle/resample_oracle.cpp, which links the very
//! libcommon.a whisper-cli is built from; testing/resample/NOTES.md has the line-by-line reading):
//!
//! 1. **Samples to f32 — dr_wav's conversions, not miniaudio's `ma_pcm_*`**: the decoder asks its WAV backend for
//!    f32 (`preferredFormat`), so every integer format is widened by `ma_dr_wav_*_to_f32`: u8 `x * (2/255)f - 1`,
//!    s16 `x * 2^-15`, s24 and s32 through a double (`x * 2^-23`, `x / 2^31`) rounded once. f32 is copied. The data
//!    chunk is clamped to what the file holds.
//! 2. **Mixdown — an average, not miniaudio's channel weights**: with one output channel and no channel map,
//!    miniaudio takes its `mono_out` path, `float t = 0; t += x[c] ...; out = t / channels` — the channel positions
//!    and the rectangular weights never apply. (Starting from +0 matters: two -0.0 channels mix to +0.0.)
//! 3. **The linear resampler** (`ma_linear_resampler`, f32, run after the mixdown at one channel): the rates reduced
//!    by their gcd; a time accumulator `inTimeInt`/`inTimeFrac` starting at 1/0; each output `x0 + (x1 - x0) * a`
//!    with `a = frac as f32 / out as f32` (miniaudio's `ma_mix_f32_fast`); and a **low-pass filter of order 4** —
//!    two biquads in transposed direct form II, Butterworth Q from `1 / (2 cos((2i + 1) π / 8))`, cutoff at half
//!    the lower reduced rate, coefficients computed in double with glibc's `sin` (miniaudio's cosine is
//!    `sin(π/2 - x)`) and rounded to f32. Downsampling filters every *input* sample before it is held; upsampling
//!    filters every *output*. The first output is always 0 (`x0` before anything is loaded).
//! 4. **The length**: `ma_decoder_get_length_in_pcm_frames`, i.e. `ma_calculate_frame_count_after_resampling` on
//!    the unreduced rates — whose rule can promise one frame more than the resampler produces; whisper-cli resizes
//!    its vector to the promise, so the missing tail stays 0.0.
//!
//! 16 kHz mono is passed through untouched; 16 kHz with more channels is mixed only.
//!
//! [`Converter`] is the streaming form: bytes of the data chunk in any chunking, state carried between pushes,
//! output identical to one whole-buffer run. [`read`] streams a file through it with one 32 KiB stack buffer, so
//! the heap holds only the output.
//!
//! Not here (miniaudio decodes them, voaice refuses by name): FLAC, MP3, Vorbis, and the WAV encodings
//! A-law, µ-law, ADPCM, f64, odd bit depths; RF64 / Wave64; the `--diarize` path (stereo kept, mixdown `L + R`).

use std::io::{Read, Seek, SeekFrom};

/// whisper's sample rate: what everything is resampled to.
pub const RATE_OUT: u32 = 16_000;
/// miniaudio's `MA_DEFAULT_RESAMPLER_LPF_ORDER` (4; `MA_MAX_FILTER_ORDER` is 8).
pub const LPF_ORDER: u32 = 4;

/// A sample format dr_wav widens to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    U8,
    S16,
    S24,
    S32,
    F32,
}

impl Sample {
    pub fn bytes(self) -> usize {
        match self {
            Sample::U8 => 1,
            Sample::S16 => 2,
            Sample::S24 => 3,
            Sample::S32 | Sample::F32 => 4,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Sample::U8 => "u8",
            Sample::S16 => "s16",
            Sample::S24 => "s24",
            Sample::S32 => "s32",
            Sample::F32 => "f32",
        }
    }
}

/// What a WAV header says, with the data chunk located and clamped to the bytes present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavFormat {
    pub rate: u32,
    pub channels: u16,
    pub sample: Sample,
    /// byte offset of the data chunk's body
    pub data_offset: u64,
    /// bytes of sample data: the chunk's size, clamped to the file
    pub data_len: u64,
}

impl WavFormat {
    pub fn block_align(&self) -> usize {
        self.sample.bytes() * self.channels as usize
    }
    /// Whole frames in the data chunk (dr_wav's `totalPCMFrameCount`).
    pub fn frames(&self) -> u64 {
        self.data_len / self.block_align() as u64
    }
    /// How many samples whisper-cli's vector holds for this file.
    pub fn output_len(&self) -> u64 {
        frame_count_after_resampling(RATE_OUT, self.rate, self.frames())
    }
}

/// miniaudio's `ma_calculate_frame_count_after_resampling`, transcribed: on the UNREDUCED rates, and not the same
/// expression as the resampler's own `get_expected_output_frame_count` (its "fractional" term divides by the output
/// rate twice), so for every rate here it adds the extra frame — one more than the resampler can make when the input
/// does not reach it. That is the length whisper-cli's vector gets.
pub fn frame_count_after_resampling(rate_out: u32, rate_in: u32, frames_in: u64) -> u64 {
    if rate_in == 0 || rate_out == 0 || frames_in == 0 {
        return 0;
    }
    if rate_out == rate_in {
        return frames_in;
    }
    let (o, i) = (rate_out as u64, rate_in as u64);
    let mut out = frames_in * o / i;
    let from_frac = (out * (i / o)) / o;
    let pre = (out * (i % o)) + from_frac;
    if pre <= frames_in {
        out += 1;
    }
    out
}

/// Parse a RIFF/WAVE header through `read_at(offset, buf)` (fills `buf` exactly or fails) over `total` bytes:
/// the `fmt ` chunk (plain or WAVE_FORMAT_EXTENSIBLE), then the `data` chunk, skipping anything between.
pub fn parse_header(
    total: u64,
    mut read_at: impl FnMut(u64, &mut [u8]) -> Result<(), String>,
) -> Result<WavFormat, String> {
    let mut h = [0u8; 12];
    if total < 12 {
        return Err("not a RIFF/WAVE file".into());
    }
    read_at(0, &mut h)?;
    if &h[0..4] != b"RIFF" || &h[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file (RF64, Wave64, FLAC, MP3 and Vorbis are not read here)".into());
    }
    let mut pos = 12u64;
    let mut fmt: Option<(u32, u16, Sample)> = None;
    while pos + 8 <= total {
        let mut c = [0u8; 8];
        read_at(pos, &mut c)?;
        let len = u32::from_le_bytes([c[4], c[5], c[6], c[7]]) as u64;
        let body = pos + 8;
        match &c[0..4] {
            b"fmt " => {
                if len < 16 || body + len > total {
                    return Err("short or truncated fmt chunk".into());
                }
                let mut f = [0u8; 40];
                let n = len.min(40) as usize;
                read_at(body, &mut f[..n])?;
                let u16le = |o: usize| u16::from_le_bytes([f[o], f[o + 1]]);
                let mut tag = u16le(0);
                let (ch, rate, align, bits) =
                    (u16le(2), u32::from_le_bytes([f[4], f[5], f[6], f[7]]), u16le(12), u16le(14));
                if tag == 0xFFFE {
                    if n < 40 {
                        return Err("WAVE_FORMAT_EXTENSIBLE with a short fmt chunk".into());
                    }
                    tag = u16le(24); // the sub-format GUID's first two bytes are the format tag
                }
                let sample = match (tag, bits) {
                    (1, 8) => Sample::U8,
                    (1, 16) => Sample::S16,
                    (1, 24) => Sample::S24,
                    (1, 32) => Sample::S32,
                    (3, 32) => Sample::F32,
                    _ => {
                        return Err(format!(
                            "format tag {tag} with {bits}-bit samples: only PCM 8/16/24/32-bit and IEEE float 32-bit are read"
                        ))
                    }
                };
                if ch == 0 || ch > 254 {
                    return Err(format!("{ch} channels (miniaudio reads 1..=254)"));
                }
                if rate == 0 {
                    return Err("sample rate 0".into());
                }
                if align as usize != sample.bytes() * ch as usize {
                    return Err(format!("block align {align} is not {} x {ch}", sample.bytes()));
                }
                fmt = Some((rate, ch, sample));
            }
            b"data" => {
                let (rate, channels, sample) = fmt.ok_or("data chunk before fmt")?;
                // dr_wav clamps the chunk to the bytes the file holds (a cut-off upload reads what is there)
                let data_len = len.min(total - body);
                return Ok(WavFormat { rate, channels, sample, data_offset: body, data_len });
            }
            _ => {}
        }
        pos = body + len + (len & 1);
    }
    Err("no data chunk".into())
}

/// One second-order section, its coefficients already divided by a0 and rounded to f32 (`ma_biquad_reinit`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// miniaudio's `ma_cosd`: the sine of the complement, through libm's `sin` (Rust's `f64::sin` is the platform
/// libm's: glibc's on the linux-gnu targets, the same function libcommon calls).
fn ma_cosd(x: f64) -> f64 {
    (std::f64::consts::PI * 0.5 - x).sin()
}

/// The low-pass cascade the linear resampler builds for reduced rates `rin -> rout` (`ma_lpf_init` with
/// `order / 2` `ma_lpf2` sections): sample rate max(rin, rout), cutoff min(rin, rout) / 2, Butterworth Q per section.
/// Odd orders (which add a first-order section) are not used by miniaudio's default and are refused.
pub fn lpf_coefficients(rin: u32, rout: u32, order: u32) -> Vec<Biquad> {
    assert!(order.is_multiple_of(2) && order <= 8, "even orders up to MA_MAX_FILTER_ORDER only");
    const PI: f64 = std::f64::consts::PI; // MA_PI_D, 3.14159265358979323846264, is this double
    let sample_rate = rin.max(rout) as f64;
    let cutoff = rin.min(rout) as f64 * 0.5 * 1.0; // lpfNyquistFactor = 1
    (0..order / 2)
        .map(|i| {
            let a = (1 + i * 2) as f64 * (PI / (order * 2) as f64);
            let q = 1.0 / (2.0 * ma_cosd(a));
            let w = 2.0 * PI * cutoff / sample_rate;
            let (s, c) = (w.sin(), ma_cosd(w));
            let al = s / (2.0 * q);
            let (b0, b1, b2) = ((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0);
            let (a0, a1, a2) = (1.0 + al, -2.0 * c, 1.0 - al);
            Biquad {
                b0: (b0 / a0) as f32,
                b1: (b1 / a0) as f32,
                b2: (b2 / a0) as f32,
                a1: (a1 / a0) as f32,
                a2: (a2 / a0) as f32,
            }
        })
        .collect()
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// miniaudio's linear resampler for one channel of f32, with its state, fed one sample at a time.
#[derive(Clone, Debug)]
pub struct Linear {
    rin: u32,
    rout: u32,
    adv_int: u32,
    adv_frac: u32,
    t_int: u32,
    t_frac: u32,
    x0: f32,
    x1: f32,
    lpf: [Biquad; 4],
    r: [[f32; 2]; 4],
    stages: usize,
}

impl Linear {
    /// The resampler whisper-cli's decoder builds (`LPF_ORDER`).
    pub fn new(rate_in: u32, rate_out: u32) -> Self {
        Self::with_order(rate_in, rate_out, LPF_ORDER)
    }

    /// Any even filter order (the discriminator uses others).
    pub fn with_order(rate_in: u32, rate_out: u32, order: u32) -> Self {
        let g = gcd(rate_in, rate_out);
        let (rin, rout) = (rate_in / g, rate_out / g);
        let c = lpf_coefficients(rin, rout, order);
        let zero = Biquad { b0: 0.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
        let mut lpf = [zero; 4];
        lpf[..c.len()].copy_from_slice(&c);
        Linear {
            rin,
            rout,
            adv_int: rin / rout,
            adv_frac: rin % rout,
            t_int: 1, // "force an input sample to always be loaded for the first output frame"
            t_frac: 0,
            x0: 0.0,
            x1: 0.0,
            lpf,
            r: [[0.0; 2]; 4],
            stages: c.len(),
        }
    }

    /// The reduced rates (in, out).
    pub fn rates(&self) -> (u32, u32) {
        (self.rin, self.rout)
    }

    /// Feed input samples; append every output they complete, stopping once `out` holds `limit` samples.
    pub fn feed(&mut self, input: impl Iterator<Item = f32>, out: &mut Vec<f32>, limit: usize) {
        match self.stages {
            0 => self.feed_n::<0>(input, out, limit),
            1 => self.feed_n::<1>(input, out, limit),
            2 => self.feed_n::<2>(input, out, limit),
            3 => self.feed_n::<3>(input, out, limit),
            _ => self.feed_n::<4>(input, out, limit),
        }
    }

    // The state lives in locals for the loop and is written back once: the loop is miniaudio's
    // `process_pcm_frames_f32_{down,up}sample` with the load and the emit interleaved per input sample, which
    // performs the same operations in the same order (a load happens exactly when inTimeInt > 0, an emit exactly when
    // it is 0) without its per-call bookkeeping.
    #[inline(always)]
    fn feed_n<const N: usize>(&mut self, input: impl Iterator<Item = f32>, out: &mut Vec<f32>, limit: usize) {
        if out.len() >= limit {
            return;
        }
        let (rout, adv_int, adv_frac, down) = (self.rout, self.adv_int, self.adv_frac, self.rin > self.rout);
        let routf = rout as f32;
        let (mut x0, mut x1, mut ti, mut tf) = (self.x0, self.x1, self.t_int, self.t_frac);
        let bq: [Biquad; N] = std::array::from_fn(|k| self.lpf[k]);
        let mut r: [[f32; 2]; N] = std::array::from_fn(|k| self.r[k]);
        // ma_biquad_process_pcm_frame_f32__direct_form_2_transposed, the sections in order
        let filter = |r: &mut [[f32; 2]; N], mut x: f32| {
            for k in 0..N {
                let b = &bq[k];
                let y = b.b0 * x + r[k][0];
                r[k][0] = b.b1 * x - b.a1 * y + r[k][1];
                r[k][1] = b.b2 * x - b.a2 * y;
                x = y;
            }
            x
        };
        'all: for s in input {
            x0 = x1;
            x1 = if down { filter(&mut r, s) } else { s };
            ti -= 1;
            while ti == 0 {
                let a = tf as f32 / routf;
                let y = x0 + (x1 - x0) * a; // ma_mix_f32_fast: r0 = y - x; r1 = r0 * a; x + r1
                out.push(if down { y } else { filter(&mut r, y) });
                ti += adv_int;
                tf += adv_frac;
                if tf >= rout {
                    tf -= rout;
                    ti += 1;
                }
                if out.len() >= limit {
                    break 'all;
                }
            }
        }
        (self.x0, self.x1, self.t_int, self.t_frac) = (x0, x1, ti, tf);
        self.r[..N].copy_from_slice(&r);
    }
}

/// u8 -> f32 as dr_wav: `x * 0.00784313725490196078f - 1`, two roundings.
#[inline(always)]
fn u8_f32(b: u8) -> f32 {
    #[allow(clippy::excessive_precision)] // dr_wav's literal, digit for digit
    const K: f32 = 0.007_843_137_254_901_960_78;
    b as f32 * K - 1.0
}
#[inline(always)]
fn s16_f32(b: &[u8]) -> f32 {
    i16::from_le_bytes([b[0], b[1]]) as f32 * (1.0 / 32768.0) // dr_wav: x * 0.000030517578125f, 2^-15 exactly
}
#[inline(always)]
fn s24_f32(b: &[u8]) -> f32 {
    let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
    (v as f64 * 0.000_000_119_209_289_550_781_25) as f32
}
#[inline(always)]
fn s32_f32(b: &[u8]) -> f32 {
    (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / 2_147_483_648.0) as f32
}
#[inline(always)]
fn f32_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The streaming converter: push the data chunk's bytes in any pieces, then [`Converter::finish`].
pub struct Converter {
    fmt: WavFormat,
    rs: Option<Linear>,
    carry: Vec<u8>,
    len: usize,
}

impl Converter {
    pub fn new(fmt: &WavFormat) -> Self {
        Self::with_order(fmt, LPF_ORDER)
    }

    /// With another low-pass order (for the discriminator; whisper-cli's is [`LPF_ORDER`]).
    pub fn with_order(fmt: &WavFormat, order: u32) -> Self {
        let rs = (fmt.rate != RATE_OUT).then(|| Linear::with_order(fmt.rate, RATE_OUT, order));
        Converter { fmt: *fmt, rs, carry: Vec::new(), len: fmt.output_len() as usize }
    }

    /// The number of samples the finished output holds.
    pub fn output_len(&self) -> usize {
        self.len
    }

    /// Convert `bytes` (the next piece of the data chunk), appending to `out`.
    pub fn push(&mut self, mut bytes: &[u8], out: &mut Vec<f32>) {
        let w = self.fmt.block_align();
        if !self.carry.is_empty() {
            let take = (w - self.carry.len()).min(bytes.len());
            self.carry.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.carry.len() < w {
                return;
            }
            let frame = std::mem::take(&mut self.carry);
            self.frames(&frame, out);
            self.carry = frame;
            self.carry.clear();
        }
        let whole = bytes.len() / w * w;
        self.frames(&bytes[..whole], out);
        self.carry.extend_from_slice(&bytes[whole..]);
    }

    /// The output, padded with zeros to the length whisper-cli's vector has (a trailing partial frame is dropped,
    /// as dr_wav drops it).
    pub fn finish(self, out: &mut Vec<f32>) {
        if out.len() < self.len {
            out.resize(self.len, 0.0);
        }
    }

    fn frames(&mut self, b: &[u8], out: &mut Vec<f32>) {
        let ch = self.fmt.channels as usize;
        match (self.fmt.sample, ch) {
            (Sample::U8, 1) => self.run(b, 1, |f| u8_f32(f[0]), out),
            (Sample::U8, 2) => self.run(b, 2, |f| mix2(u8_f32(f[0]), u8_f32(f[1])), out),
            (Sample::U8, _) => self.run(b, ch, |f| mixn(f.iter().map(|&x| u8_f32(x)), ch), out),
            (Sample::S16, 1) => self.run(b, 2, s16_f32, out),
            (Sample::S16, 2) => self.run(b, 4, |f| mix2(s16_f32(&f[..2]), s16_f32(&f[2..])), out),
            (Sample::S16, _) => self.run(b, 2 * ch, |f| mixn(f.as_chunks::<2>().0.iter().map(|c| s16_f32(c)), ch), out),
            (Sample::S24, 1) => self.run(b, 3, s24_f32, out),
            (Sample::S24, _) => self.run(b, 3 * ch, |f| mixn(f.as_chunks::<3>().0.iter().map(|c| s24_f32(c)), ch), out),
            (Sample::S32, 1) => self.run(b, 4, s32_f32, out),
            (Sample::S32, _) => self.run(b, 4 * ch, |f| mixn(f.as_chunks::<4>().0.iter().map(|c| s32_f32(c)), ch), out),
            (Sample::F32, 1) => self.run(b, 4, f32_f32, out),
            (Sample::F32, 2) => self.run(b, 8, |f| mix2(f32_f32(&f[..4]), f32_f32(&f[4..])), out),
            (Sample::F32, _) => self.run(b, 4 * ch, |f| mixn(f.as_chunks::<4>().0.iter().map(|c| f32_f32(c)), ch), out),
        }
    }

    #[inline(always)]
    fn run(&mut self, b: &[u8], w: usize, frame: impl Fn(&[u8]) -> f32, out: &mut Vec<f32>) {
        let samples = b.chunks_exact(w).map(frame);
        match &mut self.rs {
            Some(rs) => rs.feed(samples, out, self.len),
            None => {
                let room = self.len.saturating_sub(out.len());
                out.extend(samples.take(room));
            }
        }
    }
}

/// miniaudio's `mono_out` for two channels: `t = 0; t += l; t += r; t / 2`.
#[inline(always)]
fn mix2(l: f32, r: f32) -> f32 {
    (0.0 + l + r) / 2.0
}
#[inline(always)]
fn mixn(xs: impl Iterator<Item = f32>, ch: usize) -> f32 {
    let mut t = 0.0f32;
    for x in xs {
        t += x;
    }
    t / ch as f32
}

/// Read a WAV file as whisper-cli does: its 16 kHz mono f32 samples. Streams the file through a 32 KiB stack
/// buffer; the heap holds the output (allocated once, at its final length) and nothing else.
pub fn read(path: &std::path::Path) -> Result<(WavFormat, Vec<f32>), String> {
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = std::fs::File::open(path).map_err(err)?;
    let total = f.metadata().map_err(err)?.len();
    let fmt = parse_header(total, |off, buf| {
        f.seek(SeekFrom::Start(off)).map_err(err)?;
        f.read_exact(buf).map_err(err)
    })
    .map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(fmt.data_offset)).map_err(err)?;
    let mut conv = Converter::new(&fmt);
    let mut out = Vec::with_capacity(conv.output_len());
    let mut buf = [0u8; 32 * 1024];
    let mut left = fmt.data_len;
    while left > 0 {
        let want = (buf.len() as u64).min(left) as usize;
        let n = f.read(&mut buf[..want]).map_err(err)?;
        if n == 0 {
            break;
        }
        conv.push(&buf[..n], &mut out);
        left -= n as u64;
    }
    conv.finish(&mut out);
    Ok((fmt, out))
}

/// Convert WAV bytes held in memory (the same result as [`read`] on a file of those bytes).
pub fn convert(b: &[u8]) -> Result<(WavFormat, Vec<f32>), String> {
    let fmt = parse_header(b.len() as u64, |off, buf| {
        let o = off as usize;
        buf.copy_from_slice(b.get(o..o + buf.len()).ok_or("truncated header")?);
        Ok(())
    })?;
    let mut conv = Converter::new(&fmt);
    let mut out = Vec::with_capacity(conv.output_len());
    let o = fmt.data_offset as usize;
    conv.push(&b[o..o + fmt.data_len as usize], &mut out);
    conv.finish(&mut out);
    Ok((fmt, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, ch: u16, tag: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let align = ch * bits / 8;
        let mut w = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0".to_vec();
        for v in [tag, ch] {
            w.extend_from_slice(&v.to_le_bytes());
        }
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&(rate * align as u32).to_le_bytes());
        w.extend_from_slice(&align.to_le_bytes());
        w.extend_from_slice(&bits.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data.len() as u32).to_le_bytes());
        w.extend_from_slice(data);
        w
    }

    fn lcg_bytes(n: usize, mut s: u32) -> Vec<u8> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn length_rule_is_miniaudios() {
        // 48 kHz: N/3 + 1 always (the "fractional" term is tiny); 8 kHz: exactly 2N
        assert_eq!(frame_count_after_resampling(16000, 48000, 62413), 20805);
        assert_eq!(frame_count_after_resampling(16000, 48000, 1), 1);
        assert_eq!(frame_count_after_resampling(16000, 8000, 1), 2);
        assert_eq!(frame_count_after_resampling(16000, 44100, 2), 1);
        assert_eq!(frame_count_after_resampling(16000, 16000, 5), 5);
        assert_eq!(frame_count_after_resampling(16000, 44100, 0), 0);
    }

    #[test]
    fn sixteen_k_mono_passes_through() {
        // every s16 value, untouched apart from the exact 2^-15 scale; f32 bits copied, -0.0 and NaN payloads kept
        let s: Vec<u8> = (i16::MIN..=i16::MAX).flat_map(|v| v.to_le_bytes()).collect();
        let (_, y) = convert(&wav(16000, 1, 1, 16, &s)).unwrap();
        assert!(y.iter().zip(i16::MIN..=i16::MAX).all(|(&y, v)| y.to_bits() == (v as f32 / 32768.0).to_bits()));
        let bits = [0x8000_0000u32, 0x7FC0_1234, 1, 0x4080_0000];
        let f: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let (_, y) = convert(&wav(16000, 1, 3, 32, &f)).unwrap();
        assert_eq!(y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }

    #[test]
    fn mixdown_starts_from_plus_zero() {
        // miniaudio's t = 0; t += l; t += r: two -0.0 channels give +0.0, where l + r would give -0.0
        assert_eq!(mix2(-0.0, -0.0).to_bits(), 0);
        assert_eq!(mixn([-0.0f32, -0.0, -0.0].into_iter(), 3).to_bits(), 0);
        assert_eq!(mix2(0.25, 0.5), 0.375);
    }

    #[test]
    fn conversions_are_dr_wavs() {
        assert_eq!(u8_f32(0), -1.0);
        assert_eq!(u8_f32(128).to_bits(), (128.0f32 * 0.007_843_138 - 1.0).to_bits());
        assert_eq!(s24_f32(&[0, 0, 0x80]), -1.0);
        assert_eq!(s32_f32(&i32::MAX.to_le_bytes()), 1.0); // (2^31 - 1) / 2^31 rounds up to 1 in f32
        assert_eq!(s32_f32(&i32::MIN.to_le_bytes()), -1.0);
    }

    #[test]
    fn first_output_is_zero_and_filters_differ_by_order() {
        let mut a = Linear::new(48000, 16000);
        let mut o = Vec::new();
        a.feed([0.5f32; 30].into_iter(), &mut o, usize::MAX);
        assert_eq!(o[0], 0.0);
        assert_eq!(o.len(), 10);
        assert_ne!(lpf_coefficients(3, 1, 4)[0], lpf_coefficients(3, 1, 2)[0]);
    }

    #[test]
    fn streaming_equals_whole_on_random_chunks() {
        // every format and a few channel counts, at a downsampling, an upsampling and the passthrough rate; pushed
        // in random pieces (including empty ones and ones that split a frame) against one push of the whole
        let mut s = 0x2545_f491u32;
        let mut next = |m: u32| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s % m
        };
        for (tag, bits) in [(1u16, 8u16), (1, 16), (1, 24), (1, 32), (3, 32)] {
            for ch in [1u16, 2, 3] {
                for rate in [44100u32, 8000, 16000, 22050] {
                    let n = 1000 + next(500) as usize;
                    let mut data = lcg_bytes(n * (bits / 8 * ch) as usize, rate ^ bits as u32);
                    if tag == 3 {
                        // keep the floats finite and moderate: the same bytes in both runs is what matters
                        for c in data.as_chunks_mut::<4>().0 {
                            c[3] &= 0xBF;
                        }
                    }
                    let w = wav(rate, ch, tag, bits, &data);
                    let (fmt, whole) = convert(&w).unwrap();
                    let mut conv = Converter::new(&fmt);
                    let mut out = Vec::new();
                    let mut rest = &w[fmt.data_offset as usize..];
                    while !rest.is_empty() {
                        let k = (next(700) as usize).min(rest.len());
                        conv.push(&rest[..k], &mut out);
                        rest = &rest[k..];
                    }
                    conv.finish(&mut out);
                    assert_eq!(out.len(), whole.len());
                    assert!(out.iter().zip(&whole).all(|(a, b)| a.to_bits() == b.to_bits()), "{tag}/{bits} {ch}ch {rate}");
                }
            }
        }
    }

    #[test]
    fn refuses_what_it_cannot_read_exactly() {
        assert!(convert(&wav(48000, 1, 1, 12, &[0; 4])).unwrap_err().contains("12-bit"));
        assert!(convert(&wav(48000, 1, 3, 64, &[0; 8])).unwrap_err().contains("64-bit"));
        assert!(convert(&wav(48000, 1, 6, 8, &[0; 8])).unwrap_err().contains("format tag 6")); // A-law
        assert!(convert(b"RIFF\0\0\0\0WAVEdata\0\0\0\0").unwrap_err().contains("before fmt"));
        assert!(convert(b"fLaC").unwrap_err().contains("RIFF"));
    }
}
