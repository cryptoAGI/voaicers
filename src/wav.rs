// SPDX-License-Identifier: MIT OR Apache-2.0
//! The audio input: 16-bit PCM, mono, 16 kHz RIFF/WAVE. Anything else is refused with a reason, not converted
//! (resampling and channel mixing are whisper-cli's miniaudio, a later stage with its own oracle).
//!
//! Samples become f32 exactly as whisper-cli's decoder makes them: miniaudio's `ma_pcm_s16_to_f32` multiplies by
//! `0.00003051757812f`, a literal that rounds to 2^-15 exactly, so `s as f32 / 32768.0` gives the same bits.

/// Read a WAV file's samples as whisper sees them.
pub fn read(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&b).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse WAV bytes.
pub fn parse(b: &[u8]) -> Result<Vec<f32>, String> {
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12;
    let mut fmt = false;
    while pos + 8 <= b.len() {
        let len = u32le(pos + 4) as usize;
        let body = pos + 8;
        if body + len > b.len() {
            return Err("truncated chunk".into());
        }
        match &b[pos..pos + 4] {
            b"fmt " => {
                if len < 16 {
                    return Err("short fmt chunk".into());
                }
                let (tag, ch, rate, bits) = (u16le(body), u16le(body + 2), u32le(body + 4), u16le(body + 14));
                if tag != 1 || ch != 1 || rate != 16000 || bits != 16 {
                    return Err(format!(
                        "need PCM, 1 channel, 16000 Hz, 16-bit; this is format {tag}, {ch} channel(s), {rate} Hz, {bits}-bit"
                    ));
                }
                fmt = true;
            }
            b"data" => {
                if !fmt {
                    return Err("data chunk before fmt".into());
                }
                return Ok(b[body..body + len - (len & 1)]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        pos = body + len + (len & 1);
    }
    Err("no data chunk".into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn s16_scale_is_miniaudios() {
        // miniaudio's literal 0.00003051757812f is 2^-15 exactly, so x * it == x / 32768 for every i16
        #[allow(clippy::excessive_precision)] // miniaudio's literal, digit for digit
        let lit: f32 = 0.000_030_517_578_12;
        assert_eq!(lit.to_bits(), (1.0f32 / 32768.0).to_bits());
        for s in [i16::MIN, -12345, -1, 0, 1, 777, i16::MAX] {
            assert_eq!((s as f32 * lit).to_bits(), (s as f32 / 32768.0).to_bits());
        }
    }

    #[test]
    fn refuses_what_it_cannot_read_exactly() {
        let mut w = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x02\0\x80\x3e\0\0\0\0\0\0\x04\0\x10\0".to_vec();
        w.extend_from_slice(b"data\x04\0\0\0\x01\0\x02\0");
        assert!(super::parse(&w).unwrap_err().contains("2 channel"));
        w[22] = 1;
        assert_eq!(super::parse(&w).unwrap().len(), 2);
    }
}
