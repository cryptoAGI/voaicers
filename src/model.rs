// SPDX-License-Identifier: MIT OR Apache-2.0
//! The whisper.cpp model file ("ggml" legacy format, as `models/convert-pt-to-ggml.py` writes it), parsed the way
//! `whisper_model_load` (src/whisper.cpp, pinned commit in upstream/PIN) reads it, and pinned by sha256.
//!
//! Layout, all little-endian, no alignment padding anywhere:
//! ```text
//! u32 magic 0x67676d6c ("ggml")
//! i32 n_vocab n_audio_ctx n_audio_state n_audio_head n_audio_layer n_text_ctx n_text_state n_text_head
//!     n_text_layer n_mels ftype                                     (ftype % 1000 = weight type, / 1000 = qnt version)
//! i32 filters.n_mel, filters.n_fft, then n_mel*n_fft f32             (the mel filterbank, [n_mel][n_fft])
//! i32 n_vocab_file, then per token: u32 len, len bytes               (n_vocab_file may be < n_vocab: extras are named)
//! tensors until EOF: i32 n_dims, i32 name_len, i32 ttype, i32 ne[n_dims], name bytes, data (ggml_nbytes bytes)
//! ```
//! Beyond what the reference checks, this loader also requires each tensor's stored type to equal the type the
//! reference creates it with (whisper.cpp only checks the byte count, so a same-size type swap would load as
//! garbage there); it refuses rather than reinterprets.

use crate::sha256;

/// A model file voaice.rs knows by its bytes. `size` and `sha256` must both match.
pub struct Pin {
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub source: &'static str,
}

/// The pins (also in upstream/PIN). Production's other model, ggml-base.en.bin (147,964,211 bytes), is not pinned
/// yet: it has not been downloaded and hashed here, and a size alone is not a pin.
pub const PINS: &[Pin] = &[Pin {
    file: "ggml-tiny.en.bin",
    size: 77_704_715,
    sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
    source: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
}];

/// whisper.cpp's g_lang, in id order (whisper_lang_str(id) is LANGS[id]).
pub const LANGS: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
];

pub const GGML_FILE_MAGIC: u32 = 0x6767_6d6c;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hparams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    /// ftype with the quantization version removed (ftype % 1000), as whisper.cpp keeps it
    pub ftype: i32,
    /// ftype / 1000
    pub qntvr: i32,
}

impl Hparams {
    /// whisper.cpp's e_model name (by n_audio_layer)
    pub fn model_type(&self) -> &'static str {
        match self.n_audio_layer {
            4 => "tiny",
            6 => "base",
            12 => "small",
            24 => "medium",
            32 => "large",
            _ => "unknown",
        }
    }
    pub fn is_multilingual(&self) -> bool {
        self.n_vocab >= 51865
    }
    /// whisper_vocab::num_languages
    pub fn num_languages(&self) -> i32 {
        self.n_vocab - 51765 - if self.is_multilingual() { 1 } else { 0 }
    }
}

/// The ggml tensor types this stage accepts (whisper models stored as f32 or f16 — production's are f16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    fn from_ggml(t: i32) -> Result<Dtype, String> {
        match t {
            0 => Ok(Dtype::F32),
            1 => Ok(Dtype::F16),
            _ => Err(format!("ggml type {t} is not supported yet (0.0.1 reads f32 and f16 models only)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
        }
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub n_dims: usize,
    /// ggml order: ne[0] is the innermost (contiguous) dimension; unused dimensions are 1
    pub ne: [i64; 4],
    /// byte offset of the data in the file
    pub offset: usize,
    pub nbytes: usize,
}

pub struct Model {
    pub hparams: Hparams,
    pub filters_n_mel: i32,
    pub filters_n_fft: i32,
    /// [n_mel][n_fft], row-major, exactly the file's floats
    pub filters: Vec<f32>,
    /// id -> token bytes, n_vocab entries (file tokens, then the extras whisper.cpp names)
    pub vocab: Vec<Vec<u8>>,
    pub n_vocab_file: usize,
    pub tensors: Vec<Tensor>,
    /// the whole file (tensor data is read from here)
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub pin: Option<&'static Pin>,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err(format!("truncated file: {what} needs {n} bytes at offset {}", self.pos));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn i32(&mut self, what: &str) -> Result<i32, String> {
        let s = self.take(4, what)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(self.i32(what)? as u32)
    }
    fn eof(&self) -> bool {
        self.pos >= self.b.len()
    }
}

/// The special token ids whisper.cpp uses (`whisper_vocab`), shifted for multilingual vocabularies as it shifts them.
#[derive(Debug, Clone, Copy)]
pub struct Specials {
    pub eot: i32,
    pub sot: i32,
    pub translate: i32,
    pub transcribe: i32,
    pub solm: i32,
    pub prev: i32,
    pub nosp: i32,
    pub not: i32,
    pub beg: i32,
}

impl Specials {
    pub fn for_hparams(h: &Hparams) -> Specials {
        let mut s = Specials {
            eot: 50256,
            sot: 50257,
            translate: 50357,
            transcribe: 50358,
            solm: 50359,
            prev: 50360,
            nosp: 50361,
            not: 50362,
            beg: 50363,
        };
        if h.is_multilingual() {
            s.eot += 1;
            s.sot += 1;
            let dt = h.num_languages() - 98;
            for v in [&mut s.translate, &mut s.transcribe, &mut s.solm, &mut s.prev, &mut s.nosp, &mut s.not, &mut s.beg] {
                *v += dt;
            }
        }
        s
    }
}

/// The tensors whisper.cpp creates for these hparams: name -> (dtype, ne). `w` is the weight type, `v` the conv type
/// (f16 unless the weights are f32), as in whisper_model_load.
pub fn expected_tensors(h: &Hparams, w: Dtype) -> Vec<(String, Dtype, [i64; 4])> {
    let v = if w == Dtype::F32 { Dtype::F32 } else { Dtype::F16 };
    let f = Dtype::F32;
    let (sa, st) = (h.n_audio_state as i64, h.n_text_state as i64);
    let mut out = vec![
        ("encoder.positional_embedding".to_string(), f, [sa, h.n_audio_ctx as i64, 1, 1]),
        ("encoder.conv1.weight".into(), v, [3, h.n_mels as i64, sa, 1]),
        ("encoder.conv1.bias".into(), f, [1, sa, 1, 1]),
        ("encoder.conv2.weight".into(), v, [3, sa, sa, 1]),
        ("encoder.conv2.bias".into(), f, [1, sa, 1, 1]),
        ("encoder.ln_post.weight".into(), f, [sa, 1, 1, 1]),
        ("encoder.ln_post.bias".into(), f, [sa, 1, 1, 1]),
    ];
    let block = |out: &mut Vec<(String, Dtype, [i64; 4])>, p: &str, s: i64, cross: bool| {
        out.push((format!("{p}.mlp_ln.weight"), f, [s, 1, 1, 1]));
        out.push((format!("{p}.mlp_ln.bias"), f, [s, 1, 1, 1]));
        out.push((format!("{p}.mlp.0.weight"), w, [s, 4 * s, 1, 1]));
        out.push((format!("{p}.mlp.0.bias"), f, [4 * s, 1, 1, 1]));
        out.push((format!("{p}.mlp.2.weight"), w, [4 * s, s, 1, 1]));
        out.push((format!("{p}.mlp.2.bias"), f, [s, 1, 1, 1]));
        let attn = |out: &mut Vec<(String, Dtype, [i64; 4])>, a: &str| {
            out.push((format!("{p}.{a}_ln.weight"), f, [s, 1, 1, 1]));
            out.push((format!("{p}.{a}_ln.bias"), f, [s, 1, 1, 1]));
            out.push((format!("{p}.{a}.query.weight"), w, [s, s, 1, 1]));
            out.push((format!("{p}.{a}.query.bias"), f, [s, 1, 1, 1]));
            out.push((format!("{p}.{a}.key.weight"), w, [s, s, 1, 1]));
            out.push((format!("{p}.{a}.value.weight"), w, [s, s, 1, 1]));
            out.push((format!("{p}.{a}.value.bias"), f, [s, 1, 1, 1]));
            out.push((format!("{p}.{a}.out.weight"), w, [s, s, 1, 1]));
            out.push((format!("{p}.{a}.out.bias"), f, [s, 1, 1, 1]));
        };
        attn(out, "attn");
        if cross {
            attn(out, "cross_attn");
        }
    };
    for i in 0..h.n_audio_layer {
        block(&mut out, &format!("encoder.blocks.{i}"), sa, false);
    }
    out.push(("decoder.positional_embedding".into(), f, [st, h.n_text_ctx as i64, 1, 1]));
    out.push(("decoder.token_embedding.weight".into(), w, [st, h.n_vocab as i64, 1, 1]));
    out.push(("decoder.ln.weight".into(), f, [st, 1, 1, 1]));
    out.push(("decoder.ln.bias".into(), f, [st, 1, 1, 1]));
    for i in 0..h.n_text_layer {
        block(&mut out, &format!("decoder.blocks.{i}"), st, true);
    }
    out
}

impl Model {
    /// Load a model only if its bytes are a pinned file: the guard. The reason for a refusal names what differed.
    pub fn load_pinned(path: &std::path::Path) -> Result<Model, String> {
        Self::load_unverified(path)?.guard(&path.display().to_string())
    }

    /// Keep the model only if it is a pinned file; otherwise say what differed. `label` names it in the reason.
    pub fn guard(self, label: &str) -> Result<Model, String> {
        let m = self;
        if m.pin.is_none() {
            let size = m.bytes.len();
            let near = PINS.iter().find(|p| p.size as usize == size);
            return Err(match near {
                Some(p) => format!(
                    "refused: {} has the size of {} ({} bytes) but sha256 {}, not the pinned {}",
                    label,
                    p.file,
                    size,
                    m.sha256,
                    p.sha256
                ),
                None => format!(
                    "refused: {} (sha256 {}, {} bytes) is not a pinned model; pinned: {}",
                    label,
                    m.sha256,
                    size,
                    PINS.iter().map(|p| p.file).collect::<Vec<_>>().join(", ")
                ),
            });
        }
        Ok(m)
    }

    /// Read and parse without requiring a pin (the hash is still computed and recorded).
    pub fn load_unverified(path: &std::path::Path) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Model, String> {
        let sha = sha256::hex(&sha256::digest(&bytes));
        let pin = PINS.iter().find(|p| p.sha256 == sha && p.size as usize == bytes.len());
        let mut r = Reader { b: &bytes, pos: 0 };

        let magic = r.u32("magic")?;
        if magic != GGML_FILE_MAGIC {
            return Err(format!("bad magic 0x{magic:08x} (want 0x{GGML_FILE_MAGIC:08x}, \"ggml\")"));
        }
        let mut h = Hparams {
            n_vocab: r.i32("n_vocab")?,
            n_audio_ctx: r.i32("n_audio_ctx")?,
            n_audio_state: r.i32("n_audio_state")?,
            n_audio_head: r.i32("n_audio_head")?,
            n_audio_layer: r.i32("n_audio_layer")?,
            n_text_ctx: r.i32("n_text_ctx")?,
            n_text_state: r.i32("n_text_state")?,
            n_text_head: r.i32("n_text_head")?,
            n_text_layer: r.i32("n_text_layer")?,
            n_mels: r.i32("n_mels")?,
            ftype: r.i32("ftype")?,
            qntvr: 0,
        };
        h.qntvr = h.ftype / 1000;
        h.ftype %= 1000;
        for (v, what) in [
            (h.n_vocab, "n_vocab"),
            (h.n_audio_ctx, "n_audio_ctx"),
            (h.n_audio_state, "n_audio_state"),
            (h.n_audio_head, "n_audio_head"),
            (h.n_audio_layer, "n_audio_layer"),
            (h.n_text_ctx, "n_text_ctx"),
            (h.n_text_state, "n_text_state"),
            (h.n_text_head, "n_text_head"),
            (h.n_text_layer, "n_text_layer"),
            (h.n_mels, "n_mels"),
        ] {
            if !(1..=1 << 20).contains(&v) {
                return Err(format!("implausible {what} = {v}"));
            }
        }
        if h.n_text_state != h.n_audio_state {
            return Err("n_text_state != n_audio_state (whisper.cpp asserts they are equal)".into());
        }
        let wtype = match h.ftype {
            0 => Dtype::F32,
            1 => Dtype::F16,
            t => return Err(format!("ftype {t} (quantized) is not supported yet: 0.0.1 reads f32 and f16 models")),
        };

        // the mel filterbank
        let n_mel = r.i32("filters.n_mel")?;
        let n_fft = r.i32("filters.n_fft")?;
        if !(1..=4096).contains(&n_mel) || !(1..=65536).contains(&n_fft) {
            return Err(format!("implausible filterbank {n_mel} x {n_fft}"));
        }
        let fb = r.take(4 * (n_mel as usize) * (n_fft as usize), "mel filters")?;
        let filters: Vec<f32> = fb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();

        // the vocabulary
        let n_vocab_file = r.i32("vocab count")?;
        if n_vocab_file < 0 || n_vocab_file > h.n_vocab.max(n_vocab_file).min(1 << 20) {
            return Err(format!("implausible vocab count {n_vocab_file}"));
        }
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(h.n_vocab.max(n_vocab_file) as usize);
        for i in 0..n_vocab_file {
            let len = r.u32("token length")? as usize;
            vocab.push(r.take(len, &format!("token {i}"))?.to_vec());
        }
        if (n_vocab_file as usize) < h.n_vocab as usize {
            let s = Specials::for_hparams(&h);
            for i in n_vocab_file..h.n_vocab {
                let w = if i > s.beg {
                    format!("[_TT_{}]", i - s.beg)
                } else if i == s.eot {
                    "[_EOT_]".into()
                } else if i == s.sot {
                    "[_SOT_]".into()
                } else if i == s.translate {
                    "[_TRANSLATE_]".into()
                } else if i == s.transcribe {
                    "[_TRANSCRIBE_]".into()
                } else if i == s.solm {
                    "[_SOLM_]".into()
                } else if i == s.prev {
                    "[_PREV_]".into()
                } else if i == s.nosp {
                    "[_NOSP_]".into()
                } else if i == s.not {
                    "[_NOT_]".into()
                } else if i == s.beg {
                    "[_BEG_]".into()
                } else if i > s.sot && i <= s.sot + h.num_languages() {
                    // whisper_lang_str(i - sot - 1); an .en vocab names 99 of them too (ids 50258..50356)
                    let id = (i - s.sot - 1) as usize;
                    match LANGS.get(id) {
                        Some(l) => format!("[_LANG_{l}]"),
                        None => return Err(format!("token {i}: language id {id} has no name (whisper.cpp would crash)")),
                    }
                } else {
                    format!("[_extra_token_{i}]")
                };
                vocab.push(w.into_bytes());
            }
        }

        // the tensors
        let expected = expected_tensors(&h, wtype);
        let mut tensors: Vec<Tensor> = Vec::with_capacity(expected.len());
        while !r.eof() {
            let n_dims = r.i32("tensor n_dims")?;
            let name_len = r.i32("tensor name length")?;
            let ttype = r.i32("tensor type")?;
            if !(1..=4).contains(&n_dims) || !(1..=1024).contains(&name_len) {
                return Err(format!("implausible tensor header at offset {}: n_dims {n_dims}, name length {name_len}", r.pos));
            }
            let mut ne = [1i64; 4];
            for d in ne.iter_mut().take(n_dims as usize) {
                *d = r.i32("tensor ne")? as i64;
                if *d < 1 {
                    return Err(format!("tensor dimension {d} < 1"));
                }
            }
            let name = String::from_utf8(r.take(name_len as usize, "tensor name")?.to_vec())
                .map_err(|_| "tensor name is not UTF-8".to_string())?;
            let dtype = Dtype::from_ggml(ttype).map_err(|e| format!("tensor '{name}': {e}"))?;
            let Some((_, want_t, want_ne)) = expected.iter().find(|(n, _, _)| *n == name) else {
                return Err(format!("unknown tensor '{name}' in model file"));
            };
            if tensors.iter().any(|t| t.name == name) {
                return Err(format!("tensor '{name}' appears twice"));
            }
            // whisper.cpp compares ne[0..3]; ne[3] is always 1 for these tensors
            if ne != *want_ne {
                return Err(format!("tensor '{name}' has shape {ne:?}, expected {want_ne:?}"));
            }
            if dtype != *want_t {
                return Err(format!("tensor '{name}' is stored as {}, whisper.cpp creates it as {}", dtype.name(), want_t.name()));
            }
            let nbytes = ne.iter().product::<i64>() as usize * dtype.size();
            let offset = r.pos;
            r.take(nbytes, &format!("tensor '{name}' data"))?;
            tensors.push(Tensor { name, dtype, n_dims: n_dims as usize, ne, offset, nbytes });
        }
        if tensors.len() != expected.len() {
            let missing: Vec<&str> =
                expected.iter().filter(|(n, _, _)| !tensors.iter().any(|t| &t.name == n)).map(|(n, _, _)| n.as_str()).collect();
            return Err(format!("not all tensors loaded: expected {}, got {}; missing {:?}", expected.len(), tensors.len(), missing));
        }

        Ok(Model {
            hparams: h,
            filters_n_mel: n_mel,
            filters_n_fft: n_fft,
            filters,
            vocab,
            n_vocab_file: n_vocab_file as usize,
            tensors,
            sha256: sha,
            pin,
            bytes,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_bytes(&self, t: &Tensor) -> &[u8] {
        &self.bytes[t.offset..t.offset + t.nbytes]
    }

    /// A human summary.
    pub fn summary(&self) -> String {
        let h = &self.hparams;
        let mut s = String::new();
        s += &format!(
            "sha256        {}  ({})\n",
            self.sha256,
            match self.pin {
                Some(p) => format!("pinned: {}", p.file),
                None => "NOT PINNED".into(),
            }
        );
        s += &format!("size          {} bytes\n", self.bytes.len());
        s += &format!("type          {} ({}multilingual)\n", h.model_type(), if h.is_multilingual() { "" } else { "not " });
        s += &format!("n_vocab       {} ({} in file)\n", h.n_vocab, self.n_vocab_file);
        s += &format!("audio         ctx {} · state {} · heads {} · layers {}\n", h.n_audio_ctx, h.n_audio_state, h.n_audio_head, h.n_audio_layer);
        s += &format!("text          ctx {} · state {} · heads {} · layers {}\n", h.n_text_ctx, h.n_text_state, h.n_text_head, h.n_text_layer);
        s += &format!("n_mels        {}\n", h.n_mels);
        s += &format!("ftype         {} (qntvr {})\n", h.ftype, h.qntvr);
        s += &format!("filters       {} x {}\n", self.filters_n_mel, self.filters_n_fft);
        let (mut f32b, mut f16b, mut f32n, mut f16n) = (0usize, 0usize, 0, 0);
        for t in &self.tensors {
            match t.dtype {
                Dtype::F32 => {
                    f32b += t.nbytes;
                    f32n += 1
                }
                Dtype::F16 => {
                    f16b += t.nbytes;
                    f16n += 1
                }
            }
        }
        s += &format!(
            "tensors       {} ({} f16 · {:.2} MB, {} f32 · {:.2} MB; total {:.2} MB)\n",
            self.tensors.len(),
            f16n,
            f16b as f64 / 1e6,
            f32n,
            f32b as f64 / 1e6,
            (f16b + f32b) as f64 / 1e6
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_layout_counts() {
        let h = Hparams {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
            qntvr: 0,
        };
        // whisper.cpp: 10 /* input */ + 15 + 15*n_audio_layer + 24*n_text_layer is its context budget; the real
        // count is 7 + 15*L_enc + 4 + 24*L_dec
        assert_eq!(expected_tensors(&h, Dtype::F16).len(), 7 + 15 * 4 + 4 + 24 * 4);
        let s = Specials::for_hparams(&h);
        assert_eq!((s.eot, s.sot, s.beg), (50256, 50257, 50363));
    }

    #[test]
    fn refuses_garbage() {
        assert!(Model::parse(b"nope".to_vec()).err().unwrap().contains("magic"));
        let mut b = GGML_FILE_MAGIC.to_le_bytes().to_vec();
        b.extend_from_slice(&[0; 8]);
        assert!(Model::parse(b).err().unwrap().contains("truncated"));
    }
}
