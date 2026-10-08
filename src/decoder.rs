// SPDX-License-Identifier: MIT OR Apache-2.0
//! The decoder's input (0.1.2): what `whisper_build_graph_decoder` computes before its first block, and the batch
//! `whisper_decode_internal` takes it from (testing/decin/NOTES.md). Read from the pinned source and the shipped
//! `libggml-cpu.so`, then checked by the oracle on every decoder call `whisper_full` makes for the 8 recorded inputs:
//!
//! - **`cur = ggml_add(ggml_get_rows(d_te, embd), ggml_get_rows(d_pe, position))`** (whisper.cpp:2515): `d_te`, the token
//!   embedding, is **f16** `[n_vocab][n_state]`; its rows are widened by `ggml_cpu_fp16_to_fp32` (`vcvtph2ps` in blocks
//!   of 8 and 4, the table for a tail — exact either way, 0.0.3). `d_pe`, the positional embedding, is **f32**
//!   `[n_text_ctx][n_state]`; its rows are copied (`ggml_vec_cpy_f32`). The ADD is one f32 add per value, the token row
//!   its first operand.
//! - **the batch** ([`Batch`]): a prompt goes in as `whisper_batch_prep_legacy(prompt, n_past 0, seq 0)` — position
//!   `n_past + i`, one sequence id each, logits for the last row only; each next token as a one-row batch at position
//!   `prompt.len() + i` with logits on (whisper.cpp:7456).
//! - **the prompt** ([`Prompt`]): `[SOT]`, then for a multilingual model the language and the task, then `NOT` when
//!   timestamps are off (whisper.cpp:6975); before it, when there is past text, `PREV` and the last
//!   `max_prompt_ctx − 1` tokens of it (whisper.cpp:7110).
//!
//! The fast path ([`DecoderInput::run_into`]) widens each token row and adds its position row in one pass, eight values
//! at a time (F16C where the CPU has it — `vcvtph2ps` is the reference's own instruction), straight into the caller's
//! buffer; nothing is allocated per call. It runs on one thread: a decoder call has one row per token (one row for a
//! step, 226 at most for a full prompt), and the reference's split of rows between threads cannot change a bit.

use crate::f16::{fp16_to_fp32, fp32_to_fp16};
use crate::model::{Dtype, Model, Specials};

/// The batch whisper fills for a decoder call (`whisper_batch`): one entry per token row.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub token: Vec<i32>,
    pub pos: Vec<i32>,
    /// always 1 (`n_seq_id`, "here for consistency with llama.cpp")
    pub n_seq_id: Vec<i32>,
    /// `seq_id[i][0]`: the decoder the row belongs to (0 for greedy decoding at temperature 0)
    pub seq_id: Vec<i32>,
    /// whether the row's logits are copied out
    pub logits: Vec<bool>,
}

impl Batch {
    /// Room for `n` rows (whisper allocates `n_text_ctx`); filling it never allocates again.
    pub fn with_capacity(n: usize) -> Batch {
        Batch {
            token: Vec::with_capacity(n),
            pos: Vec::with_capacity(n),
            n_seq_id: Vec::with_capacity(n),
            seq_id: Vec::with_capacity(n),
            logits: Vec::with_capacity(n),
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.token.len()
    }

    fn clear(&mut self) {
        self.token.clear();
        self.pos.clear();
        self.n_seq_id.clear();
        self.seq_id.clear();
        self.logits.clear();
    }

    fn push(&mut self, token: i32, pos: i32, seq_id: i32, logits: bool) {
        self.token.push(token);
        self.pos.push(pos);
        self.n_seq_id.push(1);
        self.seq_id.push(seq_id);
        self.logits.push(logits);
    }

    /// `whisper_batch_prep_legacy(batch, tokens, n_tokens, n_past, seq_id)`: positions `n_past + i`, logits for the
    /// last row only. How whisper sends a window's prompt (n_past 0, seq 0).
    pub fn prep_legacy(&mut self, tokens: &[i32], n_past: i32, seq_id: i32) {
        assert!(!tokens.is_empty(), "a batch needs a token");
        self.clear();
        for (i, &t) in tokens.iter().enumerate() {
            self.push(t, n_past + i as i32, seq_id, i + 1 == tokens.len());
        }
    }

    /// One decoding step (whisper.cpp:7456): the decoder's last sampled token at position `prompt_len + i` (`i`
    /// steps after the prompt), logits on. With one decoder (greedy, temperature 0) this is the whole batch.
    pub fn prep_step(&mut self, token: i32, prompt_len: usize, i: usize, seq_id: i32) {
        self.clear();
        self.push(token, (prompt_len + i) as i32, seq_id, true);
    }
}

/// What decides a window's prompt (`whisper_full_params` and the model).
#[derive(Clone, Copy, Debug)]
pub struct Prompt {
    pub specials: Specials,
    pub multilingual: bool,
    /// `whisper_lang_id(language)`, used by multilingual models only
    pub lang_id: i32,
    pub translate: bool,
    pub no_timestamps: bool,
    /// `min(n_max_text_ctx, n_text_ctx / 2)` (whisper.cpp:6927)
    pub max_prompt_ctx: usize,
    /// `n_max_text_ctx > 0` and the temperature below the history cutoff (0.5)
    pub use_past: bool,
}

impl Prompt {
    /// whisper_full's defaults for `m`: English, transcribe, timestamps on, `n_max_text_ctx` 16384, temperature 0.
    pub fn for_model(m: &Model) -> Prompt {
        let h = &m.hparams;
        Prompt {
            specials: Specials::for_hparams(h),
            multilingual: h.is_multilingual(),
            lang_id: 0,
            translate: false,
            no_timestamps: false,
            max_prompt_ctx: 16384.min(h.n_text_ctx as usize / 2),
            use_past: true,
        }
    }

    /// `prompt_init` (whisper.cpp:6975): `[SOT]`, `+ [lang, task]` for a multilingual model, `+ [NOT]` without
    /// timestamps. (The first-release distilled models force `no_timestamps`; voaice's pinned models are not those.)
    pub fn init(&self) -> Vec<i32> {
        let s = &self.specials;
        let mut v = vec![s.sot];
        if self.multilingual {
            v.push(s.sot + 1 + self.lang_id);
            v.push(if self.translate { s.translate } else { s.transcribe });
        }
        if self.no_timestamps {
            v.push(s.not);
        }
        v
    }

    /// A window's prompt into `out` (whisper.cpp:7106): `[PREV] + the last min(max_prompt_ctx − 1, |past|) of past`
    /// when there is past and it is used, then `init`. `past` is `prompt_past1` — the initial prompt's tokens and the
    /// decoded text of earlier windows; whisper empties it for a window that starts within 500 frames of the end, which
    /// the caller decides (it needs the window's seek).
    pub fn window(&self, past: &[i32], out: &mut Vec<i32>) {
        out.clear();
        if self.use_past && self.max_prompt_ctx > 0 && !past.is_empty() {
            out.push(self.specials.prev);
            let n = (self.max_prompt_ctx - 1).min(past.len());
            out.extend_from_slice(&past[past.len() - n..]);
        }
        out.extend(self.init());
    }
}

/// The two embeddings the decoder's input reads.
pub struct DecoderInput {
    pub n_state: usize,
    pub n_vocab: usize,
    /// rows of the positional embedding (`n_text_ctx`)
    pub n_ctx: usize,
    /// `decoder.token_embedding.weight`, f16 `[n_vocab][n_state]` (the file's bits)
    pub te: Vec<u16>,
    /// `decoder.positional_embedding`, f32 `[n_ctx][n_state]`
    pub pe: Vec<f32>,
}

/// The model's three nodes, each `[n_tokens][n_state]`: the token rows (GET_ROWS d_te), the position rows (GET_ROWS
/// d_pe) and their sum (ADD).
#[derive(Default, Debug)]
pub struct DecinNodes {
    pub te: Vec<f32>,
    pub pe: Vec<f32>,
    pub sum: Vec<f32>,
}

/// How the model computes the input: `DecinVariant::default()` is the reference's; every other setting is a
/// discriminator (a reading the oracle must reject — or, where marked, cannot tell apart).
#[derive(Clone, Copy, Default, Debug)]
pub struct DecinVariant {
    /// f16 subnormals flushed to zero when widened (a DAZ conversion)
    pub te_ftz: bool,
    /// the token row widened through bf16 truncation of the f16 value's f32 (a lossy conversion)
    pub te_bf16: bool,
    /// the position row rounded to f16 before the add (d_pe read as if it were the weight type)
    pub pe_f16: bool,
    /// the sum rounded to f16 (the ADD's output in the weight type)
    pub sum_f16: bool,
    /// the position row as the ADD's first operand — f32 addition commutes, so this cannot be told apart
    pub pe_first: bool,
    /// the sum in double, then rounded — exact for two f32 (53 ≥ 2·24 + 2 bits), so this cannot be told apart
    pub sum_double: bool,
}

impl DecoderInput {
    pub fn new(m: &Model) -> Result<DecoderInput, String> {
        let h = &m.hparams;
        let (n_state, n_vocab, n_ctx) = (h.n_text_state as usize, h.n_vocab as usize, h.n_text_ctx as usize);
        let te = m.tensor("decoder.token_embedding.weight").ok_or("no decoder.token_embedding.weight")?;
        let pe = m.tensor("decoder.positional_embedding").ok_or("no decoder.positional_embedding")?;
        if te.dtype != Dtype::F16 || te.ne[0] as usize != n_state || te.ne[1] as usize != n_vocab {
            return Err(format!("decoder.token_embedding.weight: expected f16 [{n_state}, {n_vocab}]"));
        }
        if pe.dtype != Dtype::F32 || pe.ne[0] as usize != n_state || pe.ne[1] as usize != n_ctx {
            return Err(format!("decoder.positional_embedding: expected f32 [{n_state}, {n_ctx}]"));
        }
        Ok(DecoderInput {
            n_state,
            n_vocab,
            n_ctx,
            te: m.tensor_bytes(te).as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect(),
            pe: m.tensor_bytes(pe).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        })
    }

    /// GGML_ASSERT(i01 >= 0 && i01 < ne01) of both GET_ROWS: the reference aborts; voaice refuses.
    pub fn check(&self, tokens: &[i32], pos: &[i32]) -> Result<(), String> {
        if tokens.len() != pos.len() || tokens.is_empty() {
            return Err(format!("decoder input: {} tokens and {} positions", tokens.len(), pos.len()));
        }
        if let Some(t) = tokens.iter().find(|&&t| t < 0 || t as usize >= self.n_vocab) {
            return Err(format!("decoder input: token {t} outside the vocabulary of {}", self.n_vocab));
        }
        if let Some(p) = pos.iter().find(|&&p| p < 0 || p as usize >= self.n_ctx) {
            return Err(format!("decoder input: position {p} outside the text context of {}", self.n_ctx));
        }
        Ok(())
    }

    /// The model: the three nodes as the source reads, one value at a time.
    pub fn model(&self, tokens: &[i32], pos: &[i32], v: DecinVariant) -> Result<DecinNodes, String> {
        self.check(tokens, pos)?;
        let ns = self.n_state;
        let (mut te, mut pe, mut sum) = (vec![0f32; tokens.len() * ns], vec![0f32; tokens.len() * ns], vec![0f32; tokens.len() * ns]);
        for (i, (&t, &p)) in tokens.iter().zip(pos).enumerate() {
            for c in 0..ns {
                let h = self.te[t as usize * ns + c];
                let mut a = fp16_to_fp32(h);
                if v.te_ftz && h & 0x7C00 == 0 {
                    a = if h & 0x8000 != 0 { -0.0 } else { 0.0 };
                }
                if v.te_bf16 {
                    a = f32::from_bits(a.to_bits() & 0xFFFF_0000);
                }
                let mut b = self.pe[p as usize * ns + c];
                if v.pe_f16 {
                    b = fp16_to_fp32(fp32_to_fp16(b));
                }
                let mut s = match (v.sum_double, v.pe_first) {
                    (true, _) => (a as f64 + b as f64) as f32,
                    (false, true) => b + a,
                    (false, false) => a + b,
                };
                if v.sum_f16 {
                    s = fp16_to_fp32(fp32_to_fp16(s));
                }
                te[i * ns + c] = a;
                pe[i * ns + c] = b;
                sum[i * ns + c] = s;
            }
        }
        Ok(DecinNodes { te, pe, sum })
    }

    /// The fast path: `out[i] = widen(te[tokens[i]]) + pe[pos[i]]`, `out` the caller's (`n_tokens · n_state`).
    pub fn run_into(&self, tokens: &[i32], pos: &[i32], out: &mut [f32]) -> Result<(), String> {
        self.check(tokens, pos)?;
        let ns = self.n_state;
        if out.len() != tokens.len() * ns {
            return Err(format!("decoder input: output of {} values for {} rows of {ns}", out.len(), tokens.len()));
        }
        #[cfg(target_arch = "x86_64")]
        let simd = ns.is_multiple_of(8) && std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx");
        for ((&t, &p), o) in tokens.iter().zip(pos).zip(out.chunks_exact_mut(ns)) {
            let (te, pe) = (&self.te[t as usize * ns..][..ns], &self.pe[p as usize * ns..][..ns]);
            #[cfg(target_arch = "x86_64")]
            if simd {
                // SAFETY: the CPU has F16C and AVX (checked above); the three slices have ns values, a multiple of 8
                unsafe { x86::widen_add(te, pe, o) };
                continue;
            }
            for ((o, &h), &b) in o.iter_mut().zip(te).zip(pe) {
                *o = fp16_to_fp32(h) + b;
            }
        }
        Ok(())
    }

    /// The batch's rows into `out` (resized to `n_tokens · n_state`; its capacity is kept between calls).
    pub fn run_batch(&self, b: &Batch, out: &mut Vec<f32>) -> Result<(), String> {
        out.resize(b.n_tokens() * self.n_state, 0.0);
        self.run_into(&b.token, &b.pos, out)
    }

    /// Bytes held: the f16 token table and the f32 positions.
    pub fn bytes(&self) -> usize {
        2 * self.te.capacity() + 4 * self.pe.capacity()
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// `o = vcvtph2ps(te) + pe`, 8 at a time; the three slices of one length, a multiple of 8.
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn widen_add(te: &[u16], pe: &[f32], o: &mut [f32]) {
        let n = o.len();
        let mut i = 0;
        // SAFETY (all loads and stores): i + 8 <= n and te, pe, o have length n
        while i + 8 <= n {
            unsafe {
                let h = _mm_loadu_si128(te.as_ptr().add(i) as *const __m128i);
                let s = _mm256_add_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(pe.as_ptr().add(i)));
                _mm256_storeu_ps(o.as_mut_ptr().add(i), s);
            }
            i += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prep_legacy_is_whispers() {
        let mut b = Batch::with_capacity(448);
        b.prep_legacy(&[50360, 1, 2, 50257], 0, 0);
        assert_eq!(b.pos, [0, 1, 2, 3]);
        assert_eq!(b.logits, [false, false, false, true]);
        assert_eq!(b.n_seq_id, [1; 4]);
        b.prep_step(843, 4, 2, 0);
        assert_eq!((b.token.as_slice(), b.pos.as_slice(), b.logits.as_slice()), (&[843][..], &[6][..], &[true][..]));
    }

    #[test]
    fn widen_add_matches_the_scalar() {
        let ns = 16;
        let te: Vec<u16> = (0..3 * ns as u32).map(|i| (i * 2731 + 0x0201) as u16 & 0x7BFF).collect();
        let pe: Vec<f32> = (0..2 * ns).map(|i| (i as f32 - 7.5) * 0.37).collect();
        let d = DecoderInput { n_state: ns, n_vocab: 3, n_ctx: 2, te, pe };
        let (tok, pos) = ([2, 0, 1], [1, 0, 1]);
        let mut out = vec![0f32; 3 * ns];
        d.run_into(&tok, &pos, &mut out).unwrap();
        let want = d.model(&tok, &pos, DecinVariant::default()).unwrap().sum;
        assert_eq!(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert!(d.run_into(&[3], &[0], &mut out[..ns]).is_err());
        assert!(d.run_into(&[0], &[2], &mut out[..ns]).is_err());
    }
}
