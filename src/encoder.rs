// SPDX-License-Identifier: MIT OR Apache-2.0
//! whisper's audio encoder, whole (v0.1.0): the mel window → conv1 + bias, GELU → conv2 + bias, GELU → + positions
//! (0.0.6, 0.0.7) → four blocks of attn_ln → Q, K, V (0.0.8, 0.0.9) → flash attention ([`crate::attention`]) → out
//! projection + residual → mlp_ln → fc1, GELU, fc2 + residual (0.0.9) → ln_post (0.0.8) = `embd_enc`, bit for bit what
//! the pinned whisper.cpp's `whisper_encode_with_state` leaves in `whisper_state::embd_enc` (testing/attention/NOTES.md).
//!
//! Every stage is the fast path its own increment checked; the pipeline adds no arithmetic of its own. The buffers
//! between stages are owned by the caller ([`EncoderBuffers`], sized on the first call and then reused), so an encode
//! allocates only its threads' small panels.

use crate::attention::{n_kv_pad, Attention};
use crate::conv::ConvStage;
use crate::matmul::{Block, MlpTaps, QkvTaps};
use crate::model::Model;
use crate::norm::{LayerNorm, Node};

pub struct Encoder {
    pub conv: ConvStage,
    pub blocks: Vec<Block>,
    pub attn: Attention,
    pub ln_post: LayerNorm,
    /// encoder frames (1,500 on every whisper model) and their width
    pub n_ctx: usize,
    pub n_state: usize,
}

/// The encoder's working memory between stages, kept by the caller across calls.
#[derive(Default)]
pub struct EncoderBuffers {
    conv16: Vec<u16>,
    x: Vec<f32>,
    y: Vec<f32>,
    q: Vec<f32>,
    k16: Vec<u16>,
    v16: Vec<u16>,
    att: Vec<f32>,
    attn: Vec<f32>,
}

impl EncoderBuffers {
    /// Bytes held (the conv stage's f16 buffer, the residual stream twice, Q, K and V, the attention's output and its
    /// widened K/V).
    pub fn bytes(&self) -> usize {
        2 * (self.conv16.capacity() + self.k16.capacity() + self.v16.capacity())
            + 4 * (self.x.capacity() + self.y.capacity() + self.q.capacity() + self.att.capacity() + self.attn.capacity())
    }
}

impl Encoder {
    pub fn new(m: &Model) -> Result<Encoder, String> {
        let h = &m.hparams;
        let (n_ctx, n_state) = (h.n_audio_ctx as usize, h.n_audio_state as usize);
        if h.n_audio_head as usize * 64 != n_state {
            return Err(format!("encoder: {} heads of {} is not a head size of 64", h.n_audio_head, n_state));
        }
        let blocks = (0..h.n_audio_layer as usize).map(|il| Block::new(m, il)).collect::<Result<Vec<_>, _>>()?;
        Ok(Encoder {
            conv: ConvStage::new(m)?,
            blocks,
            attn: Attention::new(n_state, n_kv_pad(n_ctx))?,
            ln_post: LayerNorm::new(m, "encoder.ln_post")?,
            n_ctx,
            n_state,
        })
    }

    /// The 2·n_ctx mel frames from `offset` (zero past `n_len`, as whisper slices them) → `out` ([n_ctx][n_state],
    /// `embd_enc`).
    pub fn encode_into(&self, mel: &[f32], n_len: usize, offset: usize, threads: usize, buf: &mut EncoderBuffers, out: &mut [f32]) {
        let n = self.n_ctx * self.n_state;
        assert_eq!(out.len(), n, "encoder: the output is not [n_ctx][n_state]");
        for v in [&mut buf.x, &mut buf.y, &mut buf.q, &mut buf.att] {
            v.resize(n, 0.0);
        }
        buf.k16.resize(n, 0);
        buf.v16.resize(n, 0);
        self.conv.run_into(mel, n_len, offset, 2 * self.n_ctx, threads, &mut buf.conv16, &mut buf.x);
        for b in &self.blocks {
            b.qkv_into(&buf.x, threads, &mut buf.q, &mut buf.k16, &mut buf.v16, QkvTaps::default());
            self.attn.run_into(&buf.q, &buf.k16, &buf.v16, threads, &mut buf.attn, &mut buf.att);
            b.mlp_into(&buf.att, &buf.x, threads, &mut buf.y, MlpTaps::default());
            std::mem::swap(&mut buf.x, &mut buf.y);
        }
        self.ln_post.run_into(&buf.x, threads, Node::Add, out);
    }

    /// The same into a new vector, with buffers of its own.
    pub fn encode(&self, mel: &[f32], n_len: usize, offset: usize, threads: usize) -> Vec<f32> {
        let mut buf = EncoderBuffers::default();
        let mut out = vec![0.0f32; self.n_ctx * self.n_state];
        self.encode_into(mel, n_len, offset, threads, &mut buf, &mut out);
        out
    }
}
