//! JIT wrappers for Whisper.
//!
//! Every wrapper is prepared at a concrete capacity. Prefill projects encoder
//! features into the cross-attention caches once per window; the fixed-slot
//! decoder step and the aligner then read those caches.
#![allow(clippy::too_many_arguments)]

use svod_macros::jit_wrapper;

use super::model::Whisper;

#[derive(Clone)]
pub struct WhisperAlignmentModel {
    model: Whisper,
    alignment_heads: Vec<(usize, usize)>,
}

impl WhisperAlignmentModel {
    pub fn new(model: Whisper, alignment_heads: Vec<(usize, usize)>) -> Self {
        Self { model, alignment_heads }
    }

    fn forward(
        &self,
        cross_k: &svod_tensor::Tensor,
        cross_v: &svod_tensor::Tensor,
        tokens: &svod_tensor::Tensor,
    ) -> super::error::Result<svod_tensor::Tensor> {
        self.model.align_with_cross_kv(tokens, cross_k, cross_v, &self.alignment_heads)
    }
}

// Encoder-only JIT: mel `[B, n_mels, T]` → `[B, T/2, D]`.
jit_wrapper! {
    WhisperEncoderJit(Whisper) {
        mel: Tensor,

        build(mel) {
            model.encode(mel)
        }
    }
}

// Static teacher-forced alignment replay. The graph shape and selected heads
// are fixed at construction; valid token/audio lengths are host metadata.
jit_wrapper! {
    WhisperAlignmentJit(WhisperAlignmentModel) {
        cross_k: Tensor,
        cross_v: Tensor,
        tokens: Tensor,

        build(cross_k, cross_v, tokens) {
            model.forward(cross_k, cross_v, tokens)
        }
    }
}

// Prefill JIT: initial tokens `[1, init_len]` + encoder features → logits
// `[1, init_len, n_vocab]` and the packed self and cross caches. Row 0 of the
// logits depends on the first token alone, so the same graph also serves
// language detection.
jit_wrapper! {
    WhisperPrefillJit(Whisper) {
        tokens: Tensor,
        audio_features: Tensor,

        outputs { logits, self_k, self_v, cross_k, cross_v }

        build(tokens, audio_features) {
            model.decode_prefill(tokens, audio_features, 0)
        }
    }
}

// KV-cached decoder step JIT: single-token forward with K/V cache recycling.
// Inputs: token [B,1], self/cross K/V caches, self key lengths [B] (also the
// position), cross cache row map [B].
// Outputs: logits [B,n_vocab], new_self_k [B,1,n_layer*H,Dh], new_self_v [...].
// After execute: copy_output_to_self_k_cache/v_cache to append new K/V at pos.
jit_wrapper! {
    WhisperDecoderStepJit(Whisper) {
        token: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        self_key_lens: Tensor,
        cross_cache_map: Tensor,

        outputs { logits, new_self_k, new_self_v }

        build(token, self_k_cache, self_v_cache, cross_k, cross_v, self_key_lens, cross_cache_map) {
            model.decode_step(token, self_k_cache, self_v_cache, cross_k, cross_v, self_key_lens, cross_cache_map)
        }
    }
}
