//! Whisper model dimensions, size presets, and audio constants.

// ─── Audio constants (matching whisper/audio.py) ────────────────────────────

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
pub const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2;
pub const TOKENS_PER_SECOND: f32 = SAMPLE_RATE as f32 / N_SAMPLES_PER_TOKEN as f32;
pub const FRAMES_PER_SECOND: f32 = SAMPLE_RATE as f32 / HOP_LENGTH as f32;
pub const N_AUDIO_CTX: usize = N_FRAMES / 2;
pub const N_TEXT_CTX: usize = 448;

// ─── ModelDimensions (matching whisper/model.py) ─────────────────────────────

use svod_dtype::DType;

#[derive(Clone, Debug)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
    /// Compute dtype for projected weights and activations. Defaults to
    /// Float16; set to Float32 for CPU parity tests or hardware without fp16
    /// support. fp8 is viable on AMD-GPU paths only (CPU treats it as raw i8).
    pub dtype: DType,
}

impl ModelDimensions {
    pub fn is_multilingual(&self) -> bool {
        self.n_vocab >= 51865
    }

    pub fn num_languages(&self) -> usize {
        self.n_vocab - 51765 - self.is_multilingual() as usize
    }

    /// The dtype both K/V caches are stored at. The projections produce the
    /// activation dtype, so storing anything else either widens for nothing or
    /// narrows silently; fp8 is the exception, since attention cannot read it.
    pub fn cache_dtype(&self) -> DType {
        let fp8 = [DType::FP8E4M3, DType::FP8E4M3FNUZ, DType::FP8E5M2, DType::FP8E5M2FNUZ];
        if fp8.contains(&self.dtype) { DType::Float16 } else { self.dtype.clone() }
    }

    /// Known model size presets. Dims match OpenAI's checkpoints; the encoder
    /// and decoder share width and head count in every one.
    pub fn for_size(size: WhisperSize) -> Self {
        use WhisperSize::*;
        let (n_mels, n_state, n_head, n_audio_layer, n_text_layer, n_vocab) = match size {
            TinyEn => (80, 384, 6, 4, 4, 51864),
            Tiny => (80, 384, 6, 4, 4, 51865),
            BaseEn => (80, 512, 8, 6, 6, 51864),
            Base => (80, 512, 8, 6, 6, 51865),
            SmallEn => (80, 768, 12, 12, 12, 51864),
            Small => (80, 768, 12, 12, 12, 51865),
            MediumEn => (80, 1024, 16, 24, 24, 51864),
            Medium => (80, 1024, 16, 24, 24, 51865),
            LargeV1 | LargeV2 => (80, 1280, 20, 32, 32, 51865),
            LargeV3 => (128, 1280, 20, 32, 32, 51866),
            Turbo => (128, 1280, 20, 32, 4, 51866),
        };
        Self {
            n_mels,
            n_audio_ctx: N_AUDIO_CTX,
            n_audio_state: n_state,
            n_audio_head: n_head,
            n_audio_layer,
            n_vocab,
            n_text_ctx: N_TEXT_CTX,
            n_text_state: n_state,
            n_text_head: n_head,
            n_text_layer,
            dtype: DType::Float16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperSize {
    TinyEn,
    Tiny,
    BaseEn,
    Base,
    SmallEn,
    Small,
    MediumEn,
    Medium,
    LargeV1,
    LargeV2,
    LargeV3,
    Turbo,
}

impl WhisperSize {
    pub fn name(&self) -> &'static str {
        match self {
            Self::TinyEn => "tiny.en",
            Self::Tiny => "tiny",
            Self::BaseEn => "base.en",
            Self::Base => "base",
            Self::SmallEn => "small.en",
            Self::Small => "small",
            Self::MediumEn => "medium.en",
            Self::Medium => "medium",
            Self::LargeV1 => "large-v1",
            Self::LargeV2 => "large-v2",
            Self::LargeV3 => "large-v3",
            Self::Turbo => "turbo",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "tiny.en" => Some(Self::TinyEn),
            "tiny" => Some(Self::Tiny),
            "base.en" => Some(Self::BaseEn),
            "base" => Some(Self::Base),
            "small.en" => Some(Self::SmallEn),
            "small" => Some(Self::Small),
            "medium.en" => Some(Self::MediumEn),
            "medium" => Some(Self::Medium),
            "large-v1" => Some(Self::LargeV1),
            "large-v2" => Some(Self::LargeV2),
            "large-v3" => Some(Self::LargeV3),
            "large" => Some(Self::LargeV3),
            "turbo" => Some(Self::Turbo),
            _ => None,
        }
    }

    /// Cross-attention alignment heads for DTW word-level timestamps.
    /// Each tuple is (text_layer, head). Decoded from the base85 `_ALIGNMENT_HEADS`
    /// blob in OpenAI's `whisper/__init__.py`.
    pub fn alignment_heads(&self) -> &'static [(usize, usize)] {
        match self {
            Self::TinyEn => &[(1, 0), (2, 0), (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
            Self::Tiny => &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
            Self::BaseEn => &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
            Self::Base => &[(3, 1), (4, 2), (4, 3), (4, 7), (5, 1), (5, 2), (5, 4), (5, 6)],
            Self::SmallEn => &[
                (6, 6),
                (7, 0),
                (7, 3),
                (7, 8),
                (8, 2),
                (8, 5),
                (8, 7),
                (9, 0),
                (9, 4),
                (9, 8),
                (9, 10),
                (10, 0),
                (10, 1),
                (10, 2),
                (10, 3),
                (10, 6),
                (10, 11),
                (11, 2),
                (11, 4),
            ],
            Self::Small => &[(5, 3), (5, 9), (8, 0), (8, 4), (8, 7), (8, 8), (9, 0), (9, 7), (9, 9), (10, 5)],
            Self::MediumEn => &[
                (11, 4),
                (14, 1),
                (14, 12),
                (14, 14),
                (15, 4),
                (16, 0),
                (16, 4),
                (16, 9),
                (17, 12),
                (17, 14),
                (18, 7),
                (18, 10),
                (18, 15),
                (20, 0),
                (20, 3),
                (20, 9),
                (20, 14),
                (21, 12),
            ],
            Self::Medium => &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
            Self::LargeV1 => &[(9, 19), (11, 2), (11, 4), (11, 17), (22, 7), (22, 11), (22, 17), (23, 2), (23, 15)],
            Self::LargeV2 => &[
                (10, 12),
                (13, 17),
                (16, 11),
                (16, 12),
                (16, 13),
                (17, 15),
                (17, 16),
                (18, 4),
                (18, 11),
                (18, 19),
                (19, 11),
                (21, 2),
                (21, 3),
                (22, 3),
                (22, 9),
                (22, 12),
                (23, 5),
                (23, 7),
                (23, 13),
                (25, 5),
                (26, 1),
                (26, 12),
                (27, 15),
            ],
            Self::LargeV3 => {
                &[(7, 0), (10, 17), (12, 18), (13, 12), (16, 1), (17, 14), (19, 11), (21, 4), (24, 1), (25, 6)]
            }
            Self::Turbo => &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
        }
    }
}
