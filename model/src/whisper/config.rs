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

    /// Known model size presets.  Dims match OpenAI's checkpoints.
    pub fn for_size(size: WhisperSize) -> Self {
        match size {
            WhisperSize::TinyEn => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 384,
                n_audio_head: 6,
                n_audio_layer: 4,
                n_vocab: 51864,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 384,
                n_text_head: 6,
                n_text_layer: 4,
                dtype: DType::Float16,
            },
            WhisperSize::Tiny => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 384,
                n_audio_head: 6,
                n_audio_layer: 4,
                n_vocab: 51865,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 384,
                n_text_head: 6,
                n_text_layer: 4,
                dtype: DType::Float16,
            },
            WhisperSize::BaseEn => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 512,
                n_audio_head: 8,
                n_audio_layer: 6,
                n_vocab: 51864,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 512,
                n_text_head: 8,
                n_text_layer: 6,
                dtype: DType::Float16,
            },
            WhisperSize::Base => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 512,
                n_audio_head: 8,
                n_audio_layer: 6,
                n_vocab: 51865,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 512,
                n_text_head: 8,
                n_text_layer: 6,
                dtype: DType::Float16,
            },
            WhisperSize::SmallEn => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 768,
                n_audio_head: 12,
                n_audio_layer: 12,
                n_vocab: 51864,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 768,
                n_text_head: 12,
                n_text_layer: 12,
                dtype: DType::Float16,
            },
            WhisperSize::Small => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 768,
                n_audio_head: 12,
                n_audio_layer: 12,
                n_vocab: 51865,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 768,
                n_text_head: 12,
                n_text_layer: 12,
                dtype: DType::Float16,
            },
            WhisperSize::MediumEn => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 1024,
                n_audio_head: 16,
                n_audio_layer: 24,
                n_vocab: 51864,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 1024,
                n_text_head: 16,
                n_text_layer: 24,
                dtype: DType::Float16,
            },
            WhisperSize::Medium => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 1024,
                n_audio_head: 16,
                n_audio_layer: 24,
                n_vocab: 51865,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 1024,
                n_text_head: 16,
                n_text_layer: 24,
                dtype: DType::Float16,
            },
            WhisperSize::LargeV1 | WhisperSize::LargeV2 => Self {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 1280,
                n_audio_head: 20,
                n_audio_layer: 32,
                n_vocab: 51865,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 1280,
                n_text_head: 20,
                n_text_layer: 32,
                dtype: DType::Float16,
            },
            WhisperSize::LargeV3 => Self {
                n_mels: 128,
                n_audio_ctx: 1500,
                n_audio_state: 1280,
                n_audio_head: 20,
                n_audio_layer: 32,
                n_vocab: 51866,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 1280,
                n_text_head: 20,
                n_text_layer: 32,
                dtype: DType::Float16,
            },
            WhisperSize::Turbo => Self {
                n_mels: 128,
                n_audio_ctx: 1500,
                n_audio_state: 1280,
                n_audio_head: 20,
                n_audio_layer: 4,
                n_vocab: 51866,
                n_text_ctx: N_TEXT_CTX,
                n_text_state: 1280,
                n_text_head: 20,
                n_text_layer: 8,
                dtype: DType::Float16,
            },
        }
    }
}

/// Element type of the cross-attention K/V caches.
///
/// The projection produces these in the model's activation dtype and they are
/// only read back by attention, which accumulates in f32 regardless. Storing
/// them wider buys nothing and costs twice the VRAM and twice the bandwidth of
/// the kernel that streams them -- at large-v3 that is gigabytes.
pub(crate) fn cross_cache_dtype() -> DType {
    DType::Float16
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

    /// OpenAI download URL for the `.pt` checkpoint.
    pub fn url(&self) -> &'static str {
        match self {
            Self::TinyEn => {
                "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"
            }
            Self::Tiny => {
                "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
            }
            Self::BaseEn => {
                "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt"
            }
            Self::Base => {
                "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
            }
            Self::SmallEn => {
                "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt"
            }
            Self::Small => {
                "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
            }
            Self::MediumEn => {
                "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt"
            }
            Self::Medium => {
                "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
            }
            Self::LargeV1 => {
                "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt"
            }
            Self::LargeV2 => {
                "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
            }
            Self::LargeV3 => {
                "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
            }
            Self::Turbo => {
                "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"
            }
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
