//! Construction-time geometry for prepared Whisper graphs.

use super::config::{ModelDimensions, N_AUDIO_CTX, N_TEXT_CTX, WhisperSize};

/// Static capacities compiled into a Whisper pipeline. Changing this plan
/// requires constructing a new pipeline; requests never alter graph geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WhisperPlan {
    /// Windows encoded by one encoder dispatch.
    pub encoder_batch: usize,
    /// Stable rows compiled into the cached decoder-step graph.
    pub decoder_slots: usize,
    /// Windows aligned by one teacher-forced alignment dispatch.
    pub alignment_batch: usize,
}

impl WhisperPlan {
    /// Derive recognition capacities from the model geometry and memory budget.
    pub fn for_recognizer(dims: &ModelDimensions) -> Self {
        const ENCODER_SCORES_BUDGET: usize = 512 * 1024 * 1024;
        let encoder_scores = dims.n_audio_head * N_AUDIO_CTX * N_AUDIO_CTX * std::mem::size_of::<f32>();
        let encoder_batch = (ENCODER_SCORES_BUDGET / encoder_scores).clamp(1, 8);
        // Five rows keep the default beam strategy constructible even when the
        // encoder memory budget selects a smaller batch.
        Self { encoder_batch, decoder_slots: encoder_batch.max(5), alignment_batch: 1 }
    }

    /// Derive recognition and alignment capacities for a model size.
    pub fn for_model(dims: &ModelDimensions, size: WhisperSize) -> Self {
        const ALIGNMENT_OUTPUT_BUDGET: usize = 256 * 1024 * 1024;
        let mut plan = Self::for_recognizer(dims);
        let alignment_output = size.alignment_heads().len() * N_TEXT_CTX * N_AUDIO_CTX * std::mem::size_of::<f32>();
        plan.alignment_batch = (ALIGNMENT_OUTPUT_BUDGET / alignment_output).clamp(1, plan.encoder_batch);
        plan
    }

    /// Capacities for a driver that hands over one window at a time, as the
    /// ASR pipeline does for a transcriber that advances by what it decoded:
    /// the encoder and aligner at batch 1, the decoder slots kept for the beam.
    pub fn sequential(dims: &ModelDimensions, size: WhisperSize) -> Self {
        Self { encoder_batch: 1, alignment_batch: 1, ..Self::for_model(dims, size) }
    }

    /// Validate that every compiled capacity is nonzero.
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        if self.encoder_batch == 0 {
            return Err("encoder_batch must be non-zero");
        }
        if self.decoder_slots == 0 {
            return Err("decoder_slots must be non-zero");
        }
        if self.alignment_batch == 0 {
            return Err("alignment_batch must be non-zero");
        }
        Ok(())
    }
}
