//! RNN-T joint network: encoder + predictor projections combined into per-step
//! log-probabilities.

use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_tensor::nn::Module;

use crate::init::fan_in_uniform;

use crate::gigaam::Result;

/// RNN-T joint: `log_softmax(out_w · ReLU(enc_w · enc_t + enc_b + pred_w · g + pred_b) + out_b)`.
///
/// Class-axis alignment for [`RnntJoint::pad_classes`]: the vocab argmax only
/// lowers to a grouped multi-thread reduction when its axis is a multiple of
/// 16 (GigaAM's 1025 is not); 32 measured slightly slower.
pub(crate) const CLASS_ALIGN: usize = 16;

/// All Linear weights stored PyTorch-style `[out_features, in_features]` so
/// they plug straight into the `linear()` builder (which transposes
/// internally).
#[derive(Clone, Module)]
pub struct RnntJoint {
    pub enc_w: Tensor,
    pub enc_b: Tensor,
    pub pred_w: Tensor,
    pub pred_b: Tensor,
    /// `[padded_classes, joint_hidden]`; rows past `num_classes` are zero
    /// (see [`Self::pad_classes`]).
    pub out_w: Tensor,
    /// `[padded_classes]`; entries past `num_classes` are `-1e30`.
    pub out_b: Tensor,
    /// Real class count (vocab + blank); the logits width may be padded.
    pub num_classes: usize,
}

impl RnntJoint {
    pub fn empty(enc_hidden: usize, pred_hidden: usize, joint_hidden: usize, num_classes: usize) -> Self {
        Self {
            enc_w: fan_in_uniform(&[joint_hidden, enc_hidden], enc_hidden, DType::Float32),
            enc_b: fan_in_uniform(&[joint_hidden], enc_hidden, DType::Float32),
            pred_w: fan_in_uniform(&[joint_hidden, pred_hidden], pred_hidden, DType::Float32),
            pred_b: fan_in_uniform(&[joint_hidden], pred_hidden, DType::Float32),
            out_w: fan_in_uniform(&[num_classes, joint_hidden], joint_hidden, DType::Float32),
            out_b: fan_in_uniform(&[num_classes], joint_hidden, DType::Float32),
            num_classes,
        }
    }

    /// Pad the output projection to a multiple of `align` classes: zero weight
    /// rows and `-1e30` biases, so a padded class can never win the argmax and
    /// the token ids stay `< num_classes`. Idempotent; realized once here so
    /// the decode plan sees plain buffers.
    pub(crate) fn pad_classes(&mut self, align: usize) -> Result<()> {
        let extra = self.num_classes.div_ceil(align) * align - self.num_classes;
        if extra == 0 || self.out_w.dim_const(0)? > self.num_classes {
            return Ok(());
        }
        let neg = Tensor::full(&[extra], -1e30, self.out_b.dtype());
        self.out_w = self.out_w.try_pad(&[(0, extra as isize), (0, 0)])?.contiguous();
        self.out_b = Tensor::cat(&[&self.out_b, &neg], 0)?.contiguous();
        Tensor::realize_batch([&self.out_w, &self.out_b])?;
        Ok(())
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → raw logits
    /// `[1, 1, num_classes]` (pre-softmax), padding sliced off.
    fn logits(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let enc_proj = enc_t.linear().weight(&self.enc_w).bias(&self.enc_b).call()?;
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call()?;
        let summed = enc_proj.try_add(&pred_proj)?;
        let activated = summed.relu()?;
        let logits = activated.linear().weight(&self.out_w).bias(&self.out_b).call()?;
        Ok(logits.narrow(-1, 0usize, self.num_classes)?)
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → log-probs
    /// `[1, 1, num_classes]`.
    pub fn forward(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        Ok(self.logits(enc_t, g)?.log_softmax(-1isize)?)
    }

    /// Greedy variant: the device-side argmax token index `[1, 1]` (int32)
    /// over the vocab. `log_softmax` is omitted — argmax is invariant under the
    /// monotonic log-softmax, so the chosen index is identical while the host
    /// reads back a single int instead of the full vocab logit vector.
    pub fn forward_argmax(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        Ok(self.logits(enc_t, g)?.argmax(-1isize)?)
    }

    /// Encoder projection `enc_w · enc + enc_b` over a whole frame axis —
    /// hoisted out of the decode loop (`[B, T, E] → [B, T, J]`, one MFMA
    /// matmul per wave instead of a per-step row projection).
    pub fn project_encoder(&self, enc: &Tensor) -> Result<Tensor> {
        Ok(enc.linear().weight(&self.enc_w).bias(&self.enc_b).call()?)
    }

    /// Greedy argmax over PRE-PROJECTED encoder rows ([`Self::project_encoder`]).
    /// Runs over the padded class axis: padded classes never win, so the index
    /// is always `< num_classes`.
    pub fn argmax_preproj(&self, enc_proj_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call()?;
        let activated = enc_proj_t.try_add(&pred_proj)?.relu()?;
        let logits = activated.linear().weight(&self.out_w).bias(&self.out_b).call()?;
        Ok(logits.argmax(-1isize)?)
    }
}
