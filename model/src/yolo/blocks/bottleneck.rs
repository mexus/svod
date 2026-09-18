use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::conv::{YoloConv, store_channels_last, tensor_core_dtype};
use crate::state::scoped;
use crate::yolo::error::Result;

/// Standard YOLO bottleneck: two Conv+BN+SiLU layers with optional residual.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone, Module)]
pub struct YoloBottleneck {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
    /// The block's output is stored channels-last; see [`Self::channels_last`].
    #[module(skip)]
    pub channels_last: bool,
}

impl YoloBottleneck {
    /// Default: `k=(3,3)`, `e=0.5`.
    pub fn empty(in_ch: usize, out_ch: usize, shortcut: bool) -> Self {
        Self::empty_full(in_ch, out_ch, shortcut, 3, 3, 0.5)
    }

    /// Full control: separate kernel sizes for cv1/cv2 and expansion ratio.
    pub fn empty_full(in_ch: usize, out_ch: usize, shortcut: bool, k1: usize, k2: usize, e: f64) -> Self {
        let c_ = (out_ch as f64 * e) as usize;
        let add = shortcut && in_ch == out_ch;
        let cv1 = YoloConv::empty(in_ch, c_, k1, 1, true);
        Self { cv1, cv2: YoloConv::empty(c_, out_ch, k2, 1, true), add, channels_last: false }
    }

    /// Store both `cv1`'s output and the block's channels-last: `cv1` feeds
    /// only `cv2`, a 3x3, and the block's output goes to the next block's 3x3
    /// or a 1x1. The residual add stays in `cv2`'s epilogue, ahead of the store.
    pub fn channels_last(mut self) -> Self {
        self.cv1 = self.cv1.channels_last().channels_last_input();
        self.cv2 = self.cv2.channels_last_input();
        self.channels_last = true;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = scoped("cv1", || self.cv1.forward(x))?;
        let out = scoped("cv2", || self.cv2.forward(&h))?;
        let out = if self.add { out.try_add(x)? } else { out };
        if self.channels_last && tensor_core_dtype(&out.dtype()) { store_channels_last(&out) } else { Ok(out) }
    }
}
