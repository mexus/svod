use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, ConvTranspose2d, Layer, Module};

use crate::blocks::{batchnorm2d_with_eps, conv2d, conv2d_grouped};
use crate::init::fan_in_uniform;

use crate::yolo::error::Result;

/// Ultralytics' `initialize_weights` rewrites every BatchNorm's epsilon to
/// 1e-3, so YOLO checkpoints are not normalized with PyTorch's 1e-5 default.
pub const YOLO_BN_EPS: f64 = 1e-3;

fn bn(channels: usize) -> BatchNorm2d {
    batchnorm2d_with_eps(channels, YOLO_BN_EPS)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A `kernel×kernel` convolution with a bias and `kernel / 2` padding, as the
/// Detect head's final 1×1 layers use it. State-dict keys: `weight`, `bias`.
pub fn conv2d_bias(in_ch: usize, out_ch: usize, kernel: usize, stride: usize) -> Conv2d {
    let fan_in = in_ch * kernel * kernel;
    let bias = fan_in_uniform(&[out_ch], fan_in, DType::Float32);
    let p = (kernel / 2) as isize;
    Conv2d::new(fan_in_uniform(&[out_ch, in_ch, kernel, kernel], fan_in, DType::Float32), Some(bias))
        .with_stride((stride, stride))
        .with_padding(((p, p), (p, p)))
}

/// A biased transposed convolution that doubles the spatial resolution.
/// State-dict keys: `weight`, `bias`.
pub fn deconv2d_2x(in_ch: usize, out_ch: usize, kernel: usize) -> ConvTranspose2d {
    let fan_in = in_ch * kernel * kernel;
    ConvTranspose2d::new(
        fan_in_uniform(&[in_ch, out_ch, kernel, kernel], fan_in, DType::Float32),
        Some(fan_in_uniform(&[out_ch], fan_in, DType::Float32)),
    )
    .with_stride((2, 2))
}

/// Conv2d(bias=False) + BatchNorm2d + SiLU — the universal YOLO building block.
/// When `act` is `false` the activation is skipped (used by SPPF.cv1,
/// Attention projections, and PSABlock FFN output conv).
///
/// State-dict keys: `conv.weight`, `bn.{weight,bias,running_mean,running_var}`.
/// A checkpoint load folds the norm into the conv ([`fold_batchnorm`]), and a
/// conv that carries a bias is taken as already normalized.
///
/// [`fold_batchnorm`]: crate::yolo::loader::fold_batchnorm
#[derive(Clone, Module)]
pub struct YoloConv {
    pub conv: Conv2d,
    pub bn: BatchNorm2d,
    pub act: bool,
}

impl YoloConv {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        Self { conv: conv2d(out_ch, in_ch, kernel, stride, kernel / 2), bn: bn(out_ch), act }
    }

    /// Depthwise variant: `groups = gcd(in_ch, out_ch)`.
    pub fn empty_dw(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let groups = gcd(in_ch, out_ch);
        Self { conv: conv2d_grouped(out_ch, in_ch, kernel, stride, kernel / 2, groups), bn: bn(out_ch), act }
    }

    /// Accumulate the conv in `dtype` and keep the block's output there, so
    /// half-width operands still leave the norm and activation at full width.
    /// (Doing so for every block costs 5% of the forward for 0.01 px, so the
    /// default rounds the epilogue to the operand dtype.)
    pub fn with_acc_dtype(mut self, dtype: DType) -> Self {
        self.conv = self.conv.with_acc_dtype(dtype);
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = if self.conv.bias.is_some() { x } else { self.bn.forward(&x)? };
        if self.act { Ok(x.silu()?) } else { Ok(x) }
    }
}
