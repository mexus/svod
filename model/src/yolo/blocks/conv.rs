use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{BatchNorm2d, Conv2d, ConvTranspose2d, Layer, Module, StateDict, prefixed};

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
///
/// Layouts follow the tensor core, which reduces over the channels and wants a
/// fragment's K elements contiguous (at f32, which has no core on RDNA4, both
/// stay as loaded). A conv reading NCHW activations gets its
/// `k x k` weight stored taps-major, `[cout, kh, kw, cin]`, at load (under
/// BEAM a stride-2 3x3 runs 1.9-2.8x faster on gfx1201); one reading
/// channels-last activations keeps the checkpoint's `[cout, cin, kh, kw]`,
/// which measured faster there. [`Self::channels_last`] picks the output
/// layout, [`Self::channels_last_input`] declares the input's.
#[derive(Clone)]
pub struct YoloConv {
    pub conv: Conv2d,
    pub bn: BatchNorm2d,
    pub act: bool,
    /// Store the output channels-last; see [`Self::channels_last`].
    pub channels_last: bool,
    /// The input arrives channels-last; see [`Self::channels_last_input`].
    pub channels_last_input: bool,
}

impl YoloConv {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let conv = conv2d(out_ch, in_ch, kernel, stride, kernel / 2);
        Self { conv, bn: bn(out_ch), act, channels_last: false, channels_last_input: false }
    }

    /// Depthwise variant: `groups = gcd(in_ch, out_ch)`.
    pub fn empty_dw(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, act: bool) -> Self {
        let groups = gcd(in_ch, out_ch);
        let conv = conv2d_grouped(out_ch, in_ch, kernel, stride, kernel / 2, groups);
        Self { conv, bn: bn(out_ch), act, channels_last: false, channels_last_input: false }
    }

    /// Store the output channels-last, handing on the NCHW view every consumer
    /// expects. Under BEAM a 3x3 stride-1 conv reading it is 1.5-2.7x faster on
    /// gfx1201, a stride-2 conv 2-3x slower, so the producer chooses by what
    /// consumes it.
    pub fn channels_last(mut self) -> Self {
        self.channels_last = true;
        self
    }

    /// The input is stored channels-last, so the weight stays `cin`-major.
    pub fn channels_last_input(mut self) -> Self {
        self.channels_last_input = true;
        self
    }

    fn taps_major(&self) -> bool {
        !self.channels_last_input
            && tensor_core_dtype(&self.conv.weight.dtype())
            && self.conv.weight.dims().is_ok_and(|d| d.len() == 4 && d[2] * d[3] > 1)
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
        let x = if self.act { x.silu()? } else { x };
        if self.channels_last && tensor_core_dtype(&x.dtype()) { store_channels_last(&x) } else { Ok(x) }
    }
}

/// The layouts serve the tensor core, which the half-width dtypes reach; an
/// f32 model keeps the checkpoint's, which the scalar path reads faster.
pub(crate) fn tensor_core_dtype(dtype: &DType) -> bool {
    *dtype == DType::Float16 || *dtype == DType::BFloat16
}

impl Module for YoloConv {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.conv.write_state(&prefixed(prefix, "conv"), out);
        self.bn.write_state(&prefixed(prefix, "bn"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.bn.load_state_dict(sd, &prefixed(prefix, "bn"))?;
        if self.taps_major() {
            let taps_major = self.conv.weight.try_permute(&[0, 2, 3, 1])?.contiguous();
            taps_major.realize()?;
            // Kept a view: realizing it would copy the bytes back cin-major.
            self.conv.weight = taps_major.try_permute(&[0, 3, 1, 2])?;
        }
        Ok(())
    }
}

/// Realize an NCHW tensor channels-last and hand back the NCHW view of it.
/// Right after a conv the permute folds into the kernel's store.
pub(crate) fn store_channels_last(x: &Tensor) -> Result<Tensor> {
    Ok(x.try_permute(&[0, 2, 3, 1])?.contiguous().try_permute(&[0, 3, 1, 2])?)
}
