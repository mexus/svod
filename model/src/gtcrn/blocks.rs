//! GTCRN weight structs and forward passes — Rust ports of the upstream
//! PyTorch modules (`submodules/gtcrn/gtcrn.py`). All layers operate on the
//! `(B, C, T, F)` layout the encoder/decoder use.
//!
//! Every struct derives [`Module`], so the converted checkpoint's keys map 1:1
//! onto the field names; `#[module(key = "…")]` carries the handful of places
//! where PyTorch spells a submodule differently (`att_gru`, `att_fc`, the `_l0`
//! GRU suffixes). The batch norms keep PyTorch's raw
//! `running_mean`/`running_var` and fold them at forward time, so no state-dict
//! rewriting happens on the way in.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{
    BatchNorm2d, Conv2d, ConvTranspose2d, GruDirection, Layer, LayerNorm, Linear, Module, RnnLayout,
};

use crate::init::{fan_in_uniform, zeros};

use super::error::Result;

// --------------------------------------------------------------------------- //
// ERB subband transform
// --------------------------------------------------------------------------- //

/// The two frozen bias-free `nn.Linear` matrices that project between the full
/// 257-bin spectrum and the 129-band ERB representation. `erb_subband_1 = 65`
/// low bands pass through unchanged; the top `192 = 257 - 65` bins are linearly
/// combined into `64` ERB bands (and back).
#[derive(Clone, Module)]
pub struct Erb {
    /// `[erb_subband_2=64, nfreqs - erb_subband_1=192]` — analysis (bm).
    pub erb_fc: Linear,
    /// `[192, 64]` — synthesis (bs); stored as the transposed analysis matrix.
    pub ierb_fc: Linear,
}

const ERB_SUBBAND_1: usize = 65;
const ERB_SUBBAND_2: usize = 64;

impl Erb {
    pub fn empty() -> Self {
        Self {
            erb_fc: Linear::new(fan_in_uniform(&[ERB_SUBBAND_2, 192], 192, DType::Float32), None),
            ierb_fc: Linear::new(fan_in_uniform(&[192, ERB_SUBBAND_2], ERB_SUBBAND_2, DType::Float32), None),
        }
    }

    /// Analysis: `(B, C, T, 257) -> (B, C, T, 129)`. Low 65 bands pass through;
    /// high 192 bands are projected to 64 via `erb_fc`, then concatenated.
    pub fn bm(&self, x: &Tensor) -> Result<Tensor> {
        self.split_project(x, 257, &self.erb_fc)
    }

    /// Synthesis: `(B, C, T, 129) -> (B, C, T, 257)`. Low 65 bands pass through;
    /// the high 64 bands are projected back to 192 via `ierb_fc`, then
    /// concatenated (inverse of [`bm`](Self::bm)).
    pub fn bs(&self, x_erb: &Tensor) -> Result<Tensor> {
        self.split_project(x_erb, 129, &self.ierb_fc)
    }

    /// Pass the low 65 bins through and project `[65..end)` with `fc`.
    fn split_project(&self, x: &Tensor, end: usize, fc: &Linear) -> Result<Tensor> {
        let low = x.narrow(-1, 0usize, ERB_SUBBAND_1)?;
        let high = fc.forward(&x.narrow(-1, ERB_SUBBAND_1, end - ERB_SUBBAND_1)?)?;
        Ok(Tensor::cat(&[&low, &high], -1)?)
    }
}

// --------------------------------------------------------------------------- //
// SFE — Subband Feature Extraction (nn.Unfold over the F axis)
// --------------------------------------------------------------------------- //

/// Unfold a `(B,C,T,F)` tensor into `(B, C*kernel, T, F)` by extracting
/// `kernel`-wide sliding windows along the F axis (symmetric padding). This
/// mirrors `nn.Unfold(kernel_size=(1,K), stride=(1,1), padding=(0,(K-1)//2))`
/// followed by the reshape in `SFE.forward`. Built on [`Tensor::unfold`].
pub fn sfe(x: &Tensor, kernel: usize, in_channels: usize) -> Result<Tensor> {
    let pad = (kernel - 1) / 2;
    let ndim = x.ndim()?;
    let mut pad_spec = vec![(0isize, 0isize); ndim];
    pad_spec[ndim - 1] = (pad as isize, pad as isize);
    let xp = x.try_pad(&pad_spec)?;
    // unfold along F (last axis): (B,C,T,F) -> (B,C,T,n_windows,kernel).
    // n_windows == F here (symmetric pad keeps the dim), so the result is
    // (B,C,T,F,kernel).
    let unfolded = xp.unfold(-1, kernel, 1)?;
    // Move kernel next to channels: (B, C, kernel, T, F).
    let permuted = unfolded.try_permute(&[0, 1, 4, 2, 3])?;
    // Merge (C, kernel) -> C*kernel. The other axes are carried over as the
    // `SInt`s they already are, so a symbolic batch stays symbolic instead of
    // being pinned to a guessed constant.
    let s = permuted.shape()?;
    Ok(permuted.try_reshape([s[0].clone(), SInt::Const(in_channels * kernel), s[3].clone(), s[4].clone()])?)
}

// --------------------------------------------------------------------------- //
// Conv blocks
// --------------------------------------------------------------------------- //

/// A GTCRN conv is either an encoder [`Conv2d`] or a decoder
/// [`ConvTranspose2d`]. Both spell their state the same way (`weight`, `bias`),
/// so the state dict is identical either way and the direction stays a
/// construction-time choice.
#[derive(Clone, Module)]
pub enum Conv {
    Normal(Conv2d),
    Transposed(ConvTranspose2d),
}

impl Conv {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Normal(c) => c.forward(x)?,
            Self::Transposed(c) => c.forward(x)?,
        })
    }

    /// Fold a causal `(pad_size, 0)` pad on the T axis into this conv's own
    /// padding, which saves the standalone pad op.
    ///
    /// A [`Conv2d`] pads its input, so the offset simply lands on the leading T
    /// pad. A [`ConvTranspose2d`] instead *crops* by `padding`, expanding to
    /// `begin = (kH-1)·dilation - before`, so widening the input by `pad_size`
    /// in front is the same as shrinking `before` by `pad_size`. Both hold only
    /// while the T stride is 1, which is true of every GTCRN block.
    pub fn with_causal_pad(self, pad_size: usize) -> Self {
        let pad = pad_size as isize;
        match self {
            Self::Normal(c) => {
                let ((before, after), f) = c.padding;
                assert_eq!(c.stride.0, 1, "causal pad folding assumes unit stride on T");
                Self::Normal(c.with_padding(((before + pad, after), f)))
            }
            Self::Transposed(c) => {
                let ((before, after), f) = c.padding;
                assert_eq!(c.stride.0, 1, "causal pad folding assumes unit stride on T");
                Self::Transposed(c.with_padding(((before - pad, after), f)))
            }
        }
    }
}

/// [`Conv`] + [`BatchNorm2d`] + activation, matching `ConvBlock`. The
/// activation is `PReLU`, except on the last layer, which is `Tanh` and so
/// carries no slope.
#[derive(Clone, Module)]
pub struct ConvBlock {
    pub conv: Conv,
    pub bn: BatchNorm2d,
    /// PReLU slope (single weight). `None` for the `Tanh` final layer.
    #[module(key = "act.weight", optional = "!self.is_last")]
    pub act: Option<Tensor>,
    pub is_last: bool,
}

impl ConvBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.bn.forward(&self.conv.forward(x)?)?;
        Ok(match &self.act {
            Some(slope) => y.prelu(slope)?,
            None => y.tanh()?,
        })
    }
}

/// Grouped Temporal Convolution block (shuffle + depthwise-separable). Handles
/// both the encoder (`use_deconv=false`) and decoder (`use_deconv=true`)
/// variants. Operates on `(B, C=in, T, F)`; splits `C` in halves, processes
/// `x1` through the SFE→pointwise→depthwise→pointwise→TRA path, and shuffles
/// the result with the passthrough `x2`.
///
/// `depth_conv` carries the block's causal T pad in its own padding (see
/// [`Conv::with_causal_pad`]), so the forward has no separate pad op.
#[derive(Clone, Module)]
pub struct GTConvBlock {
    pub in_channels: usize,
    // point_conv1: in/2*3 -> hidden, 1x1
    pub point_conv1: Conv,
    pub point_bn1: BatchNorm2d,
    #[module(key = "point_act.weight")]
    pub point_act: Tensor,
    // depth_conv: hidden -> hidden, kernel, groups=hidden
    pub depth_conv: Conv,
    pub depth_bn: BatchNorm2d,
    #[module(key = "depth_act.weight")]
    pub depth_act: Tensor,
    // point_conv2: hidden -> in/2, 1x1
    pub point_conv2: Conv,
    pub point_bn2: BatchNorm2d,
    // TRA attention (in/2 channels)
    pub tra: Tra,
}

impl GTConvBlock {
    /// ShuffleNet channel shuffle: interleave two `(B,C,T,F)` halves into
    /// `(B, 2C, T, F)`. Mirrors `GTConvBlock.shuffle`. `half_channels` is the
    /// per-half `C` (needed to express the merged `2C` axis).
    pub fn shuffle(x1: &Tensor, x2: &Tensor, half_channels: usize) -> Result<Tensor> {
        // stack -> (B, 2, C, T, F); transpose(1,2) -> (B, C, 2, T, F);
        // flatten C,2 -> (B, 2C, T, F).
        let stacked = Tensor::stack(&[x1, x2], 1)?; // (B,2,C,T,F)
        let t = stacked.try_permute(&[0, 2, 1, 3, 4])?; // (B,C,2,T,F)
        let s = t.shape()?;
        Ok(t.try_reshape([s[0].clone(), SInt::Const(2 * half_channels), s[3].clone(), s[4].clone()])?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let halves = x.chunk(2, 1)?;
        let (x1, x2) = (&halves[0], &halves[1]);

        // SFE on x1: (B, in/2, T, F) -> (B, in/2*3, T, F).
        let half = self.in_channels / 2;
        let x1 = sfe(x1, 3, half)?;
        let h1 = self.point_conv1.forward(&x1)?;
        let h1 = self.point_bn1.forward(&h1)?;
        let h1 = h1.prelu(&self.point_act)?;
        let h1 = self.depth_conv.forward(&h1)?;
        let h1 = self.depth_bn.forward(&h1)?;
        let h1 = h1.prelu(&self.depth_act)?;
        let h1 = self.point_conv2.forward(&h1)?;
        let h1 = self.point_bn2.forward(&h1)?;
        let h1 = self.tra.forward(&h1)?;

        Self::shuffle(&h1, x2, half)
    }
}

// --------------------------------------------------------------------------- //
// TRA — Temporal Recurrent Attention
// --------------------------------------------------------------------------- //

/// `zt = mean(x², F) -> GRU(C, 2C) -> Linear(2C, C) -> sigmoid -> scale`.
/// The GRU runs along the T axis over `(B, T, C)` feature. Unidirectional.
#[derive(Clone, Module)]
pub struct Tra {
    #[module(key = "att_gru")]
    pub gru: GruWeights,
    /// `(C, 2C)` weight, `(C,)` bias
    #[module(key = "att_fc")]
    pub fc: Linear,
}

impl Tra {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T, F). zt = mean(x^2, -1) -> (B, C, T); transpose -> (B, T, C).
        let zt = x.square().mean_with().axes(-1isize).keepdim(false).call()?; // (B,C,T)
        let zt = zt.try_permute(&[0, 2, 1])?; // (B,T,C) — batch-first.
        let at = self.gru.forward(&zt)?; // (B, T, 2C)
        let at = self.fc.forward(&at)?; // (B, T, C)
        let at = at.try_permute(&[0, 2, 1])?.sigmoid()?; // (B, C, T)
        // Broadcast-multiply over F.
        Ok(x.try_mul(&at.try_unsqueeze(-1)?)?)
    }
}

// --------------------------------------------------------------------------- //
// GRU weight container (unidirectional, batch-first)
// --------------------------------------------------------------------------- //

/// A single unidirectional GRU's weights in svod's `[z, r, h]` gate order (the
/// convert script permutes PyTorch's `[r, z, n]` order at conversion time).
/// Stored in PyTorch's 4-tensor layout; packed into the `[num_directions, ...]`
/// shape svod's `gru()` expects at forward time.
#[derive(Clone, Module)]
pub struct GruWeights {
    pub hidden_size: usize,
    /// `(3H, input)`
    #[module(key = "weight_ih_l0")]
    pub weight_ih: Tensor,
    /// `(3H, H)`
    #[module(key = "weight_hh_l0")]
    pub weight_hh: Tensor,
    /// `(3H,)`
    #[module(key = "bias_ih_l0")]
    pub bias_ih: Tensor,
    /// `(3H,)`
    #[module(key = "bias_hh_l0")]
    pub bias_hh: Tensor,
}

impl GruWeights {
    /// Run the GRU over a `(B, T, input_size)` batch-first sequence, returning
    /// `(B, T, hidden_size)`. Uses `linear_before_reset=1` (PyTorch's
    /// `nn.GRU` formulation).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        run_gru(x, &[self], None)
    }

    /// Fuse two GRUs that read disjoint halves of the same input — `self` the
    /// low half, `other` the high one — into a single GRU of input `2I` and
    /// hidden `2H` whose gate matrices are **block diagonal**.
    ///
    /// `W·x + R·h = [W₁x₁ + R₁h₁ ; W₂x₂ + R₂h₂]` when the blocks are disjoint,
    /// and every gate nonlinearity is elementwise, so the merged GRU's output
    /// is exactly the two originals' outputs concatenated — one scan instead of
    /// two, at twice the flops of a matmul this small.
    ///
    /// Rows stay gate-major in svod's ONNX `[z, r, h]` spelling, so each gate
    /// owns `2H` contiguous rows and the two GRUs interleave *within* a gate:
    /// `[z₁, z₂, r₁, r₂, n₁, n₂]` over `[·₁ | 0]` / `[0 | ·₂]` column blocks.
    /// The bias follows the same order.
    fn merge(&self, other: &Self) -> Result<Self> {
        let h = self.hidden_size;
        assert_eq!(other.hidden_size, h, "a GRNN's two groups share a hidden size");
        let gate = |t: &Tensor, g: usize| -> Result<Tensor> { Ok(t.narrow(0, g * h, h)?) };
        // Two `[3H, n]` gate-major matrices -> `[6H, 2n]` block diagonal.
        let block_diag = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
            let pad = zeros(&[h, a.dim_const(1)?], DType::Float32);
            let column = |w: &Tensor, lead: bool| -> Result<Tensor> {
                let mut rows = Vec::with_capacity(6);
                for g in 0..3 {
                    let block = gate(w, g)?;
                    rows.extend(if lead { [block, pad.clone()] } else { [pad.clone(), block] });
                }
                Ok(Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?)
            };
            Ok(Tensor::cat(&[&column(a, true)?, &column(b, false)?], 1)?)
        };
        // Two `[3H]` gate-major vectors -> `[6H]`, interleaved within each gate.
        let interleave = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
            let mut parts = Vec::with_capacity(6);
            for g in 0..3 {
                parts.extend([gate(a, g)?, gate(b, g)?]);
            }
            Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?)
        };
        Ok(Self {
            hidden_size: 2 * h,
            weight_ih: block_diag(&self.weight_ih, &other.weight_ih)?,
            weight_hh: block_diag(&self.weight_hh, &other.weight_hh)?,
            bias_ih: interleave(&self.bias_ih, &other.bias_ih)?,
            bias_hh: interleave(&self.bias_hh, &other.bias_hh)?,
        })
    }
}

/// Drive `gru()` over `dirs` directions' weights, returning nn.GRU's
/// `(B, T, num_directions * hidden)` batch-first output.
///
/// svod's `gru()` takes the ONNX spelling: `w`/`r` as `[D, 3H, *]` and the bias
/// as `[D, 6H]` laid out `[w_bz, w_br, w_bh, r_bz, r_br, r_bh]`, in `[z, r, h]`
/// gate order — which is exactly how the convert script stores the checkpoint.
fn run_gru(x: &Tensor, dirs: &[&GruWeights], direction: Option<GruDirection>) -> Result<Tensor> {
    let stack = |f: fn(&GruWeights) -> &Tensor| -> Result<Tensor> {
        let parts: Vec<Tensor> = dirs.iter().map(|d| f(d).try_unsqueeze(0)).collect::<std::result::Result<_, _>>()?;
        Ok(Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?)
    };
    let bias = Tensor::cat(&[&stack(|d| &d.bias_ih)?, &stack(|d| &d.bias_hh)?], 1)?; // [D, 6H]
    let out = x
        .gru()
        .w(&stack(|d| &d.weight_ih)?)
        .r_weights(&stack(|d| &d.weight_hh)?)
        .hidden_size(dirs[0].hidden_size)
        .bias(&bias)
        .maybe_direction(direction)
        .linear_before_reset(true)
        .layout(RnnLayout::BatchFirst)
        .call()?;
    // `output` is already [batch, seq, D*hidden] for BatchFirst — the two
    // directions concatenated on the feature axis, as nn.GRU returns them.
    Ok(out.output)
}

// --------------------------------------------------------------------------- //
// GRNN — grouped RNN (two GRUs over channel-halves)
// --------------------------------------------------------------------------- //

/// `GRNN`: two GRUs over disjoint halves of the feature axis, outputs
/// concatenated. A present `_b` half makes that GRU bidirectional — the reverse
/// weights are the whole of what "bidirectional" means here, so there is no
/// separate flag.
///
/// The two groups never exchange information, so the forward runs them as one
/// [block-diagonal GRU](GruWeights::merge) — the same arithmetic in half the
/// scans, which is what this model's latency is made of. The four weight sets
/// stay stored apart, so the state dict is untouched.
#[derive(Clone, Module)]
pub struct Grnn {
    pub rnn1_f: GruWeights,
    pub rnn1_b: Option<GruWeights>,
    pub rnn2_f: GruWeights,
    pub rnn2_b: Option<GruWeights>,
}

impl Grnn {
    /// Both groups' forward GRUs as one. Built once per graph — the scan
    /// re-launches only the step, so this never enters the time loop.
    pub(super) fn merged_forward(&self) -> Result<GruWeights> {
        self.rnn1_f.merge(&self.rnn2_f)
    }

    /// `(B, T, input) -> (B, T, hidden)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fwd = self.merged_forward()?;
        let (Some(b1), Some(b2)) = (&self.rnn1_b, &self.rnn2_b) else {
            // Unidirectional: the merged output is `[y₁ | y₂]` already.
            return fwd.forward(x);
        };
        let y = run_gru(x, &[&fwd, &b1.merge(b2)?], Some(GruDirection::Bidirectional))?;
        regroup(&y, self.rnn1_f.hidden_size)
    }
}

/// `[fwd₁ | fwd₂ | rev₁ | rev₂] -> [fwd₁ | rev₁ | fwd₂ | rev₂]`, the order the
/// per-group `cat` of two bidirectional GRUs produced and the consuming FC was
/// trained against. A merged bidirectional GRU concatenates *directions*
/// outermost where the split pair concatenated *groups* outermost, so undoing
/// it is a transpose of the two 2-wide factors of the feature axis — one op on
/// the finished sequence, outside the scan.
fn regroup(y: &Tensor, h: usize) -> Result<Tensor> {
    let s = y.shape()?;
    let (b, t) = (s[0].clone(), s[1].clone());
    let split = y.try_reshape([b.clone(), t.clone(), SInt::Const(2), SInt::Const(2), SInt::Const(h)])?;
    Ok(split.try_permute(&[0, 1, 3, 2, 4])?.try_reshape([b, t, SInt::Const(4 * h)])?)
}

// --------------------------------------------------------------------------- //
// DPGRNN — dual-path grouped RNN
// --------------------------------------------------------------------------- //

/// `DPGRNN`: intra-chunk (frequency-axis) bidirectional RNN + inter-chunk
/// (time-axis) unidirectional RNN, each with FC + LayerNorm + residual add.
#[derive(Clone, Module)]
pub struct Dpgrnn {
    /// bidirectional, input=C=16 split 8/8, hidden=4 → output 16; runs along F
    pub intra_rnn: Grnn,
    pub intra_fc: Linear,
    /// `(width, hidden)` = `(33, 16)` affine, normalizing over both axes.
    pub intra_ln: LayerNorm,
    /// unidirectional, input=C=16 split 8/8, hidden=8 → output 16; runs along T
    pub inter_rnn: Grnn,
    pub inter_fc: Linear,
    pub inter_ln: LayerNorm,
}

impl Dpgrnn {
    pub(crate) const LN_EPS: f64 = 1e-8;

    /// `(B, C, T, F) -> (B, C, T, F)`. `C` = `hidden_size`, `F` = `width`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.try_permute(&[0, 2, 3, 1])?; // (B,T,F,C)
        let s = x.shape()?;
        let (b, t, f, c) = (s[0].clone(), s[1].clone(), s[2].clone(), s[3].clone());

        // Intra RNN: run over the F axis for each (B,T) slice.
        let intra_x = x.try_reshape([b.clone() * t.clone(), f.clone(), c.clone()])?;
        let intra_x = self.intra_rnn.forward(&intra_x)?;
        let intra_x = self.intra_fc.forward(&intra_x)?;
        let intra_x = intra_x.try_reshape([b.clone(), t.clone(), f.clone(), c.clone()])?;
        let intra_x = self.intra_ln.forward(&intra_x)?;
        let intra_out = x.try_add(&intra_x)?;

        // Inter RNN: run over the T axis for each (B,F) slice.
        let inter_in = intra_out.try_permute(&[0, 2, 1, 3])?; // (B,F,T,C)
        let inter_x = inter_in.try_reshape([b.clone() * f.clone(), t.clone(), c.clone()])?;
        let inter_x = self.inter_rnn.forward(&inter_x)?;
        let inter_x = self.inter_fc.forward(&inter_x)?;
        let inter_x = inter_x.try_reshape([b, f, t, c])?.try_permute(&[0, 2, 1, 3])?; // (B,T,F,C)
        let inter_x = self.inter_ln.forward(&inter_x)?;
        let inter_out = intra_out.try_add(&inter_x)?;

        Ok(inter_out.try_permute(&[0, 3, 1, 2])?) // (B,C,T,F)
    }
}
