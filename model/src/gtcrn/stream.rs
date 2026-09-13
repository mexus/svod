//! Streaming GTCRN — frame-by-frame causal speech enhancement.
//!
//! A pure-Rust port of the upstream `StreamGTCRN`
//! (`submodules/gtcrn/stream/gtcrn_stream.py`). Processes one STFT frame
//! (`T=1`) per call, threading recurrent state (conv caches, GRU hidden states)
//! between calls. Because the time dimension is pinned to 1 at JIT-prepare
//! time, the GRU recurrence unrolls exactly one IR node per call — no graph
//! explosion regardless of total audio length.
//!
//! ## State
//!
//! Three families of caches, all zero-init at start and recycled **on-device**
//! (the host never copies recurrent state — see [`forward_jit`]):
//!
//! | cache | count | per-call shape | what it holds |
//! |-------|-------|--------------|---------------|
//! | conv cache | 6 (3 enc + 3 dec) | `(1, 16, 2·d, 33)` | T-history for one depthwise conv; `d` = dilation |
//! | tra h-state | 6 | `(1, 16)` | GRU hidden state for one TRA attention block |
//! | inter h-state | 2 | `(33, 16)` | GRU hidden state for one DPGRNN's inter_rnn |
//!
//! ## Same weights as offline
//!
//! The stream model reuses [`Gtcrn`](super::Gtcrn)'s `gtcrn.safetensors`. The
//! only weight transform is the `StreamConvTranspose2d` flip: the decoder's
//! transpose-conv depth weights are rewritten as regular conv weights with
//! `flip_2d(permute(W, 1,0,2,3))`, applied once at load time in
//! [`GtcrnStream::from_state_dict`].

extern crate self as svod_model;

use std::path::Path;

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;
use svod_tensor::nn::{GruDirection, RnnLayout};

use crate::blocks::{BatchNormWeights, Conv2dWeights};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::blocks::{ConvBlock, Dpgrnn, Erb, GTConvBlock, Grnn, GruWeights, affine_ln, sfe};
use super::error::{HubSnafu, Result, StateSnafu, TensorSnafu};
use super::{C_IN, C_NET, C_SFE, DPGRNN_HIDDEN, DPGRNN_WIDTH, HUB_REPO, channel, empty_dpgrnn};

/// Encoder/decoder GTConvBlock dilations (encoder order; decoder is reversed).
const DILATIONS: [usize; 3] = [1, 2, 5];

/// History frames per depth conv = `(kT-1) * dilation = 2 * dilation`.
const fn hist(d: usize) -> usize {
    2 * d
}

// =========================================================================== //
// GtcrnStream — the streaming model
// =========================================================================== //

/// Runtime tuning for the complex-ratio-mask post-processing. Both knobs are
/// baked into the JIT graph at `prepare` time (the mask is a constant-free
/// function of the network output, so the transform compiles inline).
///
/// The trained model applies `enhanced = spec * mask` (a complex ratio mask).
/// With `scale` and `blend` the applied transform becomes:
/// ```text
/// mask_t  = mask * scale
/// crm     = spec * mask_t                      // complex multiply
/// enhanced = blend * crm + (1 - blend) * spec  // dry/wet mix
/// ```
///
/// - `scale = 1.0` (default): the mask is applied as-is — exact parity with
///   the trained reference. `scale > 1` amplifies the mask (stronger
///   suppression); `scale < 1` dampens it (closer to the input).
/// - `blend = 1.0` (default): fully enhanced output. `blend = 0` is a pure
///   passthrough; intermediate values mix the enhanced and raw spectrograms,
///   which avoids the over-suppression "musical noise" artifacts an aggressive
///   mask can introduce.
#[derive(Clone, Copy, Debug)]
pub struct MaskConfig {
    pub scale: f32,
    pub blend: f32,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self { scale: 1.0, blend: 1.0 }
    }
}

/// The streaming GTCRN network. Same topology and weights as the offline
/// [`Gtcrn`](super::Gtcrn), but the depthwise convs use cache-concat instead
/// of left-padding, and the T-axis GRUs (TRA + inter_rnn) thread their hidden
/// state. Construct via [`GtcrnStream::from_hub`] /
/// [`GtcrnStream::from_safetensors`].
#[derive(Clone)]
pub struct GtcrnStream {
    pub erb: Erb,
    encoder: [EncoderLayer; 5],
    dpgrnn1: Dpgrnn,
    dpgrnn2: Dpgrnn,
    decoder: [DecoderLayer; 5],
    /// Mask post-processing (scale + dry/wet blend). Defaults to identity.
    pub mask: MaskConfig,
}

// One model instance; never stored in a collection — the enum size is harmless
// (same call site as the offline `Gtcrn` layer enums, YOLO's `Csp`, GigaAM's
// `Stage`). Boxing would add indirection for zero benefit.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum EncoderLayer {
    Conv(ConvBlock),
    Gt(GtStreamBlock),
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum DecoderLayer {
    Gt(GtStreamBlock),
    Conv(ConvBlock),
}

/// A `GTConvBlock` variant where the depthwise conv uses cache-concat (instead
/// of left-padding) and the TRA threads its GRU hidden state. The pointwise
/// convs (1×1) and SFE are stateless and reused as-is.
#[derive(Clone)]
pub struct GtStreamBlock {
    pub in_channels: usize,
    pub point_conv1: Conv2dWeights,
    pub point_bn1: BatchNormWeights,
    pub point_act: Tensor,
    /// Depth conv. For the encoder this is a regular `Conv2d` (`transpose=false`).
    /// For the decoder it is a **flipped** weight stored as a regular Conv2d
    /// (`transpose=false`) — the `StreamConvTranspose2d` trick. The F-axis
    /// upsampling + asymmetric pad is applied in `forward_stream`.
    pub depth_conv: Conv2dWeights,
    pub depth_bn: BatchNormWeights,
    pub depth_act: Tensor,
    pub point_conv2: Conv2dWeights,
    pub point_bn2: BatchNormWeights,
    pub tra: TraStreamWeights,
    /// `true` for decoder blocks — triggers the transpose-conv-as-conv path.
    pub use_deconv: bool,
}

/// TRA weights (same tensors as offline `Tra`, but forward threads h-state).
#[derive(Clone)]
pub struct TraStreamWeights {
    pub gru: GruWeights,
    pub fc_weight: Tensor,
    pub fc_bias: Tensor,
}

impl GtcrnStream {
    /// One streaming forward step. `spec` is `(1, 257, 1, 2)` — a single STFT
    /// frame. Returns `(enhanced_spec, new_caches)`.
    ///
    /// This is a pure graph function — it takes caches as input tensors and
    /// returns new caches. The on-device recycle (writing new caches back into
    /// the input buffers) happens in [`forward_jit`], not here.
    pub fn forward_stream(
        &self,
        spec: &Tensor,
        conv_caches: &ConvCaches,
        tra_caches: &TraCaches,
        inter_caches: &InterCaches,
    ) -> Result<(Tensor, ConvCaches, TraCaches, InterCaches)> {
        // spec: (1, F, T=1, 2). Split real/imag, permute to (1, T, F).
        let spec_real = spec
            .try_shrink([None, None, None, Some((SInt::Const(0), SInt::Const(1)))])
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?
            .try_permute(&[0, 2, 1])
            .context(TensorSnafu)?;
        let spec_imag = spec
            .try_shrink([None, None, None, Some((SInt::Const(1), SInt::Const(2)))])
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?
            .try_permute(&[0, 2, 1])
            .context(TensorSnafu)?;

        let mag = spec_real.square().context(TensorSnafu)?;
        let mag = mag.try_add(&spec_imag.square().context(TensorSnafu)?).context(TensorSnafu)?;
        let eps = Tensor::full(&[1], 1e-12f32, DType::Float32).context(TensorSnafu)?;
        let spec_mag = mag.try_add(&eps).context(TensorSnafu)?.try_sqrt().context(TensorSnafu)?;
        let feat = Tensor::stack(&[&spec_mag, &spec_real, &spec_imag], 1).context(TensorSnafu)?;

        let feat = self.erb.bm(&feat)?;
        let feat = sfe(&feat, 3, C_IN)?;

        // Encoder: 2 stateless ConvBlocks, then 3 cached GTConvBlocks.
        let mut en_outs: Vec<Tensor> = Vec::with_capacity(5);
        let mut x = feat;
        for layer in &self.encoder[..2] {
            x = layer.forward(&x)?;
            en_outs.push(x.clone());
        }
        let mut new_conv_en = conv_caches.encoder.clone();
        let mut new_tra_en = tra_caches.encoder.clone();
        for (i, layer) in self.encoder[2..].iter().enumerate() {
            let (xi, nc, nt) = layer.forward_stream(&x, &conv_caches.encoder[i], &tra_caches.encoder[i])?;
            x = xi;
            en_outs.push(x.clone());
            new_conv_en[i] = nc;
            new_tra_en[i] = nt;
        }

        // DPGRNNs — thread inter_rnn h-state.
        let (x, ic0) = self.dpgrnn1.forward_stream(&x, &inter_caches.caches[0])?;
        let (mut x, ic1) = self.dpgrnn2.forward_stream(&x, &inter_caches.caches[1])?;

        // Decoder: 3 cached GTConvBlocks (reversed dilation), then 2 stateless ConvBlocks.
        let mut new_conv_de = conv_caches.decoder.clone();
        let mut new_tra_de = tra_caches.decoder.clone();
        let n = self.decoder.len();
        for (i, layer) in self.decoder[..3].iter().enumerate() {
            let skip = &en_outs[n - 1 - i];
            let added = x.try_add(skip).context(TensorSnafu)?;
            let (xi, nc, nt) = layer.forward_stream(&added, &conv_caches.decoder[i], &tra_caches.decoder[i])?;
            x = xi;
            new_conv_de[i] = nc;
            new_tra_de[i] = nt;
        }
        for (i, layer) in self.decoder[3..].iter().enumerate() {
            let skip = &en_outs[n - 1 - (3 + i)];
            let added = x.try_add(skip).context(TensorSnafu)?;
            x = layer.forward(&added)?;
        }

        // ERB synthesis + complex ratio mask.
        let m = self.erb.bs(&x)?;
        let spec_ref = spec.try_permute(&[0, 3, 2, 1]).context(TensorSnafu)?;
        // Mask post-processing: scale the mask, then dry/wet blend the CRM
        // output with the raw input. Both bake into the JIT graph as constants.
        let m = if (self.mask.scale - 1.0).abs() > f32::EPSILON {
            let s = Tensor::full(&[], self.mask.scale, DType::Float32).context(TensorSnafu)?;
            m.try_mul(&s).context(TensorSnafu)?
        } else {
            m
        };
        let mask_real = channel(&m, 0)?;
        let mask_imag = channel(&m, 1)?;
        let spec_r = channel(&spec_ref, 0)?;
        let spec_i = channel(&spec_ref, 1)?;
        // CRM: out = spec * mask (complex multiply).
        let mut out_real = spec_r
            .try_mul(&mask_real)
            .context(TensorSnafu)?
            .try_sub(&spec_i.try_mul(&mask_imag).context(TensorSnafu)?)
            .context(TensorSnafu)?;
        let mut out_imag = spec_i
            .try_mul(&mask_real)
            .context(TensorSnafu)?
            .try_add(&spec_r.try_mul(&mask_imag).context(TensorSnafu)?)
            .context(TensorSnafu)?;
        // Dry/wet blend: enhanced = blend * crm + (1 - blend) * spec_ref.
        // blend=1.0 → fully enhanced (default); blend=0.0 → passthrough.
        if (self.mask.blend - 1.0).abs() > f32::EPSILON {
            let b = Tensor::full(&[], self.mask.blend, DType::Float32).context(TensorSnafu)?;
            let one_minus_b = Tensor::full(&[], 1.0 - self.mask.blend, DType::Float32).context(TensorSnafu)?;
            out_real = out_real
                .try_mul(&b)
                .context(TensorSnafu)?
                .try_add(&spec_r.try_mul(&one_minus_b).context(TensorSnafu)?)
                .context(TensorSnafu)?;
            out_imag = out_imag
                .try_mul(&b)
                .context(TensorSnafu)?
                .try_add(&spec_i.try_mul(&one_minus_b).context(TensorSnafu)?)
                .context(TensorSnafu)?;
        }
        let spec_enh = Tensor::stack(&[&out_real, &out_imag], 1).context(TensorSnafu)?;
        let spec_enh = spec_enh.try_permute(&[0, 3, 2, 1]).context(TensorSnafu)?;

        Ok((
            spec_enh,
            ConvCaches { encoder: new_conv_en, decoder: new_conv_de },
            TraCaches { encoder: new_tra_en, decoder: new_tra_de },
            InterCaches { caches: [ic0, ic1] },
        ))
    }

    /// Allocate zero-initialized caches for a fresh stream.
    pub fn zero_caches(&self) -> (ConvCaches, TraCaches, InterCaches) {
        let half = C_NET / 2;
        let tra_hidden = half * 2;
        let conv = ConvCaches {
            encoder: core::array::from_fn(|i| {
                Tensor::zeros(&[1, C_NET, hist(DILATIONS[i]), DPGRNN_WIDTH], DType::Float32).unwrap()
            }),
            decoder: core::array::from_fn(|i| {
                // Decoder dilations are reversed: [5, 2, 1].
                let d = DILATIONS[2 - i];
                Tensor::zeros(&[1, C_NET, hist(d), DPGRNN_WIDTH], DType::Float32).unwrap()
            }),
        };
        let tra = TraCaches {
            encoder: core::array::from_fn(|_| Tensor::zeros(&[1, tra_hidden], DType::Float32).unwrap()),
            decoder: core::array::from_fn(|_| Tensor::zeros(&[1, tra_hidden], DType::Float32).unwrap()),
        };
        let inter = InterCaches {
            caches: core::array::from_fn(|_| Tensor::zeros(&[DPGRNN_WIDTH, DPGRNN_HIDDEN], DType::Float32).unwrap()),
        };
        (conv, tra, inter)
    }

    // ----------------------------------------------------------------------- //
    // Loaders (reuse the offline gtcrn.safetensors + transpose-conv flip)
    // ----------------------------------------------------------------------- //

    /// Download `gtcrn.safetensors` from [`HUB_REPO`] and load it.
    pub fn from_hub() -> Result<Self> {
        Self::from_hub_with_revision("main")
    }

    pub fn from_hub_with_revision(revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo = api.repo(hf_hub::Repo::with_revision(HUB_REPO.into(), hf_hub::RepoType::Model, revision.into()));
        let path = repo.get("gtcrn.safetensors").context(HubSnafu)?;
        Self::from_safetensors(&path)
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let sd_raw = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd_raw)
    }

    pub fn from_state_dict(sd_raw: &StateDict) -> Result<Self> {
        let sd = crate::blocks::remap::fold_batchnorm(sd_raw.clone())?;
        // Flip the decoder transpose-conv depth weights into regular-conv layout
        // (the StreamConvTranspose2d Version 2 trick: Conv2d with flipped kernel).
        let sd = flip_transpose_depth_convs(&sd)?;
        let mut model = Self::with_random_weights();
        model.load_state_dict(&sd, "").context(StateSnafu)?;
        Ok(model)
    }

    /// Build with random weights matching the GTCRN layout (for tests/JIT
    /// exercising without a checkpoint).
    pub fn with_random_weights() -> Self {
        Self {
            erb: Erb::empty(),
            encoder: default_stream_encoder(),
            dpgrnn1: empty_dpgrnn(),
            dpgrnn2: empty_dpgrnn(),
            decoder: default_stream_decoder(),
            mask: MaskConfig::default(),
        }
    }

    /// Set the mask post-processing config (scale + dry/wet blend). Must be
    /// called before JIT `prepare` — the values bake into the compiled graph.
    pub fn with_mask(mut self, mask: MaskConfig) -> Self {
        self.mask = mask;
        self
    }
}

// =========================================================================== //
// Cache containers
// =========================================================================== //

/// Per-block depthwise-conv history. One tensor per GTConvBlock (3 encoder +
/// 3 decoder), each shaped `(1, 16, 2·d, 33)`.
#[derive(Clone)]
pub struct ConvCaches {
    pub encoder: [Tensor; 3],
    pub decoder: [Tensor; 3],
}

/// Per-block TRA GRU hidden state. Each shaped `(1, 16)`.
#[derive(Clone)]
pub struct TraCaches {
    pub encoder: [Tensor; 3],
    pub decoder: [Tensor; 3],
}

/// Per-DPGRNN inter_rnn GRU hidden state. Each shaped `(33, 16)`.
#[derive(Clone)]
pub struct InterCaches {
    pub caches: [Tensor; 2],
}

// =========================================================================== //
// forward_jit — on-device cache recycle (the assign-back idiom)
// =========================================================================== //

/// JIT graph: one enhanced frame + in-place cache recycle. Each new cache is
/// written back into its own input buffer (`AFTER(in_buf, STORE(in_buf,
/// value))`), so `execute()` updates the state where the next frame reads it —
/// the host never copies recurrent state (the RN-T block decoder pattern,
/// `gigaam/rnnt/block.rs`). Read-before-write is safe: each cache input is read
/// exactly once (the cat / initial_h) before its store.
/// All JIT outputs: enhanced spec + 14 recycled caches, as a flat tuple (the
/// `jit_wrapper!` macro requires flat `Tensor`s, not arrays).
#[allow(clippy::type_complexity)]
pub fn forward_jit(
    model: &GtcrnStream,
    spec: &Tensor,
    conv_en: [&Tensor; 3],
    conv_de: [&Tensor; 3],
    tra_en: [&Tensor; 3],
    tra_de: [&Tensor; 3],
    inter: [&Tensor; 2],
) -> Result<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    let conv_caches = ConvCaches {
        encoder: [conv_en[0].clone(), conv_en[1].clone(), conv_en[2].clone()],
        decoder: [conv_de[0].clone(), conv_de[1].clone(), conv_de[2].clone()],
    };
    let tra_caches = TraCaches {
        encoder: [tra_en[0].clone(), tra_en[1].clone(), tra_en[2].clone()],
        decoder: [tra_de[0].clone(), tra_de[1].clone(), tra_de[2].clone()],
    };
    let inter_caches = InterCaches { caches: [inter[0].clone(), inter[1].clone()] };

    let (enh, new_conv, new_tra, new_inter) = model.forward_stream(spec, &conv_caches, &tra_caches, &inter_caches)?;

    let recycle = |input: &Tensor, value: &Tensor| -> Result<Tensor> {
        let out = Tensor::from_lazy(input.uop());
        out.try_assign(value).context(TensorSnafu)?;
        Ok(out)
    };

    Ok((
        enh,
        // nconv_en0..2
        recycle(conv_en[0], &new_conv.encoder[0])?,
        recycle(conv_en[1], &new_conv.encoder[1])?,
        recycle(conv_en[2], &new_conv.encoder[2])?,
        // nconv_de0..2
        recycle(conv_de[0], &new_conv.decoder[0])?,
        recycle(conv_de[1], &new_conv.decoder[1])?,
        recycle(conv_de[2], &new_conv.decoder[2])?,
        // ntra_en0..2
        recycle(tra_en[0], &new_tra.encoder[0])?,
        recycle(tra_en[1], &new_tra.encoder[1])?,
        recycle(tra_en[2], &new_tra.encoder[2])?,
        // ntra_de0..2
        recycle(tra_de[0], &new_tra.decoder[0])?,
        recycle(tra_de[1], &new_tra.decoder[1])?,
        recycle(tra_de[2], &new_tra.decoder[2])?,
        // ninter0..1
        recycle(inter[0], &new_inter.caches[0])?,
        recycle(inter[1], &new_inter.caches[1])?,
    ))
}

// =========================================================================== //
// GtcrnStreamJit — the JIT wrapper
// =========================================================================== //

#[allow(clippy::too_many_arguments)]
mod stream_jit {
    use super::*;
    use svod_macros::jit_wrapper;
    jit_wrapper! {
        GtcrnStreamJit(GtcrnStream) {
            spec: Tensor,
            conv_en0: Tensor, conv_en1: Tensor, conv_en2: Tensor,
            conv_de0: Tensor, conv_de1: Tensor, conv_de2: Tensor,
            tra_en0: Tensor, tra_en1: Tensor, tra_en2: Tensor,
            tra_de0: Tensor, tra_de1: Tensor, tra_de2: Tensor,
            inter0: Tensor, inter1: Tensor,

            outputs {
                enh,
                nconv_en0, nconv_en1, nconv_en2,
                nconv_de0, nconv_de1, nconv_de2,
                ntra_en0, ntra_en1, ntra_en2,
                ntra_de0, ntra_de1, ntra_de2,
                ninter0, ninter1,
            },

            build(spec,
                  conv_en0, conv_en1, conv_en2,
                  conv_de0, conv_de1, conv_de2,
                  tra_en0, tra_en1, tra_en2,
                  tra_de0, tra_de1, tra_de2,
                  inter0, inter1) {
                super::forward_jit(
                    model, spec,
                    [conv_en0, conv_en1, conv_en2],
                    [conv_de0, conv_de1, conv_de2],
                    [tra_en0, tra_en1, tra_en2],
                    [tra_de0, tra_de1, tra_de2],
                    [inter0, inter1],
                )
            }
        }
    }
}
pub use stream_jit::GtcrnStreamJit;

// =========================================================================== //
// Streaming block forward impls
// =========================================================================== //

impl GtStreamBlock {
    /// `(B, C, T=1, F)` + conv_cache `(B, C, hist, F)` + tra_h `(B, hidden)`
    /// → `(out (B, C, 1, F), new_conv_cache, new_tra_h)`.
    pub fn forward_stream(&self, x: &Tensor, conv_cache: &Tensor, tra_h: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let half = self.in_channels / 2;
        let halves = x.chunk(2, 1).context(TensorSnafu)?;
        let (x1, x2) = (&halves[0], &halves[1]);

        let x1 = sfe(x1, 3, half)?;
        let h1 = self.point_conv1.forward(&x1)?;
        let h1 = self.point_bn1.forward(&h1)?;
        let h1 = h1.prelu(&self.point_act).context(TensorSnafu)?;

        // Depthwise conv with cache-concat — same mechanics for encoder (conv)
        // and decoder (conv_transpose). The cache provides T-axis causality;
        // the conv's own stride/padding/dilation handle the rest.
        let (h1, new_cache) = self.stream_conv(&h1, conv_cache)?;

        let h1 = self.depth_bn.forward(&h1)?;
        let h1 = h1.prelu(&self.depth_act).context(TensorSnafu)?;
        let h1 = self.point_conv2.forward(&h1)?;
        let h1 = self.point_bn2.forward(&h1)?;

        let (h1, new_tra_h) = self.tra.forward_stream(&h1, tra_h)?;
        let out = GTConvBlock::shuffle(&h1, x2, half)?;
        Ok((out, new_cache, new_tra_h))
    }

    /// Cache-concat depthwise conv, faithfully mirroring the reference
    /// `StreamConv2d` (encoder) and `StreamConvTranspose2d` (decoder).
    ///
    /// **Encoder** (`StreamConv2d`, `convolution.py:85-93`): `cat([cache, x], T)`
    /// → `Conv2d` with `padding=(0, F_pad)` (T_pad=0 asserted; cache provides
    /// left context) → `new_cache = inp[:, :, 1:, :]`.
    ///
    /// **Decoder** (`StreamConvTranspose2d` Version 2, `convolution.py:232-262`):
    /// `cat([cache, x], T)` → symmetric F-pad of `(F_size-1)*F_dilation - F_pad`
    /// → `Conv2d` (flipped weight) with `stride=(1,1)`, `padding=(0,0)`, real
    /// dilation → `new_cache = inp[:, :, 1:, :]` (from the un-padded cat).
    fn stream_conv(&self, x: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let inp = Tensor::cat(&[cache, x], 2).context(TensorSnafu)?;
        let t_len = inp_dim(&inp, 2);

        let y = if self.use_deconv {
            // StreamConvTranspose2d (Version 2): flipped-weight Conv2d.
            // The weight was flipped at load time (flip_transpose_depth_convs).
            let p = self.depth_conv.padding;
            let d = self.depth_conv.dilation;
            let kw = self.depth_conv_weight_shape()[3]; // kernel F dim
            let f_dilation = d[1];
            let f_pad = p[1];
            // Symmetric F-pad: (F_size-1)*F_dilation - F_pad on each side.
            let f_pad_amt = (kw - 1) * f_dilation - f_pad;
            let padded = inp
                .try_pad(&[(0, 0), (0, 0), (0, 0), (f_pad_amt as isize, f_pad_amt as isize)])
                .context(TensorSnafu)?;
            // Conv2d with flipped weight, stride (1,1), padding (0,0).
            padded
                .conv2d()
                .weight(&self.depth_conv.weight)
                .groups(self.depth_conv.groups)
                .stride(&[1, 1])
                .dilation(&[d[0], d[1]])
                .padding(&[(0, 0), (0, 0)])
                .maybe_bias(self.depth_conv.bias.as_ref())
                .call()
                .context(TensorSnafu)?
        } else {
            // StreamConv2d: Conv2d with padding=(0, F_pad).
            let p = self.depth_conv.padding;
            let d = self.depth_conv.dilation;
            let s = self.depth_conv.stride;
            inp.conv2d()
                .weight(&self.depth_conv.weight)
                .groups(self.depth_conv.groups)
                .stride(&[s[0], s[1]])
                .dilation(&[d[0], d[1]])
                .padding(&[(0, 0), (p[1] as isize, p[1] as isize)])
                .maybe_bias(self.depth_conv.bias.as_ref())
                .call()
                .context(TensorSnafu)?
        };

        let new_cache =
            inp.try_shrink([None, None, Some((SInt::Const(1), SInt::Const(t_len))), None]).context(TensorSnafu)?;
        Ok((y, new_cache))
    }

    fn depth_conv_weight_shape(&self) -> Vec<usize> {
        self.depth_conv.weight.shape().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
    }
}

impl TraStreamWeights {
    /// `(B, C, T, F) + h (B, hidden)` → `((B, C, T, F), new_h (B, hidden))`.
    pub fn forward_stream(&self, x: &Tensor, h_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let sq = x.square().context(TensorSnafu)?;
        let zt = sq.mean_with().axes(-1isize).keepdim(false).call().context(TensorSnafu)?;
        let zt = zt.try_permute(&[0, 2, 1]).context(TensorSnafu)?; // (B,T,C)
        let (at, new_h) = self.gru.forward_with_state(&zt, h_prev)?;
        let at = at.linear().weight(&self.fc_weight).bias(&self.fc_bias).call().context(TensorSnafu)?;
        let at = at.try_permute(&[0, 2, 1]).context(TensorSnafu)?;
        let at = at.sigmoid().context(TensorSnafu)?;
        let at = at.try_unsqueeze(-1).context(TensorSnafu)?;
        let out = x.try_mul(&at).context(TensorSnafu)?;
        Ok((out, new_h))
    }
}

impl GruWeights {
    /// Run the GRU over `(B, T, input)` with an explicit initial hidden state,
    /// returning `(y_seq (B,T,H), y_h (B,H))`. Used by streaming TRA/inter_rnn.
    pub fn forward_with_state(&self, x: &Tensor, h0: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.hidden_size;
        let w = self.weight_ih.try_unsqueeze(0).context(TensorSnafu)?;
        let r = self.weight_hh.try_unsqueeze(0).context(TensorSnafu)?;
        let bias = Tensor::cat(&[&self.bias_ih, &self.bias_hh], 0).context(TensorSnafu)?;
        let bias = bias.try_unsqueeze(0).context(TensorSnafu)?;
        let h0 = h0.try_unsqueeze(0).context(TensorSnafu)?; // (1, B, H)
        let out = x
            .gru()
            .w(&w)
            .r_weights(&r)
            .hidden_size(h)
            .bias(&bias)
            .initial_h(&h0)
            .direction(GruDirection::Forward)
            .linear_before_reset(true)
            .layout(RnnLayout::BatchFirst)
            .call()
            .context(TensorSnafu)?;
        let y = out.y.try_squeeze(Some(2)).context(TensorSnafu)?;
        // For BatchFirst, y_h is (batch, 1, hidden) — squeeze the direction axis (dim 1).
        let y_h = out.y_h.try_squeeze(Some(1)).context(TensorSnafu)?;
        Ok((y, y_h))
    }
}

impl Dpgrnn {
    /// Streaming forward: `intra_rnn` is stateless (bidirectional, F-axis);
    /// `inter_rnn` threads its hidden state across T. Returns
    /// `(out (B,C,T,F), new_inter_h (BF, hidden))`.
    pub fn forward_stream(&self, x: &Tensor, inter_h: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.try_permute(&[0, 2, 3, 1]).context(TensorSnafu)?; // (B,T,F,C)
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].as_const().or_else(|| shape[0].vmax()).unwrap_or(1) as isize;
        let t_dim = shape[1].as_const().unwrap_or(1) as isize;
        let f_dim = shape[2].as_const().unwrap_or(self.width) as isize;
        let c_dim = shape[3].as_const().unwrap_or(self.hidden_size) as isize;
        let w = self.width as isize;
        let h = self.hidden_size as isize;

        // Intra RNN: bidirectional over F, NO state threading.
        let intra_x = x.try_reshape([b * t_dim, f_dim, c_dim]).context(TensorSnafu)?;
        let intra_x = self.intra_rnn.forward(&intra_x)?;
        let intra_x =
            intra_x.linear().weight(&self.intra_fc_weight).bias(&self.intra_fc_bias).call().context(TensorSnafu)?;
        let intra_x = intra_x.try_reshape([b, t_dim, w, h]).context(TensorSnafu)?;
        let intra_x = affine_ln(&intra_x, &self.intra_ln_weight, &self.intra_ln_bias, Self::LN_EPS)?;
        let intra_out = x.try_add(&intra_x).context(TensorSnafu)?;

        // Inter RNN: unidirectional over T, WITH state threading.
        let inter_in = intra_out.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?; // (B,F,T,C)
        let inter_x = inter_in.try_reshape([b * f_dim, t_dim, c_dim]).context(TensorSnafu)?;
        let (inter_x, new_inter_h) = self.inter_rnn.forward_with_state(&inter_x, inter_h)?;
        let inter_x =
            inter_x.linear().weight(&self.inter_fc_weight).bias(&self.inter_fc_bias).call().context(TensorSnafu)?;
        let inter_x = inter_x.try_reshape([b, w, t_dim, h]).context(TensorSnafu)?;
        let inter_x = inter_x.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
        let inter_x = affine_ln(&inter_x, &self.inter_ln_weight, &self.inter_ln_bias, Self::LN_EPS)?;
        let inter_out = intra_out.try_add(&inter_x).context(TensorSnafu)?;
        let out = inter_out.try_permute(&[0, 3, 1, 2]).context(TensorSnafu)?;
        Ok((out, new_inter_h))
    }
}

impl Grnn {
    /// Like [`forward`](Self::forward) but threads the hidden state. `h` is
    /// `(BF, hidden)` for the full grouped RNN (rnn1 gets the first half,
    /// rnn2 the second).
    pub fn forward_with_state(&self, x: &Tensor, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let halves = x.chunk(2, -1).context(TensorSnafu)?;
        let (x1, x2) = (&halves[0], &halves[1]);
        let h_halves = h.chunk(2, -1).context(TensorSnafu)?;
        let (h1, h2) = (&h_halves[0], &h_halves[1]);
        let (y1, nh1) = self.rnn1_f.forward_with_state(x1, h1)?;
        let (y2, nh2) = self.rnn2_f.forward_with_state(x2, h2)?;
        let y = Tensor::cat(&[&y1, &y2], -1).context(TensorSnafu)?;
        let new_h = Tensor::cat(&[&nh1, &nh2], -1).context(TensorSnafu)?;
        Ok((y, new_h))
    }
}

// =========================================================================== //
// EncoderLayer / DecoderLayer forward dispatch
// =========================================================================== //

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            EncoderLayer::Conv(c) => c.forward(x),
            EncoderLayer::Gt(_) => unreachable!("stream GTConvBlock uses forward_stream"),
        }
    }

    fn forward_stream(&self, x: &Tensor, conv_cache: &Tensor, tra_h: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            EncoderLayer::Gt(g) => g.forward_stream(x, conv_cache, tra_h),
            EncoderLayer::Conv(_) => unreachable!("stream ConvBlock uses forward"),
        }
    }
}

impl DecoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            DecoderLayer::Conv(c) => c.forward(x),
            DecoderLayer::Gt(_) => unreachable!("stream GTConvBlock uses forward_stream"),
        }
    }

    fn forward_stream(&self, x: &Tensor, conv_cache: &Tensor, tra_h: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            DecoderLayer::Gt(g) => g.forward_stream(x, conv_cache, tra_h),
            DecoderLayer::Conv(_) => unreachable!("stream ConvBlock uses forward"),
        }
    }
}

// =========================================================================== //
// Helpers
// =========================================================================== //

/// Read a concrete dim from a tensor (symbolic dims resolve to vmax).
fn inp_dim(t: &Tensor, axis: usize) -> usize {
    t.shape().unwrap()[axis].as_const().or_else(|| t.shape().unwrap()[axis].vmax()).unwrap_or(0)
}

/// Flip the decoder's transpose-conv depth weights into regular-conv layout.
/// Mirrors `submodules/gtcrn/stream/modules/convert.py:23-28`. The stream model
/// stores depth convs as `nn.Conv2d` (not ConvTranspose2d) with flipped kernels.
/// For the GTCRN depthwise case (groups=channels, in/g == out/g == 1), the
/// ConvTranspose2d weight `[in, out/g, kH, kW]` == Conv2d weight
/// `[out, in/g, kH, kW]` (both `[16, 1, 3, 3]`), so only the spatial flip is
/// needed (no channel permute).
fn flip_transpose_depth_convs(sd: &StateDict) -> Result<StateDict> {
    let mut out = StateDict::new();
    for (key, val) in sd.iter() {
        let is_decoder_gt = (0..=2).any(|i| key == &format!("decoder.de_convs.{i}.depth_conv.weight"));
        if is_decoder_gt {
            let flipped = val.flip(&[-2, -1]).context(TensorSnafu)?;
            out.insert(key.clone(), flipped);
        } else {
            out.insert(key.clone(), val.clone());
        }
    }
    Ok(out)
}

// =========================================================================== //
// Default architecture construction (streaming variants)
// =========================================================================== //

fn default_stream_encoder() -> [EncoderLayer; 5] {
    let mk_conv = |out_ch: usize,
                   in_ch: usize,
                   kernel: [usize; 2],
                   stride: [usize; 2],
                   padding: [usize; 2],
                   dilation: [usize; 2],
                   groups: usize,
                   is_last: bool|
     -> ConvBlock {
        ConvBlock {
            conv: Conv2dWeights::new(out_ch, in_ch, kernel, stride, padding, dilation, groups, true, false),
            bn: BatchNormWeights::empty(out_ch),
            act: (!is_last).then(|| crate::init::fan_in_uniform(&[1], 1, DType::Float32)),
            is_last,
        }
    };
    [
        EncoderLayer::Conv(mk_conv(16, C_SFE, [1, 5], [1, 2], [0, 2], [1, 1], 1, false)),
        EncoderLayer::Conv(mk_conv(16, 16, [1, 5], [1, 2], [0, 2], [1, 1], 2, false)),
        EncoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [1, 1], false)),
        EncoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [2, 1], false)),
        EncoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [5, 1], false)),
    ]
}

fn default_stream_decoder() -> [DecoderLayer; 5] {
    let mk_conv = |out_ch: usize,
                   in_ch: usize,
                   kernel: [usize; 2],
                   stride: [usize; 2],
                   padding: [usize; 2],
                   dilation: [usize; 2],
                   groups: usize,
                   is_last: bool|
     -> ConvBlock {
        ConvBlock {
            conv: Conv2dWeights::new(out_ch, in_ch, kernel, stride, padding, dilation, groups, true, true),
            bn: BatchNormWeights::empty(out_ch),
            act: (!is_last).then(|| crate::init::fan_in_uniform(&[1], 1, DType::Float32)),
            is_last,
        }
    };
    [
        DecoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [5, 1], true)),
        DecoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [2, 1], true)),
        DecoderLayer::Gt(empty_stream_gtconv([3, 3], [1, 1], [0, 1], [1, 1], true)),
        DecoderLayer::Conv(mk_conv(16, 16, [1, 5], [1, 2], [0, 2], [1, 1], 2, false)),
        DecoderLayer::Conv(mk_conv(2, 16, [1, 5], [1, 2], [0, 2], [1, 1], 1, true)),
    ]
}

fn empty_stream_gtconv(
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    use_deconv: bool,
) -> GtStreamBlock {
    let in_channels = C_NET;
    let hidden = C_NET;
    let half = in_channels / 2;
    GtStreamBlock {
        in_channels,
        point_conv1: Conv2dWeights::new(hidden, half * 3, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, use_deconv),
        point_bn1: BatchNormWeights::empty(hidden),
        point_act: crate::init::fan_in_uniform(&[1], 1, DType::Float32),
        // depth_conv: always a regular Conv2d (transpose=false). For the
        // decoder, the weight is flipped at load time (flip_transpose_depth_convs)
        // — the StreamConvTranspose2d trick (Conv2d with flipped kernel).
        depth_conv: Conv2dWeights::new(hidden, hidden, kernel, stride, padding, dilation, hidden, true, false),
        depth_bn: BatchNormWeights::empty(hidden),
        depth_act: crate::init::fan_in_uniform(&[1], 1, DType::Float32),
        point_conv2: Conv2dWeights::new(half, hidden, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, use_deconv),
        point_bn2: BatchNormWeights::empty(half),
        tra: TraStreamWeights {
            gru: GruWeights {
                hidden_size: half * 2,
                weight_ih: crate::init::fan_in_uniform(&[half * 6, half], half, DType::Float32),
                weight_hh: crate::init::fan_in_uniform(&[half * 6, half * 2], half * 2, DType::Float32),
                bias_ih: crate::init::fan_in_uniform(&[half * 6], half, DType::Float32),
                bias_hh: crate::init::fan_in_uniform(&[half * 6], half, DType::Float32),
            },
            fc_weight: crate::init::fan_in_uniform(&[half, half * 2], half * 2, DType::Float32),
            fc_bias: crate::init::fan_in_uniform(&[half], half, DType::Float32),
        },
        use_deconv,
    }
}

// =========================================================================== //
// HasStateDict
// =========================================================================== //

impl HasStateDict for GtcrnStream {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.erb.state_dict(&prefixed(prefix, "erb"));
        for (i, layer) in self.encoder.iter().enumerate() {
            let p = prefixed(prefix, &format!("encoder.en_convs.{i}"));
            sd.extend(encoder_layer_state_dict(layer, &p));
        }
        sd.extend(self.dpgrnn1.state_dict(&prefixed(prefix, "dpgrnn1")));
        sd.extend(self.dpgrnn2.state_dict(&prefixed(prefix, "dpgrnn2")));
        for (i, layer) in self.decoder.iter().enumerate() {
            let p = prefixed(prefix, &format!("decoder.de_convs.{i}"));
            sd.extend(decoder_layer_state_dict(layer, &p));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.erb.load_state_dict(sd, &prefixed(prefix, "erb"))?;
        for (i, layer) in self.encoder.iter_mut().enumerate() {
            let p = prefixed(prefix, &format!("encoder.en_convs.{i}"));
            load_encoder_layer(layer, sd, &p)?;
        }
        self.dpgrnn1.load_state_dict(sd, &prefixed(prefix, "dpgrnn1"))?;
        self.dpgrnn2.load_state_dict(sd, &prefixed(prefix, "dpgrnn2"))?;
        for (i, layer) in self.decoder.iter_mut().enumerate() {
            let p = prefixed(prefix, &format!("decoder.de_convs.{i}"));
            load_decoder_layer(layer, sd, &p)?;
        }
        Ok(())
    }
}

fn encoder_layer_state_dict(layer: &EncoderLayer, prefix: &str) -> StateDict {
    match layer {
        EncoderLayer::Conv(c) => c.state_dict(prefix),
        EncoderLayer::Gt(g) => gt_stream_state_dict(g, prefix),
    }
}

fn decoder_layer_state_dict(layer: &DecoderLayer, prefix: &str) -> StateDict {
    match layer {
        DecoderLayer::Gt(g) => gt_stream_state_dict(g, prefix),
        DecoderLayer::Conv(c) => c.state_dict(prefix),
    }
}

fn load_encoder_layer(
    layer: &mut EncoderLayer,
    sd: &StateDict,
    prefix: &str,
) -> std::result::Result<(), crate::state::Error> {
    match layer {
        EncoderLayer::Conv(c) => c.load_state_dict(sd, prefix),
        EncoderLayer::Gt(g) => load_gt_stream(g, sd, prefix),
    }
}

fn load_decoder_layer(
    layer: &mut DecoderLayer,
    sd: &StateDict,
    prefix: &str,
) -> std::result::Result<(), crate::state::Error> {
    match layer {
        DecoderLayer::Gt(g) => load_gt_stream(g, sd, prefix),
        DecoderLayer::Conv(c) => c.load_state_dict(sd, prefix),
    }
}

fn gt_stream_state_dict(g: &GtStreamBlock, prefix: &str) -> StateDict {
    let mut sd = g.point_conv1.state_dict(&prefixed(prefix, "point_conv1"));
    sd.extend(g.point_bn1.state_dict(&prefixed(prefix, "point_bn1")));
    sd.insert(prefixed(prefix, "point_act.weight"), g.point_act.clone());
    sd.extend(g.depth_conv.state_dict(&prefixed(prefix, "depth_conv")));
    sd.extend(g.depth_bn.state_dict(&prefixed(prefix, "depth_bn")));
    sd.insert(prefixed(prefix, "depth_act.weight"), g.depth_act.clone());
    sd.extend(g.point_conv2.state_dict(&prefixed(prefix, "point_conv2")));
    sd.extend(g.point_bn2.state_dict(&prefixed(prefix, "point_bn2")));
    sd.extend(tra_stream_state_dict(&g.tra, &prefixed(prefix, "tra")));
    sd
}

fn load_gt_stream(g: &mut GtStreamBlock, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
    g.point_conv1.load_state_dict(sd, &prefixed(prefix, "point_conv1"))?;
    g.point_bn1.load_state_dict(sd, &prefixed(prefix, "point_bn1"))?;
    g.point_act = get_tensor(sd, &prefixed(prefix, "point_act.weight"))?;
    g.depth_conv.load_state_dict(sd, &prefixed(prefix, "depth_conv"))?;
    g.depth_bn.load_state_dict(sd, &prefixed(prefix, "depth_bn"))?;
    g.depth_act = get_tensor(sd, &prefixed(prefix, "depth_act.weight"))?;
    g.point_conv2.load_state_dict(sd, &prefixed(prefix, "point_conv2"))?;
    g.point_bn2.load_state_dict(sd, &prefixed(prefix, "point_bn2"))?;
    load_tra_stream(&mut g.tra, sd, &prefixed(prefix, "tra"))?;
    Ok(())
}

fn tra_stream_state_dict(tra: &TraStreamWeights, prefix: &str) -> StateDict {
    let mut sd = tra.gru.state_dict(&prefixed(prefix, "att_gru"));
    sd.insert(prefixed(prefix, "att_fc.weight"), tra.fc_weight.clone());
    sd.insert(prefixed(prefix, "att_fc.bias"), tra.fc_bias.clone());
    sd
}

fn load_tra_stream(
    tra: &mut TraStreamWeights,
    sd: &StateDict,
    prefix: &str,
) -> std::result::Result<(), crate::state::Error> {
    tra.gru.load_state_dict(sd, &prefixed(prefix, "att_gru"))?;
    tra.fc_weight = get_tensor(sd, &prefixed(prefix, "att_fc.weight"))?;
    tra.fc_bias = get_tensor(sd, &prefixed(prefix, "att_fc.bias"))?;
    Ok(())
}
