use svod_dtype::DType;

use crate::Tensor;
use crate::nn::{Layer, Module};

type Result<T> = crate::Result<T>;

/// Fully connected layer: `y = (x @ weight.T) * weight_scale + bias`.
///
/// Weight shape: `[out_features, in_features]`, bias shape: `[out_features]`.
/// State-dict keys: `weight`, `bias` when the layer has one, and
/// `weight.weight_scale` for a quantized weight stored with one scale per
/// output channel (`[out_features]` or `[out_features, 1]`). The scale is
/// applied to the accumulated product, which is exact and keeps the reduce's
/// operands plain loads.
#[derive(Clone, Module)]
#[module(crate = "crate")]
pub struct Linear {
    pub weight: Tensor,
    #[module(optional)]
    pub bias: Option<Tensor>,
    #[module(key = "weight.weight_scale", optional)]
    pub weight_scale: Option<Tensor>,
}

impl Linear {
    /// Create a linear layer from existing weight and optional bias tensors.
    ///
    /// Weight must have shape `[out_features, in_features]`, bias must have shape `[out_features]`.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, weight_scale: None }
    }

    /// Multiply an accumulated product `[.., out_features]` by the per-channel
    /// weight scale, when the weight carries one.
    pub fn apply_weight_scale(&self, product: Tensor) -> Result<Tensor> {
        match &self.weight_scale {
            Some(scale) => product.try_mul(&scale.cast(product.dtype()).try_reshape([-1isize])?),
            None => Ok(product),
        }
    }

    /// Create a linear layer with a Kaiming-uniform weight and a zero bias.
    ///
    /// Weight shape: `[out_features, in_features]`. Both parameters are
    /// [`contiguous`](Tensor::contiguous), so they materialize into their own
    /// buffers instead of being fused into every consumer.
    #[track_caller]
    pub fn with_dims(in_features: usize, out_features: usize, bias: bool, dtype: DType) -> Self {
        origin_call!("Linear::with_dims");
        let weight = Tensor::kaiming_uniform_with_dtype(&[out_features, in_features], 0.0, dtype.clone())
            .expect("non-empty shape")
            .contiguous();
        Self { weight, bias: bias.then(|| Tensor::zeros(&[out_features], dtype).contiguous()), weight_scale: None }
    }
}

impl Layer for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.weight_scale.is_none() {
            return x.linear().weight(&self.weight).maybe_bias(self.bias.as_ref()).call();
        }
        let scaled = self.apply_weight_scale(x.linear().weight(&self.weight).call()?)?;
        match &self.bias {
            Some(bias) => scaled.try_add(bias),
            None => Ok(scaled),
        }
    }
}
