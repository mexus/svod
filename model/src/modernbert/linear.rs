//! `x·wᵀ (+ residual)` for the projections: tk's hand GEMM where it applies, the
//! generic `Tensor::linear` elsewhere.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::Tensor;

use super::error::{Result, TkSnafu};

/// `x` `[.., K]` · `w` `[N, K]`ᵀ, plus `residual` `[.., N]` when given. The tk
/// kernel takes 16-bit operands whose dims are static or pinned by the JIT, and
/// folds the residual into its store; it declines off its tile grid or arch, and
/// every other case is the generic GEMM with the add after it.
///
/// `x` is materialized either way — the kernel launch copies a lazy operand into
/// its own buffer, the generic path realizes it here: left lazy, its producer
/// fuses into the GEMM's K loop and is recomputed once per output feature (the
/// GEGLU's erf made the MLP's `Wo` 154 µs instead of 74 at 1×512).
pub(crate) fn linear(x: &Tensor, w: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
    let shape = x.shape()?.to_vec();
    let sixteen_bit = [DType::BFloat16, DType::Float16].contains(&x.dtype()) && x.dtype() == w.dtype();
    if sixteen_bit && shape.iter().all(|d| svod_tk::launch::pinned_dim(d).is_some()) {
        let epilogue = residual.map_or(svod_tk::Epilogue::Plain, svod_tk::Epilogue::Add);
        if let Some(y) = svod_tk::gemm_nt_with_epilogue(x, w, epilogue).context(TkSnafu)? {
            // Back to `x`'s own leading dims, a JIT-pinned batch included.
            let dims: Vec<SInt> = shape[..shape.len() - 1].iter().cloned().chain([w.dim(0)?]).collect();
            return Ok(y.try_reshape(dims)?);
        }
    }
    let y = x.contiguous().linear().weight(w).call()?;
    Ok(match residual {
        Some(residual) => y.try_add(residual)?,
        None => y,
    })
}
