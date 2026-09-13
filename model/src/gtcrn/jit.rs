//! JIT wrapper for [`Gtcrn`]. Compiles the waveform→waveform graph — analysis
//! STFT, the mask network, synthesis ISTFT — once at `prepare()` time and
//! replays it per call.
//!
//! Shape contract:
//! - `prepare(InputSpec::f32(&[B, L]))` bakes `L` (and `B`) into the plan: the
//!   frame count `L / HOP + 1` fixes the GRU recurrence, which needs a
//!   concrete sequence length. The output is `[B, (L / HOP) · HOP]`.
//! - For a different audio length, prepare a fresh wrapper (or call `prepare`
//!   again) with a new `InputSpec`.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::Gtcrn;

jit_wrapper! {
    GtcrnJit(Gtcrn) {
        waveform: Tensor,

        build(waveform) {
            model.enhance(waveform)
        }
    }
}
