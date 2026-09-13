//! JIT wrapper for [`Gtcrn`]. Compiles the spectrogram→spectrogram forward
//! graph once at `prepare()` time and replays it per call.
//!
//! Shape contract:
//! - `prepare(InputSpec::f32(&[B, 257, T, 2]))` bakes `T` (and `B`) into the
//!   plan. The GRU recurrence requires a concrete sequence length.
//! - For a different audio length, prepare a fresh wrapper (or call `prepare`
//!   again) with a new `InputSpec`.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::Gtcrn;

jit_wrapper! {
    GtcrnJit(Gtcrn) {
        spec: Tensor,

        build(spec) {
            model.forward(spec)
        }
    }
}
