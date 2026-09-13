//! Raw complex STFT / ISTFT for GTCRN, matching `torch.stft` /
//! `torch.istft` with `n_fft = 512`, `hop = 256`, `win = hann_window(512)**0.5`,
//! and `center = True`.
//!
//! Runs eagerly on CPU via `realfft` — same approach as
//! [`crate::audio::mel`](crate::audio::mel). The lazy tensor pipeline has no
//! FFT op, so the forward STFT is a host-side preprocess feeding the JIT graph
//! and the inverse STFT is a host-side post-process.
//!
//! ## Layout
//!
//! [`Stft::forward`] takes `[B, L]` PCM and returns `[B, F=257, T, 2]`
//! `(real, imag)`, matching the layout GTCRN's `forward` consumes
//! (`spec[..., 0]` = real, `spec[..., 1]` = imag). [`Stft::inverse`] takes the
//! same `[B, F, T, 2]` and returns `[B, L']` PCM via overlap-add.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use svod_tensor::Tensor;

/// STFT constants — the upstream GTCRN defaults (gtcrn.py / infer.py).
pub const N_FFT: usize = 512;
pub const HOP: usize = 256;
pub const N_BINS: usize = N_FFT / 2 + 1;

/// Periodic Hann window raised to the 0.5 power, matching
/// `torch.hann_window(N_FFT).pow(0.5)` (periodic form: `0.5·(1 − cos(2πn/N))`).
fn sqrt_hann_window() -> Vec<f32> {
    (0..N_FFT)
        .map(|n| {
            let hann = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos());
            hann.sqrt()
        })
        .collect()
}

/// Reflect-pad `x` by `pad` samples on each side, matching `torch.nn.functional
/// .pad(x, (pad, pad), mode="reflect")` (PyTorch's `center=True` STFT padding).
/// Reflect mirrors without repeating the boundary sample: `[a b c d]` padded by
/// 2 left → `[c b a b c d]`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 || pad == 0 {
        return x.to_vec();
    }
    let mut out = vec![0.0f32; n + 2 * pad];
    out[pad..pad + n].copy_from_slice(x);
    // Left wing: out[pad-1-j] mirrors x[j+1] (skip x[0] boundary).
    for j in 0..pad {
        let src = (j + 1).min(n - 1);
        out[pad - 1 - j] = x[src];
    }
    // Right wing: out[pad+n+j] mirrors x[n-2-j] (skip x[n-1] boundary).
    for j in 0..pad {
        let src = n.saturating_sub(2 + j);
        out[pad + n + j] = x[src];
    }
    out
}

/// Host-side STFT/ISTFT engine. Cheap to build (the FFT plan is cached in an
/// `Arc`); clone freely.
#[derive(Clone)]
pub struct Stft {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn realfft::ComplexToReal<f32>>,
    window: Arc<Vec<f32>>,
}

impl Stft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            r2c: planner.plan_fft_forward(N_FFT),
            c2r: planner.plan_fft_inverse(N_FFT),
            window: Arc::new(sqrt_hann_window()),
        }
    }

    /// Number of STFT frames produced by a waveform of `waveform_len` samples
    /// (after `center=True` reflect padding). Equals `1 + waveform_len / HOP`
    /// when `waveform_len` is a multiple of `HOP`.
    pub fn num_frames(waveform_len: usize) -> usize {
        // center=True pads by N_FFT/2 each side, so the padded length is
        // always len + N_FFT and the frame count reduces to this.
        1 + waveform_len / HOP
    }

    /// Forward STFT of `[B, L]` PCM → `[B, N_BINS, T, 2]` `(real, imag)`.
    /// `center=True` reflect padding is applied to each row independently.
    #[track_caller]
    pub fn forward(&self, waveform: &[f32], batch: usize) -> Vec<f32> {
        assert!(batch > 0, "STFT needs at least one row, got batch = 0");
        assert_eq!(
            waveform.len() % batch,
            0,
            "STFT input must divide evenly into rows: {} samples over {batch} rows",
            waveform.len()
        );
        let l = waveform.len() / batch;
        let n_frames = Self::num_frames(l);
        let mut out = vec![0.0f32; batch * N_BINS * n_frames * 2];

        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut scratch = self.r2c.make_scratch_vec();
        for b in 0..batch {
            // Each row reflects against its own boundaries: padding the rows as
            // one buffer would mirror the neighbouring row's samples into the
            // wings, and leave the slice arithmetic below short by `b * N_FFT`.
            let row = reflect_pad(&waveform[b * l..(b + 1) * l], N_FFT / 2);
            for f in 0..n_frames {
                let start = f * HOP;
                for i in 0..N_FFT {
                    indata[i] = row[start + i] * self.window[i];
                }
                self.r2c.process_with_scratch(&mut indata, &mut outdata, &mut scratch).expect("FFT failed");
                // Layout [B, N_BINS, T, 2]: interleave real/imag per bin.
                for (k, c) in outdata.iter().enumerate().take(N_BINS) {
                    let off = ((b * N_BINS + k) * n_frames + f) * 2;
                    out[off] = c.re;
                    out[off + 1] = c.im;
                }
            }
        }
        out
    }

    /// Inverse STFT of `[B, N_BINS, T, 2]` → `[B, L']` PCM via overlap-add with
    /// the squared analysis window as the synthesis normalization (matching
    /// `torch.istft`, which divides the overlap-add of `frame·window` by the
    /// overlap-add of `window²`).
    ///
    /// `n_frames` is inferred from `spec.len()`; the output length matches the
    /// overlap-add region (`(n_frames - 1) * HOP + N_FFT`), trimmed to drop the
    /// `center=True` padding — same length contract as `torch.istft`.
    pub fn inverse(&self, spec: &[f32], batch: usize, n_frames: usize) -> Vec<f32> {
        let out_len = (n_frames.saturating_sub(1)) * HOP + N_FFT;
        let mut signal = vec![0.0f32; batch * out_len];
        let mut norm = vec![0.0f32; batch * out_len];
        let mut indata = self.c2r.make_input_vec(); // Vec<Complex32>, len N_BINS
        let mut outdata = self.c2r.make_output_vec(); // Vec<f32>, len N_FFT
        let mut scratch = self.c2r.make_scratch_vec();

        for b in 0..batch {
            for f in 0..n_frames {
                // `make_input_vec` is exactly the one-sided spectrum
                // (`complex_len() == N_BINS`); ComplexToReal reconstructs the
                // conjugate half itself, so every slot is written below and the
                // buffer needs no clearing between frames.
                for (k, slot) in indata.iter_mut().enumerate().take(N_BINS) {
                    let off = ((b * N_BINS + k) * n_frames + f) * 2;
                    *slot = realfft::num_complex::Complex32::new(spec[off], spec[off + 1]);
                }
                // realfft's ComplexToReal requires the DC (bin 0) and Nyquist
                // (last) bins to be purely real (Hermitian symmetry). The network
                // output doesn't enforce this; torch.istft just ignores the
                // imaginary part, so we do the same.
                indata[0].im = 0.0;
                indata[N_BINS - 1].im = 0.0;
                self.c2r.process_with_scratch(&mut indata, &mut outdata, &mut scratch).expect("IFFT failed");
                let start = f * HOP;
                for (i, &sample) in outdata.iter().enumerate().take(N_FFT) {
                    let s = start + i;
                    if s < out_len {
                        let v = sample * self.window[i] / N_FFT as f32;
                        signal[b * out_len + s] += v;
                        norm[b * out_len + s] += self.window[i] * self.window[i];
                    }
                }
            }
        }

        // Normalize by the window-squared overlap-add, then trim the center=True
        // padding: PyTorch's istft drops N_FFT/2 from each end.
        let trim = N_FFT / 2;
        let inner = out_len.saturating_sub(2 * trim);
        let mut out = vec![0.0f32; batch * inner];
        for b in 0..batch {
            for i in 0..inner {
                let s = b * out_len + trim + i;
                let n = norm[s];
                out[b * inner + i] = if n > 1e-11 { signal[s] / n } else { 0.0 };
            }
        }
        out
    }

    /// Convenience: forward STFT wrapped as a `[batch, N_BINS, T, 2]` tensor.
    pub fn forward_tensor(&self, waveform: &[f32], batch: usize) -> Result<Tensor, svod_tensor::error::Error> {
        let data = self.forward(waveform, batch);
        let n_frames = Self::num_frames(waveform.len() / batch);
        Tensor::from_slice(data.as_slice()).try_reshape([batch as isize, N_BINS as isize, n_frames as isize, 2])
    }
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}
