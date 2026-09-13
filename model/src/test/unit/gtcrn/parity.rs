//! Golden parity test — loads the real GTCRN checkpoint + a PyTorch-generated
//! golden spectrogram, runs the svod forward, and compares against the
//! reference enhanced spectrogram.
//!
//! ```text
//! SVOD_GTCRN=$PWD/data/gtcrn cargo test -p svod-model --lib gtcrn::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_tensor::Tensor;

use crate::gtcrn::{Gtcrn, HOP, HUB_REPO};
use crate::state::{StateDict, load_safetensors};

fn resolve_file(name: &str) -> PathBuf {
    for dir in [std::env::var("SVOD_GTCRN").ok(), Some(format!("{}/../data/gtcrn", env!("CARGO_MANIFEST_DIR")))]
        .into_iter()
        .flatten()
    {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    crate::hub::HubRepo::open(HUB_REPO, "main")
        .and_then(|repo| repo.get(name))
        .unwrap_or_else(|e| panic!("download {name} from {HUB_REPO}: {e}"))
}

fn load_golden_vec(sd: &StateDict, key: &str) -> Vec<f32> {
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.to_vec::<f32>().unwrap()
}

/// Number of STFT frames exercised by the parity forward. Kept small: the GRU
/// recurrence unrolls one IR node per time step, so the full 611-frame golden
/// would explode the symbolic graph. 24 frames exercise every op (encoder
/// downsampling, bidirectional DPGRNN, conv-transpose decoder) while staying
/// tractable. The crop is applied to both the input spec and the reference
/// output (captured as a separate PyTorch forward on the same window), so the
/// comparison is self-consistent.
const PARITY_FRAMES: usize = 24;

#[test]
#[ignore = "heavy: real GTCRN weights + PyTorch golden (local or HF Hub download)"]
fn forward_matches_pytorch() {
    let model = Gtcrn::from_safetensors(&resolve_file("gtcrn.safetensors")).expect("load model");
    let golden = load_safetensors(&resolve_file("golden.safetensors")).expect("load golden");

    // Use the pre-cropped 24-frame spec + output (the PyTorch forward on the
    // same cropped window). The full-length `spec` / `output` keys are stored
    // transposed by the converter; the `*_crop` pair is the correct layout.
    let spec_vec = load_golden_vec(&golden, "spec_crop");
    let want = load_golden_vec(&golden, "output_crop");
    let spec = Tensor::from_slice(&spec_vec).try_reshape([1, 257, PARITY_FRAMES as isize, 2]).unwrap();

    let got = model.forward(&spec).expect("forward").to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len(), "output length mismatch");

    let delta = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    // fp32 conv/GRU, no quantization. GTCRN's output range is large (raw STFT
    // magnitudes), so bound the absolute delta by the reference's peak.
    let peak = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel = delta / peak.max(1e-6);
    assert!(rel < 1e-3, "max |delta| = {delta:.6} (rel {rel:.2e}, peak {peak:.2}); exceeds 1e-3 relative");
}

/// Relative bound on the waveform round trip, as a fraction of the golden's
/// peak. Two errors add up: the graph STFT is a windowed-DFT `conv1d` rather
/// than a radix FFT, which at this size costs 4.3e-7 relative against a host
/// `realfft` transform, and the fp32 network itself lands at 2.56e-6 absolute
/// (5.2e-7 of its own peak) on the spectrogram — [`forward_matches_pytorch`].
/// The measured waveform figure is 1.5e-6 relative on CPU and 1.9e-6 on CUDA,
/// so this leaves ~2x for another device's fma order.
const ENHANCE_REL: f32 = 4e-6;

/// Samples fed to the waveform round trip: 32 hops, i.e. 33 STFT frames —
/// the order [`PARITY_FRAMES`] keeps the GRU unroll to, and a whole number of
/// output samples (`(T - 1) · HOP == ENHANCE_SAMPLES`).
const ENHANCE_SAMPLES: usize = 32 * HOP;

/// Output samples the prefix reproduces exactly. Frame `t` reads
/// `[t·HOP - HOP, t·HOP + HOP)` of the signal, so the last frame of the prefix
/// straddles its end and reads the prefix's own reflection instead of the
/// golden's next samples; sample `n` is overlap-added from frames
/// `n/HOP` and `n/HOP + 1`, so everything below `(32 - 2) · HOP` is clean.
const ENHANCE_CLEAN: usize = ENHANCE_SAMPLES - 2 * HOP;

/// Waveform → waveform through [`Gtcrn::enhance`] (graph STFT → network →
/// graph ISTFT) against PyTorch's full-length enhanced waveform.
///
/// GTCRN is causal in time — the DPGRNN's inter-frame RNN and the TRA gates
/// run forward only, and the GT blocks pad T causally — so a prefix of the
/// golden input reproduces the full run's output over the frames it shares,
/// up to the right-edge reflection excluded by [`ENHANCE_CLEAN`].
#[test]
#[ignore = "heavy: real GTCRN weights + PyTorch golden (local or HF Hub download)"]
fn enhance_matches_pytorch() {
    let model = Gtcrn::from_safetensors(&resolve_file("gtcrn.safetensors")).expect("load model");
    let golden = load_safetensors(&resolve_file("golden.safetensors")).expect("load golden");

    let samples = load_golden_vec(&golden, "samples");
    let want = load_golden_vec(&golden, "enh");
    let noisy = Tensor::from_slice(&samples[..ENHANCE_SAMPLES])
        .try_reshape([1, ENHANCE_SAMPLES as isize])
        .expect("reshape input");

    let got = model.enhance(&noisy).expect("enhance").to_vec::<f32>().unwrap();
    assert_eq!(got.len(), ENHANCE_SAMPLES, "(T - 1) · HOP samples come back out");

    let (got, want) = (&got[..ENHANCE_CLEAN], &want[..ENHANCE_CLEAN]);
    let delta = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let peak = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel = delta / peak.max(1e-6);
    assert!(rel < ENHANCE_REL, "max |delta| = {delta:.3e} (rel {rel:.2e}, peak {peak:.3}); exceeds {ENHANCE_REL:.1e}");
}
