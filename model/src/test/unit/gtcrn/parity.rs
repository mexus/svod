//! Golden parity test — loads the real GTCRN checkpoint + a PyTorch-generated
//! golden spectrogram, runs the svod forward, and compares against the
//! reference enhanced spectrogram.
//!
//! ```text
//! SVOD_GTCRN=$PWD/data/gtcrn cargo test -p svod-model --lib gtcrn::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_tensor::Tensor;

use crate::gtcrn::{Gtcrn, HUB_REPO};
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
