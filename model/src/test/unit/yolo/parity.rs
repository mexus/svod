//! Golden parity test — downloads real YOLO26n weights from HuggingFace,
//! runs forward, and compares against a PyTorch reference output.
//!
//! ```text
//! SVOD_YOLO=$PWD/data/yolo cargo test -p svod-model --lib yolo::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_tensor::Tensor;

use crate::state::StateDict;
use crate::state::load_safetensors;
use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};

const HUB_REPO: &str = "ultralytics/yolo26n";

/// Boxes are decoded into pixels, so a deviation is only meaningful against the
/// image side: this is ~1e-4 of a 640 px coordinate, and ~20x what CPU f32
/// reassociation actually produces.
const BOX_TOL_PX: f32 = 0.05;

/// Scores share the scale PyTorch reports them on, so they keep a tolerance
/// that would catch a real numerical drift -- a wrong batch-norm epsilon moved
/// these by ~0.06.
const SCORE_TOL: f32 = 1e-3;

fn resolve_file(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_YOLO") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/yolo").join(name);
    if p.exists() {
        return p;
    }
    let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from {HUB_REPO}"))
}

fn load_golden_vec<T: Clone + Default + svod_dtype::ext::HasDType>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<T>().unwrap()
}

fn load_golden_i64(sd: &StateDict, key: &str) -> Vec<i64> {
    load_golden_vec(sd, key)
}

/// Decoded box coordinates live in pixel space, up to the image side; scores
/// are sigmoid outputs in [0, 1]. One absolute tolerance cannot serve both, so
/// split the deviation by channel: `4 + nc` channels of `anchors` each, boxes
/// first.
fn deltas_by_channel(got: &[f32], want: &[f32], channels: usize, anchors: usize) -> (f32, f32) {
    got.iter().zip(want).enumerate().fold((0.0f32, 0.0f32), |(boxes, scores), (i, (a, b))| {
        let d = (a - b).abs();
        if (i / anchors) % channels < 4 { (boxes.max(d), scores) } else { (boxes, scores.max(d)) }
    })
}

#[test]
#[ignore = "heavy: real YOLO26n weights + PyTorch golden (local or HF Hub download)"]
fn detect_output_matches_pytorch() {
    let weights = resolve_file("model.safetensors");
    let golden_path = resolve_file("golden.safetensors");

    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Detect::from_safetensors(&weights, cfg).expect("load model");

    let golden = load_safetensors(&golden_path).expect("load golden");

    let image_shape = load_golden_i64(&golden, "images_shape");
    let images_vec = load_golden_vec::<f32>(&golden, "images");
    let images = Tensor::from_slice(&images_vec)
        .try_reshape(image_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
        .unwrap();

    let out = model.forward(&images).expect("forward");
    out.realize().unwrap();

    let got = out.as_vec::<f32>().unwrap();
    let want = load_golden_vec::<f32>(&golden, "output");
    assert_eq!(got.len(), want.len(), "output length mismatch");

    let dims = out.dims().unwrap();
    let (channels, anchors) = (dims[1], dims[2]);
    let (box_delta, score_delta) = deltas_by_channel(&got, &want, channels, anchors);

    assert!(box_delta < BOX_TOL_PX, "max box |delta| = {box_delta:.6} px exceeds {BOX_TOL_PX}");
    assert!(score_delta < SCORE_TOL, "max score |delta| = {score_delta:.6} exceeds {SCORE_TOL:e}");
}
