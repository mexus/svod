use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Module, StateDict};
use test_case::test_case;

use crate::yolo::YoloConv;
use crate::yolo::loader::fold_batchnorm;

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7919 % 97) as f32 / 97.0 - 0.5) * scale + offset).collect()
}

fn unfolded_state(cin: usize, cout: usize, k: usize) -> StateDict {
    let mut sd = StateDict::new();
    let t = |data: Vec<f32>, shape: &[isize]| Tensor::from_slice(data).try_reshape(shape.to_vec()).unwrap();
    sd.insert(
        "conv.weight".into(),
        t(ramp(cout * cin * k * k, 0.2, 0.0), &[cout as isize, cin as isize, k as isize, k as isize]),
    );
    sd.insert("bn.weight".into(), t(ramp(cout, 0.5, 1.0), &[cout as isize]));
    sd.insert("bn.bias".into(), t(ramp(cout, 0.3, 0.0), &[cout as isize]));
    sd.insert("bn.running_mean".into(), t(ramp(cout, 0.4, 0.0), &[cout as isize]));
    sd.insert("bn.running_var".into(), t(ramp(cout, 0.5, 1.0), &[cout as isize]));
    sd
}

/// Folding is value-preserving: the biased conv alone reproduces conv + norm.
#[test_case(4, 8, 3, true; "3x3 with activation")]
#[test_case(6, 6, 1, false; "1x1 without activation")]
fn a_folded_conv_matches_conv_then_norm(cin: usize, cout: usize, k: usize, act: bool) {
    let sd = unfolded_state(cin, cout, k);
    let mut plain = YoloConv::empty(cin, cout, k, 1, act);
    plain.load_state_dict(&sd, "").unwrap();
    let mut folded = YoloConv::empty(cin, cout, k, 1, act);
    folded.load_state_dict(&fold_batchnorm(&sd).unwrap(), "").unwrap();
    assert!(folded.conv.bias.is_some(), "the fold leaves a bias on the conv");
    assert!(plain.conv.bias.is_none());

    let x = Tensor::from_slice(ramp(cin * 25, 2.0, 0.1)).try_reshape([1, cin as isize, 5, 5]).unwrap();
    let want = plain.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let got = folded.forward(&x).unwrap().to_vec::<f32>().unwrap();
    let max = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 1e-5, "folded conv drifts by {max}");
}

/// Only a `conv.weight` with a full `bn.*` beside it folds; anything else, such
/// as the head's biased final convs or the norm keys themselves, passes through.
#[test]
fn the_fold_leaves_other_keys_alone() {
    let mut sd = unfolded_state(2, 2, 1);
    sd.insert("head.2.weight".into(), Tensor::zeros(&[2, 2, 1, 1], DType::Float32));
    sd.insert("head.2.bias".into(), Tensor::zeros(&[2], DType::Float32));
    sd.insert("lone.conv.weight".into(), Tensor::zeros(&[2, 2, 1, 1], DType::Float32));
    let folded = fold_batchnorm(&sd).unwrap();
    assert_eq!(folded.len(), sd.len() + 1, "exactly one bias appears");
    assert!(folded.contains_key("conv.bias"));
    assert!(!folded.contains_key("lone.conv.bias"), "no norm, nothing to fold");
    for key in ["bn.weight", "bn.running_var", "head.2.weight", "lone.conv.weight"] {
        assert!(std::sync::Arc::ptr_eq(&sd[key].uop(), &folded[key].uop()), "{key} is handed straight through");
    }
}

/// The folded weight keeps the checkpoint's dtype; the fold itself runs in f32.
#[test]
fn the_fold_keeps_the_weight_dtype() {
    let sd: StateDict = unfolded_state(2, 4, 3).into_iter().map(|(k, t)| (k, t.cast(DType::Float16))).collect();
    let folded = fold_batchnorm(&sd).unwrap();
    assert_eq!(folded["conv.weight"].dtype(), DType::Float16);
    assert_eq!(folded["conv.bias"].dtype(), DType::Float16);
}
