//! GTCRN cheap default-tier tests — build the symbolic forward graph and assert
//! output shapes, plus a state-dict round-trip. No `.realize()`, no checkpoint,
//! milliseconds total (mirrors the silero_vad / resnet default-tier pattern).

use std::sync::Arc;

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use crate::gtcrn::{Gtcrn, N_FREQ};

/// `forward` on the `[1, 257, T, 2]` spectrogram returns a fully concrete
/// `[1, 257, T, 2]`. Catches axis/permute bugs across
/// ERB→SFE→encoder→DPGRNN→decoder→mask without any kernel compile; a dim that
/// only has a symbolic upper bound fails here rather than being accepted.
#[test]
fn forward_shape() {
    let model = Gtcrn::with_random_weights();
    let t = 16;
    let spec = Tensor::zeros(&[1, N_FREQ, t, 2], DType::Float32);
    let out = model.forward(&spec).unwrap();
    assert_eq!(out.dims().expect("concrete output shape"), vec![1, N_FREQ, t, 2]);
    assert_eq!(out.dtype(), DType::Float32);
}

/// The whole network is a multiplicative mask on the input spectrogram, so a
/// zero spectrogram must come back exactly zero however the weights fall. This
/// is the cheapest end-to-end assertion that actually *runs* the graph: it
/// catches a decoder skip-connection or mask-product wiring bug that a shape
/// check cannot see.
///
/// 8 frames for coverage, not out of necessity: two is enough to pass now that
/// `Tensor::pool` clamps the window arithmetic that used to underflow when a
/// dilation exceeds the output extent, which the encoder's last GT block (T
/// dilated by 5) reaches at every short input.
#[test]
fn zero_spec_stays_zero() {
    let model = Gtcrn::with_random_weights();
    let spec = Tensor::zeros(&[1, N_FREQ, 8, 2], DType::Float32);
    let out = model.forward(&spec).unwrap();
    out.realize().unwrap();
    assert!(out.as_vec::<f32>().unwrap().iter().all(|&v| v == 0.0), "mask output is not identically zero");
}

/// Every weight the checkpoint carries, with the shape it carries it in. The
/// transposed decoder convs store `[in, out/groups, kH, kW]` while the encoder
/// stores `[out, in/groups, kH, kW]`, so a swapped constructor argument shows
/// up here and nowhere else until a checkpoint is loaded.
const EXPECTED_SHAPES: &[(&str, &[usize])] = &[
    ("erb.erb_fc.weight", &[64, 192]),
    ("erb.ierb_fc.weight", &[192, 64]),
    ("encoder.en_convs.0.conv.weight", &[16, 9, 1, 5]),
    ("encoder.en_convs.0.conv.bias", &[16]),
    ("encoder.en_convs.0.bn.running_var", &[16]),
    ("encoder.en_convs.0.act.weight", &[1]),
    ("encoder.en_convs.1.conv.weight", &[16, 8, 1, 5]), // groups=2
    ("encoder.en_convs.2.point_conv1.weight", &[16, 24, 1, 1]),
    ("encoder.en_convs.2.depth_conv.weight", &[16, 1, 3, 3]), // groups=16
    ("encoder.en_convs.2.point_conv2.weight", &[8, 16, 1, 1]),
    ("encoder.en_convs.2.tra.att_gru.weight_ih_l0", &[48, 8]),
    ("encoder.en_convs.2.tra.att_gru.weight_hh_l0", &[48, 16]),
    ("encoder.en_convs.2.tra.att_fc.weight", &[8, 16]),
    ("dpgrnn1.intra_rnn.rnn1_f.weight_ih_l0", &[12, 8]),
    ("dpgrnn1.intra_rnn.rnn1_b.weight_hh_l0", &[12, 4]),
    ("dpgrnn1.inter_rnn.rnn2_f.weight_hh_l0", &[24, 8]),
    ("dpgrnn1.intra_ln.weight", &[33, 16]),
    ("dpgrnn2.inter_fc.bias", &[16]),
    ("decoder.de_convs.0.point_conv1.weight", &[24, 16, 1, 1]),
    ("decoder.de_convs.0.depth_conv.weight", &[16, 1, 3, 3]),
    ("decoder.de_convs.0.point_conv2.weight", &[16, 8, 1, 1]),
    ("decoder.de_convs.3.conv.weight", &[16, 8, 1, 5]), // deconv, groups=2
    ("decoder.de_convs.4.conv.weight", &[16, 2, 1, 5]),
    ("decoder.de_convs.4.bn.weight", &[2]),
];

/// The converted checkpoint (`scripts/convert_gtcrn.py`) holds exactly this
/// many tensors; the emitted dict must be key-for-key the same set.
const CHECKPOINT_KEYS: usize = 249;

/// Emitted keys and shapes match the checkpoint, and a reload carries every
/// value across: reloading into a second model and re-emitting must yield the
/// *same tensors*, which a loader that silently skips a key cannot do.
#[test]
fn state_dict_round_trip() {
    let model = Gtcrn::with_random_weights();
    let sd = model.state_dict("");
    assert_eq!(sd.len(), CHECKPOINT_KEYS);

    for (key, shape) in EXPECTED_SHAPES {
        let t = sd.get(*key).unwrap_or_else(|| panic!("missing key: {key}"));
        assert_eq!(&t.dims().unwrap(), shape, "wrong shape for {key}");
    }

    // The unidirectional inter_rnn has no reverse weights, and nothing carries
    // the retired folded-BN `invstd`.
    assert!(!sd.contains_key("dpgrnn1.inter_rnn.rnn1_b.weight_ih_l0"), "inter_rnn is unidirectional");
    assert!(!sd.keys().any(|k| k.ends_with(".invstd")), "batch norms keep PyTorch's raw running stats");
    // The final decoder block ends in Tanh, so it has no PReLU slope.
    assert!(!sd.contains_key("decoder.de_convs.4.act.weight"), "the last block is Tanh");

    let mut reloaded = Gtcrn::with_random_weights();
    reloaded.load_state_dict(&sd, "").expect("load round-trip");
    let out = reloaded.state_dict("");

    assert_eq!(out.len(), sd.len());
    for (key, want) in &sd {
        let got = out.get(key).unwrap_or_else(|| panic!("key dropped by the round trip: {key}"));
        assert!(Arc::ptr_eq(&got.uop(), &want.uop()), "{key} was not taken from the state dict");
    }
}

/// A checkpoint missing a required weight is an error, not a silent fallback to
/// the randomly initialized parameter.
#[test]
fn load_state_dict_rejects_a_missing_key() {
    let mut sd = Gtcrn::with_random_weights().state_dict("");
    sd.remove("encoder.en_convs.2.depth_conv.weight").expect("key present");
    let err = Gtcrn::from_state_dict(&sd).err().expect("missing weight must fail");
    assert!(err.to_string().contains("depth_conv.weight"), "unhelpful error: {err}");
}

/// `Conv::with_causal_pad` claims that folding a GT block's leading `(2d, 0)`
/// pad into the depth conv's own padding is exact — for a `Conv2d` by widening
/// the leading pad, for a `ConvTranspose2d` (whose padding *crops*) by
/// narrowing it. Both must be bit-for-bit identical to the explicit pad, at
/// every dilation GTCRN uses.
#[test]
fn causal_pad_folds_into_conv_padding() {
    use svod_tensor::nn::{Conv2d, ConvTranspose2d, Layer};

    let rand = |shape: &[usize]| Tensor::uniform_with_dtype(shape, -1.0, 1.0, DType::Float32).unwrap().contiguous();
    let x = rand(&[1, 4, 12, 9]);
    for d in [1usize, 2, 5] {
        let pad = (2 * d) as isize;
        let prepad = |x: &Tensor| x.try_pad(&[(0, 0), (0, 0), (pad, 0), (0, 0)]).unwrap();
        let w = rand(&[4, 1, 3, 3]);

        let conv = Conv2d::new(w.clone(), None).with_groups(4).with_dilation((d, 1));
        let explicit = conv.clone().with_padding(((0, 0), (1, 1))).forward(&prepad(&x)).unwrap();
        let folded = conv.with_padding(((pad, 0), (1, 1))).forward(&x).unwrap();
        assert_eq!(explicit.to_vec::<f32>().unwrap(), folded.to_vec::<f32>().unwrap(), "conv2d, dilation {d}");

        let deconv = ConvTranspose2d::new(w, None).with_groups(4).with_dilation((d, 1));
        let explicit = deconv.clone().with_padding(((pad, pad), (1, 1))).forward(&prepad(&x)).unwrap();
        let folded = deconv.with_padding(((0, pad), (1, 1))).forward(&x).unwrap();
        assert_eq!(explicit.to_vec::<f32>().unwrap(), folded.to_vec::<f32>().unwrap(), "deconv, dilation {d}");
    }
}
