//! GigaAM RN-T decode tests: the predictor's gather embedding, the joint's
//! padded class axis, and the block plan's structure.

use svod_arch::rnnt::{BatchBlockStep, LaneDecode, RnntDecoder, RnntOpts};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::gigaam::rnnt::RnntBlockBackend;
use crate::gigaam::rnnt::block::DECODE_WINDOW;
use crate::gigaam::rnnt::joint::{CLASS_ALIGN, RnntJoint};
use crate::gigaam::rnnt::predictor::RnntPredictor;
use crate::gigaam::{GigaAm, TransducerConfig};

/// `[b, 1]` i64 token ids: the extremes (0, blank) plus a deterministic
/// spread over the vocab.
fn token_ids(b: usize, num_classes: usize) -> Tensor {
    let ids: Vec<i64> = (0..b)
        .map(|i| match i {
            0 => 0,
            1 => num_classes as i64 - 1,
            _ => ((i * 7919) % num_classes) as i64,
        })
        .collect();
    Tensor::from_slice(&ids).try_reshape([b as isize, 1]).expect("ids [b, 1]")
}

/// Uniform `[-0.5, 0.5)` f32 tensor, so logits and activations have both signs.
fn centered(shape: &[usize]) -> Tensor {
    Tensor::rand(shape).expect("rand").try_sub(0.5f32).expect("shift")
}

fn f32s(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().expect("read f32")
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs {y}");
    }
}

// ─── Predictor: gather embedding ──────────────────────────────────────────

/// `forward_parts` embeds `prev` with a row gather; it must match the one-hot
/// `Tensor::embedding` reference fed through the same LSTM stack, for ids
/// that include row 0 and the blank row.
#[test_case(1, 8, 16, 1; "single lane")]
#[test_case(4, 33, 16, 1; "unaligned vocab")]
#[test_case(6, 1025, 32, 2; "gigaam vocab, two layers")]
fn predictor_gather_embedding_matches_one_hot(b: usize, num_classes: usize, p: usize, layers: usize) {
    let predictor = RnntPredictor::empty(p, layers, num_classes);
    let ids = token_ids(b, num_classes);
    let h = centered(&[layers, b, p]);
    let c = centered(&[layers, b, p]);

    let (g, new_h, new_c) = predictor.forward_parts(&ids, &h, &c).expect("forward_parts");

    let emb = predictor.embed.embedding(&ids).expect("embedding").try_squeeze(Some(1)).expect("[b, p]");
    let (top, ref_h, ref_c) = predictor.lstm.step_stacked(&emb, &h, &c).expect("reference lstm");
    let flat =
        |s: Tensor| s.try_permute(&[1, 0, 2]).unwrap().try_reshape([b as isize, 1, (layers * p) as isize]).unwrap();

    assert_eq!(g.dims().unwrap(), vec![b, 1, p]);
    assert_close(&f32s(&g), &f32s(&top), 1e-6, "g");
    assert_close(&f32s(&new_h), &f32s(&flat(ref_h)), 1e-6, "h");
    assert_close(&f32s(&new_c), &f32s(&flat(ref_c)), 1e-6, "c");
}

// ─── Joint: padded class axis ─────────────────────────────────────────────

/// Padding the output projection to a multiple of `CLASS_ALIGN` must not
/// change any argmax (the padded logits are `-1e30`), keep every index below
/// the real class count, and leave `forward`'s log-probs at the real width.
#[test_case(8, 1; "already aligned")]
#[test_case(33, 4; "unaligned, window")]
#[test_case(1025, 4; "gigaam vocab")]
fn padded_joint_argmax_matches_unpadded(num_classes: usize, w: usize) {
    const B: usize = 6;
    let (enc_hidden, pred_hidden, joint_hidden) = (16, 16, 32);
    let joint = RnntJoint::empty(enc_hidden, pred_hidden, joint_hidden, num_classes);
    let mut padded = joint.clone();
    padded.pad_classes(CLASS_ALIGN).expect("pad");
    let width = num_classes.div_ceil(CLASS_ALIGN) * CLASS_ALIGN;
    assert_eq!(padded.out_w.dims().unwrap(), vec![width, joint_hidden]);
    assert_eq!(padded.out_b.dims().unwrap(), vec![width]);
    assert_eq!(padded.num_classes, num_classes);
    padded.pad_classes(CLASS_ALIGN).expect("pad again");
    assert_eq!(padded.out_w.dims().unwrap(), vec![width, joint_hidden], "pad_classes must be idempotent");

    let enc_proj = centered(&[B, w, joint_hidden]);
    let g = centered(&[B, 1, pred_hidden]);
    let plain = joint.argmax_preproj(&enc_proj, &g).expect("argmax").to_vec::<i32>().expect("ids");
    let via_pad = padded.argmax_preproj(&enc_proj, &g).expect("padded argmax").to_vec::<i32>().expect("ids");
    assert_eq!(plain.len(), B * w);
    assert!(via_pad.iter().all(|&t| (t as usize) < num_classes), "padded class leaked into the argmax: {via_pad:?}");
    assert_eq!(via_pad, plain);

    let enc = centered(&[B, w, enc_hidden]);
    let logp = joint.forward(&enc, &g).expect("log-probs");
    let logp_pad = padded.forward(&enc, &g).expect("padded log-probs");
    assert_eq!(logp_pad.dims().unwrap(), vec![B, w, num_classes]);
    assert_close(&f32s(&logp_pad), &f32s(&logp), 1e-5, "log_softmax");
}

// ─── Block plan ───────────────────────────────────────────────────────────

/// The decode plan reduces over the padded class axis only (no kernel carries
/// the raw class count as a loop dim, some carry the padded one), and the
/// tape never carries a padded class id: every token slot (emitting or not)
/// stays below the real class count across a full wave. Compiles the block
/// JIT on random weights: a few seconds on CPU.
#[test]
fn rnnt_block_plan_reduces_over_padded_classes_only() {
    const LANES: usize = 2;
    const MAX_T: usize = 48;
    const NUM_CLASSES: usize = 33;
    const PADDED: usize = NUM_CLASSES.div_ceil(CLASS_ALIGN) * CLASS_ALIGN;

    let mut cfg = super::super::batch::test_config();
    cfg.transducer = Some(TransducerConfig {
        pred_hidden: 16,
        pred_rnn_layers: 1,
        joint_hidden: 24,
        num_classes: NUM_CLASSES,
        max_symbols_per_step: 3,
        vocabulary: (0..NUM_CLASSES - 1).map(|i| i.to_string()).collect(),
        sentencepiece: false,
    });
    let d_model = cfg.d_model;
    let model = GigaAm::with_random_weights(cfg);
    let mut backend: RnntBlockBackend = RnntBlockBackend::from_model(model, LANES, MAX_T).expect("block backend");

    let names = backend.kernel_names().expect("kernel names");
    let with_dim = |d: usize| names.iter().filter(|n| n.split('_').any(|dim| dim == d.to_string())).collect::<Vec<_>>();
    assert!(
        with_dim(NUM_CLASSES).is_empty(),
        "kernels over the raw class axis {NUM_CLASSES}: {:?}",
        with_dim(NUM_CLASSES)
    );
    assert!(!with_dim(PADDED).is_empty(), "no kernel over the padded class axis {PADDED}: {names:?}");

    let valid = [MAX_T, MAX_T - 7];
    let frames: Vec<Vec<f32>> = valid
        .iter()
        .enumerate()
        .map(|(lane, &n)| (0..n * d_model).map(|i| ((i + lane) % 13) as f32 * 0.05 - 0.3).collect())
        .collect();
    backend.reset().expect("reset");
    backend.bind_batch(&frames, &valid).expect("bind");
    let mut emitted = 0;
    for _ in 0..64 {
        let tapes = backend.run_block().expect("block");
        assert!(tapes.tokens.iter().all(|&t| (t as usize) < NUM_CLASSES), "padded class in tape: {:?}", tapes.tokens);
        emitted += tapes.emit.iter().filter(|&&e| e != 0).count();
        if !tapes.active_any {
            assert!(emitted > 0, "random weights emitted nothing; the leak check saw no real tokens");
            return;
        }
    }
    panic!("wave never finished");
}

/// The WIND window is a pure performance knob, so every `W >= 1` must produce
/// the same emission stream. `W == 1` is the case that folds: the one-element
/// window makes `first_nb` a constant 0, `emit.select(first_nb, 0)` collapses
/// to `0`, and the per-step token buffers drop out of the frame-tape kernel's
/// ABI — the plan then carries fewer compact buffers than the CALL has
/// arguments, which used to fail `prepare`. The raw tapes are *not* comparable
/// across windows (a wider window collapses a blank run into one step, so
/// emissions land on different tape slots); the decoded `(token, frame)` stream
/// is the invariant.
#[test_case(1; "per-frame baseline")]
#[test_case(2; "two-frame window")]
#[test_case(8; "eight-frame window")]
fn rnnt_block_decode_is_window_invariant(window: usize) {
    const LANES: usize = 2;
    const MAX_T: usize = 32;
    const NUM_CLASSES: usize = 12;

    let mut cfg = super::super::batch::test_config();
    cfg.transducer = Some(TransducerConfig {
        pred_hidden: 16,
        pred_rnn_layers: 1,
        joint_hidden: 16,
        num_classes: NUM_CLASSES,
        max_symbols_per_step: 3,
        vocabulary: (0..NUM_CLASSES - 1).map(|i| i.to_string()).collect(),
        sentencepiece: false,
    });
    let d_model = cfg.d_model;
    let decoder =
        RnntDecoder::new((0..NUM_CLASSES - 1).map(|i| i.to_string()).collect(), RnntOpts { max_symbols_per_step: 3 });

    let valid = [MAX_T, MAX_T - 9];
    let frames: Vec<Vec<f32>> = valid
        .iter()
        .enumerate()
        .map(|(lane, &n)| (0..n * d_model).map(|i| ((i + lane) % 11) as f32 * 0.07 - 0.35).collect())
        .collect();

    fn decode<const W: usize>(
        model: GigaAm,
        frames: &[Vec<f32>],
        valid: &[usize],
        decoder: &RnntDecoder,
    ) -> Vec<LaneDecode> {
        let mut backend = RnntBlockBackend::<W>::from_model(model, LANES, MAX_T).expect("block backend");
        backend.bind_batch(frames, valid).expect("bind");
        decoder.decode_batch_blocks(valid, &mut backend).expect("decode")
    }

    // `with_random_weights` draws from the global RNG, which seeds itself from the
    // wall clock unless a test called `manual_seed`. Whether a random transducer
    // reaches a non-blank class before running out of frames is then part of the
    // draw — about one draw in thirty emits nothing, which makes the comparison
    // below vacuous and failed CI on an unrelated commit. Seeding would only fix
    // the draw, not the arithmetic: the emission also turns on float results that
    // differ between this host and the runner. So redraw until the reference has
    // something to compare, which holds whatever the platform computes.
    let (model, reference) = (0..8)
        .find_map(|_| {
            let model = GigaAm::with_random_weights(cfg.clone());
            let decoded = decode::<{ DECODE_WINDOW }>(model.clone(), &frames, &valid, &decoder);
            decoded.iter().any(|(_, emissions)| !emissions.is_empty()).then_some((model, decoded))
        })
        .expect("eight random transducers all emitted nothing; the fixture is no longer decoding");
    let under_test = match window {
        1 => decode::<1>(model, &frames, &valid, &decoder),
        2 => decode::<2>(model, &frames, &valid, &decoder),
        8 => decode::<8>(model, &frames, &valid, &decoder),
        w => panic!("unmapped window {w}"),
    };
    assert_eq!(under_test, reference, "W={window} decoded differently from W={DECODE_WINDOW}");
}
