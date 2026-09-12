//! Property tests for the 64-bit word split in [`pm_long_decomp`]: backends without
//! native i64 (WebGPU) must reproduce the native `<<` / `>>` result bit for bit.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, ScalarDType};
use svod_ir::ops;
use svod_ir::rewrite::graph_rewrite_bottom_up;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::{Op, UOp};

use crate::devectorize::pm_long_decomp;
use crate::test::support::prelude::*;

/// Fold a fully constant word expression, mirroring what the backend would compute.
fn eval_word(expr: &Arc<UOp>) -> ConstValue {
    eval_typed(expr, &Bindings::none()).expect("word expression must fold")
}

/// The low 32 bits of a word constant.
fn word_bits(value: ConstValue) -> u32 {
    match value {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        other => panic!("word is not an integer: {other:?}"),
    }
}

/// The 64-bit constant `value` at `from`.
pub fn long_const(from: ScalarDType, value: u64) -> Arc<UOp> {
    let long = DType::Scalar(from);
    UOp::const_(long, if from == ScalarDType::Int64 { ConstValue::Int(value as i64) } else { ConstValue::UInt(value) })
}

/// A leftover 64-bit node means the rewrite stalled and the input came back untouched.
fn assert_fully_split(root: &Arc<UOp>) {
    let long = |node: &Arc<UOp>| matches!(node.dtype().base(), ScalarDType::Int64 | ScalarDType::UInt64);
    assert!(root.toposort().iter().all(|node| !long(node)), "64-bit node survived decomposition: {}", root.tree());
}

/// Split `STORE(buffer[at], value)` with `pm_long_decomp`; return each word's bits and address.
pub fn split_store(from: ScalarDType, at: i64, value: Arc<UOp>) -> [(u32, i64); 2] {
    let indices = vec![UOp::const_(DType::Index, ConstValue::Int(at))];
    let index = UOp::index().buffer(buffer_of(8, from)).indices(indices).call().unwrap();
    let decomposed = graph_rewrite_bottom_up(&pm_long_decomp(), index.store(value), &mut ());
    assert_fully_split(&decomposed);

    let mut words = [None; 2];
    for node in decomposed.toposort() {
        let Op::Store(ops::Store { index, value, .. }) = node.op() else { continue };
        let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("a split store addresses an INDEX") };
        let address = eval_word(indices.last().expect("INDEX carries an index")).try_int().expect("address");
        let word = node.tag().as_ref().expect("split store is word-tagged")[1];
        words[word] = Some((word_bits(eval_word(value)), address));
    }
    [words[0].expect("low word"), words[1].expect("high word")]
}

/// `[low, high]` for `STORE(index, value)`; the words must land on adjacent elements.
fn split_long(from: ScalarDType, value: Arc<UOp>) -> [u32; 2] {
    let [(low, low_at), (high, high_at)] = split_store(from, 1, value);
    assert_eq!([low_at, high_at], [2, 3], "{from:?} word addresses");
    [low, high]
}

/// `a * b` and `-a` at 64 bits, plus the float cast that reads both words.
pub fn assert_long_arithmetic(a: u64, b: u64, from: ScalarDType) {
    let folded = |value: u64| [value as u32, (value >> 32) as u32];
    let mul = Op::Binary(BinaryOp::Mul, long_const(from, a), long_const(from, b));
    assert_eq!(split_long(from, UOp::new(mul, DType::Scalar(from))), folded(a.wrapping_mul(b)), "{from:?} mul");
    let neg = Op::Unary(svod_ir::UnaryOp::Neg, long_const(from, a));
    assert_eq!(split_long(from, UOp::new(neg, DType::Scalar(from))), folded(a.wrapping_neg()), "{from:?} neg");
    // A cast away from a long is not itself split, so only the stall is observable.
    let cast = long_const(from, a).cast(DType::Float32);
    assert_fully_split(&graph_rewrite_bottom_up(&pm_long_decomp(), cast, &mut ()));
}

proptest! {
    #![proptest_config(cheap())]

    /// `x << s` / `x >> s` must equal the native 64-bit shift at both words, for `s < 64`.
    #[test]
    fn long_shift_words_match_native(
        value in any::<u64>(),
        shift in 0u64..64,
        signed in any::<bool>(),
        right in any::<bool>(),
    ) {
        let from = if signed { ScalarDType::Int64 } else { ScalarDType::UInt64 };
        let op = if right { BinaryOp::Shr } else { BinaryOp::Shl };
        let native = match op {
            BinaryOp::Shl => value << shift,
            BinaryOp::Shr if signed => ((value as i64) >> shift) as u64,
            _ => value >> shift,
        };
        let expr = UOp::new(Op::Binary(op, long_const(from, value), long_const(from, shift)), DType::Scalar(from));
        let [(low, low_at), (high, high_at)] = split_store(from, 1, expr);
        prop_assert_eq!([low_at, high_at], [2, 3], "{:?} word addresses", from);
        prop_assert_eq!([low, high], [native as u32, (native >> 32) as u32]);
    }

    /// The word split of `a * b` and `-a` must equal the native 64-bit result.
    #[test]
    fn long_arithmetic_words_match_native(a in any::<u64>(), b in any::<u64>(), signed in any::<bool>()) {
        assert_long_arithmetic(a, b, if signed { ScalarDType::Int64 } else { ScalarDType::UInt64 });
    }
}
