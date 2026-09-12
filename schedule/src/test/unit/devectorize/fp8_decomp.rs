//! Storage-dtype decomposition: FP8/BF16 widening (`pm_float_decomp`), the AMD non-native FP8 ALU widening, and the 64-bit word split (`pm_long_decomp`), plus the target table that picks them.
use super::helpers::*;
use crate::devectorize::{Fp8DecompCtx, amd_non_native_fp8_patterns, pm_float_decomp, pm_long_decomp};
use crate::optimizer::{Renderer, apply_dtype_decomps, get_dtype_decomps};
use crate::test::property::long_shift::{assert_long_arithmetic, long_const, split_store};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_ir::rewrite::graph_rewrite_bottom_up;
use svod_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp, ops};
use test_case::test_case;
fn decompose_to(from: ScalarDType, to: ScalarDType, root: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite_bottom_up(&pm_float_decomp(), root, &mut Fp8DecompCtx { from, to })
}
fn decompose(from: ScalarDType, root: Arc<UOp>) -> Arc<UOp> {
    decompose_to(from, ScalarDType::Float16, root)
}
fn store_value_dtypes(root: &Arc<UOp>) -> Vec<DType> {
    root.toposort()
        .into_iter()
        .filter_map(|node| match node.op() {
            Op::Store(ops::Store { value, .. }) => Some(value.dtype()),
            _ => None,
        })
        .collect()
}
/// Widening an FP8 load must not drop the gate/alt pair the late gater installed.
#[test]
fn fp8_decomp_preserves_alt_on_gated_load() {
    let alt = UOp::const_(DType::Scalar(ScalarDType::FP8E5M2), ConstValue::Float(0.0));
    let index = index(buffer_of(64, ScalarDType::FP8E5M2), 0);
    let load = UOp::load().index(index).alt(alt).gate(UOp::native_const(false)).call();
    let decomposed = decompose(ScalarDType::FP8E5M2, load);
    let gated: Vec<_> = decomposed
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Load(ops::Load { gate: Some(_), .. })))
        .collect();
    assert!(!gated.is_empty(), "the gated load must survive decomposition:\n{}", decomposed.tree());
    assert!(
        gated.iter().all(|node| matches!(node.op(), Op::Load(ops::Load { alt: Some(_), .. }))),
        "{}",
        decomposed.tree()
    );
}
#[test]
fn vector_fp8_load_decomposes_to_scalar_loads_and_stack() {
    let lanes = DType::Scalar(ScalarDType::FP8E4M3).vec(4).unwrap();
    let index = shaped_index(buffer_of(4, ScalarDType::FP8E4M3), 0..4).with_dtype(lanes.clone());
    let decomposed = decompose(ScalarDType::FP8E4M3, UOp::load().index(index).dtype(lanes).call());
    assert!(matches!(decomposed.op(), Op::Stack(..)), "{}", decomposed.tree());
    assert_eq!(decomposed.dtype(), DType::Float16);
    assert_eq!(loads(&decomposed), 4);
}
/// Both directions are rewritten: the STORE narrows to the uint8 storage form and the LOAD reads it back, leaving no FNUZ node behind.
#[test]
fn fnuz_store_and_load_are_both_decomposed() {
    let index = index(buffer_of(4, ScalarDType::FP8E4M3FNUZ), 0);
    let value = UOp::const_(DType::Scalar(ScalarDType::FP8E4M3FNUZ), ConstValue::Float(1.0));
    let decomposed = decompose(ScalarDType::FP8E4M3FNUZ, UOp::sink(vec![store(index.clone(), value), load(index)]));
    assert!(
        !decomposed.toposort().iter().any(|u| u.dtype().base() == ScalarDType::FP8E4M3FNUZ),
        "{}",
        decomposed.tree()
    );
    assert!(store_value_dtypes(&decomposed).contains(&DType::UInt8), "{}", decomposed.tree());
    assert_eq!(loads(&decomposed), 1, "one uint8 reload:\n{}", decomposed.tree());
    let reloaded = first_op(&decomposed, |op| matches!(op, Op::Load(..))).expect("the uint8 reload");
    assert_eq!(reloaded.dtype(), DType::UInt8, "{}", decomposed.tree());
}
const FP8_TO_HALF: &[(ScalarDType, ScalarDType)] = &[
    (ScalarDType::FP8E4M3, ScalarDType::Float16),
    (ScalarDType::FP8E5M2, ScalarDType::Float16),
    (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16),
    (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),
];
const FNUZ_TO_HALF: &[(ScalarDType, ScalarDType)] =
    &[(ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16), (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16)];
const WEBGPU: &[(ScalarDType, ScalarDType)] = &[
    (ScalarDType::Int64, ScalarDType::Int32),
    (ScalarDType::FP8E4M3, ScalarDType::Float32),
    (ScalarDType::FP8E5M2, ScalarDType::Float32),
    (ScalarDType::Float16, ScalarDType::Float32),
    (ScalarDType::BFloat16, ScalarDType::Float32),
    (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float32),
    (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float32),
];
/// Which storage dtypes need decomposing is a property of the target, not of the AST.
#[test_case(Renderer::cpu(), FP8_TO_HALF; "cpu lacks every fp8 encoding")]
#[test_case(Renderer::for_amd_arch(AmdArch::Gfx942), FNUZ_TO_HALF; "cdna3 renders ocp fp8 natively")]
#[test_case(Renderer::for_amd_arch(AmdArch::Gfx950), FNUZ_TO_HALF; "cdna4 renders ocp fp8 natively")]
#[test_case(Renderer::for_amd_arch(AmdArch::Gfx1151), FP8_TO_HALF; "rdna3.5 has no fp8 at all")]
#[test_case(Renderer::webgpu(), WEBGPU; "webgpu lacks 64-bit integers and every sub-f32 float")]
fn dtype_decomposition_mapping_is_target_sensitive(renderer: Renderer, expected: &[(ScalarDType, ScalarDType)]) {
    let values = [
        (ScalarDType::FP8E4M3, ConstValue::Float(1.0)),
        (ScalarDType::FP8E4M3FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::Float16, ConstValue::Float(1.0)),
        (ScalarDType::BFloat16, ConstValue::Float(1.0)),
        (ScalarDType::Int64, ConstValue::Int(1)),
        (ScalarDType::UInt64, ConstValue::UInt(1)),
    ];
    let sink = UOp::sink(values.into_iter().map(|(dt, value)| UOp::const_(DType::Scalar(dt), value)).collect());
    assert_eq!(get_dtype_decomps(&sink, &renderer).as_slice(), expected);
}
/// The combined pass must commit weak dtypes before decomposing, or the stored value keeps a weak type the narrowing rules never match.
#[test]
fn combined_dtype_pass_commits_weak_stores_before_decomposition() {
    let root = UOp::sink(vec![
        store(index(buffer_of(4, ScalarDType::FP8E4M3), 0), UOp::const_(DType::WeakFloat, ConstValue::Float(1.5))),
        store(index(buffer_of(4, ScalarDType::BFloat16), 0), UOp::const_(DType::WeakFloat, ConstValue::Float(-2.0))),
    ]);
    let decomposed = apply_dtype_decomps(root, Renderer::webgpu());
    assert!(
        !decomposed.toposort().iter().any(|u| matches!(u.dtype().base(), ScalarDType::FP8E4M3 | ScalarDType::BFloat16)),
        "{}",
        decomposed.tree()
    );
    let dtypes = store_value_dtypes(&decomposed);
    assert!(dtypes.contains(&DType::UInt8) && dtypes.contains(&DType::UInt16), "{}", decomposed.tree());
}
/// Same for the word split: a weak 64-bit value must be committed to `Int64` before it can be cut into two `Int32` words.
#[test]
fn combined_dtype_pass_commits_long_weak_store_before_word_split() {
    let value = UOp::new(
        Op::Binary(
            BinaryOp::Shl,
            UOp::const_(DType::WeakInt, ConstValue::Int(0x1_0000_0000)),
            UOp::const_(DType::WeakInt, ConstValue::Int(0x7654_3210)),
        ),
        DType::WeakInt,
    );
    let decomposed = apply_dtype_decomps(
        UOp::sink(vec![store(index(buffer_of(4, ScalarDType::Int64), 0), value)]),
        Renderer::webgpu(),
    );
    assert_eq!(store_value_dtypes(&decomposed), vec![DType::Int32; 2], "{}", decomposed.tree());
}
/// The two words of a split 64-bit STORE address *adjacent* elements of the doubled 32-bit buffer, at `2*i` and `2*i+1`.
#[test_case(0)]
#[test_case(3)]
fn long_store_words_address_adjacent_elements(at: i64) {
    for from in [ScalarDType::Int64, ScalarDType::UInt64] {
        let split = split_store(from, at, long_const(from, 0xdead_beef_feed_face));
        assert_eq!(split.map(|(_, address)| address), [2 * at, 2 * at + 1], "{from:?} at {at}");
        assert_eq!(split.map(|(word, _)| word), [0xfeed_face, 0xdead_beef], "{from:?} at {at}");
    }
}
/// `any::<u64>()` never samples these, so the property test cannot reach the carry and all-ones boundaries of the multiply word split.
#[test_case(u64::MAX, u64::MAX; "all ones")]
#[test_case(0x0000_0000_ffff_ffff, 0x0000_0000_0000_0002; "low word carry")]
fn long_arithmetic_word_split_matches_native_at_boundaries(a: u64, b: u64) {
    for from in [ScalarDType::Int64, ScalarDType::UInt64] {
        assert_long_arithmetic(a, b, from);
    }
}
fn long_bin(from: ScalarDType, op: BinaryOp, a: u64, b: u64) -> Arc<UOp> {
    UOp::new(Op::Binary(op, long_const(from, a), long_const(from, b)), DType::Scalar(from))
}
/// Split a 64-bit expression into words with `pm_long_decomp` and fold each word to its stored bits; a surviving 64-bit node means the split stalled.
fn split_words(from: ScalarDType, value: Arc<UOp>) -> [u32; 2] {
    let decomposed = graph_rewrite_bottom_up(&pm_long_decomp(), store(index(buffer_of(8, from), 1), value), &mut ());
    assert!(
        !decomposed
            .toposort()
            .iter()
            .any(|node| matches!(node.dtype().base(), ScalarDType::Int64 | ScalarDType::UInt64)),
        "a 64-bit node survived the split:\n{}",
        decomposed.tree()
    );
    let mut words = [None; 2];
    for node in decomposed.toposort() {
        let Op::Store(ops::Store { value, .. }) = node.op() else { continue };
        words[node.tag().as_ref().expect("split store is word-tagged")[1]] = Some(fold(value, &mut HashMap::new()));
    }
    let word = |value: Option<ConstValue>| match value.expect("both word stores must fold") {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        other => panic!("word is not an integer: {other:?}"),
    };
    [word(words[0]), word(words[1])]
}
/// Fold a decomposed expression. The support evaluator reinterprets a `Float` with its f64 bit pattern whatever the
/// node's width, so a `Float32 -> UInt32` bitcast folds to zero; this fold reads the pattern at the source dtype. The
/// memo keeps the shared div/mod DAG linear.
fn fold(expr: &Arc<UOp>, memo: &mut HashMap<u64, ConstValue>) -> ConstValue {
    if let Some(value) = memo.get(&expr.id) {
        return *value;
    }
    let dtype = expr.dtype().base();
    let value = match expr.op() {
        Op::Const(constant) => constant.0,
        Op::Cast(ops::Cast { src, .. }) => fold(src, memo).cast(&DType::Scalar(dtype)).expect("fold: cast"),
        Op::BitCast(ops::BitCast { src, .. }) => {
            let bits = match (src.dtype().base(), fold(src, memo)) {
                (ScalarDType::Float32, ConstValue::Float(value)) => (value as f32).to_bits() as u64,
                (ScalarDType::Float64, ConstValue::Float(value)) => value.to_bits(),
                (_, ConstValue::Int(value)) => value as u64,
                (_, ConstValue::UInt(value)) => value,
                (_, ConstValue::Bool(value)) => value as u64,
                (_, other) => return other,
            };
            reinterpret_bits(bits, dtype)
        }
        Op::Unary(op, src) => eval_unary_op_typed(*op, fold(src, memo), dtype).expect("fold: unary"),
        Op::Binary(op, lhs, rhs) => {
            eval_binary_op_typed(*op, fold(lhs, memo), fold(rhs, memo), dtype).expect("fold: binary")
        }
        Op::Ternary(op, a, b, c) => {
            eval_ternary_op_typed(*op, fold(a, memo), fold(b, memo), fold(c, memo), dtype).expect("fold: ternary")
        }
        other => panic!("fold: unsupported {other:?}"),
    };
    memo.insert(expr.id, value);
    value
}
fn reinterpret_bits(bits: u64, dtype: ScalarDType) -> ConstValue {
    match dtype {
        ScalarDType::Int8 => ConstValue::Int(bits as u8 as i8 as i64),
        ScalarDType::Int16 => ConstValue::Int(bits as u16 as i16 as i64),
        ScalarDType::Int32 => ConstValue::Int(bits as u32 as i32 as i64),
        ScalarDType::Int64 => ConstValue::Int(bits as i64),
        ScalarDType::UInt8 => ConstValue::UInt(bits as u8 as u64),
        ScalarDType::UInt16 => ConstValue::UInt(bits as u16 as u64),
        ScalarDType::UInt32 => ConstValue::UInt(bits as u32 as u64),
        ScalarDType::UInt64 => ConstValue::UInt(bits),
        ScalarDType::Float32 => ConstValue::Float(f32::from_bits(bits as u32) as f64),
        ScalarDType::Float64 => ConstValue::Float(f64::from_bits(bits)),
        _ => ConstValue::Invalid,
    }
}
/// Every word-wise 64-bit ALU op splits into two 32-bit words that reproduce the native result, signed and unsigned.
#[test_case(ScalarDType::Int64, BinaryOp::Add, 0x1_0000_0000, 0xffff_ffff; "add carries into the high word")]
#[test_case(ScalarDType::Int64, BinaryOp::Sub, 0x1_0000_0000, 1; "sub borrows from the high word")]
#[test_case(ScalarDType::Int64, BinaryOp::And, 0xff00_ff00_ff00_ff00, 0x0ff0_0ff0_0ff0_0ff0; "and")]
#[test_case(ScalarDType::Int64, BinaryOp::Or, 0xff00_ff00_ff00_ff00, 0x0ff0_0ff0_0ff0_0ff0; "or")]
#[test_case(ScalarDType::Int64, BinaryOp::Xor, 0xff00_ff00_ff00_ff00, 0x0ff0_0ff0_0ff0_0ff0; "xor")]
#[test_case(ScalarDType::Int64, BinaryOp::CDiv, 7, 2; "positive quotient")]
#[test_case(ScalarDType::Int64, BinaryOp::CDiv, -7i64 as u64, 2; "negative dividend truncates toward zero")]
#[test_case(ScalarDType::Int64, BinaryOp::CDiv, 7, -2i64 as u64; "negative divisor")]
#[test_case(ScalarDType::Int64, BinaryOp::CDiv, -7i64 as u64, -2i64 as u64; "both negative")]
#[test_case(ScalarDType::Int64, BinaryOp::CMod, 7, 2; "positive remainder")]
#[test_case(ScalarDType::Int64, BinaryOp::CMod, -7i64 as u64, 2; "remainder follows the dividend sign")]
#[test_case(ScalarDType::Int64, BinaryOp::CMod, 7, -2i64 as u64; "negative divisor remainder")]
#[test_case(ScalarDType::UInt64, BinaryOp::CDiv, (1 << 40) + 5, 3; "unsigned quotient wider than a word")]
#[test_case(ScalarDType::UInt64, BinaryOp::CMod, (1 << 40) + 5, 3; "unsigned remainder wider than a word")]
#[test_case(ScalarDType::UInt64, BinaryOp::CDiv, u64::MAX, 0x1_0000_0001; "all ones over a two-word divisor")]
fn long_word_ops_split_into_words(from: ScalarDType, op: BinaryOp, a: u64, b: u64) {
    let (signed_a, signed_b) = (a as i64, b as i64);
    let expected = match op {
        BinaryOp::Add => a.wrapping_add(b),
        BinaryOp::Sub => a.wrapping_sub(b),
        BinaryOp::And => a & b,
        BinaryOp::Or => a | b,
        BinaryOp::Xor => a ^ b,
        BinaryOp::CDiv if from == ScalarDType::Int64 => (signed_a / signed_b) as u64,
        BinaryOp::CMod if from == ScalarDType::Int64 => (signed_a % signed_b) as u64,
        BinaryOp::CDiv => a / b,
        BinaryOp::CMod => a % b,
        _ => unreachable!("table only names word-wise ops"),
    };
    assert_eq!(
        split_words(from, long_bin(from, op, a, b)),
        [expected as u32, (expected >> 32) as u32],
        "{from:?} {op:?}"
    );
}
/// AMD LLVM takes OCP FP8 storage, conversions and MFMA operands, but not ordinary FP8 ALU: every ALU/cast node is computed in f32 and cast back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp8Op {
    Unary,
    Binary,
    Compare,
    Where,
    MulAccLike,
    CastTo,
    CastFrom,
}
fn fp8_node(op: Fp8Op) -> Arc<UOp> {
    let fp8 = DType::Scalar(ScalarDType::FP8E4M3);
    let (lhs, rhs) =
        (UOp::const_(fp8.clone(), ConstValue::Float(1.0)), UOp::const_(fp8.clone(), ConstValue::Float(2.0)));
    match op {
        Fp8Op::Unary => UOp::new(Op::Unary(UnaryOp::Neg, lhs), fp8),
        Fp8Op::Binary => UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), fp8),
        Fp8Op::Compare => UOp::new(Op::Binary(BinaryOp::Lt, lhs, rhs), DType::Bool),
        Fp8Op::Where => UOp::try_where(UOp::native_const(true), lhs, rhs).unwrap(),
        Fp8Op::MulAccLike => UOp::new(Op::Ternary(TernaryOp::MulAcc, lhs.clone(), rhs, lhs), fp8),
        Fp8Op::CastTo => UOp::new(Op::Cast(ops::Cast { src: UOp::native_const(1i32), dtype: fp8.clone() }), fp8),
        Fp8Op::CastFrom => UOp::new(Op::Cast(ops::Cast { src: lhs, dtype: DType::Int32 }), DType::Int32),
    }
}
/// Every non-native FP8 ALU/cast node widens through f32. The comparison keeps its Bool result (only the operands widen)
/// and the casts keep their own result dtype around an inner f32 CAST; the expectation is derived from the row's op.
#[test_case(Fp8Op::Unary; "unary")]
#[test_case(Fp8Op::Binary; "binary")]
#[test_case(Fp8Op::Compare; "a comparison widens only its operands")]
#[test_case(Fp8Op::Where; "select")]
#[test_case(Fp8Op::MulAccLike; "other ternary")]
#[test_case(Fp8Op::CastTo; "cast to fp8 goes through f32")]
#[test_case(Fp8Op::CastFrom; "cast from fp8 reads f32")]
fn amd_widens_fp8_alu_through_float32(op: Fp8Op) {
    let fp8 = DType::Scalar(ScalarDType::FP8E4M3);
    let widened = rewrite(amd_non_native_fp8_patterns(), fp8_node(op));
    if op == Fp8Op::Compare {
        assert_eq!(widened.dtype(), DType::Bool);
        assert!(
            widened.op().sources().iter().all(|source| source.dtype().base() == ScalarDType::Float32),
            "{}",
            widened.tree()
        );
    } else if matches!(op, Fp8Op::CastTo | Fp8Op::CastFrom) {
        let Op::Cast(ops::Cast { src, dtype }) = widened.op() else { panic!("expected CAST: {}", widened.tree()) };
        let expected = if op == Fp8Op::CastTo { fp8 } else { DType::Int32 };
        assert_eq!(dtype, &expected);
        assert!(
            matches!(src.op(), Op::Cast(ops::Cast { dtype, .. }) if *dtype == DType::Float32),
            "{}",
            widened.tree()
        );
    } else {
        let Op::Cast(ops::Cast { src, dtype }) = widened.op() else {
            panic!("expected CAST(ALU<f32>):\n{}", widened.tree())
        };
        assert_eq!(dtype.base(), ScalarDType::FP8E4M3);
        assert_eq!(src.dtype(), DType::Float32);
        assert!(
            src.op().sources().iter().all(|source| source.dtype().base() != ScalarDType::FP8E4M3),
            "{}",
            widened.tree()
        );
    }
}
/// WMMA and memory nodes are opcodes the pattern never matches: FP8 storage and MFMA operands are native on AMD.
#[test]
fn amd_leaves_wmma_and_memory_nodes_alone() {
    let load = load(index(buffer_of(8, ScalarDType::FP8E4M3), 0));
    let wmma = UOp::wmma(load.clone(), load.clone(), load.clone(), wmma_metadata("fp8", None));
    for node in [load, wmma] {
        assert_same!(rewrite(amd_non_native_fp8_patterns(), node.clone()), node);
    }
}
proptest! {
    #![proptest_config(cheap())]
    /// `FP8 -> f32 -> FP8` is the identity on every finite normal E4M3 encoding: the upcast widens the storage word and the STORE narrows it back.
    #[test]
    fn fp8_round_trips_through_a_wider_float(bits in any::<u8>()) {
        let (exponent, mantissa) = ((bits >> 3) & 0x0f, bits & 0x07);
        prop_assume!(exponent != 0 && (exponent != 0x0f || mantissa != 0x07));
        prop_assert_eq!(round_trip(bits), ConstValue::UInt(bits as u64));
    }
}
/// `pm_float_decomp` flushes a zero exponent field, so every E4M3 zero/subnormal narrows back to a zero of the same sign rather than to its own encoding.
#[test_case(0x00, 0x00; "positive zero")]
#[test_case(0x80, 0x80; "negative zero")]
#[test_case(0x01, 0x00; "smallest positive subnormal")]
#[test_case(0x07, 0x00; "largest positive subnormal")]
#[test_case(0x87, 0x80; "largest negative subnormal")]
fn fp8_zero_exponents_flush_to_zero(bits: u8, expected: u8) {
    assert_eq!(round_trip(bits), ConstValue::UInt(expected as u64));
}
/// Upcast an E4M3 storage word to f32, store it back, and fold the narrowed word.
fn round_trip(bits: u8) -> ConstValue {
    let fp8 = DType::Scalar(ScalarDType::FP8E4M3);
    let value = UOp::new(
        Op::BitCast(ops::BitCast { src: UOp::const_(DType::UInt8, ConstValue::UInt(bits as u64)), dtype: fp8.clone() }),
        fp8,
    );
    let decomposed = decompose_to(
        ScalarDType::FP8E4M3,
        ScalarDType::Float32,
        store(index(buffer_of(8, ScalarDType::FP8E4M3), 0), value),
    );
    let stored = first_op(&decomposed, |op| matches!(op, Op::Store(..))).expect("decomposed STORE");
    let Op::Store(ops::Store { value, .. }) = stored.op() else { unreachable!() };
    fold(value, &mut HashMap::new())
}
