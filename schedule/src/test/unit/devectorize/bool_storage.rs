//! `bool_storage_patterns`: bool LOAD/STORE go through uint8 storage so LLVM never sees an `i1` with garbage high
//! bits (tinygrad's PTX/NIR bool rules).
use super::helpers::*;
use std::sync::Arc;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{Op, SInt, UOp, ops};
use test_case::test_case;
/// A bool LOAD becomes `CAST(LOAD<uint8>, bool)`; every other element type is left alone.
#[test_case(ScalarDType::Bool; "bool loads through uint8")]
#[test_case(ScalarDType::Float32; "float32 untouched")]
#[test_case(ScalarDType::Int32; "int32 untouched")]
fn load_uses_uint8_storage_only_for_bool(scalar: ScalarDType) {
    let result = apply_bool_storage(&load(index(buffer_of(64, scalar), 0)));
    if scalar != ScalarDType::Bool {
        assert_op!(result, Op::Load(..));
        assert_eq!(result.dtype(), DType::Scalar(scalar));
        return;
    }
    let Op::Cast(ops::Cast { src, dtype }) = result.op() else { panic!("expected CAST(LOAD), got {}", result.tree()) };
    assert_eq!(*dtype, DType::Bool);
    assert_op!(src, Op::Load(..));
    assert_eq!(src.dtype(), DType::UInt8);
}
/// The vector path widens the whole shaped base, not just scalar lanes.
#[test]
fn shaped_bool_load_widens_its_vector_base() {
    let lanes = DType::Bool.vec(4).unwrap();
    let index = shaped_index(buffer_of(64, ScalarDType::Bool), 0..4).with_dtype(lanes.clone());
    let result = apply_bool_storage(&UOp::load().index(index).dtype(lanes.clone()).call());
    let Op::Cast(ops::Cast { src, dtype }) = result.op() else { panic!("expected CAST(LOAD), got {}", result.tree()) };
    assert_eq!(*dtype, lanes);
    assert_eq!(src.dtype(), DType::UInt8.vec(4).unwrap());
    assert_eq!(src.shape().unwrap().unwrap().as_slice(), &[SInt::Const(4)]);
}
/// A bool STORE casts its value to uint8 first; other element types keep theirs, and an Invalid value has no bool
/// storage form yet.
#[test_case(UOp::native_const(true), ScalarDType::Bool, Some(ScalarDType::UInt8); "bool stores as uint8")]
#[test_case(bool_values([true, false, true, false]), ScalarDType::Bool, Some(ScalarDType::UInt8); "shaped bool stores as uint8")]
#[test_case(UOp::native_const(3.0f32), ScalarDType::Float32, Some(ScalarDType::Float32); "float32 untouched")]
#[test_case(UOp::invalid_marker(), ScalarDType::Bool, None; "invalid store waits for the final decomposition")]
fn store_uses_uint8_storage_only_for_bool(value: Arc<UOp>, buffer: ScalarDType, expected: Option<ScalarDType>) {
    let original = store(index(buffer_of(64, buffer), 0), value);
    let result = apply_bool_storage(&original);
    let Some(expected) = expected else {
        assert_same!(result, original);
        return;
    };
    let Op::Store(ops::Store { value, .. }) = result.op() else { panic!("expected STORE, got {}", result.tree()) };
    assert_eq!(value.dtype().base(), expected, "{}", result.tree());
}
/// No backend renders a bool bitcast: both directions become a CAST.
#[test_case(UOp::var("p", DType::Bool, 0, 1), DType::UInt8; "bool source")]
#[test_case(UOp::var("b", DType::UInt8, 0, 1), DType::Bool; "bool destination")]
fn bitcast_through_bool_becomes_cast(src: Arc<UOp>, dtype: DType) {
    let bitcast = UOp::new(Op::BitCast(ops::BitCast { src, dtype: dtype.clone() }), dtype.clone());
    let result = apply_bool_storage(&bitcast);
    assert!(matches!(result.op(), Op::Cast(ops::Cast { dtype: got, .. }) if got == &dtype), "{}", result.tree());
    assert!(!has_op(&result, |op| matches!(op, Op::BitCast(..))), "{}", result.tree());
}
/// The gate and its alt survive the storage rewrite, with the alt widened to uint8.
#[test]
fn gated_bool_load_keeps_gate_and_converts_alt() {
    let index = index(buffer_of(64, ScalarDType::Bool), 0);
    let load = UOp::load().index(index).alt(UOp::native_const(true)).gate(UOp::native_const(false)).call();
    let result = apply_bool_storage(&load);
    let Op::Cast(ops::Cast { src, .. }) = result.op() else { panic!("expected CAST(LOAD), got {}", result.tree()) };
    let Op::Load(ops::Load { alt: Some(alt), gate: Some(_), .. }) = src.op() else {
        panic!("the late LOAD gate and alt must both survive: {}", src.tree())
    };
    assert_eq!(alt.dtype(), DType::UInt8);
}
/// The full pass reaches the same bool storage form.
#[test]
fn devectorize_lowers_bool_loads() {
    let result = apply_devectorize(&load(index(buffer_of(64, ScalarDType::Bool), 0)));
    assert!(matches!(result.op(), Op::Cast(ops::Cast { src, .. }) if src.dtype() == DType::UInt8), "{}", result.tree());
    assert_eq!(result.dtype(), DType::Bool);
}
