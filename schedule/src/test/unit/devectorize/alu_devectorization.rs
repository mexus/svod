//! `no_vectorized_alu` (tinygrad `devectorizer.py:219-223`): a vector ALU op becomes a STACK of scalar ones; a scalar
//! op is left alone.
use super::helpers::*;
use std::sync::Arc;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp, ops};
use test_case::test_case;
#[derive(Clone, Copy, Debug)]
enum Alu {
    Binary(BinaryOp),
    Unary(UnaryOp),
    Cast,
    Where,
    MulAcc,
}
fn lane(elem: ScalarDType, index: i64, base: i64) -> Arc<UOp> {
    match elem {
        ScalarDType::Float32 => UOp::native_const((base + index) as f32),
        _ => UOp::const_(DType::Int64, ConstValue::Int(base + index)),
    }
}
/// The scalar form of the expression when `width == 1`, the vector form otherwise.
fn build(alu: Alu, width: usize, elem: ScalarDType) -> Arc<UOp> {
    let vector = |dtype: DType| if width == 1 { dtype } else { dtype.vec(width).unwrap() };
    let dtype = vector(DType::Scalar(elem));
    let operand = |base: i64| {
        if width == 1 {
            lane(elem, 0, base)
        } else {
            UOp::stack((0..width as i64).map(|i| lane(elem, i, base)).collect())
        }
    };
    let (a, b) = (operand(0), operand(10));
    match alu {
        Alu::Binary(op) => UOp::new(Op::Binary(op, a, b), dtype),
        Alu::Unary(op) => UOp::new(Op::Unary(op, a), dtype),
        Alu::Cast => a.cast(vector(DType::Int64)),
        Alu::Where => {
            let condition =
                if width == 1 { UOp::native_const(true) } else { bool_values((0..width).map(|i| i % 2 == 0)) };
            UOp::new(Op::Ternary(TernaryOp::Where, condition, a, b), dtype)
        }
        Alu::MulAcc => UOp::try_mulacc(a.clone(), b, a).expect("MulAcc"),
    }
}
fn is_lane_of(alu: Alu, uop: &Arc<UOp>) -> bool {
    match (alu, uop.op()) {
        (Alu::Binary(op), Op::Binary(got, ..)) => *got == op,
        (Alu::Unary(op), Op::Unary(got, ..)) => *got == op,
        (Alu::Cast, Op::Cast(..))
        | (Alu::Where, Op::Ternary(TernaryOp::Where, ..))
        | (Alu::MulAcc, Op::Ternary(TernaryOp::MulAcc, ..)) => true,
        _ => false,
    }
}
#[test_case(Alu::Binary(BinaryOp::Add), 4, ScalarDType::Float32; "add")]
#[test_case(Alu::Binary(BinaryOp::Sub), 4, ScalarDType::Float32; "sub")]
#[test_case(Alu::Binary(BinaryOp::Mul), 8, ScalarDType::Float32; "mul vec8")]
#[test_case(Alu::Binary(BinaryOp::Add), 16, ScalarDType::Float32; "add vec16")]
#[test_case(Alu::Binary(BinaryOp::Add), 4, ScalarDType::Int64; "integer add")]
#[test_case(Alu::Binary(BinaryOp::And), 4, ScalarDType::Int64; "bitwise and")]
#[test_case(Alu::Unary(UnaryOp::Neg), 4, ScalarDType::Float32; "neg")]
#[test_case(Alu::Unary(UnaryOp::Sqrt), 4, ScalarDType::Float32; "sqrt")]
#[test_case(Alu::Unary(UnaryOp::Exp2), 4, ScalarDType::Float32; "exp2")]
#[test_case(Alu::Cast, 4, ScalarDType::Float32; "cast")]
#[test_case(Alu::Where, 4, ScalarDType::Float32; "where select")]
#[test_case(Alu::MulAcc, 4, ScalarDType::Float32; "mulacc")]
fn vector_alu_becomes_a_stack_of_scalar_lanes(alu: Alu, width: usize, elem: ScalarDType) {
    let result = apply_no_vectorized_alu(&build(alu, width, elem));
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected a STACK of lanes: {}", result.tree()) };
    assert_eq!(sources.len(), width);
    assert!(sources.iter().all(|lane| is_lane_of(alu, lane) && lane.dtype().vcount() == 1), "{}", result.tree());
}
#[test_case(Alu::Binary(BinaryOp::Add); "binary")]
#[test_case(Alu::Unary(UnaryOp::Sqrt); "unary")]
#[test_case(Alu::Cast; "cast")]
#[test_case(Alu::Where; "where select")]
#[test_case(Alu::MulAcc; "mulacc")]
fn scalar_alu_is_left_alone(alu: Alu) {
    let scalar = build(alu, 1, ScalarDType::Float32);
    let result = apply_no_vectorized_alu(&scalar);
    assert_same!(result, scalar);
    assert!(is_lane_of(alu, &result));
}
/// A broadcast scalar operand is scalarized along with the vector one.
#[test]
fn broadcast_operand_is_scalarized_with_its_vector_partner() {
    let add = UOp::new(
        Op::Binary(BinaryOp::Add, float_values((0..4).map(|i| i as f64)), UOp::native_const(10.0f32).broadcast(4)),
        DType::Float32.vec(4).unwrap(),
    );
    let result = apply_no_vectorized_alu(&add);
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected a STACK of lanes: {}", result.tree()) };
    assert_eq!(sources.len(), 4);
    assert!(sources.iter().all(|lane| matches!(lane.op(), Op::Binary(BinaryOp::Add, ..))));
}
/// A source narrower than the result has no lane to select, so the op stays vectorized.
#[test]
fn mismatched_source_extent_is_not_scalarized() {
    let add = UOp::new(
        Op::Binary(BinaryOp::Add, float_values((0..4).map(|i| i as f64)), UOp::native_const(10.0f32)),
        DType::Float32.vec(4).unwrap(),
    );
    assert!(matches!(apply_no_vectorized_alu(&add).op(), Op::Binary(BinaryOp::Add, ..)));
}
/// A shaped LOAD has no STACK to index into, so each lane reads it back through a scalar INDEX rather than being
/// duplicated.
#[test]
fn shaped_load_lanes_go_through_index_extraction() {
    let offsets = UOp::stack(smallvec::smallvec![UOp::index_const(0), UOp::index_const(1)]);
    let address = UOp::index().buffer(UOp::param(0, 8, DType::Float32, None)).indices(vec![offsets]).call().unwrap();
    let value = load(address);
    let result = apply_no_vectorized_alu(&value.add(&value));
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected shaped result to become STACK") };
    assert_eq!(sources.len(), 2);
    for source in sources {
        let Op::Binary(BinaryOp::Add, lhs, rhs) = source.op() else { panic!("expected scalar ADD") };
        assert!(matches!(lhs.op(), Op::Index(ops::Index { buffer, indices })
            if matches!(buffer.op(), Op::Load(..)) && indices.len() == 1 && matches!(indices[0].op(), Op::Const(_))));
        assert!(Arc::ptr_eq(lhs, rhs));
        assert_eq!(lhs.dtype(), DType::Float32);
    }
}
/// A STACK of Invalid markers is a shaped value like any other: each lane selects its own marker instead of the whole
/// STACK being treated as one Invalid.
#[test]
fn shaped_invalid_is_indexed_lane_by_lane() {
    let invalid = UOp::stack((0..4).map(|_| UOp::invalid_marker()).collect());
    let indices = UOp::stack((0..4).map(UOp::index_const).collect());
    let where_op = UOp::new(
        Op::Ternary(TernaryOp::Where, bool_values([true, false, true, false]), invalid, indices),
        DType::WeakInt,
    );
    let result = apply_no_vectorized_alu(&where_op);
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected scalarized WHERE lanes") };
    for element in sources {
        let Op::Ternary(TernaryOp::Where, _, invalid, _) = element.op() else { panic!("expected WHERE lane") };
        assert!(UOp::is_invalid_marker(invalid), "STACK indexing must select the corresponding Invalid lane");
    }
}
