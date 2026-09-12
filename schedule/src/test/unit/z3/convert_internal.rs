use super::*;
use test_case::test_case;

use crate::test::support::prelude::*;
use crate::test::unit::z3::helpers::{assert_valid, assert_valid_bool, assert_valid_in};

/// `op(lhs, rhs)` exactly as written. The `UOp` constructors normalise: `try_sub` is
/// `add(rhs.neg())` and `neg` is `mul(x, -1)` for a non-bool, so neither the `Sub` nor the
/// `Neg` arm of the converter is reachable through them.
fn raw_binary(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), dtype)
}

/// `op(src)` exactly as written; see [`raw_binary`].
fn raw_unary(op: UnaryOp, src: &Arc<UOp>) -> Arc<UOp> {
    let dtype = src.dtype();
    UOp::new(Op::Unary(op, src.clone()), dtype)
}

/// Floor division expressed through Z3's own Euclidean `div`/`mod`, so the expected value
/// is derived independently of the converter's truncate-then-correct formulation. Euclidean
/// division already floors for a positive divisor; for a negative one it rounds up, so the
/// quotient steps down by one exactly when the division is inexact.
fn z3_floor_div(a: &Int, b: &Int) -> Int {
    let (quotient, remainder) = (a.div(b), a.modulo(b));
    let rounds_up = Bool::and(&[b.lt(Int::from_i64(0)), remainder.eq(Int::from_i64(0)).not()]);
    rounds_up.ite(&(quotient.clone() - 1), &quotient)
}

/// `a - b * floor(a / b)`: the remainder that carries the *divisor's* sign.
fn z3_floor_mod(a: &Int, b: &Int) -> Int {
    a - z3_floor_div(a, b) * b
}

/// The single Z3 integer the graph denotes, converted in `context`.
fn int_of(context: &mut Z3Context, expr: &Arc<UOp>) -> Int {
    context.convert_uop(expr).expect("conversion should succeed").as_int().expect("an integer expression")
}

/// Refute `expr != expected` in `context`, which carries the variable bounds that
/// `convert_uop` asserted; `Sat` on the equality alone would accept a fresh variable.
fn assert_converts_to(context: &mut Z3Context, expr: &Arc<UOp>, expected: &Int) {
    let converted = int_of(context, expr);
    assert_valid_in(context.solver(), &converted, expected);
}

#[test_case(UOp::const_(DType::Int32, ConstValue::Int(42)), 42 ; "a signed constant")]
#[test_case(UOp::native_const(7u32), 7 ; "an unsigned constant")]
#[test_case(UOp::const_(DType::Int8, ConstValue::Int(-5)), -5 ; "a narrow signed constant")]
fn convert_const_pins_its_value(constant: Arc<UOp>, expected: i64) {
    assert_valid(&int_of(&mut Z3Context::new(), &constant), &Int::from_i64(expected));
}

#[test]
fn convert_bool_const_pins_its_value() {
    let converted = Z3Context::new().convert_uop(&UOp::native_const(true)).expect("conversion should succeed");
    assert_valid_bool(&converted.as_bool().expect("a boolean expression"), &Bool::from_bool(true));
}

/// The declared bounds are asserted: `outside` is refuted and `inside` stays satisfiable.
fn assert_bounds(expr: &Arc<UOp>, outside: i64, inside: i64) {
    for (value, expected) in [(outside, z3::SatResult::Unsat), (inside, z3::SatResult::Sat)] {
        let mut context = Z3Context::new();
        let converted = int_of(&mut context, expr);
        let solver = context.solver();
        solver.assert(converted.eq(Int::from_i64(value)));
        assert_eq!(solver.check(), expected, "value {value} of {}", expr.tree());
    }
}

#[test]
fn convert_asserts_operand_bounds() {
    assert_bounds(&UOp::var("x", DType::Int32, 0, 100), 101, 100);
    assert_bounds(&range_symbolic(UOp::native_const(8i32), 0), 8, 7);
}

/// Each op must land on the Z3 operation it claims, expressed here with atoms
/// built directly on the Z3 side so a converter that returned a fresh variable
/// for every node cannot pass.
#[test_case(|v| v.x.add(&v.c(3)), |x| x + Int::from_i64(3) ; "an addition keeps both operands")]
#[test_case(|v| v.x.sub(&v.c(3)), |x| x - Int::from_i64(3) ; "a subtraction negates the right operand")]
#[test_case(|v| raw_binary(BinaryOp::Sub, &v.x, &v.c(3)), |x| x - Int::from_i64(3) ; "an authored SUB node subtracts")]
#[test_case(|v| v.x.mul(&v.c(3)), |x| x * Int::from_i64(3) ; "a multiplication keeps both operands")]
#[test_case(|v| v.x.neg(), |x| -x ; "negation flips the sign")]
#[test_case(|v| raw_unary(UnaryOp::Neg, &v.x), |x| -x ; "an authored NEG node flips the sign")]
#[test_case(|v| v.x.max(&v.c(3)), |x| x.gt(Int::from_i64(3)).ite(&x, &Int::from_i64(3)) ; "max selects the larger operand")]
#[test_case(|v| v.x.cdiv(&v.c(3)), |x| z3_cdiv(&x, &Int::from_i64(3)) ; "CDiv is truncated division")]
#[test_case(|v| v.x.cmod(&v.c(3)), |x| z3_cmod(&x, &Int::from_i64(3)) ; "CMod follows CDiv")]
#[test_case(|v| v.x.try_div(&v.c(3)).unwrap(), |x| &x / &Int::from_i64(3) ; "FloorDiv is euclidean on a non-negative range")]
#[test_case(|v| v.x.mod_(&v.c(3)), |x| z3_cmod(&x, &Int::from_i64(3)) ; "FloorMod matches CMod on a non-negative range")]
fn convert_integer_operations_to_their_z3_form(expr: Term, expected: fn(Int) -> Int) {
    let vars = TestVars::new();
    let mut context = Z3Context::new();
    assert_converts_to(&mut context, &expr(&vars), &expected(Int::new_const("x")));
}

/// Svod's `FloorDiv`/`FloorMod` round toward negative infinity; C truncates toward zero, so
/// the converter emits `z3_cdiv` plus a correction term. Every row above is written over the
/// non-negative `x`, where the two agree and that correction is identically `false` —
/// replacing it with `Bool::from_bool(false)` passes them all. These rows run over `a`, which
/// straddles zero, against an expected value derived from Z3's Euclidean `div`/`mod` instead.
#[test_case(|v| v.a.try_div(&v.n).unwrap(), z3_floor_div, z3_cdiv ; "a negative dividend rounds down")]
#[test_case(|v| v.a.mod_(&v.n), z3_floor_mod, z3_cmod ; "the remainder keeps the divisor's sign")]
#[test_case(|v| raw_binary(BinaryOp::FloorDiv, &v.a, &v.c(-3)), z3_floor_div, z3_cdiv ; "a negative divisor rounds down")]
#[test_case(|v| raw_binary(BinaryOp::FloorMod, &v.a, &v.c(-3)), z3_floor_mod, z3_cmod ; "a negative divisor gives a non-positive remainder")]
fn convert_floor_division_applies_the_floor_correction(
    expr: Term,
    flooring: fn(&Int, &Int) -> Int,
    truncating: fn(&Int, &Int) -> Int,
) {
    let vars = TestVars::new();
    let expr = expr(&vars);
    let (lhs, rhs) = match expr.op() {
        Op::Binary(_, lhs, rhs) => (int_of(&mut Z3Context::new(), lhs), int_of(&mut Z3Context::new(), rhs)),
        other => panic!("expected a division, got {other:?}"),
    };

    let mut context = Z3Context::new();
    assert_converts_to(&mut context, &expr, &flooring(&lhs, &rhs));

    // And the row has to discriminate: over *these* operands the flooring and the
    // truncating form must be different functions, or asserting the first proves nothing
    // about whether the converter applied its correction at all.
    let solver = context.solver();
    solver.assert(flooring(&lhs, &rhs).eq(truncating(&lhs, &rhs)).not());
    assert_eq!(solver.check(), z3::SatResult::Sat, "floor and truncate agree everywhere on {}", expr.tree());
}

#[test_case(|v| (v.x.lt(&v.c(50)), Int::new_const("x").lt(Int::from_i64(50))) ; "less-than")]
#[test_case(|v| (v.x.try_cmpeq(&v.c(50)).unwrap(), Int::new_const("x").eq(Int::from_i64(50))) ; "equality")]
#[test_case(|v| (v.x.try_cmpne(&v.c(50)).unwrap(), Int::new_const("x").eq(Int::from_i64(50)).not()) ; "inequality")]
#[test_case(|v| (v.x.lt(&v.c(50)).and_(&v.y.lt(&v.c(20))), Bool::and(&[Int::new_const("x").lt(Int::from_i64(50)), Int::new_const("y").lt(Int::from_i64(20))])) ; "a conjunction")]
#[test_case(|v| (v.x.lt(&v.c(50)).or_(&v.y.lt(&v.c(20))), Bool::or(&[Int::new_const("x").lt(Int::from_i64(50)), Int::new_const("y").lt(Int::from_i64(20))])) ; "a disjunction")]
fn convert_boolean_operations_to_their_z3_form(case: fn(&TestVars) -> (Arc<UOp>, Bool)) {
    let vars = TestVars::new();
    let (expr, expected) = case(&vars);
    let converted = Z3Context::new().convert_uop(&expr).expect("conversion should succeed");
    assert_valid_bool(&converted.as_bool().expect("a boolean expression"), &expected);
}

#[test]
fn convert_where_and_mulacc_to_their_z3_form() {
    let vars = TestVars::new();
    let mut context = Z3Context::new();
    let (condition, x, y) = (Int::new_const("x").lt(Int::from_i64(50)), Int::new_const("x"), Int::new_const("y"));
    let where_ = UOp::try_where(vars.x.lt(&vars.c(50)), vars.x.clone(), vars.y.clone()).unwrap();
    assert_converts_to(&mut context, &where_, &condition.ite(&x, &y));
    let mulacc = UOp::try_mulacc(vars.x.clone(), vars.y.clone(), vars.c(1)).unwrap();
    assert_converts_to(&mut context, &mulacc, &(x * y + Int::from_i64(1)));
}

/// The cast arm's subtlety: a widening cast is bound to its source, but a narrowing
/// cast must not be, or the solver goes globally UNSAT.
#[test]
fn convert_cast_binds_the_result_only_when_the_source_range_fits() {
    let mut context = Z3Context::new();
    let widened = int_of(&mut context, &UOp::var("wide", DType::Int8, 0, 100).cast(DType::Int32));
    assert_valid_in(context.solver(), &widened, &Int::new_const("wide"));
    let mut context = Z3Context::new();
    let narrowed = int_of(&mut context, &UOp::var("narrow", DType::Int32, 0, 1000).cast(DType::Int8));
    let solver = context.solver();
    solver.assert(narrowed.eq(Int::new_const("narrow")).not());
    assert_eq!(solver.check(), z3::SatResult::Sat, "a narrowing cast keeps a fresh bounded variable");
}

/// Every rejection has to name its own variant, so a regression that widens one arm into
/// another (e.g. `UnsupportedOp` for a load) is visible.
#[test_case(|_| UOp::native_const(1.0f32), |e| matches!(e, ConversionError::UnsupportedType { .. }) ; "a float constant has no encoding")]
#[test_case(|v| v.bounded.try_exp2().unwrap(), |e| matches!(e, ConversionError::UnsupportedUnaryOp { .. }) ; "an unhandled unary op")]
#[test_case(|v| v.x.try_xor_op(&v.y).unwrap(), |e| matches!(e, ConversionError::UnsupportedBinaryOp { .. }) ; "an unhandled binary op")]
#[test_case(|v| v.x.and_(&v.y), |e| matches!(e, ConversionError::UnsupportedOperation { .. }) ; "bitwise AND between integers")]
#[test_case(|v| UOp::try_where(v.p.clone(), v.p.clone(), v.q.clone()).unwrap(), |e| matches!(e, ConversionError::UnsupportedOperation { .. }) ; "a WHERE over bool branches")]
#[test_case(|v| v.x.lt(&v.c(0)).not(), |e| matches!(e, ConversionError::TypeMismatch { .. }) ; "a boolean not has no integer source")]
#[test_case(|v| v.p.and_(&v.q), |e| matches!(e, ConversionError::UnsupportedOperation { .. }) ; "boolean variables convert to integers, so a conjunction is rejected")]
#[test_case(|v| unboundable(v.x.dtype()), |e| matches!(e, ConversionError::UnsupportedOp { .. }) ; "a load has no encoding")]
fn unconvertible_graphs_report_their_variant(case: Term, expected: fn(&ConversionError) -> bool) {
    let error = Z3Context::new().convert_uop(&case(&TestVars::new())).expect_err("the graph must not convert");
    assert!(expected(&error), "unexpected variant for {error:?}");
}
