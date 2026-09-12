//! Late decomposition patterns — tinygrad `decompositions.py:321-367`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, ConstValue as CV, Op, UOp, UnaryOp};

use test_case::test_case;

use crate::rangeify::patterns::{
    pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul,
};
use crate::symbolic::{pm_fold_cast_const, symbolic_simple};
use crate::test::support::prelude::*;

/// The full late table: the decomposition plus the folds the backend applies
/// around it, so a row pins the final form the renderer sees.
fn late(matcher: &'static crate::TypedPatternMatcher, root: Arc<UOp>) -> Arc<UOp> {
    rewrite(&(symbolic_simple() + pm_fold_cast_const() + matcher), root)
}

/// A dividend the vmin analysis proves non-negative.
fn x() -> Arc<UOp> {
    UOp::variable("x".into(), 0, 9999, DType::Int32)
}

/// A dividend whose declared range straddles zero, so no analysis can prove its
/// sign: the only operand that reaches the signed arm of `pm_div_to_shr`.
fn signed() -> Arc<UOp> {
    UOp::variable("s".into(), -9999, 9999, DType::Int32)
}

/// Assert `result` is `op(x, rhs)`.
fn assert_strength_reduced(result: &Arc<UOp>, op: BinaryOp, rhs: i64) {
    let Op::Binary(actual, lhs, actual_rhs) = result.op() else { panic!("expected {op:?}, got {}", result.tree()) };
    assert_eq!(*actual, op, "{}", result.tree());
    assert!(Arc::ptr_eq(lhs, &x()), "LHS must be the original operand");
    assert_const!(actual_rhs, rhs);
}

/// Power-of-two `%`, `*` and `//` become bit ops on the same operand.
#[test_case(pm_mod_to_and, |x, c| x.mod_(&c), 2, 1, BinaryOp::And ; "modulo 2")]
#[test_case(pm_mod_to_and, |x, c| x.mod_(&c), 8, 7, BinaryOp::And ; "modulo 8")]
#[test_case(pm_mod_to_and, |x, c| x.mod_(&c), 1024, 1023, BinaryOp::And ; "modulo 1024")]
#[test_case(pm_mul_to_shl, |x, c| x.mul(&c), 2, 1, BinaryOp::Shl ; "times 2")]
#[test_case(pm_mul_to_shl, |x, c| x.mul(&c), 8, 3, BinaryOp::Shl ; "times 8")]
#[test_case(pm_mul_to_shl, |x, c| x.mul(&c), 256, 8, BinaryOp::Shl ; "times 256")]
#[test_case(pm_div_to_shr, |x, c| x.cdiv(&c), 2, 1, BinaryOp::Shr ; "over 2")]
#[test_case(pm_div_to_shr, |x, c| x.cdiv(&c), 8, 3, BinaryOp::Shr ; "over 8")]
#[test_case(pm_div_to_shr, |x, c| x.cdiv(&c), 256, 8, BinaryOp::Shr ; "over 256")]
fn a_power_of_two_becomes_a_bit_op(
    matcher: fn() -> &'static crate::TypedPatternMatcher,
    build: fn(Arc<UOp>, Arc<UOp>) -> Arc<UOp>,
    divisor: i64,
    expected_rhs: i64,
    expected_op: BinaryOp,
) {
    assert_strength_reduced(&late(matcher(), build(x(), UOp::index_const(divisor))), expected_op, expected_rhs);
}

/// `x // 2^n` biases before shifting only when the dividend can be negative:
/// `(x + (x < 0).where(n - 1, 0)) >> n` corrects the round-towards-zero. The
/// non-negative row is the other half of the same choice — without it, deleting
/// the signed arm outright would still leave a green test.
#[test_case(x(), false ; "a non-negative dividend shifts directly")]
#[test_case(signed(), true ; "a signed dividend is biased first")]
fn only_a_signed_dividend_is_biased_before_the_right_shift(dividend: Arc<UOp>, biased: bool) {
    let int = |v: i64| UOp::const_(DType::Int32, CV::Int(v));
    let shifted = if biased {
        let negative = dividend.try_cmplt(&int(0)).expect("cmplt");
        let adjustment = UOp::try_where(negative, int(7), int(0)).expect("where");
        dividend.try_add(&adjustment).expect("add")
    } else {
        dividend.clone()
    };
    let expected = late(pm_div_to_shr(), shifted.try_shr_op(&int(3)).expect("shr"));

    assert_same!(late(pm_div_to_shr(), dividend.cdiv(&UOp::index_const(8))), expected);
}

/// Non-powers of two stay as they are, and the trivial identities fold to the
/// operand itself. `pm_neg_from_mul` is guarded on `-1` alone: any other scale
/// must survive as a MUL rather than turning into a NEG.
#[test_case(pm_mod_to_and, |c| x().mod_(&c), 7, Some(BinaryOp::FloorMod) ; "modulo 7")]
#[test_case(pm_mul_to_shl, |c| x().mul(&c), 7, Some(BinaryOp::Mul) ; "times 7")]
#[test_case(pm_div_to_shr, |c| x().cdiv(&c), 7, Some(BinaryOp::CDiv) ; "over 7")]
#[test_case(pm_div_to_shr, |c| x().cdiv(&c), 1, Some(BinaryOp::CDiv) ; "over 1 is not a zero shift")]
#[test_case(pm_mul_to_shl, |c| x().mul(&c), 1, None ; "multiply by one is the identity")]
#[test_case(pm_neg_from_mul, |c| x().mul(&c), 7, Some(BinaryOp::Mul) ; "times seven is not a negation")]
#[test_case(pm_neg_from_mul, |c| x().mul(&c), -2, Some(BinaryOp::Mul) ; "times minus two is not a negation")]
fn no_strength_reduction_without_a_power_of_two(
    matcher: fn() -> &'static crate::TypedPatternMatcher,
    build: fn(Arc<UOp>) -> Arc<UOp>,
    operand: i64,
    expected: Option<BinaryOp>,
) {
    let result = late(matcher(), build(UOp::index_const(operand)));
    match expected {
        Some(expected) => {
            let actual = assert_op!(result, Op::Binary(op, _, _) => op);
            assert_eq!(*actual, expected, "{}", result.tree());
        }
        None => assert_same!(result, x()),
    }
}

/// `x * -1 → NEG(x)`: the codegen-facing form of the canonical MUL, and
/// `x + (-y) → x - y`, the SUB the backend renders instead of an add of a NEG.
#[test]
fn negations_lower_to_the_codegen_ops() {
    let negated = late(pm_neg_from_mul(), x().mul(&UOp::index_const(-1)));
    let Op::Unary(UnaryOp::Neg, inner) = negated.op() else { panic!("expected NEG, got {}", negated.tree()) };
    assert_same!(inner, x());

    let y = TestVars::new().y;
    let subtracted = late(pm_neg_from_mul(), x().add(&y.neg()));
    let Op::Binary(BinaryOp::Sub, lhs, rhs) = subtracted.op() else {
        panic!("expected SUB, got {}", subtracted.tree())
    };
    assert_same!(lhs, x());
    assert_same!(rhs, y);
}

/// FDIV → MUL by reciprocal (decompositions.py:364-366).
#[test_case(2.0, 0.5 ; "half")]
#[test_case(4.0, 0.25 ; "quarter")]
#[test_case(5.0, 0.2 ; "fifth")]
#[test_case(0.5, 2.0 ; "reciprocal below one")]
fn dividing_by_a_float_constant_becomes_a_reciprocal_multiply(divisor: f32, reciprocal: f32) {
    let division = UOp::native_const(100.0f32).try_div(&UOp::native_const(divisor)).expect("div");

    let result = rewrite(pm_fdiv_to_mul(), division);

    let Op::Binary(BinaryOp::Mul, _, rhs) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    let Op::Const(c) = rhs.op() else { panic!("expected a constant reciprocal, got {}", result.tree()) };
    let ConstValue::Float(f) = c.0 else { panic!("expected a float reciprocal, got {:?}", c.0) };
    assert!((f - reciprocal as f64).abs() < 1e-6, "expected {reciprocal}, got {f}");
}

/// A zero divisor is a non-finite reciprocal: the rule must decline and leave the
/// FDIV so IEEE-754 semantics survive.
#[test]
fn a_zero_divisor_is_not_rewritten_to_a_reciprocal() {
    let division =
        UOp::new(Op::Binary(BinaryOp::Fdiv, UOp::native_const(1.0f32), UOp::native_const(0.0f32)), DType::Float32);

    assert_same!(rewrite(pm_fdiv_to_mul(), division.clone()), division);
}

// ===== comparison negations (decompositions.py:354-361) =====

/// Negating an integer `<` flips it to the complementary `<` with a shifted
/// bound; a two-sided band collapses to an equality.
#[test]
fn negated_integer_comparisons_become_the_complementary_bound() {
    let five = UOp::index_const(5);

    let not_lt = late(pm_comparison_negations(), x().try_cmplt(&five).expect("cmplt").not());
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = not_lt.op() else { panic!("expected LT, got {}", not_lt.tree()) };
    assert_const!(lhs, 4);
    assert_same!(rhs, x());

    let not_gt = late(pm_comparison_negations(), five.try_cmplt(&x()).expect("cmplt").not());
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = not_gt.op() else { panic!("expected LT, got {}", not_gt.tree()) };
    assert_same!(lhs, x());
    assert_const!(rhs, 6);

    let above = UOp::index_const(3).try_cmplt(&x()).expect("cmplt");
    let below = x().try_cmplt(&UOp::index_const(5)).expect("cmplt");
    let band = late(pm_comparison_negations(), above.try_and_op(&below).expect("and"));
    let Op::Binary(BinaryOp::Eq, lhs, rhs) = band.op() else { panic!("expected EQ, got {}", band.tree()) };
    let (var, konst) = if matches!(lhs.op(), Op::Const(_)) { (rhs, lhs) } else { (lhs, rhs) };
    assert_same!(var, x());
    assert_const!(konst, 4);
}

/// `x * -1 < c` moves the negation onto the bound: `-c < x`. Against another
/// scaled operand `y * c` it flips the sides and negates the scale instead, so
/// the comparison never has to materialise `-x`.
#[test]
fn negated_operands_move_the_bound_or_flip_the_comparison() {
    let lt = x().mul(&UOp::index_const(-1)).try_cmplt(&UOp::index_const(5)).expect("cmplt");

    let result = late(pm_comparison_negations(), lt);
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected LT, got {}", result.tree()) };
    assert_const!(lhs, -5);
    assert_same!(rhs, x());

    let vars = TestVars::new();
    let scaled = vars.a.mul(&UOp::index_const(-1)).try_cmplt(&vars.y.mul(&vars.y.const_like(4i64))).expect("cmplt");

    let result = late(pm_comparison_negations(), scaled);
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected LT, got {}", result.tree()) };
    let Op::Binary(BinaryOp::Mul, scaled, factor) = lhs.op() else {
        panic!("expected MUL on the left, got {}", result.tree())
    };
    assert_same!(scaled, vars.y);
    assert_const!(factor, -4);
    assert_same!(rhs, vars.a);
}

// ===== renderer-gated late rewrites =====

/// Fast integer division for a non-power-of-two divisor is opt-in; the
/// power-of-two shift is not gated.
#[test]
fn fast_integer_division_is_explicitly_opt_in() {
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let division = x().cdiv(&UOp::native_const(7i32));
    let modulo = x().cmod(&UOp::native_const(7i32));

    let disabled =
        crate::optimizer::apply_late_rewrites(UOp::sink(vec![division.clone(), modulo.clone()]), &renderer, true);
    assert!(has_op(&disabled, |op| matches!(op, Op::Binary(BinaryOp::CDiv, ..))), "{}", disabled.tree());
    assert!(has_op(&disabled, |op| matches!(op, Op::Binary(BinaryOp::CMod, ..))), "{}", disabled.tree());

    let enabled = crate::optimizer::apply_late_rewrites(UOp::sink(vec![division, modulo]), &renderer, false);
    assert!(
        !has_op(&enabled, |op| matches!(op, Op::Binary(BinaryOp::CDiv | BinaryOp::CMod, ..))),
        "{}",
        enabled.tree()
    );

    let power_of_two = crate::optimizer::apply_late_rewrites(x().cdiv(&UOp::native_const(8i32)), &renderer, true);
    assert!(matches!(power_of_two.op(), Op::Binary(BinaryOp::Shr, ..)), "{}", power_of_two.tree());
}

/// Weak lanes must be concretised before the final render, or the renderer mints
/// weak scalar constants it cannot type.
#[test]
fn weak_lowering_concretizes_a_weak_vconst_before_the_final_rewrite() {
    let lanes = UOp::vconst((0..4).map(CV::Int).collect(), DType::WeakInt);

    let lowered = rewrite_with(
        &crate::symbolic::pm_lower_index_dtype(),
        &mut crate::symbolic::WeakMemo::default(),
        UOp::sink(vec![lanes]),
    );
    let result = rewrite(crate::optimizer::final_rewrite_patterns(), lowered);

    assert!(result.toposort().iter().all(|u| !u.dtype().is_weak()), "{}", result.tree());
    let source = expect_sink(&result)[0].clone();
    assert!(matches!(source.op(), Op::VConst(svod_ir::ops::VConst { values }) if values.len() == 4));
    assert_eq!(source.dtype(), DType::Int32.vec(4).expect("vector dtype"));
}
