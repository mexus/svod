//! Property tests for the symbolic optimizer's algebraic rules. The tables are replayed
//! at every integer and float dtype: a rule that only fires at Int32 stops firing for the
//! narrow and unsigned families the schedulers also emit.

use std::sync::Arc;

use proptest::prelude::*;

use crate::symbolic::symbolic;
use svod_dtype::{DType, ScalarDType};
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::{Op, UOp};

use crate::test::property::checks::{folded_binary, same_value, same_value_over};
use crate::test::property::generators::{arb_int_property_dtype, arb_op_tree_up_to, arb_property_dtype, build_binary};
use crate::test::support::prelude::*;

use svod_ir::test::property::generators::*;

/// `(op, identity, also_folds_on_the_left)`: `x op identity == x`.
const IDENTITIES: &[(BinaryOp, i64, bool)] = &[
    (BinaryOp::Add, 0, true),
    (BinaryOp::Sub, 0, false),
    (BinaryOp::Mul, 1, true),
    (BinaryOp::FloorDiv, 1, false),
    (BinaryOp::Or, 0, false),
    (BinaryOp::Xor, 0, false),
];

/// The identities that also hold for floats, which are signed-zero aware: `x + 0.0`
/// is *not* the identity under IEEE 754 (`-0.0 + 0.0 == 0.0`), `x + (-0.0)` is;
/// subtraction needs the positive zero, and float division is `FDIV`, which carries
/// the same `x / 1 -> x` rule. `OR`/`XOR` are integer-only.
const FLOAT_IDENTITIES: &[(BinaryOp, f64)] =
    &[(BinaryOp::Add, -0.0), (BinaryOp::Sub, 0.0), (BinaryOp::Mul, 1.0), (BinaryOp::FloorDiv, 1.0)];

/// `(op, absorbing element)`: `x op absorbing == absorbing`, either way round.
const ABSORBING: &[(BinaryOp, i32)] = &[(BinaryOp::Mul, 0), (BinaryOp::And, 0)];

/// `x op unit` must fold back to the already-canonical `canonical`.
fn assert_identity(
    op: BinaryOp,
    unit: &Arc<UOp>,
    folds_on_the_left: bool,
    canonical: &Arc<UOp>,
) -> Result<(), TestCaseError> {
    let binary = |lhs: Arc<UOp>, rhs: Arc<UOp>| build_binary(op, lhs, rhs).expect("the identity is in the op's domain");
    let mut orders = vec![binary(canonical.clone(), unit.clone())];
    if folds_on_the_left {
        orders.push(binary(unit.clone(), canonical.clone()));
    }
    for expr in orders {
        let folded = rewrite(Matchers::simple(), expr.clone());
        prop_assert!(
            Arc::ptr_eq(&folded, canonical),
            "the identity {} must fold {op:?} away\n{}",
            unit.tree(),
            expr.tree()
        );
    }
    Ok(())
}

/// The whole identity table over `canonical`, at the dtype family it was drawn in.
fn identities_hold(canonical: Arc<UOp>, float: bool) -> Result<(), TestCaseError> {
    let canonical = rewrite(Matchers::simple(), canonical);
    if float {
        for &(op, identity) in FLOAT_IDENTITIES {
            assert_identity(op, &canonical.f(identity), false, &canonical)?;
        }
    } else {
        for &(op, identity, folds_on_the_left) in IDENTITIES {
            assert_identity(op, &canonical.c(identity), folds_on_the_left, &canonical)?;
        }
    }
    Ok(())
}

proptest! {
    // Several per-dtype properties were merged into the dtype-parameterised
    // `identities_fold_at_every_dtype`, so the block budget is scaled by the dtypes it
    // covers; the Int32 depth lives in its own block below.
    #![proptest_config(ProptestConfig::with_cases(3 * CHEAP))]

    /// The whole identity table on an arbitrary operand tree, not on the two leaf
    /// shapes `arb_simple_uop` draws, and at every integer and float dtype: the
    /// identities may not depend on the storage width, and the float family uses the
    /// signed-zero-aware table.
    ///
    /// This is a *breadth* check. Spread over eleven dtypes it leaves each of them a
    /// few dozen cases, so the depth at the dtype the tables were written at lives in
    /// `identities_fold_at_int32` below.
    #[test]
    fn identities_fold_at_every_dtype(case in arb_property_dtype().prop_flat_map(|dtype| (Just(dtype.clone()), arb_op_tree_up_to(dtype, 2)))) {
        let (dtype, tree) = case;
        identities_hold(tree, dtype.is_float())?;
    }

    /// `x op absorbing == absorbing`. The value is pinned first, then the node: the
    /// only live zero is the one the table built, so identity follows from interning.
    #[test]
    fn absorbing_operand_swallows_the_expression(x in arb_op_tree_up_to(DType::Int32, 2)) {
        let canonical = rewrite(Matchers::simple(), x);
        for &(op, value) in ABSORBING {
            let absorbing = canonical.c(value as i64);
            for (side, expr) in [
                ("right", build_binary(op, canonical.clone(), absorbing.clone())),
                ("left", build_binary(op, absorbing.clone(), canonical.clone())),
            ] {
                let expr = expr.expect("zero is in every op's domain");
                let simplified = rewrite(Matchers::simple(), expr.clone());
                prop_assert_eq!(simplified.dtype(), canonical.dtype(), "the absorbing element keeps the dtype");
                assert_const!(simplified, 0);
                same_value(&expr, &simplified)?;
                prop_assert!(Arc::ptr_eq(&simplified, &absorbing), "{op:?} on the {side} must return the absorbing node");
            }
        }
    }

    /// Integer self comparisons are decided without looking at the value.
    #[test]
    fn self_comparison_folds_to_a_constant(x in arb_op_tree_up_to(DType::Int32, 2)) {
        let canonical = rewrite(Matchers::simple(), x);
        for (expr, expected) in [
            (canonical.try_cmplt(&canonical).unwrap(), false),
            (canonical.try_cmpne(&canonical).unwrap(), false),
            (canonical.try_cmpeq(&canonical).unwrap(), true),
        ] {
            let folded = rewrite(symbolic(), expr);
            assert_const!(folded, expected);
        }
    }

    /// `x / x` folds to 1 only when the declared range excludes zero.
    #[test]
    fn self_division_folds_only_away_from_zero(min in 0i64..20, span in 0i64..20) {
        let x = UOp::var("x", DType::Int32, min, min + span);
        let simplified = rewrite(Matchers::simple(), x.try_div(&x).unwrap());
        if min > 0 {
            assert_const!(simplified, 1);
        } else {
            prop_assert!(matches!(simplified.op(), Op::Binary(BinaryOp::FloorDiv, ..)), "x / x must survive when x may be zero, got\n{}", simplified.tree());
        }
    }

    /// A binary op over two constants folds to the evaluator's value; the operands
    /// are `i16` so the check needs no overflow guard.
    #[test]
    fn constant_operands_fold_to_the_evaluated_value(
        a in any::<i16>(),
        b in any::<i16>(),
        divisor in any::<i16>().prop_filter("nonzero", |value| *value != 0),
    ) {
        let konst = |value: i64| UOp::const_(DType::Int32, ConstValue::Int(value));
        for (op, rhs) in [(BinaryOp::Add, b), (BinaryOp::Mul, b), (BinaryOp::FloorDiv, divisor)] {
            let expression = build_binary(op, konst(a as i64), konst(rhs as i64)).unwrap();
            let expected = eval_typed(&expression, &Bindings::none()).expect("constants evaluate");
            let folded = rewrite(Matchers::simple(), expression);
            assert_const!(folded, expected);
        }
    }

    /// `(a // b) // c` collapses to `a // (b * c)`. `a`'s range always reaches
    /// `b * c`, so the generator never has to skip a case.
    #[test]
    fn nested_div_collapse(a_max in 64i64..=100, b in 2..8i32, c in 2..8i32) {
        let a = UOp::var("a", DType::Int32, 0, a_max);
        let div = a.try_div(&UOp::native_const(b)).unwrap().try_div(&UOp::native_const(c)).unwrap();
        let (var, divisor) = folded_binary(symbolic(), div, BinaryOp::FloorDiv)?;
        prop_assert!(Arc::ptr_eq(&var, &a));
        assert_const!(divisor, (b as i64) * (c as i64));
    }

    /// `(a * b) * c` collapses to `a * (b * c)`.
    #[test]
    fn nested_mul_collapse(a in arb_var_uop(DType::Int32), b in 2..20i32, c in 2..20i32) {
        let mul = a.try_mul(&UOp::native_const(b)).unwrap().try_mul(&UOp::native_const(c)).unwrap();
        let (var, factor) = folded_binary(symbolic(), mul, BinaryOp::Mul)?;
        prop_assert!(Arc::ptr_eq(&var, &a));
        assert_const!(factor, (b as i64) * (c as i64));
    }

    /// `(a % b) % b` collapses to `a % b`. `a`'s range always reaches `b`, otherwise
    /// range analysis folds the modulo to `a` before the idempotence rule can fire.
    #[test]
    fn mod_idempotence(b in 2..100i32, span in 0..100i64) {
        let a = UOp::var("a", DType::Int32, 0, (b as i64) + span);
        let divisor = UOp::native_const(b);
        let nested = a.try_mod(&divisor).unwrap().try_mod(&divisor).unwrap();
        let (var, actual) = folded_binary(Matchers::simple(), nested, BinaryOp::FloorMod)?;
        prop_assert!(Arc::ptr_eq(&var, &a) && Arc::ptr_eq(&actual, &divisor));
    }

    /// `(a + b) + c` collapses to `a + (b + c)`, or to `a` alone when the sum cancels.
    #[test]
    fn nested_add_collapse(a in arb_var_uop(DType::Int32), b in -100..100i32, c in -100..100i32) {
        let add = a.try_add(&UOp::native_const(b)).unwrap().try_add(&UOp::native_const(c)).unwrap();
        let simplified = rewrite(symbolic(), add);
        let sum = (b as i64) + (c as i64);
        match simplified.op() {
            Op::Binary(BinaryOp::Add, var, addend) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                assert_const!(addend, sum);
            }
            Op::Binary(BinaryOp::Sub, var, subtrahend) => {
                prop_assert!(Arc::ptr_eq(var, &a));
                assert_const!(subtrahend, -sum);
            }
            Op::DefineVar(..) => {
                prop_assert!(Arc::ptr_eq(&simplified, &a));
                prop_assert_eq!(sum, 0, "a bare variable may only come out when the constants cancel");
            }
            other => prop_assert!(false, "expected Add, Sub or the bare variable, got {other:?}"),
        }
    }

    /// `(a - b) - c` collapses to tinygrad's subtraction form, `a + -(b + c)`.
    #[test]
    fn nested_sub_collapse(a in arb_var_uop(DType::Int32), b in 1..100i32, c in 1..100i32) {
        let sub = a.try_sub(&UOp::native_const(b)).unwrap().try_sub(&UOp::native_const(c)).unwrap();
        let (var, addend) = folded_binary(symbolic(), sub, BinaryOp::Add)?;
        prop_assert!(Arc::ptr_eq(&var, &a));
        assert_const!(addend, -((b as i64) + (c as i64)));
    }

    /// `(a * b) // b` cancels back to `a`.
    #[test]
    fn mul_div_inverse(a in arb_var_uop(DType::Int32), b in 1..100i32) {
        let b = UOp::native_const(b);
        let simplified = rewrite(Matchers::simple(), a.try_mul(&b).unwrap().try_div(&b).unwrap());
        prop_assert!(Arc::ptr_eq(&simplified, &a), "got {}", simplified.tree());
    }
}

proptest! {
    // The depth HEAD ran the identity table at. The dtype sweep above spreads its budget
    // over eleven dtypes weighted half to floats, which leaves Int32 — the dtype the rules
    // were written at and the one every scheduler emits most — a few dozen cases.
    #![proptest_config(ProptestConfig::with_cases(4 * CHEAP))]

    /// The identity table over deep Int32 operand trees.
    #[test]
    fn identities_fold_at_int32(tree in arb_op_tree_up_to(DType::Int32, 2)) {
        identities_hold(tree, false)?;
    }
}

/// The divmod box is `(a_max + 1) * (b_max + 1)` points, at most `8 * 8`. The sweep has to
/// cover it whole — `range_points` enumerates every point only once its cap reaches the
/// product — because a soundness claim that silently skipped part of the declared range is
/// not a soundness claim.
const DIVMOD_BOX: usize = 64;

/// `factor_a * a + factor_b * b + offset` over `a ∈ [0, a_max]`, `b ∈ [0, b_max]`:
/// the affine inputs the divmod rules are written for.
fn divmod_expression(factor_a: i64, factor_b: i64, offset: i64, a_max: i64, b_max: i64) -> Arc<UOp> {
    let a = UOp::variable("a".into(), 0, a_max, DType::Int32);
    let b = UOp::variable("b".into(), 0, b_max, DType::Int32);
    let mut expr = UOp::index_const(offset);
    for (factor, operand) in [(factor_a, &a), (factor_b, &b)] {
        if factor != 0 {
            expr = expr.try_add(&UOp::index_const(factor).try_mul(operand).unwrap()).unwrap();
        }
    }
    expr
}

/// `(affine expression, divisor)`: the generator domain both soundness properties draw from.
fn arb_divmod_case() -> impl Strategy<Value = (Arc<UOp>, Arc<UOp>)> {
    (-20i64..20, -20i64..20, -20i64..20, 2i64..16, 1i64..8, 1i64..8).prop_map(
        |(factor_a, factor_b, offset, divisor, a_max, b_max)| {
            (divmod_expression(factor_a, factor_b, offset, a_max, b_max), UOp::index_const(divisor))
        },
    )
}

/// The simple tier's rewrite of `original` must agree with it at every point of the box.
fn assert_divmod_sound(original: Arc<UOp>) -> Result<(), TestCaseError> {
    let rewritten = rewrite(Matchers::simple(), original.clone());
    same_value_over(&original, &rewritten, DIVMOD_BOX)
}

proptest! {
    // The modulo and the division rewrite are two separate properties over one verbatim
    // generator domain, not one property with a boolean: merged behind `divide in
    // any::<bool>()` each branch would see only half the budget.
    #![proptest_config(ProptestConfig::with_cases(4 * EQUIVALENCE))]

    /// The affine modulo rewrite must preserve the value at every point of the box.
    #[test]
    fn divmod_mod_rewrites_are_sound(case in arb_divmod_case()) {
        let (expr, divisor) = case;
        assert_divmod_sound(expr.try_mod(&divisor).expect("MOD accepts an index divisor"))?;
    }

    /// The affine division rewrite must preserve the value at every point of the box.
    #[test]
    fn divmod_idiv_rewrites_are_sound(case in arb_divmod_case()) {
        let (expr, divisor) = case;
        assert_divmod_sound(expr.try_div(&divisor).expect("DIV accepts an index divisor"))?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2 * EQUIVALENCE))]

    /// The affine congruence rules must fire and must not change the value at the
    /// narrow dtype they were derived at.
    #[test]
    fn affine_congruence_rewrites_preserve_exact_typed_runtime(divisor in 8i64..=16, offset in 0i64..=2, max in 2i64..=5) {
        let dtype = DType::Int16;
        let x = UOp::var("affine_x", dtype.clone(), 0, max);
        let divisor_uop = UOp::const_(dtype.clone(), ConstValue::Int(divisor));
        let constant = |value: i64| UOp::const_(dtype.clone(), ConstValue::Int(value));
        let numerator = x.mul(&constant(divisor + 1)).add(&constant(offset));
        for original in [numerator.mod_(&divisor_uop), numerator.floor_div(&divisor_uop)] {
            let rewritten = rewrite(symbolic(), original.clone());
            prop_assert!(!Arc::ptr_eq(&original, &rewritten), "congruence rule did not fire for {}", original.tree());
            for value in 0..=max {
                let bindings = Bindings::at("affine_x", value);
                prop_assert_eq!(
                    eval_typed(&original, &bindings),
                    eval_typed(&rewritten, &bindings),
                    "typed affine rewrite mismatch for original {} and replacement {}",
                    original.tree(),
                    rewritten.tree(),
                );
            }
        }
    }

    /// Divmod rewrites must reproduce the wrapping runtime result at *every* integer
    /// dtype, sampled over the whole declared range rather than one point.
    ///
    /// The dtype comes from the integer generator: half of `arb_property_dtype`'s draws
    /// are floats, and every one of them leaves through `integer_bounds`' `None` without
    /// checking anything.
    #[test]
    fn divmod_rewrites_preserve_wrapping_at_every_integer_dtype(
        dtype in arb_int_property_dtype(),
        factor in 1i64..=8,
        offset in 0i64..=8,
        window in arb_dtype_window(),
        raw in any::<i64>(),
    ) {
        // `UInt64` and the width-less integer dtypes have no declarable window.
        let Some((low, high)) = window.of(&dtype) else { return Ok(()) };
        let x = UOp::var("typed_x", dtype.clone(), low, high);
        let divisor = UOp::const_(dtype, ConstValue::Int(factor));
        let expression = x.mul(&divisor).add(&UOp::const_(x.dtype(), ConstValue::Int(offset)));
        let point = Bindings::at("typed_x", raw.clamp(low, high));
        for original in [expression.floor_div(&divisor), expression.mod_(&divisor)] {
            let rewritten = rewrite(symbolic(), original.clone());
            prop_assert_eq!(fold_at(&original, &point), fold_at(&rewritten, &point), "wrapping mismatch at the sampled point for\n{}", original.tree());
            same_value(&original, &rewritten)?;
        }
    }
}

/// Where inside a dtype's own range a declared window sits.
///
/// The window has to stay bounded: `range_points` computes `hi - lo + 1` in `i64`, which
/// overflows on a full `Int64` range, and a rewrite is only promised *within* the range it
/// was shown. Anchoring it at the dtype's minimum or maximum rather than around zero is
/// what lets `Int8` reach -128 and `UInt8` reach 255 — the values where a wrapping rule
/// actually breaks, and which a window clamped to ±64 never sees.
#[derive(Debug, Clone, Copy)]
struct DTypeWindow {
    anchor: usize,
    half: i64,
}

impl DTypeWindow {
    fn of(self, dtype: &DType) -> Option<(i64, i64)> {
        let (min, max) = integer_bounds(dtype)?;
        let span = self.half.saturating_mul(2);
        Some(match self.anchor {
            0 => (min, min.saturating_add(span).min(max)),
            1 => (max.saturating_sub(span).max(min), max),
            _ => (min.max(-self.half), max.min(self.half)),
        })
    }
}

fn arb_dtype_window() -> impl Strategy<Value = DTypeWindow> {
    (0usize..3, 1i64..=64).prop_map(|(anchor, half)| DTypeWindow { anchor, half })
}

/// The inclusive value range of an integer scalar dtype, or `None` where the window
/// would overflow `i64` (`UInt64`) or the dtype is not a plain integer.
fn integer_bounds(dtype: &DType) -> Option<(i64, i64)> {
    let scalar = dtype.scalar()?;
    let unbounded =
        matches!(&scalar, ScalarDType::UInt64 | ScalarDType::Index | ScalarDType::WeakInt | ScalarDType::Void);
    (scalar.is_int() && !unbounded).then(|| (scalar.min_value() as i64, scalar.max_value() as i64))
}
