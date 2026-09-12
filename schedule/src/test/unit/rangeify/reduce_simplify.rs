//! `reduce_unparented` (drop ranges the source never reads, factor constants out of a MUL chain) and `reduce_collapse` (lift a range-independent body out).

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{BinaryOp, ConstValue, Op, ReduceOp, UOp, ops};
use test_case::test_case;

use super::helpers::reduce_range;
use crate::rangeify::indexing::range_size_as_i64;
use crate::rangeify::patterns::pm_reduce_simplify;
use crate::rangeify::transforms::reduce_collapse;
use crate::test::support::prelude::{Bindings, fold_at};

fn collapse(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce(ops::Reduce { src, ranges, .. }) = reduce.op() else { return None };
    reduce_collapse(src, ranges)
}

fn simplify(reduce: &Arc<UOp>) -> Arc<UOp> {
    simplified(reduce).expect("the pattern must fire")
}

/// The simplified node, or `None` when the pass declines.
fn simplified(reduce: &Arc<UOp>) -> Option<Arc<UOp>> {
    match pm_reduce_simplify().rewrite(reduce, &mut ()) {
        svod_ir::RewriteResult::Rewritten(out) => Some(out),
        _ => None,
    }
}

fn has_op_kind(uop: &Arc<UOp>, pred: impl Fn(&Op) -> bool) -> bool {
    uop.toposort().iter().any(|n| pred(n.op()))
}

fn has_reduce(uop: &Arc<UOp>) -> bool {
    has_op_kind(uop, |op| matches!(op, Op::Reduce(..) | Op::ReduceAxis(..)))
}

fn has_range(uop: &Arc<UOp>) -> bool {
    has_op_kind(uop, |op| matches!(op, Op::Range(..)))
}

// ===== reduce_unparented =====

/// A range the source never reads is folded into arithmetic on the source: ADD scales it, MUL raises it to the extent, MAX is idempotent. Tinygrad 8c8b43de handles exactly these three; MIN is deliberately absent.
#[test_case(ReduceOp::Add, BinaryOp::Mul ; "add becomes a multiply by the extent")]
#[test_case(ReduceOp::Mul, BinaryOp::Pow ; "mul becomes a power of the extent")]
fn an_unparented_range_scales_the_source(op: ReduceOp, expected: BinaryOp) {
    let reduce = UOp::native_const(5i32).reduce(smallvec![reduce_range(10, 0)], op);
    let result = simplify(&reduce);
    assert!(matches!(result.op(), Op::Binary(actual, _, _) if *actual == expected), "{}", result.tree());
    // The scale is the range extent, reached through the cast the pass applies.
    let Op::Binary(_, _, scale) = result.op() else { unreachable!() };
    assert_eq!(extent_of(scale), 10, "the scale must be the extent, got {}", scale.tree());
}

/// The integer a constant-or-cast-of-constant node holds.
fn extent_of(uop: &Arc<UOp>) -> i64 {
    match uop.op() {
        Op::Const(value) => value.0.try_int().expect("integer extent"),
        Op::Cast(ops::Cast { src, .. }) => extent_of(src),
        other => panic!("expected a constant extent, got {other:?}\n{}", uop.tree()),
    }
}

#[test]
fn max_is_idempotent_over_an_unparented_range() {
    let src = UOp::native_const(5i32);
    let reduce = src.clone().reduce(smallvec![reduce_range(10, 0)], ReduceOp::Max);
    assert!(Arc::ptr_eq(&simplify(&reduce), &src));
}

#[test]
fn min_is_not_an_unparented_fold() {
    assert!(simplified(&UOp::native_const(42i32).reduce(smallvec![reduce_range(5, 0)], ReduceOp::Min)).is_none());
}

#[test]
fn a_range_the_source_reads_is_not_unparented() {
    let range = reduce_range(10, 0);
    assert!(simplified(&Arc::clone(&range).reduce(smallvec![range], ReduceOp::Add)).is_none());
}

/// Two unparented ranges fold one at a time into a nested product: `5 * 3 * 4`.
#[test]
fn every_unparented_range_folds_into_its_own_factor() {
    let reduce = UOp::native_const(5i32).reduce(smallvec![reduce_range(3, 0), reduce_range(4, 1)], ReduceOp::Add);
    let result = simplify(&reduce);
    let Op::Binary(BinaryOp::Mul, inner, outer) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    assert!(matches!(inner.op(), Op::Binary(BinaryOp::Mul, _, _)), "{}", result.tree());
    assert_eq!(fold_at(inner, &Bindings::none()), Some(ConstValue::Int(15)), "5 * 3");
    assert_eq!(fold_at(outer, &Bindings::none()), Some(ConstValue::Int(4)), "the second extent");
    assert_eq!(fold_at(&result, &Bindings::none()), Some(ConstValue::Int(60)));
}

/// A mix keeps the parented range inside the REDUCE and scales by the other.
#[test]
fn a_parented_range_stays_inside_the_reduce() {
    let (parented, unparented) = (reduce_range(5, 0), reduce_range(10, 1));
    let src = UOp::native_const(3i32).try_add(&parented.cast(DType::Int32)).expect("add");
    let reduce = src.reduce(smallvec![parented.clone(), unparented], ReduceOp::Add);
    let result = simplify(&reduce);
    let Op::Binary(BinaryOp::Mul, inner, _) = result.op() else { panic!("expected MUL, got {}", result.tree()) };
    let Op::Reduce(ops::Reduce { ranges, .. }) = inner.op() else {
        panic!("expected an inner REDUCE, got {}", result.tree())
    };
    assert_eq!(ranges.as_slice().len(), 1);
    assert!(Arc::ptr_eq(&ranges[0], &parented));
}

/// Constant factors in a MUL chain lift out of ADD unconditionally, and out of MAX only when non-negative (a negative factor inverts the ordering).
#[test_case(ReduceOp::Add, 3, true ; "add with a positive factor")]
#[test_case(ReduceOp::Add, -1, true ; "add with a negative factor")]
#[test_case(ReduceOp::Max, 3, true ; "max with a positive factor")]
#[test_case(ReduceOp::Max, -1, false ; "max with a negative factor")]
fn a_constant_factor_lifts_out_of_the_reduce(op: ReduceOp, factor: i64, lifts: bool) {
    let range = reduce_range(10, 0);
    let src = range.cast(DType::Int32).mul(&UOp::native_const(factor as i32));
    let reduce = src.reduce(smallvec![range], op);
    let lifted = match pm_reduce_simplify().rewrite(&reduce, &mut ()) {
        svod_ir::RewriteResult::Rewritten(result) => {
            matches!(result.op(), Op::Binary(BinaryOp::Mul, _, f)
                if matches!(f.op(), Op::Const(c) if c.0 == ConstValue::Int(factor)))
        }
        _ => false,
    };
    assert_eq!(lifted, lifts);
}

/// `reduce_mul_chain` refuses floats outright: reassociating a float product changes rounding, so the factor must stay
/// inside. ADD reduces are claimed by the collapse rule first, so this drives the MAX path, where the chain rule is the
/// only candidate.
#[test]
fn a_float_product_does_not_lift() {
    let range = reduce_range(10, 0);
    let src = range.cast(DType::Float32).mul(&UOp::native_const(2.5f32));
    let reduce = src.reduce(smallvec![range], ReduceOp::Max);
    assert!(simplified(&reduce).is_none(), "the chain rule must decline a float source: {}", reduce.tree());
}

/// A symbolic MAX factor has no sound `vmin`, so it cannot be proven non-negative and must stay inside the reduce. The chain rule then has nothing left to hoist, which is why the rewrite declines outright.
#[test]
fn a_symbolic_max_factor_without_bounds_stays_inside() {
    let range = reduce_range(10, 0);
    let unbounded = UOp::var("u", DType::Int32, i64::MIN, i64::MAX);
    let src = range.cast(DType::Int32).mul(&unbounded);
    let reduce = src.reduce(smallvec![range], ReduceOp::Max);
    assert!(
        matches!(pm_reduce_simplify().rewrite(&reduce, &mut ()), svod_ir::RewriteResult::NoMatch),
        "no factor may be hoisted out of a MAX with an unbounded multiplier"
    );
}

/// Several constants in one chain lift together into a single factor.
#[test]
fn several_constant_factors_lift_into_one_product() {
    let range = reduce_range(10, 0);
    let range_int = range.cast(DType::Int32);
    let src = UOp::native_const(2i32).mul(&range_int).mul(&UOp::native_const(5i32));
    let reduce = src.reduce(smallvec![range], ReduceOp::Add);
    let result = simplify(&reduce);
    // Every factor on the outer MUL chain was lifted out; their product is 2 * 5.
    let mut factors = Vec::new();
    let mut node = result.clone();
    while let Op::Binary(BinaryOp::Mul, left, right) = node.op() {
        factors.push(fold_at(right, &Bindings::none()).expect("a hoisted constant"));
        node = left.clone();
    }
    assert!(has_reduce(&node), "the range stays inside: {}", result.tree());
    factors.sort_by_key(|value| value.try_int().expect("integer factor"));
    assert_eq!(factors, vec![ConstValue::Int(2), ConstValue::Int(5)], "both constants lifted");
}

// ===== reduce_collapse =====

/// A body that does not read the range collapses: neither the RANGE nor the REDUCE survives, and the dtype is unchanged.
#[test_case(ReduceOp::Add ; "add")]
#[test_case(ReduceOp::Mul ; "mul")]
#[test_case(ReduceOp::Max ; "max")]
#[test_case(ReduceOp::Min ; "min")]
fn a_range_independent_body_collapses(op: ReduceOp) {
    let src = UOp::native_const(2.5f64);
    let reduce = src.clone().reduce(smallvec![reduce_range(100, 0)], op);
    let result = collapse(&reduce).expect("a range-independent body must collapse");
    assert!(!has_range(&result));
    assert!(!has_reduce(&result));
    assert_eq!(result.dtype(), src.dtype());
}

#[test]
fn independent_ranges_all_collapse_together() {
    let reduce = UOp::native_const(5i32).reduce(smallvec![reduce_range(10, 0), reduce_range(20, 1)], ReduceOp::Add);
    let result = collapse(&reduce).expect("both ranges must collapse");
    assert!(!has_range(&result));
}

/// Symbolic simplification runs first, so a body that only *looks* like it reads the range collapses once the algebra cancels.
#[test_case(ReduceOp::Add, 0i32 ; "x plus zero")]
#[test_case(ReduceOp::Mul, 1i32 ; "x times one")]
fn algebra_runs_before_the_collapse(op: ReduceOp, identity: i32) {
    let x = UOp::native_const(42i32);
    let (binary, src) = match op {
        ReduceOp::Add => (BinaryOp::Add, x.try_add(&UOp::native_const(identity))),
        _ => (BinaryOp::Mul, x.try_mul(&UOp::native_const(identity))),
    };
    let reduce = src.expect("identity op").reduce(smallvec![reduce_range(10, 0)], op);
    let result = collapse(&reduce).expect("the identity must cancel and let the reduce collapse");
    assert!(!has_range(&result));
    assert!(!has_reduce(&result));
    assert!(
        !has_op_kind(&result, |op| matches!(op, Op::Binary(actual, _, _) if *actual == binary)),
        "the identity operand must be gone: {}",
        result.tree()
    );
}

/// Neither a body that reads the range nor a REDUCE with no ranges is a collapse candidate.
#[test_case(true ; "the body reads the range")]
#[test_case(false ; "there are no ranges")]
fn a_reduce_that_is_not_range_independent_does_not_collapse(reads_range: bool) {
    let range = reduce_range(10, 0);
    let reduce = if reads_range {
        range.cast(DType::Int32).try_add(&UOp::native_const(1i32)).expect("add").reduce(smallvec![range], ReduceOp::Add)
    } else {
        UOp::native_const(5i32).reduce(smallvec![], ReduceOp::Add)
    };
    assert!(collapse(&reduce).is_none());
}

/// The arange fold: `sum(r in [0, 32) of (r + v < 31 ? 0 : 1))` must collapse to bound arithmetic with no RANGE left.
/// ADD is commutative and morok's canonical ordering puts the substituted scalar first whenever it sorts below the
/// RANGE, so the Lt lift has to fire for either operand order — tinygrad's UPat matches commutative sources in both
/// positions (`codegen/simplify.py:101`).
#[test_case(true ; "range on the left")]
#[test_case(false ; "range on the right")]
fn reduce_collapse_lifts_a_commutative_add_in_either_order(range_first: bool) {
    let range = reduce_range(32, 0);
    let scalar = UOp::var("in0", range.dtype(), 0, 31);
    let sum = if range_first { range.try_add(&scalar) } else { scalar.try_add(&range) }
        .expect("range and scalar share a dtype");
    let bound = UOp::const_(sum.dtype(), ConstValue::Int(31));
    let gate = sum.try_cmplt(&bound).expect("comparison against a same-dtype bound");
    let body = UOp::try_where(gate, UOp::native_const(0i32), UOp::native_const(1i32)).expect("both branches Int32");
    let reduce = body.reduce(smallvec![range], ReduceOp::Add);
    let result = collapse(&reduce).expect("the arange fold must collapse this reduce");
    assert!(!has_range(&result), "reduce_collapse left a RANGE behind: {}", result.tree());
    assert!(!has_reduce(&result), "reduce_collapse left a REDUCE behind: {}", result.tree());
}

/// A gate whose index is in the cone but outside the range set still leaves the range in the result, so the collapse declines rather than dropping the axis.
#[test]
fn a_gate_reading_a_foreign_range_does_not_collapse() {
    let collapsed_range = reduce_range(8, 0);
    let foreign = reduce_range(8, 1);
    let body =
        UOp::try_where(foreign.cast(DType::Bool), UOp::native_const(1i32), UOp::native_const(0i32)).expect("gate");
    let reduce = body.reduce(smallvec![collapsed_range], ReduceOp::Add);
    assert!(collapse(&reduce).is_none(), "a foreign RANGE keeps the reduce alive");
}

// ===== range_size_as_i64 =====

/// Only a RANGE with a constant extent has a size; `no_range` truth table rows live in `range_load_guards.rs`.
#[test]
fn only_a_constant_range_reports_a_size() {
    assert_eq!(range_size_as_i64(&UOp::range_const(100, 0)), Some(100));
    assert_eq!(range_size_as_i64(&reduce_range(42, 1)), Some(42));
    let symbolic = UOp::range(UOp::define_var("N".to_string(), 0, 1000), 0);
    assert_eq!(range_size_as_i64(&symbolic), None);
    assert_eq!(range_size_as_i64(&UOp::native_const(100i32)), None);
}
