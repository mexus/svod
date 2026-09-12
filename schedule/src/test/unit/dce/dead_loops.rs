//! Dead loop elimination: a `Range` folds to `Const(0)` when it is provably empty and to
//! its single value when it provably has one iteration.
//!
//! END/REDUCE empty-range folds are deliberately absent: they conflated the trivial
//! `Range(end=1)` value with dead-range markers; `reduce_to_acc` handles those instead.

use smallvec::smallvec;
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::{ConstValue, ReduceOp};
use svod_ir::{AxisId, AxisType, Op, UOp};
use test_case::test_case;

use crate::symbolic::dce::is_empty_range;
use crate::test::support::prelude::*;

/// A `Range` over an unsigned extent, whose `vmax` is a `ConstValue::UInt`. It is the only
/// shape that reaches `is_empty_range`'s `Range` arm without an `Int` bound, so it is what
/// pins the "an unsigned bound is never negative" reasoning the arm rests on; an unsigned
/// *constant* would fall through to `_ => false` and say nothing about ranges at all.
fn unsigned_range(end: u64) -> Arc<UOp> {
    let extent = UOp::const_(DType::UInt32, ConstValue::UInt(end));
    UOp::range_axis_dtype(extent, AxisId::Renumbered(0), AxisType::Global, DType::UInt32)
}

/// `is_empty_range` fires only for a provably empty `Range`, only for an integer
/// `vmax < 0`, and also for the weak `Const(0)` a dead range rewrites to.
#[test_case(|| global_range(0, 0), true ; "a zero trip count")]
#[test_case(|| global_range(1, 0), false ; "one iteration is not empty")]
#[test_case(|| global_range(2, 0), false ; "two iterations are not empty")]
#[test_case(|| UOp::const_(DType::WeakInt, ConstValue::Int(0)), true ; "the weak zero dead-range marker")]
#[test_case(|| UOp::const_(DType::WeakInt, ConstValue::Int(1)), false ; "a nonzero weak constant")]
#[test_case(|| UOp::native_const(0i32), false ; "a strong zero constant is not a range marker")]
#[test_case(|| unsigned_range(1), false ; "an unsigned range's bound can never be negative")]
#[test_case(|| unsigned_range(4), false ; "a wider unsigned range is not empty either")]
#[test_case(|| UOp::native_const(0.0f32), false ; "a float is not a range")]
fn is_empty_range_fires_only_on_provably_empty_ranges(candidate: fn() -> Arc<UOp>, empty: bool) {
    assert_eq!(is_empty_range(&candidate()), empty, "{}", candidate().tree());
}

/// `Some((value, dtype))` when the range must fold, `None` when it must survive. The dtype
/// matters: the empty arm emits a *weak* `UOp::index_const(0)` while the trivial arm keeps
/// the range's own dtype, so a `Const(0, Int32)` regression would slip past a value-only check.
#[test_case(|| global_range(0, 0), Some((ConstValue::Int(0), DType::WeakInt)) ; "a dead range folds to a weak zero")]
#[test_case(|| global_range(-5, 0), Some((ConstValue::Int(0), DType::WeakInt)) ; "a negative trip count is dead")]
#[test_case(|| range_symbolic(UOp::native_const(-10i32).max(&UOp::native_const(0i32)), 0), Some((ConstValue::Int(0), DType::WeakInt)) ; "an end clamped to zero is dead")]
#[test_case(|| range_symbolic(UOp::variable("size".into(), 0, 5, DType::Int32).try_sub(&UOp::native_const(10i32)).unwrap(), 0), Some((ConstValue::Int(0), DType::WeakInt)) ; "a symbolically empty end is dead")]
#[test_case(|| global_range(1, 0), Some((ConstValue::Int(0), DType::Index)) ; "one iteration folds to its only value")]
#[test_case(|| range(1, AxisType::Unroll, 0), Some((ConstValue::Int(0), DType::WeakInt)) ; "a trivial unroll axis folds to its value in its own dtype")]
#[test_case(|| global_range(2, 0), None ; "two iterations must not fold")]
fn dead_loop_ranges_fold_or_survive(range: fn() -> Arc<UOp>, expected: Option<(ConstValue, DType)>) {
    let range = range();
    let folded = rewrite(Matchers::dce(), range.clone());
    match expected {
        Some((value, dtype)) => {
            assert_const_value(&folded, value);
            assert_eq!(folded.dtype(), dtype, "the fold must keep the range dtype\n{}", folded.tree());
        }
        None => assert_same!(folded, range),
    }
}

/// `END` over an empty range list is the identity on its computation
/// (`ir/src/uop/constructors/control.rs`): there is no loop to close, so no END node is
/// built at all. The short-circuit is what keeps a fully folded loop nest from leaving a
/// `Void`-typed END behind, and nothing else in the repo asserts it.
#[test]
fn end_without_ranges_returns_the_computation() {
    let computation = UOp::noop();
    assert_same!(Arc::clone(&computation).end(smallvec![]), computation);
}

/// A trivial `Range(Unroll, end=1)` rewrites to the weak `Const(0)` marker that a re-added
/// END/REDUCE empty-range fold would consume, collapsing the reduction to its identity.
#[test]
fn a_trivial_reduce_axis_does_not_erase_the_reduction() {
    let reduce = reduce(UOp::native_const(1.0f32), vec![range(1, AxisType::Unroll, 0)], ReduceOp::Add);
    let folded = rewrite(Matchers::dce(), reduce);
    assert!(has_op(&folded, |op| matches!(op, Op::Reduce(..))), "the reduction must survive\n{}", folded.tree());
}
