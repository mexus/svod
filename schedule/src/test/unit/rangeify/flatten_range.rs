//! `flatten_range`: canonicalise the RANGE *expressions* an END closes over, and
//! keep reduction backedges out of the flattening.

use std::sync::Arc;

use svod_ir::{Op, UOp};
use test_case::test_case;

use crate::rangeify::transforms::{flatten_range_impl, flatten_ranges};
use crate::test::support::prelude::*;

fn nested_ends(depth: usize) -> Arc<UOp> {
    (0..depth).fold(UOp::native_const(1.0f32), |inner, i| {
        inner.end(smallvec::smallvec![global_range(10 * (i as i64 + 1), i)])
    })
}

/// Only the explicit ended-range sources are flattened — computation ENDs are
/// left nested, matching tinygrad. Returning `Some` for an unchanged END would
/// also spin the rewrite engine, so a single flat range yields `None` too.
#[test_case(|| nested_ends(1) ; "one end")]
#[test_case(|| nested_ends(2) ; "two nested ends")]
#[test_case(|| nested_ends(3) ; "three nested ends")]
#[test_case(|| UOp::native_const(1.0f32) ; "not an end at all")]
#[test_case(|| index(buffer(1), 0).store(UOp::native_const(1.0f32)) ; "store without ranges")]
fn nothing_to_canonicalize_returns_none(build: fn() -> Arc<UOp>) {
    let root = build();
    assert!(flatten_range_impl(&root).is_none());
    assert_same!(flatten_ranges(&root), root);
}

/// An END whose range source is an *expression* over ranges is rewritten to close
/// over the ranges themselves, keeping the computation untouched. A BOOL/VOID
/// source of an END is a reduction backedge, not a range: it must survive the
/// flattening and stay behind the iterations it guards.
#[test]
fn a_range_expression_is_split_and_a_boolean_backedge_survives() {
    let add = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let combined = global_range(10, 0).add(&global_range(20, 1));

    let flattened =
        flatten_range_impl(&add.clone().end(smallvec::smallvec![combined])).expect("the expression flattens");

    let (computation, ranges) = expect_end(&flattened);
    assert_same!(computation, add);
    assert_eq!(ranges.len(), 2);

    let backedge = bool_values([true, false]);
    let end = UOp::native_const(1.0f32).end(smallvec::smallvec![backedge.clone(), global_range(10, 0)]);

    let flattened = flatten_range_impl(&end).expect("the range source moves");

    let (_, ranges) = expect_end(&flattened);
    assert_eq!(ranges.len(), 2, "{}", flattened.tree());
    assert_eq!(expect_range_extent(&ranges[0]), 10, "the real range comes first");
    assert_same!(ranges[1], backedge);
    assert!(has_op(&flattened, |op| matches!(op, Op::Range(..))));
}
