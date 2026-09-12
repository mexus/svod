use super::*;
use std::sync::Arc;
use svod_dtype::DType;
use test_case::test_case;

use crate::test::support::prelude::*;

/// `value` inside a loop that closes `ranges`.
fn loop_(value: Arc<UOp>, ranges: impl IntoIterator<Item = Arc<UOp>>) -> Arc<UOp> {
    value.end(ranges.into_iter().collect())
}

/// A loop body that carries no computation, so only the closed range matters.
fn counted(range: &Arc<UOp>) -> Arc<UOp> {
    loop_(UOp::native_const(1.0f32), [range.clone()])
}

/// Ranges closed by the same END are not chained: predecessor edges only come from loops
/// that actually follow one another, so the first loop of a group has no predecessor.
#[test_case(1; "one range")]
#[test_case(2; "two ranges closed together")]
fn ranges_closed_by_one_end_have_no_predecessor_edges(count: usize) {
    let ranges: Vec<_> = (0..count).map(|axis| global_range(10, axis)).collect();
    let ctx = CFGContext::new(&UOp::sink(vec![loop_(UOp::native_const(1.0f32), ranges.iter().cloned())]));
    assert!(!ctx.has_edges());
    assert_eq!(ctx.edge_count(), 0);
    assert!(ranges.iter().all(|range| ctx.get_predecessor(range).is_none()));
}

/// The positive path: the second sibling RANGE must wait for the first sibling's END.
#[test]
fn a_second_sibling_range_waits_for_the_first_end() {
    let (first, second) = (global_range(10, 0), global_range(10, 1));
    let first_end = counted(&first);
    let ctx = CFGContext::new(&UOp::sink(vec![first_end.clone(), counted(&second)]));
    assert_eq!(ctx.edge_count(), 1);
    assert!(ctx.has_edges());
    assert_same!(Arc::clone(ctx.get_predecessor(&second).expect("the second sibling must be ordered")), first_end);
    assert!(ctx.get_predecessor(&first).is_none(), "the first sibling starts the group");
}

/// Siblings are ordered by how many of the other siblings they depend on, not by toposort
/// order: `dependent` closes `base` yet must still follow the independent sibling.
#[test]
fn sibling_edges_follow_dependency_count_not_toposort_order() {
    let (base_range, dependent_range, independent_range) =
        (global_range(10, 0), global_range(10, 1), global_range(10, 2));
    let base = counted(&base_range);
    let dependent = loop_(base.clone(), [dependent_range.clone()]);
    let independent = counted(&independent_range);
    let ctx = CFGContext::new(&UOp::sink(vec![base.clone(), dependent, independent.clone()]));
    assert_eq!(ctx.edge_count(), 2);
    assert_same!(Arc::clone(ctx.get_predecessor(&independent_range).expect("independent follows base")), base);
    let after_independent = ctx.get_predecessor(&dependent_range).expect("dependent must follow independent");
    assert_same!(Arc::clone(after_independent), independent);
    assert!(ctx.get_predecessor(&base_range).is_none());
}

/// An END nested inside another END is not a sibling of its parent: its RANGE waits for the
/// parent's RANGE, and the parent keeps no predecessor.
#[test]
fn an_inner_range_waits_for_its_parent_range() {
    let (inner_range, outer_range) = (global_range(10, 0), global_range(10, 1));
    let body = UOp::native_const(1.0f32).add(&outer_range.cast(DType::Float32));
    let inner_end = loop_(body, [inner_range.clone()]);
    let ctx = CFGContext::new(&UOp::sink(vec![loop_(inner_end, [outer_range.clone()])]));
    assert_eq!(ctx.edge_count(), 1);
    let inner_predecessor = ctx.get_predecessor(&inner_range).expect("the inner loop waits for its parent");
    assert_same!(Arc::clone(inner_predecessor), outer_range);
    assert!(ctx.get_predecessor(&outer_range).is_none(), "the parent is not a sibling of its own body");
}

/// An edge that would close a cycle is malformed input: the predecessor's subtree already
/// contains the RANGE the later sibling closes.
#[test]
#[should_panic(expected = "would create cycle")]
fn an_edge_that_would_close_a_cycle_is_rejected() {
    let (closed, consumed) = (global_range(10, 0), global_range(10, 1));
    let first = loop_(UOp::native_const(1.0f32).add(&consumed.cast(DType::Float32)), [closed]);
    let second = counted(&consumed);
    let _ = CFGContext::new(&UOp::sink(vec![first, second]));
}
