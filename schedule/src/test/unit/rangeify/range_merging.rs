//! `merge_consumer_ranges`: one range per dimension across every consumer, and
//! the realize decision that falls out of it.

use std::sync::Arc;

use svod_ir::{AxisType, BinaryOp, Op, SInt, TernaryOp, UOp, UOpKey};

use crate::rangeify::indexing::{IndexingContext, all_ranges_same};
use crate::rangeify::merge_consumer_ranges;
use crate::test::support::prelude::*;

/// A range gated by `i < bound`, as a consumer with padding produces. `bound` of
/// `None` gates it by the literal `true`, the shape a merged guard short-circuits.
fn gated(idx: &Arc<UOp>, bound: Option<i64>) -> Arc<UOp> {
    let valid = match bound {
        Some(bound) => idx.try_cmplt(&UOp::index_const(bound)).expect("cmplt"),
        None => UOp::native_const(true),
    };
    UOp::try_where(valid, idx.clone(), UOp::invalid_marker()).expect("where")
}

fn realize_axes(ctx: &IndexingContext, uop: &Arc<UOp>) -> Option<Option<Vec<usize>>> {
    ctx.realize_map.get(&UOpKey(uop.clone())).cloned()
}

/// `all_ranges_same` is the merge's decision procedure. Vacuously true for zero
/// or one entry (tinygrad `helpers.py:31`), and pointer-based beyond that.
#[test]
fn ranges_are_the_same_only_when_they_are_the_same_node() {
    let mut ctx = IndexingContext::new();
    let (r0, r1) = (ctx.new_range(&SInt::Const(10), AxisType::Loop), ctx.new_range(&SInt::Const(20), AxisType::Loop));

    assert!(all_ranges_same(&[]));
    assert!(all_ranges_same(std::slice::from_ref(&r0)));
    assert!(all_ranges_same(&[r0.clone(), r0.clone()]));
    assert!(!all_ranges_same(&[r0.get_idx(), r1.get_idx()]));
}

/// A plain range is its own index and is unconditionally valid; a gated one
/// splits back into the two.
#[test]
fn a_gated_range_decomposes_into_its_index_and_its_condition() {
    let mut ctx = IndexingContext::new();
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    assert_same!(idx.get_idx(), idx);
    assert_const!(idx.get_valid(), true);

    let (narrow, valid) = (gated(&idx, Some(5)), gated(&idx, Some(5)).get_valid());
    let wide = gated(&idx, Some(8));
    assert_same!(narrow.get_idx(), idx);
    assert_same!(narrow.get_valid(), valid);
    assert_op!(narrow, Op::Ternary(TernaryOp::Where, ..));
    let Op::Ternary(TernaryOp::Where, _, _, otherwise) = narrow.op() else { unreachable!() };
    assert!(UOp::is_invalid_marker(otherwise));

    let merged = merge_consumer_ranges(&buffer(10), &[vec![narrow], vec![wide]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert_op!(merged[0], Op::Ternary(TernaryOp::Where, ..));
    let Op::Ternary(TernaryOp::Where, disjunction, merged_idx, _) = merged[0].op() else { unreachable!() };
    assert_same!(merged_idx, idx);
    assert!(matches!(disjunction.op(), Op::Binary(BinaryOp::Or, _, _)), "got {}", disjunction.tree());
}

/// Consumers that agree on a dimension pass their range straight through; one
/// that disagrees forces a fresh range and a realize. With PCONTIG=0 a single
/// disagreeing dim realizes them all (tinygrad `indexing.py:217`).
#[test]
fn consumer_ranges_converge_or_realize() {
    let mut ctx = IndexingContext::new();
    let storage = buffer(100);
    let r = ctx.new_range(&SInt::Const(100), AxisType::Loop);
    let a = ctx.new_range(&SInt::Const(100), AxisType::Loop);
    let b = ctx.new_range(&SInt::Const(100), AxisType::Loop);

    let merged = merge_consumer_ranges(&storage, &[vec![r.clone()], vec![r.clone()]], &mut ctx).expect("merge");
    assert_eq!(merged.len(), 1);
    assert!(Arc::ptr_eq(&merged[0], &r), "nothing to reconcile, so the range is passed through");
    assert!(realize_axes(&ctx, &storage).is_none());

    let merged = merge_consumer_ranges(&storage, &[vec![a.clone()], vec![b.clone()]], &mut ctx).expect("merge");
    assert_eq!(merged.len(), 1);
    assert!(!Arc::ptr_eq(&merged[0], &a) && !Arc::ptr_eq(&merged[0], &b));
    assert_eq!(realize_axes(&ctx, &storage), Some(Some(vec![0])));

    let wide = buffer(200).try_reshape(&smallvec::smallvec![SInt::Const(10), SInt::Const(20)]).expect("reshape");
    let mut ctx = IndexingContext::new();
    let shared_dim = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let (c, d) = (ctx.new_range(&SInt::Const(20), AxisType::Loop), ctx.new_range(&SInt::Const(20), AxisType::Loop));

    let consumers = [vec![shared_dim.clone(), c.clone()], vec![shared_dim.clone(), d]];
    let merged = merge_consumer_ranges(&wide, &consumers, &mut ctx).expect("merge");
    assert_eq!(merged.len(), 2);
    assert!(!Arc::ptr_eq(&merged[0], &shared_dim), "the agreeing dim is realized too");
    assert!(!Arc::ptr_eq(&merged[1], &c), "the disagreeing dim gets a fresh range");
    assert_eq!(realize_axes(&ctx, &wide), Some(Some(vec![0, 1])));
}

/// `all_same([])` is true upstream, so a dim with no consumer ranges does not
/// drag the other dims into a realize — but it has nothing to inherit either, so
/// it gets a fresh range and is realized on its own. The empty consumer list is
/// the only way into that branch; every other call site passes at least one.
#[test]
fn a_dimension_with_no_consumers_is_realized_on_its_own() {
    let mut ctx = IndexingContext::new();
    let storage = buffer(10);

    let merged = merge_consumer_ranges(&storage, &[], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert_op!(merged[0], Op::Range(..));
    assert_eq!(realize_axes(&ctx, &storage), Some(Some(vec![0])));
}

/// A consumer with more ranges than the source has dims is truncated, not
/// rejected: `all_rngs` is sized by the source shape.
#[test]
fn a_consumer_with_more_ranges_than_dimensions_is_truncated() {
    let mut ctx = IndexingContext::new();
    let storage = buffer(10);
    let used = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let extra = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let merged = merge_consumer_ranges(&storage, &[vec![used.clone(), extra]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1, "only the source's own dimension survives");
    assert_same!(merged[0], used);
    assert!(realize_axes(&ctx, &storage).is_none());
}

/// A guard that is literally true (or an OR of such guards) short-circuits: the
/// dimension unwraps to the bare index instead of a WHERE over `true`.
#[test]
fn an_always_valid_guard_short_circuits_to_the_bare_index() {
    let mut ctx = IndexingContext::new();
    let storage = buffer(10);
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let truthy = gated(&idx, None);

    let merged = merge_consumer_ranges(&storage, &[vec![truthy.clone()], vec![truthy]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert_same!(merged[0].get_idx(), idx);
    assert_const!(merged[0].get_valid(), true);
    assert!(realize_axes(&ctx, &storage).is_none());
}
