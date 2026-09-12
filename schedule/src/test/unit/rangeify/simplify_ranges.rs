//! `pm_simplify_ranges`: narrow each RANGE to the largest bound its consumers
//! can prove, and merge adjacent constant ranges.

use std::sync::Arc;

use svod_ir::{AxisType, Op, ReduceOp, UOp};
use test_case::test_case;

use crate::rangeify::{SimplifyRangesContext, pm_simplify_ranges};
use crate::test::support::prelude::*;

/// `INDEX(buffer, [range gated by `range < bound`])` — the shape a padded or
/// shrunk access takes after rangeify.
fn gated_index(range: &Arc<UOp>, bound: i64) -> Arc<UOp> {
    let gate = range.try_cmplt(&range.const_like(bound)).expect("cmplt");
    index_of(buffer(16), range.valid(gate))
}

fn gated_load(range: &Arc<UOp>, bound: i64) -> Arc<UOp> {
    load(gated_index(range, bound))
}

fn simplify(sink: Arc<UOp>) -> Arc<UOp> {
    rewrite_with(&pm_simplify_ranges(), &mut SimplifyRangesContext::default(), sink)
}

/// The surviving extent of the range renumbered `axis`.
fn narrowed_end(root: &Arc<UOp>, axis: usize) -> i64 {
    let range = root
        .ranges()
        .into_iter()
        .find(|r| matches!(r.op(), Op::Range(svod_ir::ops::Range { axis_id: svod_ir::AxisId::Renumbered(id), .. }) if *id == axis))
        .expect("range must remain in the rewritten graph");
    expect_range_extent(&range)
}

fn loop_range(axis: usize, end: i64) -> Arc<UOp> {
    range(end, AxisType::Loop, axis)
}

/// A gated access narrows its range to the gate's bound; the largest bound wins
/// when several consumers disagree, and a REDUCE-protected axis is untouched.
#[test_case(0, 7, 7 ; "bounded load")]
#[test_case(1, 5, 5 ; "bounded store")]
#[test_case(3, 6, 16 ; "reduce range is protected")]
fn a_bounded_access_narrows_its_range(axis: usize, bound: i64, expected: i64) {
    let r = if axis == 3 { range(16, AxisType::Reduce, axis) } else { loop_range(axis, 16) };
    let index = gated_index(&r, bound);
    let sink = if axis == 3 {
        UOp::sink(vec![load(index).reduce(smallvec::smallvec![r], ReduceOp::Add)])
    } else {
        UOp::sink(vec![load(index)])
    };

    assert_eq!(narrowed_end(&simplify(sink), axis), expected);
}

/// Several gated accessors of one range: the largest gate bound wins, and an
/// ungated accessor keeps the range at its original extent.
#[test]
fn conflicting_gates_choose_the_largest_bound() {
    let conflicting = loop_range(2, 16);
    let result = simplify(UOp::sink([4, 9].map(|bound| gated_load(&conflicting, bound)).to_vec()));
    assert_eq!(narrowed_end(&result, 2), 9);

    let ungated = loop_range(4, 16);
    let indirect = loop_range(5, 16);
    let gate = indirect.add(&indirect.const_like(1)).try_cmplt(&indirect.const_like(8)).expect("cmplt");
    let result = simplify(UOp::sink(
        [index_of(buffer(16), ungated), index_of(buffer(16), indirect.valid(gate))].map(load).to_vec(),
    ));
    assert_eq!(narrowed_end(&result, 4), 16, "an ungated access leaves the extent alone");
    assert_eq!(narrowed_end(&result, 5), 16, "a gate on `r + 1` is not the canonical `r < c`");
}

/// A range narrowed by one access must not be shrunk when another access uses it
/// in a later, ungated index position. The narrowing itself is pinned by
/// `a_narrowed_range_stays_a_range_of_its_axis_type`.
#[test]
fn narrowing_respects_ungated_uses() {
    let (r, q) = (loop_range(6, 16), loop_range(7, 16));
    let narrow = gated_index(&r, 4);
    let matrix = buffer(256)
        .try_reshape(&smallvec::smallvec![svod_ir::SInt::Const(16), svod_ir::SInt::Const(16)])
        .expect("reshape");
    let wide_gate = q.try_cmplt(&q.const_like(2)).expect("cmplt");
    let wide = UOp::index().buffer(matrix).indices(vec![q.valid(wide_gate), r.clone()]).call().expect("index");

    let result = simplify(UOp::sink([narrow, wide].map(load).to_vec()));

    assert_eq!(narrowed_end(&result, 6), 16, "r is used ungated in the second index");
    assert_eq!(narrowed_end(&result, 7), 2, "q is gated everywhere it is used");
}

fn merge_adjacent(ranges: smallvec::SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
    crate::rangeify::transforms::simplify_merge_adjacent(
        &mut SimplifyRangesContext::default(),
        &UOp::native_const(1.0f32).end(ranges),
    )
}

/// `simplify_merge_adjacent` folds two adjacent constant ranges into one, which
/// is what lets the indexer drop a divmod pair.
#[test]
fn adjacent_const_ranges_merge_into_one() {
    let ranges = smallvec::smallvec![UOp::range(UOp::index_const(10), 20), UOp::range(UOp::index_const(20), 21)];
    let merged = merge_adjacent(ranges).expect("two constant ranges merge");

    let (_, ranges) = expect_end(&merged);
    assert_eq!(ranges.len(), 1);
    assert_eq!(expect_range_extent(&ranges[0]), 200);
}

/// A symbolic range end must not be merged. The merge is a wash on divmod count
/// but turns a constant axis into a symbolic one, and every downstream opt filter
/// (upcast, unroll, local dims, tensor cores) is constant-only — so the merged
/// kernel loses the tensor-core path the constant axis would have taken. A
/// zero-sized range is not adjacent to anything either: the `s0 <= 0` guard
/// rejects it rather than folding an empty axis away.
#[test]
fn symbolic_and_empty_range_ends_block_the_merge() {
    let (symbolic, constant) =
        (UOp::range(UOp::define_var("b".into(), 1, 8), 22), UOp::range(UOp::index_const(20), 23));
    assert!(merge_adjacent(smallvec::smallvec![symbolic.clone(), constant.clone()]).is_none());
    assert!(merge_adjacent(smallvec::smallvec![constant.clone(), symbolic.clone()]).is_none());
    assert!(merge_adjacent(smallvec::smallvec![symbolic.clone(), symbolic]).is_none());

    let empty = UOp::range(UOp::index_const(0), 24);
    assert!(merge_adjacent(smallvec::smallvec![empty.clone(), constant]).is_none());
    assert!(merge_adjacent(smallvec::smallvec![UOp::range(UOp::index_const(20), 26), empty]).is_none());
}

/// The narrowed range keeps its axis type and stays a RANGE rather than being
/// replaced by the bound constant.
#[test_case(8, 7, AxisType::Loop ; "loop axis stays a range")]
#[test_case(9, 5, AxisType::Reduce ; "reduce axis stays a range")]
fn a_narrowed_range_stays_a_range_of_its_axis_type(axis: usize, bound: i64, axis_type: AxisType) {
    let r = if axis_type == AxisType::Reduce { range(16, AxisType::Reduce, axis) } else { loop_range(axis, 16) };
    let result = simplify(UOp::sink(vec![gated_load(&r, bound)]));
    let narrowed = &result.ranges()[0];
    assert_op!(narrowed, Op::Range(..));
    assert_eq!(range_axis_type(narrowed), axis_type);
    assert_eq!(expect_range_extent(narrowed), bound);
}
