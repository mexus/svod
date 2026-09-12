//! Graph shapes `rangeify` must accept, and the dead-axis predicate it folds
//! size-1 and size-0 iteration spaces with.

use std::sync::Arc;

use svod_ir::{AxisType, DType, Op, SInt, UOp};
use test_case::test_case;

use crate::rangeify::indexing::is_dead_axis;
use crate::rangeify::transforms::rangeify;
use crate::test::support::prelude::*;

fn accepted_symbolic_range_size() -> Arc<UOp> {
    stage(UOp::native_const(1.0f32), vec![range_symbolic(UOp::var("size", DType::Index, 0, 1024), 0)])
}

fn accepted_symbolic_range_sizes() -> Arc<UOp> {
    let ranges =
        (0..2).map(|i| range_symbolic(UOp::var(format!("size{i}").as_str(), DType::Index, 0, 1024), i)).collect();
    stage(UOp::native_const(2.0f32), ranges)
}

fn accepted_symbolic_range_arithmetic() -> Arc<UOp> {
    let n = UOp::variable("n".into(), 0, 512, DType::Int32);
    let size = n.try_mul(&UOp::index_const(2)).expect("mul");
    stage(UOp::native_const(3.0f32), vec![range_symbolic(size, 0)])
}

fn accepted_mixed_const_and_symbolic_ranges() -> Arc<UOp> {
    let symbolic = range_symbolic(param(0, 1, DType::Index), 1);
    stage(UOp::native_const(1.0f32), vec![range(10, AxisType::Loop, 0), symbolic])
}

/// `STAGE(STAGE(STAGE(x, r0), r1), r2)` — each level buffers a different extent.
fn accepted_nested_stage() -> Arc<UOp> {
    (0..3).fold(UOp::native_const(1.0f32), |inner, i| stage(inner, vec![range(5 * (i as i64 + 1), AxisType::Loop, i)]))
}

/// One STAGE read by two independent consumers.
fn accepted_stage_with_two_consumers() -> Arc<UOp> {
    let staged = stage(UOp::native_const(1.0f32), vec![range(10, AxisType::Loop, 0)]);
    let buf_shape = staged.shape().expect("shape").expect("static shape");
    let ones: svod_ir::shape::Shape = buf_shape.iter().map(|_| SInt::Const(1)).collect();
    let broadcast =
        |v: f32| UOp::native_const(v).try_reshape(&ones).expect("reshape").try_expand(buf_shape).expect("expand");
    UOp::sink(vec![staged.try_add(&broadcast(2.0)).expect("add"), staged.try_mul(&broadcast(3.0)).expect("mul")])
}

/// One compute staged twice with different iteration spaces.
fn accepted_compute_staged_twice() -> Arc<UOp> {
    let compute = UOp::native_const(1.0f32);
    UOp::sink(vec![
        stage(compute.clone(), vec![range(10, AxisType::Loop, 0)]),
        stage(compute, vec![range(20, AxisType::Loop, 1)]),
    ])
}

/// A permuted view of a buffer: the index expressions rangeify builds for it are
/// what the symbolic simplification has to survive.
fn accepted_permuted_buffer() -> Arc<UOp> {
    let src = buffer(6);
    src.try_reshape(&smallvec::smallvec![SInt::Const(2), SInt::Const(3)])
        .expect("reshape")
        .try_permute(vec![1, 0])
        .expect("permute")
}

/// Every accepted graph builds and rangeifies without losing an axis or trapping:
/// the shapes below are what the frontend mints for symbolic and staged programs.
#[test_case(accepted_symbolic_range_size ; "symbolic range size")]
#[test_case(accepted_symbolic_range_sizes ; "two symbolic range sizes")]
#[test_case(accepted_symbolic_range_arithmetic ; "symbolic range size from arithmetic")]
#[test_case(accepted_mixed_const_and_symbolic_ranges ; "const and symbolic ranges mixed")]
#[test_case(accepted_nested_stage ; "three nested stages")]
#[test_case(accepted_stage_with_two_consumers ; "stage read by two consumers")]
#[test_case(accepted_compute_staged_twice ; "compute staged twice")]
#[test_case(accepted_permuted_buffer ; "permuted buffer view")]
fn rangeify_lowers_every_accepted_graph(build: fn() -> Arc<UOp>) {
    let root = build();
    let (result, _ctx) = rangeify(Arc::clone(&root)).expect("rangeify must accept the graph");
    assert_eq!(result.dtype(), root.dtype());
}

/// `is_dead_axis` is `vmax < 1`: extent 0 and 1 collapse, extent 2 and up survive.
#[test_case(0, true ; "empty range")]
#[test_case(1, true ; "singleton range")]
#[test_case(2, false ; "two element range")]
#[test_case(10, false ; "ten element range")]
fn dead_axis_by_extent(extent: i64, dead: bool) {
    assert_eq!(is_dead_axis(&range(extent, AxisType::Loop, 0)), dead);
}

#[test]
fn only_ranges_can_be_dead_axes() {
    let constant = UOp::index_const(0);
    assert!(!is_dead_axis(&constant));
    assert!(!is_dead_axis(&constant.try_add(&constant).expect("add")));
}

/// The graph builders above must actually contain the shapes they are named for;
/// otherwise the acceptance table is vacuous.
#[test]
fn the_accepted_shapes_are_the_shapes_they_name() {
    assert!(has_op(&accepted_nested_stage(), |op| matches!(op, Op::Stage(..))));
    assert!(has_op(&accepted_permuted_buffer(), |op| matches!(op, Op::Permute(..))));
    assert!(has_op(&accepted_symbolic_range_size(), |op| matches!(op, Op::Stage(..))));
}
