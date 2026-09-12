//! End-to-end behaviour of `run_rangeify` / `rangeify` / `try_get_kernel_graph`.

use std::{f32::consts::PI, sync::Arc};

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisType, CallInfo, ConstValue, Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;

use super::helpers::{assert_same_ptr, count_kernels, count_stores, extract_kernel, reduce_range};
use crate::rangeify::{rangeify, run_rangeify, try_get_kernel_graph};
use crate::test::support::build::{buffer, has_op, index, range, store};

fn rangeify_unwrap(uop: Arc<UOp>) -> Arc<UOp> {
    rangeify(uop).expect("rangeify").0
}

/// A matcher that never fires, so the bottom-up table is the only one rewriting.
struct NoRewrite;

impl svod_ir::Matcher<()> for NoRewrite {
    fn rewrite(&self, _uop: &Arc<UOp>, _ctx: &mut ()) -> svod_ir::RewriteResult {
        svod_ir::RewriteResult::NoMatch
    }
}

/// Peels DETACH, the marker whose survival says whether a traversal reached in.
struct StripDetach;

impl svod_ir::Matcher<()> for StripDetach {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> svod_ir::RewriteResult {
        match uop.op() {
            Op::Detach(ops::Detach { src }) => svod_ir::RewriteResult::Rewritten(src.clone()),
            _ => svod_ir::RewriteResult::NoMatch,
        }
    }
}

#[track_caller]
fn opaque_body(uop: &Arc<UOp>) -> Arc<UOp> {
    match uop.op() {
        Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body.clone(),
        op => panic!("expected an opaque root, got {op:?}"),
    }
}

fn store_at_zero(value: Arc<UOp>) -> Arc<UOp> {
    store(index(buffer(100), 0), value)
}

fn reductions(uop: &Arc<UOp>) -> Vec<Arc<UOp>> {
    uop.toposort().into_iter().filter(|node| matches!(node.op(), Op::Reduce(..))).collect()
}

// ===== run_rangeify =====

#[test]
fn tensor_reduce_becomes_a_loop_reduce_over_ranges() {
    let source = buffer(6).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).expect("reshape");
    let tensor_reduce = source.try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce axis");
    let (rangeified, _ctx) = run_rangeify(UOp::sink(vec![tensor_reduce.contiguous()])).expect("run_rangeify");

    let reductions = reductions(&rangeified);
    assert!(!reductions.is_empty());
    assert!(
        reductions
            .iter()
            .all(|node| matches!(node.op(), Op::Reduce(ops::Reduce { ranges, num_axes: 0, .. }) if !ranges.is_empty())),
        "every REDUCE must carry explicit ranges and no tensor axes"
    );
}

#[test_case(&[(0, 0), (0, 0)]; "no padding")]
#[test_case(&[(1, 1), (2, 2)]; "symmetric")]
#[test_case(&[(3, 0), (0, 5)]; "one sided")]
fn run_rangeify_lowers_every_pad(pads: &[(usize, usize)]) {
    let dims: smallvec::SmallVec<[SInt; 4]> = smallvec![SInt::Const(4), SInt::Const(5)];
    let source = buffer(20).try_reshape(&dims).expect("reshape");
    let padded = source.try_pad(&pads.iter().map(|&(lo, hi)| (lo.into(), hi.into())).collect::<Vec<_>>()).expect("pad");

    let (rangeified, _ctx) = run_rangeify(UOp::sink(vec![padded.contiguous()])).expect("run_rangeify");
    assert!(
        !rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Pad(..) | Op::ReduceAxis(..))),
        "rangeify must lower every PAD:\n{}",
        rangeified.tree()
    );
}

/// CALL and FUNCTION bodies are opaque by default: only an explicit full
/// traversal reaches into them.
#[test_case(true ; "call body")]
#[test_case(false ; "function body")]
fn run_rangeify_preserves_opaque_bodies_by_default(function: bool) {
    let reduced = UOp::param(0, 8, DType::Float32, None).try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce");
    let arg = buffer(8);
    let (rangeified, _ctx) = if function {
        run_rangeify(reduced.function(smallvec![arg], CallInfo::default())).expect("run_rangeify")
    } else {
        run_rangeify(reduced.call(smallvec![arg], CallInfo::default())).expect("run_rangeify")
    };

    let body = match rangeified.op() {
        Op::Call(ops::Call { body, .. }) | Op::Function(ops::Function { body, .. }) => body.clone(),
        op => panic!("expected an opaque root, got {op:?}"),
    };
    assert!(
        body.toposort().iter().any(|u| matches!(u.op(), Op::Reduce(ops::Reduce { num_axes: 1, .. }))),
        "run_rangeify must not rewrite the body by default:\n{}",
        rangeified.tree()
    );
}

/// The two `graph_rewrite_with_bpm` forms are a pair: the preserve-calls one
/// leaves a CALL/FUNCTION body alone, and the plain one reaches into it. Pinning
/// only the first would let the second stop traversing without a failure.
#[test_case(true ; "call body")]
#[test_case(false ; "function body")]
fn only_an_explicit_full_traversal_rewrites_inside_an_opaque_body(call: bool) {
    let detached = UOp::native_const(1.0f32).detach();
    let arg = UOp::native_const(2.0f32);
    let opaque = if call {
        detached.call(smallvec![arg], CallInfo::default())
    } else {
        detached.function(smallvec![arg], CallInfo::default())
    };
    let has_detach = |body: &Arc<UOp>| has_op(body, |op| matches!(op, Op::Detach(..)));

    let preserved = svod_ir::rewrite::graph_rewrite_with_bpm_preserve_calls(&NoRewrite, &StripDetach, opaque, &mut ());
    assert!(has_detach(&opaque_body(&preserved)), "the preserve-calls form must leave the body alone");

    let full = svod_ir::rewrite::graph_rewrite_with_bpm(&NoRewrite, &StripDetach, preserved, &mut ());
    assert!(!has_detach(&opaque_body(&full)), "an explicit full rewrite must reach into the body");
}

// ===== Full pipeline =====

/// DETACH and CONTIGUOUS_BACKWARD are stripped by `earliest_rewrites`, ahead of
/// `run_rangeify`. Pattern-level coverage lives in `patterns.rs`.
#[test]
fn autograd_markers_are_gone_after_the_full_pipeline() {
    let x = UOp::native_const(1.0f32);
    for marked in [x.detach(), x.contiguous_backward()] {
        assert_same_ptr(&rangeify_unwrap(marked), &x);
    }
}

/// A kernel graph is void iff it actually owns a kernel: one CALL, one STORE.
#[test_case(super::store_at_zero(UOp::native_const(1.0f32)) ; "store")]
#[test_case(super::store_at_zero(UOp::native_const(2.0f32)).end(smallvec![range(100, AxisType::Loop, 0)]) ; "end of store")]
#[test_case(super::store_at_zero(UOp::native_const(2.0f32).try_add(&UOp::native_const(3.0f32)).expect("add")) ; "store of arithmetic")]
fn rangeify_then_kernel_split_produces_one_void_kernel(root: Arc<UOp>) {
    let (kernel, _ctx) = try_get_kernel_graph(rangeify_unwrap(root)).expect("kernel split");
    assert_eq!(kernel.dtype(), DType::Void);
    assert_eq!(count_kernels(&kernel), 1, "one kernel per STORE:\n{}", kernel.tree());
    assert_eq!(count_stores(&kernel), 1, "the kernel body owns exactly one STORE:\n{}", kernel.tree());
    assert!(extract_kernel(&kernel).is_some());
}

/// LOADs feeding a STORE stay inside one kernel; the CALL owns every buffer.
#[test_case(1 ; "one load")]
#[test_case(2 ; "two loads")]
fn loads_feeding_one_store_split_into_one_kernel(loads: usize) {
    let load = |buf| UOp::load().index(index(buf, 0)).call();

    let value = (1..loads).fold(load(buffer(100)), |acc, _| acc.try_add(&load(buffer(100))).expect("add"));
    let root = store(index(buffer(100), 0), value);

    let (result, _ctx) = try_get_kernel_graph(root).expect("kernel split");
    assert_eq!(count_kernels(&result), 1);
}

/// Reductions over a range-independent source lose the loop entirely: ADD scales
/// the value by the extent, MAX leaves it untouched. MIN is deliberately absent
/// upstream — see `reduce_simplify::min_is_not_an_unparented_fold`.
#[test_case(ReduceOp::Add, 10, ConstValue::Int(50) ; "add scales by the extent")]
#[test_case(ReduceOp::Max, 5, ConstValue::Int(5) ; "max is idempotent")]
fn unparented_reductions_collapse_to_a_constant(op: ReduceOp, extent: i64, expected: ConstValue) {
    let reduce = UOp::native_const(5i32).reduce(smallvec![reduce_range(extent, 0)], op);

    let result = rangeify_unwrap(reduce);
    assert!(matches!(result.op(), Op::Const(c) if c.0 == expected), "got {}", result.tree());
}

/// A source that reads the range cannot be collapsed — the REDUCE survives.
#[test]
fn range_dependent_reductions_are_kept() {
    let range = reduce_range(10, 0);
    let src = range.cast(DType::Int32).try_add(&UOp::native_const(1i32)).expect("add");
    let reduce = src.reduce(smallvec![range], ReduceOp::Add);

    let result = rangeify_unwrap(reduce);
    assert!(!reductions(&result).is_empty(), "got {}", result.tree());
}

/// `split_reduceop` materialises an intermediate (a CONTIGUOUS) only once the
/// reduced extent passes its threshold. Threshold arithmetic: `split_reduceop.rs`.
#[test_case(1_000, false ; "below threshold")]
#[test_case(100_000, true ; "above threshold")]
fn large_reductions_are_split_in_two_stages(size: usize, split: bool) {
    let reduce = buffer(size).try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");

    let rangeified = rangeify_unwrap(reduce);
    let has_contiguous = rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous(..)));
    assert_eq!(has_contiguous, split);
    assert_eq!(rangeified.dtype(), DType::Float32);
}

/// A constant reduced over two ranges is not range-independent per axis: the
/// REDUCE survives with both axes, and the axes are the ones that were passed in.
#[test]
fn a_multi_range_reduction_survives_the_pipeline() {
    let ranges: Vec<Arc<UOp>> = (0..2).map(|i| reduce_range(8 >> i, i)).collect();
    let (rangeified, _ctx) = run_rangeify(
        UOp::const_(DType::Float32, ConstValue::Float(PI as f64)).reduce(ranges.clone().into(), ReduceOp::Add),
    )
    .expect("run_rangeify");

    let reductions = reductions(&rangeified);
    assert_eq!(reductions.len(), 1, "the reduction must survive:\n{}", rangeified.tree());
    let Op::Reduce(ops::Reduce { ranges: kept, num_axes, .. }) = reductions[0].op() else {
        panic!("expected a REDUCE, got {}", reductions[0].tree())
    };
    assert_eq!(*num_axes, 0, "the tensor axes are gone");
    assert_eq!(kept.len(), 2, "both axes survive:\n{}", rangeified.tree());
    for (actual, expected) in kept.iter().zip(&ranges) {
        assert!(Arc::ptr_eq(actual, expected), "the surviving axes are the ones that were passed in");
    }
    assert_eq!(rangeified.dtype(), DType::Float32);
}

/// `RangeifyResult::context` reports the counter the pass used and an empty
/// `range_map` (the map is only populated by `IndexingContext` inside the pass).
#[test]
fn the_rangeify_result_reports_the_axis_counter_and_an_empty_map() {
    let result =
        crate::rangeify::rangeify_with_map(UOp::sink(vec![buffer(4).try_add(&buffer(4)).expect("add").contiguous()]))
            .expect("rangeify_with_map");

    let axes = result.sink.toposort().iter().filter(|node| matches!(node.op(), Op::Range(..))).count();
    assert!(axes > 0, "an elementwise graph is rangeified into explicit axes:\n{}", result.sink.tree());
    assert!(result.context.range_map.is_empty(), "the public range map is not populated by this entry point");
    assert!(
        result.context.range_counter >= axes,
        "the counter is the allocator high-water mark, so it covers every surviving RANGE"
    );
}

/// `uop_list` is the deduplicated backward slice of the result.
#[test]
fn the_rangeify_result_lists_the_deduplicated_backward_slice() {
    let result =
        crate::rangeify::rangeify_with_map(UOp::sink(vec![buffer(4).contiguous()])).expect("rangeify_with_map");
    let listed = result.uop_list.len();
    let unique: std::collections::HashSet<u64> = result.uop_list.iter().map(|node| node.id).collect();
    assert_eq!(listed, unique.len(), "the slice must not repeat nodes");
    assert!(listed > 0);
}
