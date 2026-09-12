//! Structure of the graph `try_get_kernel_graph` hands back.

use std::sync::Arc;

use crate::rangeify::try_get_kernel_graph;
use crate::test::support::prelude::*;
use svod_ir::{Op, UOp};

/// Two `[3,4]` views of distinct buffers, added and materialised — the graph
/// `Tensor::from_slice(a) + Tensor::from_slice(b)` lowers to.
fn added_reshaped_buffers() -> Arc<UOp> {
    let view = || {
        buffer(12).try_reshape(&smallvec::smallvec![svod_ir::SInt::Const(3), svod_ir::SInt::Const(4)]).expect("reshape")
    };
    UOp::sink(vec![view().try_add(&view()).expect("add").contiguous()])
}

/// Only GLOBAL stages become STORE/BUFFER; a LOCAL stage left at the top level
/// is not a legal kernel graph and must be reported, not silently accepted.
#[test]
fn an_outer_local_stage_fails_the_kernel_graph_boundary() {
    let range = global_range(16, 0);
    let root = UOp::sink(vec![
        stage(UOp::native_const(1.0f32), vec![range.clone()]),
        stage_with(UOp::native_const(2.0f32), vec![range], svod_ir::BufferizeOpts::local()),
    ]);

    let Err(error) = try_get_kernel_graph(root) else { panic!("outer local STAGE must be rejected") };
    assert!(matches!(error, crate::rangeify::KernelGraphError::Spec { .. }), "unexpected error: {error}");
}

/// A STAGE becomes a CALL over its own buffer: the CALL args stay BUFFERs,
/// because PARAMs live in the body.
#[test]
fn a_stage_lowers_to_a_call_over_its_own_buffer() {
    let staged = stage(UOp::native_const(std::f32::consts::PI), vec![global_range(20, 0)]);

    let (result, _ctx) = try_get_kernel_graph(staged).expect("kernel split");
    let kernel = first_call(&result).expect("CALL");

    assert_op!(expect_call(&kernel), Op::Sink(..));
    let svod_ir::ops::Call { args, .. } = assert_op!(kernel, Op::Call(c) => c);
    let [arg] = args.as_slice() else { panic!("expected the single staged buffer, got {args:?}") };
    assert!(matches!(arg.op(), Op::Buffer(..)), "CALL args stay BUFFERs; PARAMs live in the body");
}

/// Rangeify lowers every RESHAPE, and the split reaches the input buffers
/// through an INDEX over a BUFFER or a PARAM.
#[test]
fn rangeify_lowers_every_reshape_and_the_split_indexes_the_input_buffers() {
    let (rangeified, _ctx) = crate::rangeify::rangeify(added_reshaped_buffers()).expect("rangeify");
    assert!(!has_op(&rangeified, |op| matches!(op, Op::Reshape(..))), "{}", rangeified.tree());

    let (kernel_graph, _ctx) = try_get_kernel_graph(rangeified).expect("kernel split");
    assert!(
        has_op(
            &kernel_graph,
            |op| matches!(op, Op::Index(svod_ir::ops::Index { buffer, .. }) if matches!(buffer.op(), Op::Buffer(..) | Op::Param(..)))
        ),
        "input buffers must be reached through INDEX:\n{}",
        kernel_graph.tree()
    );
}
