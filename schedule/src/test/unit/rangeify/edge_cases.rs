//! Degenerate iteration spaces: no ranges at all, and a zero-sized range.

use svod_ir::{Op, UOp};

use crate::rangeify::{RangeifyBufferContext, bufferize_to_store, try_get_kernel_graph};
use crate::test::support::prelude::*;

fn scalar_stage() -> std::sync::Arc<UOp> {
    stage(UOp::native_const(42.0f32), vec![])
}

/// A rangeless STAGE stores without an END wrapper, still reaches the kernel
/// boundary, and produces a CALL whose body is the rangeless STORE.
#[test]
fn rangeless_stage_stores_without_an_end_wrapper() {
    let result = bufferize_to_store(&scalar_stage(), &mut RangeifyBufferContext::new()).expect("scalar STAGE converts");

    let (passthrough, deps) = expect_after(&result);
    assert!(matches!(passthrough.op(), Op::Buffer(..)));
    let [store] = deps.as_slice() else { panic!("expected exactly one dep") };
    assert!(matches!(store.op(), Op::Store(..)), "no ranges means STORE is not wrapped in END: {}", store.tree());

    let (result, _ctx) = try_get_kernel_graph(scalar_stage()).expect("kernel split");
    let kernel = first_call(&result).expect("the scalar stage must become a CALL");
    assert!(has_op(&expect_call(&kernel), |op| matches!(op, Op::Store(..))));
}

/// Tinygrad's `assert size > 0`: an empty range cannot back a buffer.
#[test]
#[should_panic(expected = "Cannot allocate buffer: range vmax resolved to")]
fn zero_sized_range_cannot_be_allocated() {
    bufferize_to_store(&stage(UOp::native_const(1.0f32), vec![global_range(0, 0)]), &mut RangeifyBufferContext::new());
}

/// A LOCAL stage is not a global buffer: `bufferize_to_store` declines it so that
/// `pm_add_local_buffers` can lower it later.
#[test]
fn a_local_stage_is_not_bufferized_here() {
    let staged = stage_with(UOp::native_const(1.0f32), vec![global_range(4, 0)], svod_ir::BufferizeOpts::local());

    assert!(bufferize_to_store(&staged, &mut RangeifyBufferContext::new()).is_none());
}

/// Tinygrad-aligned: closing zero ranges is the identity, not a fresh END node.
#[test]
fn end_over_no_ranges_returns_the_computation() {
    let computation = UOp::noop();
    assert_same!(computation.clone().end(smallvec::SmallVec::new()), computation);
}
