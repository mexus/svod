//! One CALL per global STAGE, each wrapping exactly one STORE, and the buffer
//! memoisation that keeps a STAGE lowered once.
//!
//! Fusion-level counts over realistic graphs live in `fusion.rs`.

use std::sync::Arc;

use svod_ir::UOp;
use test_case::test_case;

use crate::rangeify::{RangeifyBufferContext, transforms::bufferize_to_store, try_get_kernel_graph};
use crate::test::support::prelude::*;

#[test_case(|| stage(UOp::native_const(1.0f32), vec![global_range(10, 0)]), 1 ; "one stage")]
#[test_case(|| UOp::sink(vec![
    stage(UOp::native_const(1.0f32), vec![global_range(10, 0)]),
    stage(UOp::native_const(2.0f32), vec![global_range(20, 1)]),
]), 2 ; "two independent stages")]
fn each_global_stage_becomes_one_call_with_one_store(build: fn() -> Arc<UOp>, expected: usize) {
    let (result, _ctx) = try_get_kernel_graph(build()).expect("kernel split");
    assert_eq!(kernels(&result), expected);
    assert_eq!(
        count(&result, |node| matches!(node.op(), svod_ir::Op::Store(..))),
        expected,
        "each CALL body owns exactly one STORE"
    );
}

/// Lowering the same STAGE twice reuses its storage — the `lunique` slot is
/// consumed once — even though each lowering rebuilds the AFTER wrapper around
/// it. A distinct STAGE allocates its own slot.
#[test]
fn stage_identity_decides_buffer_reuse() {
    let mut ctx = RangeifyBufferContext::new();
    let r = global_range(5, 0);
    let first = stage(UOp::native_const(42i32), vec![r.clone()]);
    let second = stage(UOp::native_const(43i32), vec![r]);

    let first_lowering = bufferize_to_store(&first, &mut ctx).expect("first lowers");
    let after_repeat = bufferize_to_store(&first, &mut ctx).expect("first re-lowers");

    assert!(
        Arc::ptr_eq(&expect_after(&first_lowering).0.buf_uop(), &expect_after(&after_repeat).0.buf_uop()),
        "the same STAGE reuses its buffer"
    );
    assert_eq!(ctx.lunique_counter, 1, "a repeated STAGE must not allocate a second slot");

    let other = bufferize_to_store(&second, &mut ctx).expect("second lowers");
    assert!(!Arc::ptr_eq(&expect_after(&other).0.buf_uop(), &expect_after(&first_lowering).0.buf_uop()));
    assert_eq!(ctx.lunique_counter, 2, "the distinct STAGE allocates its own slot");
}
