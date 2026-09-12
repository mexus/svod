//! `RangeifyBufferContext`: independent slot counters, the buffer map, and the
//! bound-variable table.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;

use crate::rangeify::{LocalAddBufferContext, RangeifyBufferContext};
use crate::test::support::prelude::*;

/// The counters are disjoint: global, local, range and `lunique` slots each
/// advance on their own, and the graph seeds only the `lunique` one.
#[test]
fn the_counters_are_independent_and_the_lunique_seed_moves_one() {
    let mut ctx = RangeifyBufferContext::new();
    assert_eq!((ctx.global_counter, ctx.local_counter, ctx.range_counter), (0, 0, 0));
    assert_eq!([ctx.next_global(), ctx.next_global(), ctx.next_global()], [0, 1, 2]);
    assert_eq!([ctx.next_local(), ctx.next_local()], [0, 1]);
    assert_eq!([ctx.next_range()], [0]);
    assert_eq!((ctx.global_counter, ctx.local_counter, ctx.range_counter), (3, 2, 1));
    assert_eq!([ctx.next_lunique(), ctx.next_lunique()], [0, 1]);

    let mut seeded = RangeifyBufferContext::with_lunique_start(7);
    assert_eq!([seeded.next_lunique(), seeded.next_lunique()], [7, 8]);
    assert_eq!(seeded.global_counter, 0, "the seed only moves the lunique counter");
}

#[test]
fn a_mapped_buffer_reads_back_by_uop_identity() {
    let mut ctx = RangeifyBufferContext::new();
    let original = UOp::native_const(1.0f32);
    let replacement = param(0, 1, DType::Float32);

    assert!(!ctx.has_buffer(&original));
    ctx.map_buffer(original.clone(), replacement.clone());

    assert_same!(ctx.get_buffer(&original).expect("mapped"), replacement);
}

/// A tracked var keeps its UOP and bound value; a name binds once, so rebinding
/// replaces the entry rather than appending and the latest value wins.
#[test]
fn tracked_vars_keep_one_latest_valued_entry_per_name() {
    let mut ctx = RangeifyBufferContext::new();
    let var = UOp::scalar_param(3, Some("test_var".to_string()), DType::Int32, 0, 10);

    assert!(ctx.vars.is_empty());
    ctx.add_var(var.clone(), Some(1));
    ctx.add_var(var.clone(), Some(5));

    let (stored_uop, stored_val) = ctx.vars.get("test_var").expect("tracked");
    assert_eq!(stored_uop.id, var.id);
    assert_eq!(*stored_val, Some(5));
    assert_eq!(ctx.vars.len(), 1);
}

/// The per-kernel context keeps binding order for CALL ABI parity: rebinding a
/// name swaps the old entry out and appends the new one, so each name occupies
/// exactly one, latest-valued slot.
#[test]
fn local_add_var_appends_one_entry_per_name() {
    let mut ctx = LocalAddBufferContext::new();
    let var = UOp::scalar_param(0, Some("x".to_string()), DType::Int32, 0, 10);
    let first = UOp::noop();
    let second = UOp::native_const(1.0f32);

    ctx.add_var(first, var.clone(), Some(1));
    ctx.add_var(second.clone(), var, Some(2));

    assert_eq!(ctx.vars.len(), 1);
    let (binding, (_, value)) = ctx.vars.iter().next().expect("one entry");
    assert_eq!(*value, Some(2));
    assert!(Arc::ptr_eq(&binding.0, &second), "the newest binding is the one that survives");
}

/// A var with no name is not part of the launch ABI and is not tracked, in
/// either context.
#[test]
fn an_anonymous_var_is_not_tracked() {
    let mut ctx = RangeifyBufferContext::new();
    ctx.add_var(UOp::noop(), Some(3));
    assert!(ctx.vars.is_empty());

    let mut local = LocalAddBufferContext::new();
    local.add_var(UOp::noop(), UOp::scalar_param(0, None, DType::Int32, 0, 10), Some(3));
    assert!(local.vars.is_empty());
}
