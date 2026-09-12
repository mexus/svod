//! `to_param_patterns`: the rewrite `split_store` runs over a kernel body to turn
//! storage into codegen PARAMs, unbind scalars, canonicalise range ids, and peel
//! AFTER ordering wrappers.

use std::sync::Arc;

use svod_ir::{AxisId, AxisType, ConstValue, DType, Op, UOp};
use test_case::test_case;

use crate::rangeify::{RangeifyBufferContext, patterns::to_param_patterns};
use crate::test::support::prelude::*;

fn apply(uop: &Arc<UOp>, ctx: &mut RangeifyBufferContext) -> Option<Arc<UOp>> {
    match to_param_patterns().rewrite(uop, ctx) {
        svod_ir::pattern::RewriteResult::Rewritten(result) => Some(result),
        _ => None,
    }
}

/// A BUFFER becomes the next codegen PARAM slot and is mapped to it, so every
/// later read of the same BUFFER reuses the slot.
#[test]
fn buffers_are_numbered_into_dense_param_slots() {
    let mut ctx = RangeifyBufferContext::new();

    for slot in 0..2 {
        let storage = buffer(100 * (slot + 1));
        let param = apply(&storage, &mut ctx).expect("a BUFFER becomes a PARAM");

        let svod_ir::ops::Param { arg, .. } = assert_op!(param, Op::Param(p) => p);
        assert_eq!(arg.slot, slot);
        assert_eq!(arg.device, Some(svod_dtype::DeviceSpec::Cpu));
        assert_eq!(ctx.global_counter, slot + 1);
        assert_same!(ctx.get_buffer(&storage).expect("mapped"), param);
    }
}

/// A bound scalar becomes a scalar PARAM and its value moves into `ctx.vars`, to
/// be passed at launch.
#[test]
fn a_bound_variable_becomes_a_scalar_param_and_a_launch_value() {
    let mut ctx = RangeifyBufferContext::new();
    let var = UOp::variable("x".to_string(), 0, 10, DType::WeakInt);
    let bind = var.bind(UOp::const_(DType::WeakInt, ConstValue::Int(5)));

    let param = apply(&bind, &mut ctx).expect("a BIND unbinds");

    let svod_ir::ops::Param { arg, .. } = assert_op!(param, Op::Param(p) => p);
    assert!(arg.addrspace.is_none());
    assert_eq!(ctx.vars.get("x").expect("the value is recorded for launch").1, Some(5));
}

/// Unrenumbered ranges get sequential canonical ids, keeping their axis type and
/// extent node; an already-renumbered range is left alone.
#[test]
fn unrenumbered_ranges_are_numbered_sequentially() {
    let mut ctx = RangeifyBufferContext::new();
    let axis_types = [AxisType::Loop, AxisType::Loop, AxisType::Reduce];

    for (i, axis_type) in axis_types.into_iter().enumerate() {
        let original = unrenumbered_range(
            &UOp::range_axis(
                UOp::const_(DType::WeakInt, ConstValue::Int(10 * (i as i64 + 1))),
                AxisId::Renumbered(0),
                axis_type,
            ),
            i + 5,
        );
        let renumbered = apply(&original, &mut ctx).expect("an unrenumbered range is renumbered");

        let (end, axis_id, kept) = expect_range(&renumbered);
        assert_eq!(axis_id, AxisId::Renumbered(i));
        assert_eq!(kept, axis_type, "renumbering must not change the axis type");
        // Node identity, not just the value: renumbering must carry the extent
        // node over rather than mint an equal one.
        assert_same!(end, expect_range(&original).0);
    }
    assert_eq!(ctx.range_counter, axis_types.len());
}

/// An empty range materialises as index 0, and it still consumes its canonical
/// id: the counter is what keeps later ranges distinct.
#[test]
fn a_zero_extent_range_is_rewritten_and_consumes_its_id() {
    let mut ctx = RangeifyBufferContext::new();
    let (end, _, axis_type) = expect_range(&range(0, AxisType::Loop, 0));
    let unrenumbered = UOp::new(
        Op::Range(svod_ir::ops::Range {
            end,
            axis_id: AxisId::Unrenumbered(0),
            axis_type,
            deps: smallvec::SmallVec::new(),
        }),
        DType::WeakInt,
    );

    let result = apply(&unrenumbered, &mut ctx).expect("an Unrenumbered RANGE always rewrites");

    assert!(matches!(result.op(), Op::Const(c) if c.0 == ConstValue::Int(0)), "{}", result.tree());
    assert_eq!(result.dtype(), DType::WeakInt, "the zero stays weak until target-width lowering");
    assert_eq!(ctx.range_counter, 1, "a materialized RANGE consumes its canonical id");
}

/// The RANGE the pass is expected to canonicalise: a canonical `range` with its
/// id swapped back to `Unrenumbered`.
fn unrenumbered_range(model: &Arc<UOp>, id: usize) -> Arc<UOp> {
    let (end, _, axis_type) = expect_range(model);
    UOp::range_axis_dtype(end, AxisId::Unrenumbered(id), axis_type, model.dtype())
}

/// Already-canonical nodes and values with nothing to rewrite are left alone. Id
/// zero is its own row: it is the id a fresh counter would hand out, so a pass
/// that renumbered unconditionally would still look idle on the other row.
#[test_case(|| range(10, AxisType::Loop, 5) ; "already renumbered")]
#[test_case(|| range(10, AxisType::Loop, 0) ; "already renumbered as zero")]
#[test_case(|| UOp::native_const(42i32) ; "a bare const")]
fn canonical_nodes_are_left_alone(build: fn() -> Arc<UOp>) {
    assert!(apply(&build(), &mut RangeifyBufferContext::new()).is_none());
}

// ===== AFTER peeling and buffer tracking =====

/// AFTER is an ordering wrapper: the pattern unwraps it to the storage it passes
/// through. Global storage is recorded in the buffer map so later readers pick up
/// the same ordering edge; LOCAL and REG storage is kernel-scoped and
/// synchronised by BARRIER instead, so it must not be tracked.
#[test_case(param(11, 1024, DType::Float32), true ; "global param is tracked")]
#[test_case(UOp::buffer_id(Some(0)), true ; "unique buffer is tracked")]
#[test_case(UOp::buffer(1, 1024, DType::Float32, svod_dtype::AddrSpace::Local, None), false ; "local buffer is not tracked")]
#[test_case(UOp::buffer(2, 1024, DType::Float32, svod_dtype::AddrSpace::Reg, None), false ; "register buffer is not tracked")]
fn after_unwraps_to_its_storage_and_tracks_only_global(storage: Arc<UOp>, tracked: bool) {
    let mut ctx = RangeifyBufferContext::new();
    let after = storage.clone().after(smallvec::smallvec![UOp::noop()]);

    let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps");

    assert_same!(unwrapped, storage);
    assert_eq!(ctx.has_buffer(&storage), tracked);
    if tracked {
        assert_same!(ctx.get_buffer(&storage).expect("tracked"), after);
    }
}

/// Multi-device wrappers resolve to a single representative buffer — the first of
/// an MSTACK, the selected one of an MSELECT — and that buffer, not the wrapper,
/// is what the AFTER is recorded against. LOCAL storage is kernel-scoped and stays
/// untracked.
#[test]
fn after_sees_through_multi_device_wrappers() {
    let local = || UOp::buffer(1, 1024, DType::Float32, svod_dtype::AddrSpace::Local, None);
    let global = || UOp::buffer_id(Some(1));
    let is_local = |uop: &Arc<UOp>| matches!(uop.op(), Op::Buffer(svod_ir::ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_dtype::AddrSpace::Local));

    for (wrap, representative) in [
        (mstack(global(), UOp::buffer_id(Some(2))), global()),
        (mselect(global()), global()),
        (mstack(local(), local()), local()),
        (mselect(local()), local()),
    ] {
        let mut ctx = RangeifyBufferContext::new();
        let after = wrap.after(smallvec::smallvec![UOp::noop()]);
        let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps the wrapper");

        assert_same!(unwrapped, representative);
        assert_eq!(ctx.has_buffer(&representative), !is_local(&representative));
        if !is_local(&representative) {
            assert_same!(ctx.get_buffer(&representative).expect("tracked"), after);
        }
    }
}

fn mstack(first: Arc<UOp>, second: Arc<UOp>) -> Arc<UOp> {
    let dtype = first.dtype();
    UOp::new(Op::MStack(svod_ir::ops::MStack { buffers: smallvec::smallvec![first, second] }), dtype)
}

fn mselect(storage: Arc<UOp>) -> Arc<UOp> {
    let dtype = storage.dtype();
    UOp::new(Op::MSelect(svod_ir::ops::MSelect { buffer: storage, device_index: 0 }), dtype)
}

/// A documented gap, kept red on purpose: `apply` unwraps the AFTER to the first
/// MSTACK buffer correctly, but `ctx.buffer_map` never records the MSTACK, so a
/// later reader cannot pick up the ordering edge.
#[test]
#[ignore = "MSTACK/AFTER handling not fully implemented yet"]
fn an_after_with_no_deps_still_tracks_its_mstack() {
    let mut ctx = RangeifyBufferContext::new();
    let first = UOp::buffer_id(Some(1));
    let stacked = mstack(first.clone(), UOp::buffer_id(Some(2)));
    let after = stacked.clone().after(smallvec::SmallVec::new());

    let unwrapped = apply(&after, &mut ctx).expect("AFTER unwraps");

    assert_same!(unwrapped, first);
    assert!(ctx.buffer_map.contains_key(&svod_ir::UOpKey(stacked)));
}
