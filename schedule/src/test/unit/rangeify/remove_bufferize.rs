//! `pm_remove_bufferize`: inline `INDEX(STAGE)` when the compute is cheaper to
//! recompute than to buffer, plus the two NOOP collapses that fall out of it.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{BufferizeOpts, Op, ReduceOp, UOp, ops};
use test_case::test_case;

use super::helpers::{assert_no_match, loop_range, reduce_range, rewritten};
use crate::rangeify::patterns::pm_remove_bufferize;
use crate::test::support::build::{param as shared_param, stage, stage_with};
use crate::test::support::prelude::assert_op;

// ============================================================================
// Helper builders
// ============================================================================

fn alu_param(slot: usize, size: usize) -> Arc<UOp> {
    shared_param(slot, size, DType::Float32)
}

fn non_removable(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    stage_with(
        compute,
        ranges,
        BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Local, removable: false },
    )
}

/// `INDEX(STAGE(compute, buf_ranges), idx_ranges)` with a removable STAGE.
fn index_bufferize(compute: Arc<UOp>, buf_ranges: Vec<Arc<UOp>>, idx_ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::index().buffer(stage(compute, buf_ranges)).indices(idx_ranges).call().expect("INDEX construction")
}

/// `INDEX(<one buffer>, address)`.
fn read_at(buffer: Arc<UOp>, address: &Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![address.clone()]).call().expect("INDEX")
}

/// `read_at(param(slot, 8), address)` — an ALU-typed PARAM read.
fn read_param(slot: usize, address: &Arc<UOp>) -> Arc<UOp> {
    read_at(alu_param(slot, 8), address)
}

/// The result of the rewrite, or the input when the pattern declines.
fn inlines(idx: &Arc<UOp>) -> bool {
    matches!(pm_remove_bufferize().rewrite(idx, &mut ()), svod_ir::RewriteResult::Rewritten(_))
}

/// The pass declines `idx`, with the tree on failure.
#[track_caller]
fn assert_kept(idx: &Arc<UOp>) {
    assert_no_match(&pm_remove_bufferize(), idx, &mut ());
}

/// `read(s0) + read(s1) + ...` over `count` distinct ALU PARAMs.
fn param_chain(count: usize, address: &Arc<UOp>) -> Arc<UOp> {
    (1..count).fold(read_param(0, address), |acc, slot| acc.try_add(&read_param(slot, address)).expect("add"))
}

/// An MStack over one LOCAL buffer.
fn local_mstack(slot: usize, len: usize) -> Arc<UOp> {
    let local = UOp::buffer(slot, len, DType::Float32, AddrSpace::Local, None);
    UOp::new(
        Op::MStack(ops::MStack { buffers: smallvec![local] }),
        DType::Float32.ptr(Some(len), AddrSpace::Local).expect("local ptr"),
    )
}

// ============================================================================
// Rule 1: INDEX(STAGE) — the cost heuristic
// ============================================================================

fn contiguous() -> Arc<UOp> {
    UOp::native_const(1.0f32).contiguous()
}

fn copy() -> Arc<UOp> {
    UOp::native_const(1.0f32).copy_to_device(DeviceSpec::Cpu)
}

/// An always-run source has effects (or a transfer-sized destination) that
/// inlining would duplicate or resize.
#[test_case(super::contiguous ; "contiguous")]
#[test_case(super::copy ; "copy")]
#[test_case(UOp::noop ; "noop")]
fn always_run_sources_are_kept(build: fn() -> Arc<UOp>) {
    let address = loop_range(8, 0);
    assert_kept(&index_bufferize(build(), vec![address.clone()], vec![address]));
}

#[test]
fn a_non_removable_stage_is_kept() {
    // Multi-consumer realize boundary — inlining would duplicate compute into
    // every consumer's kernel.
    let address = loop_range(8, 0);
    let compute = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    assert_kept(&read_at(non_removable(compute, vec![address.clone()]), &address));
}

/// Inlining is worth it up to three distinct Param/Stage/MStack accesses; the
/// cutoff is `> 3`.
#[test_case(3 ; "at the threshold")]
fn the_accessed_buffer_count_decides_inlining(params: usize) {
    let address = loop_range(8, 0);
    let idx = index_bufferize(param_chain(params, &address), vec![address.clone()], vec![address]);

    assert!(inlines(&idx));
}

#[test]
fn four_accessed_buffers_are_over_the_cutoff() {
    let address = loop_range(8, 0);
    let idx = index_bufferize(param_chain(4, &address), vec![address.clone()], vec![address]);

    assert_kept(&idx);
}

#[test]
fn after_stops_the_accessed_buffer_walk() {
    // Three params sit behind an AFTER's ordering dep; this compute reads only
    // the buffer the AFTER passes through. Walking into the dep would count 4
    // buffers and keep the bufferize; the AFTER costs its own buffer, once.
    let address = loop_range(8, 0);
    let ordered = read_param(1, &address)
        .try_add(&read_param(2, &address))
        .expect("add")
        .try_add(&read_param(3, &address))
        .expect("add");
    let idx = index_bufferize(
        read_at(alu_param(0, 8).after(smallvec![ordered]), &address),
        vec![address.clone()],
        vec![address],
    );
    assert!(inlines(&idx), "the AFTER's deps must not count toward the >3 cutoff");
}

#[test_case(|outer: &Arc<UOp>, inner: &Arc<UOp>| read_at(alu_param(0, 32), &inner.cast(DType::Index))
    .try_add(&outer.cast(DType::Int32))
    .expect("add")
    .reduce(smallvec![inner.clone()], ReduceOp::Add), false ; "a buffer read in the reduce body")]
#[test_case(|_outer: &Arc<UOp>, inner: &Arc<UOp>| UOp::native_const(2.0f32).reduce(smallvec![inner.clone()], ReduceOp::Add), true ; "a range-only body")]
fn a_reduce_body_decides_inlining(build: fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>, inlines_body: bool) {
    let outer = loop_range(8, 0);
    let inner = reduce_range(4, 1);
    let idx = index_bufferize(build(&outer, &inner), vec![outer.clone()], vec![outer]);

    assert_eq!(inlines(&idx), inlines_body);
}

#[test]
fn const_range_keys_are_skipped_during_substitution() {
    // Mix CONST and RANGE in buf_ranges. The CONST slot is a broadcast dim
    // (not a real range key) and must be skipped during substitution; the
    // inlined result must reference the live `idx_r` from the index side.
    let buf_r = UOp::range_const(8, 0);
    let idx_r = loop_range(8, 1);

    let idx = index_bufferize(
        buf_r.clone(),
        vec![UOp::index_const(0), buf_r.clone()],
        vec![UOp::index_const(0), idx_r.clone()],
    );
    let inlined = rewritten(&pm_remove_bufferize(), &idx, &mut ());
    assert!(Arc::ptr_eq(&inlined, &idx_r), "substitution must replace buf_r with idx_r (CONST keys skipped)");
}

#[test]
fn an_invalid_index_value_is_not_substituted() {
    // A dead-load `Invalid` index value must NOT be substituted into the inlined
    // compute — doing so would poison the expression, and its buffer range stays.
    let (buf_r0, buf_r1) = (loop_range(8, 0), loop_range(8, 1));
    let idx0 = loop_range(8, 2);
    let compute = buf_r0.try_add(&buf_r1).expect("add");
    let buf = stage(compute, vec![buf_r0.clone(), buf_r1.clone()]);
    let idx = UOp::index().buffer(buf).indices(vec![idx0.clone(), UOp::invalid_marker()]).call().expect("INDEX");

    let inlined = rewritten(&pm_remove_bufferize(), &idx, &mut ());
    assert!(inlined.any_in_subtree(|n| Arc::ptr_eq(n, &idx0)), "the live index substitutes buf_r0");
    assert!(inlined.any_in_subtree(|n| Arc::ptr_eq(n, &buf_r1)), "the buffer range paired with Invalid must be kept");
    assert!(!inlined.any_in_subtree(UOp::is_invalid_marker), "Invalid must not be inlined");
}

// ============================================================================
// The traversal boundary: what `collect` stops at, counts, and walks past
// ============================================================================

#[test_case(|address: &Arc<UOp>| read_at(alu_param(0, 8), address) ; "one buffer")]
#[test_case(|address: &Arc<UOp>| {
    let shared = read_param(0, address);
    read_at(alu_param(1, 8).try_add(&shared).expect("add").try_add(&shared).expect("add").try_add(&shared).expect("add"), address)
} ; "a duplicated buffer read")]
#[test_case(|address: &Arc<UOp>| read_at(local_mstack(700, 4), address) ; "an MStack")]
#[test_case(|address: &Arc<UOp>| read_at(alu_param(0, 8).after(smallvec![read_param(1, address)]), address) ; "a buffer behind an AFTER")]
#[test_case(|address: &Arc<UOp>| read_at(alu_param(0, 8).after(smallvec![UOp::noop()]), address) ; "an ordered buffer")]
#[test_case(|address: &Arc<UOp>| read_at(stage(param_chain(3, address), vec![address.clone()]), address) ; "a GLOBAL STAGE")]
#[test_case(|address: &Arc<UOp>| {
    let local = stage_with(
        read_param(0, address).try_add(&read_param(1, address)).expect("add"),
        vec![address.clone()],
        BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Local, removable: true },
    );
    read_at(local, address)
} ; "a LOCAL STAGE")]
#[test_case(|address: &Arc<UOp>| {
    let stored = read_at(alu_param(0, 8).after(smallvec![UOp::noop()]), address).try_add(&read_param(1, address)).expect("add");
    read_at(alu_param(9, 8), address).store(stored)
} ; "a STORE in the compute cone")]
fn a_compute_cone_inlines(build: fn(&Arc<UOp>) -> Arc<UOp>) {
    let address = loop_range(8, 0);
    let idx = index_bufferize(build(&address), vec![address.clone()], vec![address]);
    assert!(inlines(&idx), "the index must inline its STAGE");
}

// ============================================================================
// Rule 2: STORE(x, x) → NOOP, Rule 3: END(NOOP, ..) → NOOP
// ============================================================================

/// Both NOOP collapses are rules of the same matcher and both rewrite the whole
/// node to NOOP.
#[test_case(|address: Arc<UOp>| {
    let idx = UOp::index().buffer(alu_param(0, 8)).indices(vec![address]).call().expect("INDEX");
    idx.store(idx.clone())
} ; "a store of its own index")]
#[test_case(|address: Arc<UOp>| UOp::noop().end(smallvec![address]) ; "an end of a noop")]
fn the_noop_collapses_rewrite_to_a_noop(build: fn(Arc<UOp>) -> Arc<UOp>) {
    let node = build(loop_range(8, 0));
    let folded = rewritten(&pm_remove_bufferize(), &node, &mut ());
    assert_op!(folded, Op::Noop);
}

/// A STORE to a different index keeps its value: the rule is pointer equality,
/// not "a STORE is a noop".
#[test]
fn a_store_to_a_different_index_survives() {
    let address = loop_range(8, 0);
    let first = read_param(0, &address);
    assert_kept(&first.store(read_param(1, &address)));
}
