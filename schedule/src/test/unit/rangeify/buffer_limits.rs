//! Device buffer-limit enforcement: when a kernel would bind more buffers than
//! the device allows, elementwise sub-expressions are materialised.
//!
//! The limit is passed to `buffer_limit_patterns` explicitly, so every row runs
//! on CPU without a device feature gate.

use std::sync::Arc;

use svod_device::DeviceSpec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, BufferizeOpts, Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;

use super::helpers::assert_same_ptr;
use super::helpers::{count_stages, loop_range};
use crate::rangeify::indexing::IndexingContext;
use crate::rangeify::patterns::{buffer_limit_patterns, extract_device_from_graph, is_elementwise};
use crate::rewrite::graph_rewrite;

/// A 40-element GLOBAL BUFFER with an explicit slot; the shape of every read.
fn global_at(slot: usize) -> Arc<UOp> {
    UOp::buffer(slot, 40, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu))
}

/// A codegen PARAM with an explicit slot.
fn storage_param(slot: usize) -> Arc<UOp> {
    UOp::param(slot, 40, DType::Float32, Some(DeviceSpec::Cpu))
}

fn read(storage: Arc<UOp>, address: &Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(storage).indices(vec![address.clone()]).call().expect("INDEX")
}

/// `(((s0 + s1) + s2) + ...)` over `count` distinct storages built by `storage`.
fn chain_over(count: usize, address: &Arc<UOp>, storage: impl Fn(usize) -> Arc<UOp>) -> Arc<UOp> {
    (1..count).fold(read(storage(0), address), |acc, slot| acc.try_add(&read(storage(slot), address)).expect("add"))
}

/// The first STAGE's ranges; panics when nothing was materialised.
fn stage_ranges(structure: &Arc<UOp>) -> smallvec::SmallVec<[Arc<UOp>; 4]> {
    structure
        .toposort()
        .into_iter()
        .find_map(|u| match u.op() {
            Op::Stage(ops::Stage { ranges, .. }) => Some(ranges.clone()),
            _ => None,
        })
        .expect("buffer limit should materialize an operand")
}

/// A fresh LOOP range of constant extent 10.
fn loop_10(ctx: &mut IndexingContext) -> Arc<UOp> {
    ctx.new_range(&SInt::Const(10), AxisType::Loop)
}

/// The output buffer takes one slot, so a limit of 31 means at most 30 inputs
/// before an elementwise operand has to be materialised. BUFFERs and PARAMs cost
/// the same: model weights arrive as PARAMs. The verdict is pinned in both
/// directions — below the boundary the whole graph must come back untouched.
#[test_case(global_at, 30, false ; "global buffers one below the limit")]
#[test_case(global_at, 31, true ; "global buffers at the limit")]
#[test_case(global_at, 35, true ; "global buffers well over the limit")]
#[test_case(storage_param, 30, false ; "codegen params one below the limit")]
#[test_case(storage_param, 31, true ; "codegen params at the limit")]
#[test_case(storage_param, 35, true ; "codegen params well over the limit")]
fn a_storage_input_costs_an_argument_slot(factory: fn(usize) -> Arc<UOp>, inputs: usize, materializes: bool) {
    let mut ctx = IndexingContext::new();
    let address = loop_10(&mut ctx);
    let computation = chain_over(inputs, &address, factory);

    let result = graph_rewrite(&buffer_limit_patterns(31), computation.clone(), &mut ctx);

    assert_eq!(count_stages(&result) > count_stages(&computation), materializes, "{inputs} inputs");
    if !materializes {
        assert_same_ptr(&result, &computation);
    }
}

/// A WHERE's condition, true and false arms all count toward the same limit.
#[test]
fn a_ternary_operand_tree_is_materialized_too() {
    let mut ctx = IndexingContext::new();
    let address = loop_10(&mut ctx);

    let mut cond = read(global_at(1), &address).try_cmplt(&read(global_at(0), &address)).expect("cmplt");
    for slot in (2..10).step_by(2) {
        let cmp = read(global_at(slot + 1), &address).try_cmplt(&read(global_at(slot), &address)).expect("cmplt");
        cond = cond.try_and_op(&cmp).expect("and");
    }
    let on_false = read(global_at(11), &address).try_add(&read(global_at(12), &address)).expect("add");
    let where_op = UOp::try_where(cond, read(global_at(10), &address), on_false).expect("where");

    let result = graph_rewrite(&buffer_limit_patterns(10), where_op.clone(), &mut ctx);
    assert!(count_stages(&result) > count_stages(&where_op));
}

/// Already-materialised operands are not re-staged. The STAGE has to be GLOBAL:
/// `collect_accessed_buffers` only counts GLOBAL ones, so a LOCAL STAGE would
/// pass by being invisible to the limit rather than by being skipped.
#[test]
fn an_existing_stage_is_not_materialized_again() {
    let mut ctx = IndexingContext::new();
    let ranges = vec![loop_10(&mut ctx)];

    let staged = UOp::stage_global(read(global_at(0), &ranges[0]), ranges.clone());
    let indexed = UOp::index().buffer(staged).indices(ranges).call().expect("INDEX");

    let result = graph_rewrite(&buffer_limit_patterns(31), indexed.clone(), &mut ctx);
    assert_eq!(count_stages(&result), count_stages(&indexed));
}

// ===== range-id allocation for the ranges the new STAGE carries =====

/// A range created during indexing and then collapsed by dead-axis cleanup still
/// consumed its id: the STAGE must use the next id, not reuse the collapsed one.
#[test]
fn a_collapsed_range_still_consumes_its_axis_id() {
    let mut ctx = IndexingContext::new();
    let visible = ctx.new_range(&SInt::Const(10), AxisType::Weak);
    let _collapsed = ctx.new_range_from_uop(&UOp::index_const(1), AxisType::Weak);

    let result = graph_rewrite(&buffer_limit_patterns(3), chain_over(3, &visible, global_at), &mut ctx);

    let stage_ids: Vec<_> = stage_ranges(&result)
        .iter()
        .filter_map(|r| match r.op() {
            Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(stage_ids, vec![AxisId::Unrenumbered(2)]);
    assert_eq!(ctx.range_counter(), 3);
}

/// A DEVICE range is a launch lane, not an allocated axis: the STAGE keeps it
/// verbatim alongside the freshly numbered WEAK range.
#[test]
fn a_device_range_is_carried_through_without_renumbering() {
    let mut ctx = IndexingContext::new();
    let weak = ctx.new_range(&SInt::Const(10), AxisType::Weak);
    let _collapsed = ctx.new_range_from_uop(&UOp::index_const(1), AxisType::Weak);
    let launched = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(7), AxisType::Device);

    let mixed = read(global_at(0), &weak).try_add(&read(global_at(1), &launched)).expect("add");
    let root = mixed.try_add(&read(global_at(2), &weak)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(3), root, &mut ctx);
    let stage_ranges = stage_ranges(&result);

    assert!(stage_ranges.iter().any(|r| Arc::ptr_eq(r, &launched)));
    assert!(stage_ranges.iter().any(|r| {
        matches!(r.op(), Op::Range(ops::Range { axis_id: AxisId::Unrenumbered(2), axis_type: AxisType::Weak, .. }))
    }));
    assert_eq!(ctx.range_counter(), 3);
}

// ===== helpers the pattern is built on =====

/// Only elementwise nodes are candidates for materialisation — a leaf has
/// nothing to materialise into. The set is tinygrad's `GroupOp.Elementwise`:
/// unary, binary and ternary ALU plus CAST and BITCAST.
#[test]
fn alu_and_cast_nodes_are_elementwise() {
    let (a, b) = (UOp::native_const(1.0f32), UOp::native_const(2.0f32));
    assert!(is_elementwise(&a.try_add(&b).expect("add")));
    assert!(is_elementwise(&UOp::try_where(UOp::native_const(true), a.clone(), b.clone()).expect("where")));
    assert!(is_elementwise(&a.try_sqrt().expect("sqrt")));
    assert!(is_elementwise(&a.cast(DType::Float64)));
    assert!(is_elementwise(&a.bitcast(DType::UInt32)));
    assert!(is_elementwise(&a.neg()));
    assert!(!is_elementwise(&a));
    assert!(!is_elementwise(&global_at(1)));
}

/// The device probe reads the first device off a BUFFER, a COPY or an ALLREDUCE
/// target, or a STAGE's own options.
#[test]
fn the_device_comes_from_the_storage_or_the_collective() {
    assert_eq!(extract_device_from_graph(&global_at(1)), Some(DeviceSpec::Cpu));
    assert_eq!(
        extract_device_from_graph(&UOp::native_const(1.0f32).copy_to_device(DeviceSpec::Cpu)),
        Some(DeviceSpec::Cpu)
    );

    let staged = UOp::stage(
        UOp::native_const(1.0f32),
        vec![loop_range(4, 0)],
        BufferizeOpts { device: Some(DeviceSpec::Amd { device_id: 0 }), ..BufferizeOpts::local() },
    );
    assert_eq!(extract_device_from_graph(&staged), Some(DeviceSpec::Amd { device_id: 0 }));

    let reduced = UOp::allreduce(UOp::native_const(1.0f32), DeviceSpec::Cuda { device_id: 0 }, ReduceOp::Add);
    assert_eq!(extract_device_from_graph(&reduced), Some(DeviceSpec::Cuda { device_id: 0 }));

    assert_eq!(extract_device_from_graph(&UOp::native_const(1.0f32)), None);
}

// ===== what counts as a kernel argument =====

/// LOCAL storage is compiler-managed: it lives inside the kernel and never binds
/// an argument, so no amount of it can trip the limit.
#[test]
fn local_storage_does_not_consume_an_argument_slot() {
    let mut ctx = IndexingContext::new();
    let address = loop_10(&mut ctx);
    let computation = chain_over(40, &address, |slot| UOp::buffer(slot, 40, DType::Float32, AddrSpace::Local, None));

    let result = graph_rewrite(&buffer_limit_patterns(31), computation.clone(), &mut ctx);
    assert_same_ptr(&result, &computation);
}

/// AFTER is a buffer identity: the kernels it orders against write the buffer,
/// they are not read by this one. Walking into its dependencies counted a whole
/// producer cone against a kernel that binds a single argument.
#[test]
fn an_after_costs_one_argument_not_its_producer_cone() {
    let mut ctx = IndexingContext::new();
    let address = loop_10(&mut ctx);

    let deps: smallvec::SmallVec<[Arc<UOp>; 4]> = (2..42).map(|slot| read(global_at(slot), &address)).collect();
    let ordered = read(global_at(0).after(deps), &address);
    let root = ordered.try_add(&read(global_at(1), &address)).expect("add").try_add(&ordered).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);
    assert_same_ptr(&result, &root);
}

/// `GroupOp.Elementwise` is ALU plus the casts (tinygrad `uop/__init__.py:112`),
/// so a CAST operand is a materialisation candidate like any binary one.
#[test]
fn a_cast_operand_is_materialized() {
    let mut ctx = IndexingContext::new();
    let address = loop_10(&mut ctx);

    let wide = chain_over(30, &address, global_at).cast(DType::Float64);
    let root = wide.try_add(&read(global_at(30), &address).cast(DType::Float64)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);
    assert!(count_stages(&result) > count_stages(&root));
}

/// A closed reduce range is not an axis of the new STAGE: putting one on made
/// the range substitution rebuild every producer that binds it, so copies
/// compounded through a chain of bufferized kernels until the graph exploded.
#[test]
fn a_closed_reduce_range_is_not_an_axis_of_the_new_stage() {
    let mut ctx = IndexingContext::new();
    let outer = loop_10(&mut ctx);
    let inner = ctx.new_range(&SInt::Const(4), AxisType::Reduce);

    let reduced = chain_over(30, &inner, global_at).reduce(smallvec::smallvec![inner.clone()], ReduceOp::Add);
    let over_limit = reduced.try_add(&read(global_at(30), &outer)).expect("add");
    let root = over_limit.try_add(&read(global_at(31), &outer)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);

    let stage_ranges = stage_ranges(&result);
    assert_eq!(stage_ranges.len(), 1, "only the open outer range is an axis, got {stage_ranges:?}");
    assert!(stage_ranges.iter().all(|r| !Arc::ptr_eq(r, &inner)));
    // The REDUCE keeps the axis it closes: it was never substituted.
    assert!(result.toposort().iter().any(|u| Arc::ptr_eq(u, &inner)));
}
