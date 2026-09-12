//! `buffer_folding`: noop STAGE removal, constant propagation through
//! STAGE / INDEX / COPY / MSTACK, and the SLICE guard on the noop rule.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{ConstValue, Op, UOp, ops};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::rangeify::patterns::buffer_folding;
use crate::test::support::prelude::*;

fn fold(root: Arc<UOp>) -> Arc<UOp> {
    rewrite(&buffer_folding(), root)
}

fn range(end: i64, axis_id: usize) -> Arc<UOp> {
    global_range(end, axis_id)
}

/// `INDEX(STAGE(x, R), R) → x` — the buffer would be read back at exactly the
/// coordinates it was written at, so it is a noop.
#[test]
fn a_stage_read_at_its_own_ranges_folds_away() {
    let x = param(1, 10, DType::Float32);
    let r = range(10, 0);

    let staged = stage(Arc::clone(&x), vec![r.clone()]);
    let result = fold(index_of(staged, r));

    assert_same!(result, x);
}

/// With several ranges the fold still removes the STAGE, but the index has to be
/// relinearised, so the result is a view of `x` rather than `x` itself.
#[test]
fn a_multi_range_noop_stage_is_removed_but_reindexed() {
    let x = param(1, 200, DType::Float32);
    let ranges = vec![range(10, 0), range(20, 1)];

    let staged = stage(Arc::clone(&x), ranges.clone());
    let result = fold(UOp::index().buffer(staged).indices(ranges).call().expect("index"));

    assert!(!has_op(&result, |op| matches!(op, Op::Stage(..))), "{}", result.tree());
    assert!(result.toposort().iter().any(|node| Arc::ptr_eq(node, &x)), "{}", result.tree());
}

/// A STAGE read at other ranges is kept, and a STAGE whose compute is a SLICE is
/// a real copy: the source buffer is a window into storage that is not shaped
/// like the STAGE, so the noop rule must decline.
#[test]
fn stages_that_are_not_noops_are_kept() {
    let staged = stage(param(1, 1024, DType::Float32), vec![range(10, 0)]);
    let indexed = index_of(staged, range(10, 1));
    assert_same!(fold(indexed.clone()), indexed);

    let r = range(10, 0);
    let slice = buffer(8).contiguous_slice(4, 0, DType::Float32);
    let indexed = index_of(stage(slice, vec![r.clone()]), r);
    assert_same!(fold(indexed.clone()), indexed);
}

/// The noop fold is structural — it does not care what the staged compute is.
#[test]
fn the_noop_fold_applies_to_arbitrary_compute() {
    let compute = UOp::var("x", DType::Float32, 0, 100).try_add(&UOp::var("y", DType::Float32, 0, 100)).expect("add");
    let r = range(10, 0);

    let staged = stage(Arc::clone(&compute), vec![r.clone()]);
    assert_same!(fold(index_of(staged.clone(), r.clone())), compute);
}

fn staged(c: Arc<UOp>) -> Arc<UOp> {
    stage(c, vec![range(10, 0)])
}

fn indexed(c: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(c).indices(vec![range(10, 0), range(20, 1)]).call().expect("index")
}

fn copied(c: Arc<UOp>) -> Arc<UOp> {
    c.copy(DeviceSpec::Cuda { device_id: 0 })
}

fn staged_then_indexed(c: Arc<UOp>) -> Arc<UOp> {
    let r = range(15, 0);
    index_of(stage(c, vec![r.clone()]), r)
}

/// A constant has no storage to allocate, index into, or transfer: every wrapper
/// folds straight back to it.
#[test_case(staged, DType::Int32, ConstValue::Int(42) ; "stage of int")]
#[test_case(staged, DType::Bool, ConstValue::Bool(true) ; "stage of bool")]
#[test_case(staged, DType::Float32, ConstValue::Float(std::f64::consts::PI) ; "stage of float")]
#[test_case(indexed, DType::Float32, ConstValue::Float(2.5) ; "index of const")]
#[test_case(copied, DType::Int32, ConstValue::Int(99) ; "copy of const")]
#[test_case(staged_then_indexed, DType::Int32, ConstValue::Int(123) ; "index of stage of const")]
fn constants_fold_out_of_every_wrapper(wrap: fn(Arc<UOp>) -> Arc<UOp>, dtype: DType, value: ConstValue) {
    let c = UOp::const_(dtype, value);
    assert_same!(fold(wrap(Arc::clone(&c))), c);
}

/// A multi-device stack of constants is constant too: the INDEX reads the first
/// buffer, which is a CONST, so nothing is left to allocate.
#[test]
fn an_mstack_of_constants_folds() {
    let c = UOp::native_const(7i32);
    let stacked = UOp::new(Op::MStack(ops::MStack { buffers: smallvec::smallvec![c.clone(), UOp::noop()] }), c.dtype());

    let result = fold(index_of(stacked, range(10, 0)));

    assert!(!has_op(&result, |op| matches!(op, Op::MStack(..))), "{}", result.tree());
    assert_const!(result, 7);
}

#[test]
fn buffer_folding_leaves_unrelated_nodes_alone() {
    let c = UOp::native_const(1.0f32);
    assert!(matches!(buffer_folding().rewrite(&c, &mut ()), RewriteResult::NoMatch));
    assert_const!(fold(c.clone()), 1.0f32);
}
