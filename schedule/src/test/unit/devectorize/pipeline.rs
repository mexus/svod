//! End-to-end `devectorize()`.
use super::helpers::*;
use proptest::prelude::*;
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::InScopeRangesProperty;
use svod_ir::{AxisId, AxisType, Op, UOp, ops};
use test_case::test_case;
/// A shaped memory read of `n` lanes becomes `n` scalar LOADs under one STACK, whatever the offsets or element type.
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3]; "contiguous")]
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3, 4, 5, 6, 7]; "eight wide output upcast")]
#[test_case(ScalarDType::Float32, &[0, 2, 4, 6]; "strided")]
#[test_case(ScalarDType::Float32, &[3, 4, 5, 6]; "unaligned start")]
#[test_case(ScalarDType::Float32, &[0, 1, 2]; "three lanes")]
#[test_case(ScalarDType::Float32, &[0, 1, 2, 3, 4]; "five lanes")]
#[test_case(ScalarDType::Float32, &[9000, 9001, 9002, 9003]; "large offset")]
#[test_case(ScalarDType::Float16, &[0, 1, 2, 3]; "half precision")]
#[test_case(ScalarDType::Int8, &[0, 1, 2, 3]; "int8")]
#[test_case(ScalarDType::UInt8, &[0, 1, 2, 3]; "uint8")]
#[test_case(ScalarDType::Int32, &[0, 1, 2, 3]; "int32")]
fn shaped_load_becomes_one_scalar_load_per_lane(scalar: ScalarDType, offsets: &[i64]) {
    let address = shaped_addr(&buffer_of(16384, scalar), offsets.iter().copied());
    let result = apply_devectorize(&load(address));
    assert_vcount(&result, offsets.len());
    assert_eq!(loads(&result), offsets.len());
    let Op::Stack(ops::Stack { sources }) = result.op() else {
        panic!("expected a STACK of lanes:\n{}", result.tree())
    };
    assert_eq!(sources.len(), offsets.len());
    assert!(sources.iter().all(|lane| lane.dtype() == DType::Scalar(scalar)));
}
/// Wide vectors are scalarized the same way, without a width cap.
#[test_case(32; "vec32")]
#[test_case(64; "vec64")]
fn wide_shaped_load_is_fully_scalarized(width: usize) {
    let result = apply_devectorize(&load(iota_addr(&buffer(16384), width)));
    assert_vcount(&result, width);
    assert_eq!(loads(&result), width);
}
/// `c[0..4] = a[0..4] + b[0..4]` scalarizes on both sides, and no memory op keeps a vector dtype.
#[test]
fn shaped_elementwise_kernel_scalarizes_loads_and_stores() {
    let load4 = |buffer: Arc<UOp>| load(iota_addr(&buffer, 4));
    let sum = load4(buffer(64)).add(&load4(buffer(64)));
    let result = apply_devectorize(&store(iota_addr(&buffer(64), 4), sum));
    assert_eq!(loads(&result), 8);
    assert_eq!(stores(&result), 4);
    assert!(
        !result
            .toposort()
            .iter()
            .any(|node| { matches!(node.op(), Op::Load(..) | Op::Store(..)) && node.dtype().vcount() > 1 })
    );
}
#[test]
fn sink_scalarizes_every_shaped_store() {
    let store4 = |value| store(iota_addr(&buffer(64), 4), value);
    let sink = UOp::sink(vec![store4(float_values((0..4).map(|i| i as f64))), store4(float_values([9.0; 4]))]);
    assert_eq!(stores(&apply_devectorize(&sink)), 8);
}
/// A loop-dependent address (`range * 4 + lane`) scalarizes like a constant one.
#[test]
fn loop_dependent_shaped_load_is_scalarized() {
    let buffer = UOp::param(20000, 256, DType::Float32, None);
    let base = range(64, AxisType::Loop, 0).mul(&UOp::index_const(4));
    let offsets = UOp::stack((0..4).map(|lane| base.add(&UOp::index_const(lane))).collect());
    let index = UOp::new(Op::Index(ops::Index { buffer, indices: smallvec::smallvec![offsets] }), DType::Float32);
    let result = apply_devectorize(&load(index));
    assert_vcount(&result, 4);
    assert_eq!(loads(&result), 4);
}
/// A shaped STORE into a register file keeps every lane inside the enclosing loop.
#[test]
fn shaped_register_store_preserves_outer_range() {
    let outer = UOp::range_axis(UOp::index_const(4), AxisId::Unrenumbered(0), AxisType::Loop);
    let register = UOp::buffer(0, 2, DType::Float32, AddrSpace::Reg, None);
    let result = apply_devectorize(&register.after(vec![outer.clone()].into()).store(float_values([0.0, 0.0])));
    let stores: Vec<_> = result.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..))).collect();
    assert_eq!(stores.len(), 2);
    assert!(stores.iter().all(|store| InScopeRangesProperty::get(store).iter().any(|range| *range == outer.id)));
}
/// A memory address stays `INDEX(PARAM, flat_offset)` with a scalar offset; only a value-space INDEX (into a STACK)
/// keeps a shape.
#[test]
fn flat_2d_memory_index_and_shaped_value_index_remain_distinct() {
    let buffer = UOp::param(22000, 64, DType::Float32, None);
    let row = UOp::range_const(8, 22001);
    let row_offset = row.mul(&UOp::index_const(8));
    let offsets = UOp::stack((0..4).map(|lane| row_offset.add(&UOp::index_const(lane))).collect());
    let memory_index = UOp::index().buffer(buffer.clone()).indices(vec![offsets]).call().unwrap();
    let result = apply_devectorize(&load(memory_index));
    for node in result.toposort().into_iter().filter(|node| matches!(node.op(), Op::Index(..))) {
        let Op::Index(ops::Index { buffer: address, indices }) = node.op() else { unreachable!() };
        if address.addrspace().is_some() {
            assert!(
                Arc::ptr_eq(address, &buffer),
                "memory lane must remain INDEX(PARAM, flat_offset):\n{}",
                node.tree()
            );
            assert_eq!(indices.len(), 1);
            assert!(indices[0].shape().unwrap().unwrap().is_empty());
        }
    }
    let shaped = reshape_to(&UOp::stack((0i32..4).map(UOp::native_const).collect()), &[2, 2]);
    let shaped_index =
        UOp::index().buffer(shaped.clone()).indices(vec![row.mod_(&UOp::index_const(2))]).call().unwrap();
    assert!(shaped_index.addrspace().is_none());
    assert_eq!(shaped_index.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    assert!(matches!(shaped_index.op(), Op::Index(ops::Index { buffer: source, .. }) if Arc::ptr_eq(source, &shaped)));
}
/// A scalar access is already devectorized and must survive untouched.
#[test]
fn scalar_memory_ops_pass_through() {
    let address = index(buffer(64), 5);
    let devectorized = apply_devectorize(&address);
    assert_op!(devectorized, Op::Index(..));
    let result = apply_devectorize(&load(address));
    assert_op!(result, Op::Load(..));
    assert_eq!(result.dtype(), DType::Float32);
}
/// Devectorize runs tinygrad's `symbolic_simple` tier, which does not flatten SINK.
#[test]
fn sink_structure_is_preserved() {
    assert!(
        matches!(apply_devectorize(&UOp::sink(vec![])).op(), Op::Sink(ops::Sink { sources, .. }) if sources.is_empty())
    );
    let result = apply_devectorize(&UOp::sink(vec![UOp::noop()]));
    assert!(
        matches!(result.op(), Op::Sink(ops::Sink { sources, .. }) if sources.len() == 1 && matches!(sources[0].op(), Op::Noop)),
        "devectorize must not run the larger sym cleanup tier:\n{}",
        result.tree()
    );
}
fn scalar_dtype_strategy() -> impl Strategy<Value = ScalarDType> {
    prop_oneof![
        Just(ScalarDType::Float32),
        Just(ScalarDType::Float16),
        Just(ScalarDType::Int8),
        Just(ScalarDType::Int32),
        Just(ScalarDType::UInt8),
    ]
}
proptest! {
    #![proptest_config(cheap())]
    /// `devectorize` is a single `graph_rewrite` (tinygrad `codegen/__init__.py:333`), so it must reach a fixed point
    /// in one pass for every shaped access.
    #[test]
    fn devectorize_is_idempotent(
        offsets in prop::collection::vec(0i64..64, 1..8),
        dtype in scalar_dtype_strategy(),
        as_store in any::<bool>(),
    ) {
        let buffer = if as_store { buffer(4096) } else { buffer_of(4096, dtype) };
        let address = shaped_addr(&buffer, offsets.iter().copied());
        let root = if as_store { store(address, float_values((0..offsets.len()).map(|i| i as f64))) } else { load(address) };
        let once = apply_devectorize(&root);
        prop_assert!(Arc::ptr_eq(&apply_devectorize(&once), &once), "not idempotent:\n{}", once.tree());
    }
}
