//! Kernel counts the rangeify + kernel-split pipeline produces for the graph
//! shapes fusion has to get right. Every row is materialised with CONTIGUOUS so
//! the fusion decision — not the sink — decides the kernel boundary.

use std::sync::Arc;

use svod_ir::{DType, Op, ReduceOp, UOp, ops};
use test_case::test_case;

use crate::rangeify::{rangeify_with_map, try_get_kernel_graph};
use crate::test::support::prelude::*;

fn reshaped(src: Arc<UOp>, rows: i64, cols: i64) -> Arc<UOp> {
    let new_shape = stack([UOp::index_const(rows), UOp::index_const(cols)]);
    UOp::new(Op::Reshape(ops::Reshape { src, new_shape }), DType::Float32)
}

fn add() -> Arc<UOp> {
    buffer(100).try_add(&buffer(100)).expect("add")
}

#[test_case(|| UOp::sink(vec![add()]), 1 ; "a + b is one kernel")]
#[test_case(|| UOp::sink(vec![add().try_add(&buffer(100)).expect("add")]), 1 ; "a + b + c is one kernel")]
#[test_case(|| UOp::sink(vec![reshaped(add(), 10, 10)]), 1 ; "reshape does not break fusion")]
#[test_case(|| UOp::sink(vec![UOp::new(Op::Permute(ops::Permute { src: reshaped(add(), 10, 10), axes: vec![1, 0] }), DType::Float32)]), 1 ; "permute does not break fusion")]
#[test_case(|| UOp::sink(vec![reshaped(buffer(100), 10, 10).try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce")]), 1 ; "reduce is one kernel")]
#[test_case(|| UOp::sink(vec![reshaped(add(), 10, 10).try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce")]), 1 ; "elementwise fuses into the reduce")]
#[test_case(|| UOp::sink(vec![add().contiguous().try_mul(&buffer(100)).expect("mul")]), 2 ; "contiguous forces a second kernel")]
#[test_case(|| { let shared = add(); UOp::sink(vec![shared.try_mul(&buffer(100)).expect("mul"), shared.try_mul(&buffer(100)).expect("mul")]) }, 2 ; "shared add is inlined into both outputs")]
#[test_case(|| UOp::sink(vec![UOp::native_const(1.0f32)]), 1 ; "a bare const still writes its output")]
#[test_case(|| UOp::sink(vec![]), 0 ; "empty sink launches nothing")]
fn kernel_count(build: fn() -> Arc<UOp>, expected: usize) {
    let built = build();
    let sources = expect_sink(&built);
    let root = UOp::sink(sources.iter().map(|source| source.contiguous()).collect());
    let rangeified = rangeify_with_map(root).expect("rangeify");
    let (graph, _) = try_get_kernel_graph(rangeified.sink).expect("kernel graph");
    assert_eq!(kernels(&graph), expected);
}
