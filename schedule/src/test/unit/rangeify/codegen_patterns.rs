//! `rangeify_codegen_patterns`: CONTIGUOUS stripping, the hints it carries into
//! `LocalAddBufferContext::opts`, and the NOOP → zero materialisation.
//!
//! Mirrors tinygrad's `test_rangeify.py` `Tensor.contiguous(arg=(Opt(...),))`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, ContiguousHint, Op, UOp, ops};
use test_case::test_case;

use crate::rangeify::kernel::LocalAddBufferContext;
use crate::rangeify::patterns::rangeify_codegen_patterns;
use crate::test::support::prelude::*;

fn apply(uop: Arc<UOp>) -> (Arc<UOp>, LocalAddBufferContext) {
    let mut ctx = LocalAddBufferContext::new();
    let result = rewrite_with(&rangeify_codegen_patterns(), &mut ctx, uop);
    (result, ctx)
}

fn hint(op: &str, axis: Option<usize>, arg: Option<i64>) -> ContiguousHint {
    ContiguousHint { op: op.to_string(), axis, arg }
}

/// The CONTIGUOUS marker is a scheduling instruction, not a value: it is
/// stripped and its source returned. Void NOOPs and plain values are not touched.
#[test]
fn contiguous_is_stripped_and_plain_values_are_left_alone() {
    let tensor = UOp::native_const(42.0f32);
    let opts = vec![hint("LOCAL", Some(2), Some(8))];
    for wrapped in [tensor.clone().contiguous(), tensor.clone().contiguous_with_opts(opts)] {
        assert!(Arc::ptr_eq(&apply(wrapped).0, &tensor));
    }

    for untouched in [UOp::noop(), UOp::native_const(1.0f32)] {
        assert_same!(apply(untouched.clone()).0, untouched);
    }
}

/// A NOOP carrying a real dtype is a value hole: it lowers to a zero of its own
/// dtype, scalar or vector.
#[test_case(DType::Int32 ; "scalar integer noop")]
#[test_case(DType::Float32 ; "scalar float noop")]
#[test_case(DType::Bool ; "bool noop")]
fn a_non_void_noop_lowers_to_a_scalar_zero(dtype: DType) {
    let noop = UOp::new(Op::Noop, dtype.clone());
    let (result, _) = apply(noop);

    assert_eq!(result.dtype(), dtype);
    assert_const!(result, ConstValue::zero(dtype.base()));
}

#[test]
fn a_vector_noop_lowers_to_a_stack_of_scalar_zeros() {
    let dtype = DType::Int32.vec(4).expect("vector dtype");
    let (result, _) = apply(UOp::new(Op::Noop, dtype.clone()));

    let Op::Stack(ops::Stack { sources }) = result.op() else {
        panic!("a vector noop must materialise as a STACK, got {}", result.tree())
    };
    assert_eq!(sources.len(), 4);
    assert!(sources.iter().all(|lane| lane.dtype() == DType::Int32));
    assert!(sources.iter().all(|lane| matches!(lane.op(), Op::Const(c) if c.0 == ConstValue::Int(0))));
}

fn build_no_hints() -> Vec<ContiguousHint> {
    Vec::new()
}

fn build_one_hint() -> Vec<ContiguousHint> {
    vec![hint("UPCAST", Some(0), Some(4))]
}

/// An opt with no axis, e.g. NOLOCALS.
fn build_axisless_hint() -> Vec<ContiguousHint> {
    vec![hint("NOLOCALS", None, None)]
}

fn build_mixed_hints() -> Vec<ContiguousHint> {
    vec![hint("UPCAST", Some(0), Some(4)), hint("UNROLL", Some(1), Some(4))]
}

/// tinygrad's `test_upcast_01_unroll_01`.
fn build_four_hints() -> Vec<ContiguousHint> {
    vec![
        hint("UPCAST", Some(0), Some(4)),
        hint("UPCAST", Some(1), Some(4)),
        hint("UNROLL", Some(0), Some(4)),
        hint("UNROLL", Some(1), Some(4)),
    ]
}

/// Every hint reaches `ctx.opts` verbatim and in order.
#[test_case(build_no_hints ; "no hints")]
#[test_case(build_one_hint ; "one hint")]
#[test_case(build_axisless_hint ; "hint without an axis")]
#[test_case(build_mixed_hints ; "upcast and unroll")]
#[test_case(build_four_hints ; "four hints")]
fn contiguous_hints_are_collected_in_order(build: fn() -> Vec<ContiguousHint>) {
    let hints = build();
    let (_result, ctx) = apply(UOp::native_const(1.0f32).contiguous_with_opts(hints.clone()));
    assert_eq!(ctx.opts.as_slice(), hints.as_slice());
}
