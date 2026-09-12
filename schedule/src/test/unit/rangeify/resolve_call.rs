//! `resolve_calls`: FUNCTION inlining through PARAM substitution, and every
//! boundary that must stay opaque.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, CallInfo, Error, Op, UOp, ops};
use test_case::test_case;

use crate::rangeify::{rangeify, transforms::resolve_calls};
use crate::test::support::prelude::*;

/// FUNCTION bodies are TUPLE-wrapped value producers, so inlining yields the substituted TUPLE.
fn peel_tuple(uop: &Arc<UOp>) -> &Arc<UOp> {
    match uop.op() {
        Op::Tuple(ops::Tuple { src }) if src.len() == 1 => &src[0],
        _ => uop,
    }
}

fn no_function(uop: &Arc<UOp>) -> bool {
    !has_op(uop, |op| matches!(op, Op::Function(..)))
}

/// `body` as a FUNCTION over `args`, with the default (non-precompile) `CallInfo`.
fn function_of(body: &Arc<UOp>, args: SmallVec<[Arc<UOp>; 4]>) -> Arc<UOp> {
    body.function(args, CallInfo::default())
}

/// A `precompile` FUNCTION: `body` applied to `arg` must stay opaque.
fn precompile_function(body: &Arc<UOp>, arg: &Arc<UOp>) -> Arc<UOp> {
    body.function(smallvec![arg.clone()], CallInfo { precompile: true, ..CallInfo::default() })
}

fn add_of_params() -> Arc<UOp> {
    UOp::param(0, 8, DType::Float32, None).try_add(&UOp::param(1, 8, DType::Float32, None)).expect("add")
}

fn buffer8() -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32)
}

/// The direct entry point and the pipeline that embeds it must consume a FUNCTION alike.
fn resolve_entry(f: Arc<UOp>) -> Arc<UOp> {
    resolve_calls(f).expect("resolve_calls should succeed")
}

fn pipeline_entry(f: Arc<UOp>) -> Arc<UOp> {
    rangeify(f).expect("rangeify should succeed").0
}

/// The direct entry point and the pipeline that embeds it must consume a FUNCTION
/// alike. Only the direct resolver leaves the actuals in place — the pipeline
/// bufferizes them — so the operand identity is pinned per row rather than
/// skipped.
#[test_case(resolve_entry, true ; "resolve_calls")]
#[test_case(pipeline_entry, false ; "the rangeify pipeline")]
fn a_function_is_inlined_through_its_parameters(entry: fn(Arc<UOp>) -> Arc<UOp>, keeps_actuals: bool) {
    let (a0, a1) = (buffer8(), buffer8());
    let resolved = entry(function_of(&add_of_params(), smallvec![a0.clone(), a1.clone()]));
    let inlined = peel_tuple(&resolved);
    let Op::Binary(BinaryOp::Add, lhs, rhs) = inlined.op() else {
        panic!("expected the inlined add body, got {}", inlined.tree())
    };
    assert_eq!(
        Arc::ptr_eq(lhs, &a0) && Arc::ptr_eq(rhs, &a1),
        keeps_actuals,
        "the inlined operands are the actuals\n{}",
        inlined.tree()
    );
    assert!(lhs.dtype() == DType::Float32 && rhs.dtype() == DType::Float32);
    assert!(no_function(&resolved));
}

fn program_body_fixture() -> Arc<UOp> {
    let sink = UOp::sink(vec![]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    UOp::program(sink, info, None, None, None)
}

fn kernel_sink_body_fixture() -> Arc<UOp> {
    UOp::sink_with_info(vec![UOp::native_const(1.0f32)], svod_ir::KernelInfo::default())
}

/// A SINK whose metadata is an unrelated marker, so opacity cannot be keyed on `KernelInfo`.
fn sink_with_unrelated_metadata_fixture() -> Arc<UOp> {
    #[derive(Debug)]
    struct UnrelatedMarker;
    UOp::sink(vec![UOp::param(0, 8, DType::Float32, None)]).with_metadata(UnrelatedMarker)
}

fn nested_call_body_fixture() -> Arc<UOp> {
    UOp::native_const(1.0f32).call(smallvec![], CallInfo::default())
}

fn function_inside_a_call_body_fixture() -> Arc<UOp> {
    function_of(&add_of_params(), smallvec![buffer8(), buffer8()]).call(smallvec![], CallInfo::default())
}

/// PROGRAM/SINK are tinygrad's `_OPAQUE_CALL_BODIES` (`ops.py:933`); a value body is opaque once
/// CALL-wrapped, and a body that is itself a CALL stays nested.
#[test_case(add_of_params ; "arithmetic body")]
#[test_case(program_body_fixture ; "program body")]
#[test_case(kernel_sink_body_fixture ; "kernel sink body")]
#[test_case(sink_with_unrelated_metadata_fixture ; "sink carrying unrelated metadata")]
#[test_case(nested_call_body_fixture ; "nested call body")]
#[test_case(function_inside_a_call_body_fixture ; "function inside the call body")]
fn a_call_body_is_never_inlined(build: fn() -> Arc<UOp>) {
    let body = build();
    let args = smallvec![buffer8(), buffer8()];
    let call = body.clone().call(args.clone(), CallInfo::default());
    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    let ops::Call { body: resolved_body, args: resolved_args, .. } = unwrap_op!(resolved, Op::Call(c) => c);
    assert_same!(resolved_body, body);
    assert!(resolved_args.iter().zip(&args).all(|(a, b)| Arc::ptr_eq(a, b)));
}

/// A `precompile` FUNCTION is opaque: it survives with its TUPLE body and actuals intact.
#[test]
fn a_precompile_function_is_preserved_intact() {
    let arg = buffer8();
    let sqrt = UOp::param(0, 8, DType::Float32, None).try_sqrt().unwrap();
    let resolved = resolve_calls(precompile_function(&sqrt, &arg)).expect("resolve_calls should succeed");
    let ops::Function { body, args, info } = unwrap_op!(resolved, Op::Function(f) => f);
    let src = unwrap_op!(body, Op::Tuple(ops::Tuple { src }) => src);
    assert_eq!(src.len(), 1);
    assert!(matches!(src[0].op(), Op::Unary(..)));
    assert_eq!(args.len(), 1);
    assert_same!(args[0], arg);
    assert!(info.precompile);
}

/// The traversal stops at a precompile FUNCTION (`transforms.rs:116-117`): a
/// plain FUNCTION nested inside its implementation body is not resolved away.
#[test]
fn a_nested_function_under_a_precompile_function_is_left_unresolved() {
    let inner = function_of(&UOp::param(0, 8, DType::Float32, None), smallvec![buffer8()]);
    let outer = inner.function(smallvec![], CallInfo { precompile: true, ..CallInfo::default() });

    let resolved = resolve_calls(outer).expect("an opaque outer FUNCTION must preserve its body");

    let ops::Function { body, info, .. } = unwrap_op!(resolved, Op::Function(f) => f);
    assert!(info.precompile);
    assert!(
        has_op(body, |op| matches!(op, Op::Function(ops::Function { info, .. }) if !info.precompile)),
        "the nested FUNCTION must survive:\n{}",
        body.tree()
    );
}

#[test]
fn a_gettuple_over_a_precompile_function_is_preserved() {
    let actual = buffer8();
    let formal = UOp::param(0, 8, DType::Float32, None);
    let gettuple = precompile_function(&formal, &actual).try_gettuple(0).unwrap();
    let resolved = resolve_calls(gettuple).expect("a precompiled FUNCTION must remain opaque");
    let ops::GetTuple { src, index } = unwrap_op!(resolved, Op::GetTuple(g) => g);
    assert_eq!(*index, 0, "only element 0 is peeled");
    let ops::Function { args, info, .. } = unwrap_op!(src, Op::Function(f) => f);
    assert!(info.precompile);
    assert_eq!(args.len(), 1);
    assert_same!(args[0], actual);
}

/// Nothing to resolve: a non-TUPLE/non-FUNCTION source and a FUNCTION-free graph both come back identical.
#[test_case(UOp::native_const(1.0f32) ; "a bare constant")]
#[test_case(UOp::sink(vec![add_of_params()]) ; "a graph without functions")]
fn a_root_without_functions_is_returned_unchanged(root: Arc<UOp>) {
    assert_same!(resolve_calls(root.clone()).expect("resolve_calls should succeed"), root);
}

/// The peel itself: a GETTUPLE over a FUNCTION body TUPLE resolves to the selected element.
#[test]
fn a_gettuple_over_a_function_tuple_resolves_to_its_element() {
    let function = function_of(&UOp::param(0, 8, DType::Float32, None), smallvec![buffer8()]);
    let gettuple = function.try_gettuple(0).expect("GETTUPLE of a FUNCTION body TUPLE");
    let resolved = resolve_calls(gettuple).expect("resolve_calls should succeed");
    assert!(no_function(&resolved), "the FUNCTION is resolved away:\n{}", resolved.tree());
}

/// Tinygrad parity: BIND is value-producing, so a FUNCTION wrapping it inlines like any value
/// body; with no PARAMs the substitution is a no-op and the value is the TUPLE-wrapped body.
#[test]
fn a_bind_body_function_is_inlined() {
    let var = UOp::define_var("N".to_string(), 0, 32);
    let function = function_of(&var.bind(UOp::index_const(8)), smallvec![]);
    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    assert_op!(peel_tuple(&resolved), Op::Bind(..));
    assert!(no_function(&resolved));
}

/// Sparse formal slots are valid; non-contiguous ones leave the unused actuals untouched.
#[test]
fn a_non_contiguous_formal_slot_substitutes_only_the_slots_it_uses() {
    let body = UOp::param(0, 8, DType::Float32, None).try_add(&UOp::param(2, 8, DType::Float32, None)).unwrap();
    let (a0, a2) = (buffer8(), buffer8());
    let function = function_of(&body, smallvec![a0.clone(), buffer8(), a2.clone()]);
    let resolved = resolve_calls(function).expect("unused argument slots should be allowed");
    let nodes = resolved.toposort();
    assert!(!nodes.iter().any(|u| matches!(u.op(), Op::Param(..))));
    assert!(nodes.iter().any(|u| Arc::ptr_eq(u, &a0)));
    assert!(nodes.iter().any(|u| Arc::ptr_eq(u, &a2)));
}

/// A missing slot, a different extent and a different dtype are all typed errors.
#[test]
fn a_mismatched_actual_argument_is_a_typed_error() {
    let missing_slot = function_of(&add_of_params(), smallvec![buffer8()]);
    let sqrt = |buffer| function_of(&UOp::param(0, 8, DType::Float32, None).try_sqrt().unwrap(), smallvec![buffer]);
    let wrong_shape = sqrt(UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32));
    let wrong_dtype = sqrt(UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Int32));
    let err = |f| match resolve_calls(f) {
        Err(err) => err,
        Ok(resolved) => panic!("expected a typed error, got {}", resolved.tree()),
    };
    assert!(matches!(err(missing_slot), Error::CallFormalSlotMissing { slot: 1, arg_count: 1 }));
    assert!(matches!(err(wrong_shape), Error::CallArgShapeMismatch { arg_index: 0, .. }));
    assert!(matches!(err(wrong_dtype), Error::CallArgDTypeMismatch { arg_index: 0, .. }));
}

/// Resolving twice is the same as resolving once.
#[test]
fn resolving_is_idempotent() {
    let function = function_of(&add_of_params(), smallvec![buffer8(), buffer8()]);
    let once = resolve_calls(function).expect("first pass");
    let twice = resolve_calls(once.clone()).expect("second pass");
    assert_same!(once, twice);
}

#[test]
fn the_pipeline_substitutes_an_expression_valued_function_result_shape() {
    let p1 = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let extent = p1.try_add(&p1).unwrap();
    let formal = UOp::param_with_shape(0, &smallvec![svod_ir::SInt::Symbolic(extent)], DType::Float32, None);
    let actual_dim = UOp::define_var("actual".into(), 1, 8);
    let actual_extent = actual_dim.try_add(&actual_dim).unwrap();
    let actual =
        UOp::param_with_shape(7, &smallvec![svod_ir::SInt::Symbolic(actual_extent.clone())], DType::Float32, None);
    let output = function_of(&formal, smallvec![actual, actual_dim]).try_gettuple(0).unwrap();

    assert_eq!(output.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Symbolic(actual_extent)]);
    let (resolved, _) = rangeify(output).expect("rangeify should consume substituted call shape");
    assert!(no_function(&resolved));
}

#[test]
fn the_pipeline_preserves_a_kernel_call_body_boundary() {
    let detached = UOp::native_const(1.0f32).detach();
    let body = UOp::sink_with_info(vec![detached], svod_ir::KernelInfo::default());
    let function = body.call(smallvec![], CallInfo::default());
    let (out, _) = rangeify(function).expect("rangeify should succeed");
    let call_node = first_call(&out).expect("kernel call should be preserved");
    let ops::Call { body, .. } = unwrap_op!(call_node, Op::Call(c) => c);
    assert!(has_op(body, |op| matches!(op, Op::Detach(..))), "must not rewrite inside a preserved kernel call body");
}
