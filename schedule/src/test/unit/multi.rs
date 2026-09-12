use crate::multi::{lower_allreduce_pm, multi_pm, validate_no_unresolved_allreduce, validate_supported_subset};
use crate::optimizer::apply_pre_optimization;
use crate::rangeify::rangeify_with_map;
use crate::test::support::prelude::*;
use smallvec::smallvec;
use std::sync::Arc;
use svod_device::DeviceSpec;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{BinaryOp, Error, Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;
/// An 8-element buffer viewed as `[2, 4]`, so both axes can carry a layout.
fn matrix() -> Arc<UOp> {
    buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap()
}
fn sharded(axis: usize) -> Arc<UOp> {
    UOp::multi(matrix(), axis)
}
fn add(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Float32)
}
fn multi_rewrite(node: Arc<UOp>) -> Arc<UOp> {
    rewrite(&multi_pm(), node)
}
fn reduced_multi(axis: usize, op: ReduceOp) -> Arc<UOp> {
    UOp::multi(matrix(), axis).try_reduce_axis(op, vec![axis]).unwrap()
}
/// A shard-axis reduce over two explicit shards: everything the collective rewrite needs except a supported
/// `reduce_op`, so the `Add`/`Max` allowlist is the only thing that can reject it.
fn sharded_reduce(op: ReduceOp) -> Arc<UOp> {
    UOp::multi(UOp::mstack(smallvec![buffer(4), buffer(4)]), 0).try_reduce_axis(op, vec![0]).unwrap()
}
/// The per-shard `MULTI` that `multi_pm` produces, with its shard axis.
fn per_shard(result: &Arc<UOp>) -> (Arc<UOp>, usize) {
    let Op::Multi(ops::Multi { src, axis }) = result.op() else {
        panic!("expected a per-shard MULTI:\n{}", result.tree())
    };
    (src.clone(), *axis)
}
#[derive(Clone, Copy, Debug)]
enum MSelectStage {
    MultiPm,
    PreOptimization,
    Rangeify,
}
#[test_case(MSelectStage::MultiPm; "multi_pm selects the shard")]
#[test_case(MSelectStage::PreOptimization; "the per-kernel optimizer does not repeat multi_pm")]
#[test_case(MSelectStage::Rangeify; "rangeify resolves mselect before movement lowering")]
fn mselect_resolves_to_its_shard_through_each_stage(stage: MSelectStage) {
    let shard1 = buffer(6);
    let stacked = UOp::mstack(smallvec![buffer(6), shard1.clone()]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    match stage {
        MSelectStage::MultiPm => assert_same!(multi_rewrite(stacked.mselect(1)), shard1),
        MSelectStage::PreOptimization => {
            let result = apply_pre_optimization(reshaped.mselect(1)).unwrap();
            assert!(matches!(result.op(), Op::MSelect(..)), "the per-kernel optimizer must not rerun multi_pm");
        }
        MSelectStage::Rangeify => {
            let result = rangeify_with_map(UOp::sink(vec![reshaped.mselect(1)])).unwrap();
            assert!(result.uop_list.iter().any(|node| Arc::ptr_eq(node, &shard1)));
            assert!(!has_op(&result.sink, |op| matches!(op, Op::MSelect(..))));
        }
    }
}
#[test]
fn same_axis_alu_runs_per_shard() {
    let local0 = buffer(8);
    let local1 = buffer(8);
    let (src, axis) = per_shard(&multi_rewrite(add(UOp::multi(local0.clone(), 0), UOp::multi(local1.clone(), 0))));
    assert_eq!(axis, 0);
    assert!(matches!(src.op(), Op::Binary(BinaryOp::Add, a, b) if Arc::ptr_eq(a, &local0) && Arc::ptr_eq(b, &local1)));
}
#[test]
fn a_scalar_operand_needs_no_layout_of_its_own() {
    let local = buffer(8);
    let scalar = UOp::native_const(2.0f32);
    let result = multi_rewrite(add(UOp::multi(local.clone(), 0), scalar.clone()));
    let (src, axis) = per_shard(&result);
    assert_eq!(axis, 0);
    assert!(matches!(src.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if Arc::ptr_eq(lhs, &local) && Arc::ptr_eq(rhs, &scalar)));
    validate_supported_subset(&result).unwrap();
}
#[test]
fn permute_remaps_the_shard_axis() {
    let local = buffer(6).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let permute =
        UOp::new(Op::Permute(ops::Permute { src: UOp::multi(local.clone(), 0), axes: vec![1, 0] }), DType::Float32);
    let (src, axis) = per_shard(&multi_rewrite(permute));
    assert_eq!(axis, 1);
    assert!(
        matches!(src.op(), Op::Permute(ops::Permute { src: inner, axes }) if Arc::ptr_eq(inner, &local) && axes == &[1, 0])
    );
}
/// Non-shard-axis movement inside the shard runs per shard and keeps its layout.
#[test_case(true; "flip")]
#[test_case(false; "zero pad")]
fn movement_across_a_non_shard_axis_runs_per_shard(flip: bool) {
    let local = matrix();
    let root = if flip {
        UOp::new(Op::Flip(ops::Flip { src: UOp::multi(local.clone(), 0), axes: vec![false, true] }), DType::Float32)
    } else {
        UOp::new(
            Op::Pad(ops::Pad {
                src: UOp::multi(local.clone(), 0),
                begin_pads: stack([UOp::index_const(0), UOp::index_const(0)]),
                end_pads: stack([UOp::index_const(0), UOp::index_const(1)]),
            }),
            DType::Float32,
        )
    };
    let result = multi_rewrite(root.clone());
    let (src, axis) = per_shard(&result);
    assert_eq!(axis, 0);
    assert!(matches!(src.op(), Op::Flip(..) | Op::Pad(..)), "{}", result.tree());
    assert_eq!(src.op().sources().len(), root.op().sources().len());
    validate_supported_subset(&result).unwrap();
}
#[derive(Clone, Copy, Debug)]
enum Wrapper {
    Cast,
    BitCast,
    Contiguous,
    Detach,
    ContiguousBackward,
}
/// The dtype/contiguity wrappers are transparent to the shard boundary.
#[test_case(Wrapper::Cast; "cast")]
#[test_case(Wrapper::BitCast; "bitcast")]
#[test_case(Wrapper::Contiguous; "contiguous")]
#[test_case(Wrapper::Detach; "detach")]
#[test_case(Wrapper::ContiguousBackward; "contiguous backward")]
fn dtype_and_contiguity_wrappers_pass_through_the_shard_layout(wrapper: Wrapper) {
    let local = buffer(8);
    let sharded = UOp::multi(local.clone(), 0);
    let root = match wrapper {
        Wrapper::Cast => UOp::new(Op::Cast(ops::Cast { src: sharded, dtype: DType::Float16 }), DType::Float16),
        Wrapper::BitCast => UOp::new(Op::BitCast(ops::BitCast { src: sharded, dtype: DType::Int32 }), DType::Int32),
        Wrapper::Contiguous => {
            UOp::new(Op::Contiguous(ops::Contiguous { src: sharded, opts: Vec::new() }), DType::Float32)
        }
        Wrapper::Detach => sharded.detach(),
        Wrapper::ContiguousBackward => sharded.contiguous_backward(),
    };
    let dtype = root.dtype();
    let result = multi_rewrite(root);
    let (src, axis) = per_shard(&result);
    assert_eq!(axis, 0);
    assert!(Arc::ptr_eq(&src.op().sources()[0], &local), "{}", result.tree());
    assert_eq!(src.dtype(), dtype);
}
#[test]
fn a_reduce_over_another_axis_keeps_the_shard_layout() {
    let local = buffer(8);
    let reduce = UOp::multi(local.clone(), 1).reduce_with_num_axes(smallvec![], ReduceOp::Add, 1);
    let (src, axis) = per_shard(&multi_rewrite(reduce));
    assert_eq!(axis, 0);
    assert!(matches!(src.op(), Op::Reduce(ops::Reduce { src: inner, num_axes: 1, .. }) if Arc::ptr_eq(inner, &local)));
}
#[test]
fn a_non_sharded_reduce_axis_runs_per_shard_before_rangeify() {
    let local = matrix();
    let reduced = UOp::multi(local.clone(), 1).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = multi_rewrite(reduced.clone());
    let (src, axis) = per_shard(&rewritten);
    assert_eq!(axis, 0);
    assert!(matches!(src.op(), Op::Reduce(ops::Reduce { src: inner, ranges, num_axes: 1, .. })
        if Arc::ptr_eq(inner, &local) && ranges.is_empty()));
    validate_supported_subset(&rewritten).unwrap();
    let rangeified = rangeify_with_map(UOp::sink(vec![reduced])).unwrap();
    assert!(has_op(&rangeified.sink, |op| matches!(op, Op::Multi(ops::Multi { axis: 0, .. }))));
    assert!(!has_op(
        &rangeified.sink,
        |op| matches!(op, Op::Reduce(ops::Reduce { src, .. }) if matches!(src.op(), Op::Multi(..)))
    ));
}
/// Forms `multi_pm` must leave alone: single-device, missing resharding metadata, or a reduction shape the per-shard
/// rewrite cannot express.
#[test_case(add(UOp::multi(buffer(8), 0), UOp::multi(buffer(8), 1)); "mixed shard axes")]
#[test_case(UOp::new(Op::Reshape(ops::Reshape { src: UOp::multi(buffer(8), 0), new_shape: UOp::index_const(8) }), DType::Float32); "reshape without a shard count")]
#[test_case(UOp::mstack(smallvec![buffer(8), buffer(8)]).mselect(2); "mselect out of range")]
#[test_case(UOp::native_const(1i32).mselect(0); "mselect of a non-movement source")]
#[test_case(add(buffer(8), buffer(8)); "single-device graph")]
#[test_case(UOp::multi(buffer(8), 0).reduce_with_num_axes(smallvec![], ReduceOp::Add, 0); "tensor reduction without axes")]
#[test_case(UOp::multi(buffer(8), 0).reduce_with_num_axes(smallvec![reduce_range(4, 3)], ReduceOp::Add, 1); "reduction with explicit ranges")]
#[test_case(UOp::multi(UOp::mstack(smallvec![buffer(4)]), 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap(); "single-shard collective")]
#[test_case(heterogeneous_shard_reduce(); "shards of different dtypes")]
#[test_case(sharded_reduce(ReduceOp::Mul); "product is not a supported collective")]
fn multi_pm_leaves_unsupported_forms_alone(node: Arc<UOp>) {
    assert_same!(multi_rewrite(node.clone()), node);
}
fn heterogeneous_shard_reduce() -> Arc<UOp> {
    UOp::multi(UOp::mstack(smallvec![buffer(4), buffer_of(4, ScalarDType::Int32)]), 0)
        .try_reduce_axis(ReduceOp::Add, vec![0])
        .unwrap()
}
#[test_case(add(sharded(0), sharded(1)), |e| matches!(e, Error::MultiAxisMismatch { .. }); "mixed shard axes")]
#[test_case(UOp::multi(sharded(0), 0), |e| matches!(e, Error::MultiNested { .. }); "nested multi")]
#[test_case(
    UOp::new(Op::Reshape(ops::Reshape { src: sharded(0), new_shape: UOp::index_const(8) }), DType::Float32),
    |e| matches!(e, Error::MultiMovementUnsupported { operation: "RESHAPE", .. }); "reshape across the shard boundary")]
#[test_case(
    UOp::new(Op::Flip(ops::Flip { src: sharded(0), axes: vec![true, false] }), DType::Float32),
    |e| matches!(e, Error::MultiMovementUnsupported { operation: "FLIP", axis: 0, .. }); "flip of the shard axis")]
#[test_case(add(sharded(0), matrix()), |e| matches!(e, Error::MultiLayoutMissing { axis: 0, .. }); "operand without a layout")]
#[test_case(reduced_multi(0, ReduceOp::Add), |e| matches!(e, Error::MultiReductionAcrossShardAxis { axis: 0 }); "sum across the shard axis without explicit shards")]
#[test_case(sharded_reduce(ReduceOp::Mul), |e| matches!(e, Error::MultiReductionAcrossShardAxis { axis: 0 }); "product is not a supported collective")]
#[test_case(heterogeneous_shard_reduce(), |e| matches!(e, Error::MultiUnsupported { operation: "MULTI", reason, .. } if reason.contains("identical dtype and shape")); "shards of different dtypes")]
#[test_case(
    UOp::allreduce(UOp::mstack(smallvec![buffer_of(4, ScalarDType::Bool), buffer_of(4, ScalarDType::Bool)]), DeviceSpec::Cpu, ReduceOp::Add),
    |e| matches!(e, Error::MultiUnsupported { operation: "ALLREDUCE", reason: "host collective dtype is not supported" }); "unsupported collective dtype")]
fn rangeify_rejects_unsupported_multi_forms_with_typed_errors(node: Arc<UOp>, expected: fn(&Error) -> bool) {
    let err = rangeify_with_map(UOp::sink(vec![node])).err().expect("unsupported MULTI form");
    assert!(expected(&err), "unexpected error: {err:?}");
}
#[test]
fn rangeify_runs_multi_before_tagging() {
    let result = rangeify_with_map(UOp::sink(vec![add(UOp::multi(buffer(8), 0), UOp::multi(buffer(8), 0))])).unwrap();
    assert!(result.uop_list.iter().all(|node| {
        !matches!(node.op(), Op::Binary(..))
            || node.op().sources().iter().all(|source| !matches!(source.op(), Op::Multi(..)))
    }));
    assert!(has_op(&result.sink, |op| matches!(op, Op::Multi(..))));
}
/// Independent graph outputs may each have their own single-axis layout.
#[test]
fn independent_outputs_may_have_different_single_axis_layouts() {
    validate_supported_subset(&UOp::sink(vec![sharded(0), sharded(1)])).unwrap();
}
/// A reduction over the shard axis becomes a per-shard local reduce feeding one ALLREDUCE; a non-leading shard axis
/// is permuted to the front first.
#[test_case(0, ReduceOp::Add, false; "leading shard axis")]
#[test_case(1, ReduceOp::Add, true; "non-leading shard axis")]
#[test_case(0, ReduceOp::Max, false; "max collective")]
fn shard_axis_reduce_emits_local_reduce_then_allreduce(axis: usize, reduce_op: ReduceOp, permuted: bool) {
    let shard0 = matrix();
    let shard1 = matrix();
    let shards = UOp::mstack(smallvec![shard0.clone(), shard1.clone()]);
    let reduced = UOp::multi(shards, axis).try_reduce_axis(reduce_op, vec![axis]).unwrap();
    let rewritten = multi_rewrite(reduced.clone());
    let Op::AllReduce(ops::AllReduce { src, reduce_op: collective, .. }) = rewritten.op() else {
        panic!("expected ALLREDUCE, got {:?}", rewritten.op());
    };
    assert_eq!(collective, &reduce_op);
    let Op::MStack(ops::MStack { buffers }) = src.op() else { panic!("expected local reduction MSTACK") };
    assert_eq!(buffers.len(), 2);
    for (local, shard) in buffers.iter().zip([shard0, shard1]) {
        let Op::Reduce(ops::Reduce { src, ranges, num_axes: 1, .. }) = local.op() else {
            panic!("expected tensor REDUCE")
        };
        assert!(ranges.is_empty());
        if permuted {
            assert!(
                matches!(src.op(), Op::Permute(ops::Permute { src, axes }) if Arc::ptr_eq(src, &shard) && axes == &[1, 0])
            );
        } else {
            assert!(Arc::ptr_eq(src, &shard));
        }
    }
    validate_supported_subset(&rewritten).unwrap();
    let rangeified = rangeify_with_map(UOp::sink(vec![reduced])).unwrap();
    assert!(!has_op(&rangeified.sink, |op| matches!(op, Op::Reduce(ops::Reduce { num_axes, .. }) if *num_axes != 0)));
}
#[test]
fn reduced_precision_cast_is_restored_around_collective() {
    let low0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let low1 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let reduced = UOp::multi(UOp::mstack(smallvec![low0.cast(DType::Float32), low1.cast(DType::Float32)]), 0)
        .try_reduce_axis(ReduceOp::Add, vec![0])
        .unwrap();
    let rewritten = multi_rewrite(reduced);
    let Op::Cast(ops::Cast { src: collective, dtype: DType::Scalar(ScalarDType::Float32) }) = rewritten.op() else {
        panic!("expected widened result cast, got {:?}", rewritten.op());
    };
    let Op::AllReduce(ops::AllReduce { src, .. }) = collective.op() else { panic!("expected ALLREDUCE") };
    let Op::MStack(ops::MStack { buffers }) = src.op() else { panic!("expected MSTACK") };
    assert!(buffers.iter().all(|local| local.dtype() == DType::Float16));
}
/// The host collective loweres to an opaque call whose output aliases the first materialized shard, and no ALLREDUCE
/// survives codegen.
#[test]
fn allreduce_lowers_to_opaque_host_call_before_program_codegen() {
    let local0 = buffer(4).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let local1 = buffer(4).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let allreduce = UOp::allreduce(UOp::mstack(smallvec![local0, local1]), DeviceSpec::Cpu, ReduceOp::Add);
    let lowered = rewrite(&lower_allreduce_pm(), allreduce.clone());
    validate_no_unresolved_allreduce(&lowered).unwrap();
    let Op::After(ops::After { deps, .. }) = lowered.op() else { panic!("expected AFTER output") };
    let Op::Call(ops::Call { body, args, .. }) = deps[0].op() else { panic!("expected host collective CALL") };
    assert!(matches!(
        body.op(),
        Op::CustomFunction(ops::CustomFunction {
            kind: svod_ir::CustomFunctionKind::AllReduce { reduce_op: ReduceOp::Add },
            ..
        })
    ));
    assert_eq!(args.len(), 3, "output plus two explicit shard buffers");
    assert!(matches!(args[0].op(), Op::Contiguous(..)));
    assert!(Arc::ptr_eq(&args[0], &args[1]), "collective output must alias materialized shard zero");
    assert!(matches!(body.op(), Op::CustomFunction(ops::CustomFunction { attrs, .. }) if attrs.len() == args.len()));
    assert!(!has_op(&lowered, |op| matches!(op, Op::AllReduce(..))));
    let rangeified = rangeify_with_map(UOp::sink(vec![allreduce])).unwrap();
    assert!(!has_op(&rangeified.sink, |op| matches!(op, Op::AllReduce(..))));
    assert!(has_op(&rangeified.sink, |op| matches!(
        op,
        Op::CustomFunction(ops::CustomFunction { kind: svod_ir::CustomFunctionKind::AllReduce { .. }, .. })
    )));
}
