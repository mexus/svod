//! Rule-level tests for Tinygrad's `spec_program` at pinned commit 8c8b43de. `spec_tensor` is used where useful to
//! prove that a rejection comes from the program rule rather than an inherited `spec_shared` rule.
use crate::optimizer::apply_pre_optimization;
use crate::spec::{SpecError, spec_hcq, spec_tensor, type_verify, verify_kernel_graph, verify_no_legacy_index_dtype};
use crate::test::support::prelude::*;
use crate::test::unit::devectorize::helpers::{apply_spec_program, codegen_param};
use smallvec::smallvec;
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, CallInfo, ConstValueHash, Op, ParamArg, ReduceOp, UOp, ops};
use test_case::test_case;
fn global_param(slot: usize) -> Arc<UOp> {
    codegen_param(slot, DType::Float32, AddrSpace::Global, None)
}
fn int_const(dtype: DType, value: i64) -> Arc<UOp> {
    UOp::const_(dtype, ConstValue::Int(value))
}
fn integer_const(dtype: DType, value: u64) -> Arc<UOp> {
    let constant = if dtype.is_unsigned() { ConstValue::UInt(value) } else { ConstValue::Int(value as i64) };
    UOp::const_(dtype, constant)
}
fn program_err(root: &Arc<UOp>) -> String {
    apply_spec_program(root).expect_err("expected spec_program rejection").to_string()
}
fn structured_buffer(addrspace: AddrSpace, device: Option<DeviceSpec>) -> Arc<UOp> {
    UOp::new(
        Op::Buffer(ops::Buffer {
            shape: int_const(DType::Int32, 4),
            arg: ParamArg::buffer(3, DType::Float32, addrspace, device).into(),
        }),
        DType::Float32,
    )
}
fn float_const(dtype: DType, value: f64) -> Arc<UOp> {
    UOp::new(Op::Const(ConstValueHash(ConstValue::Float(value))), dtype)
}
fn float_vconst(dtype: DType, values: [f64; 2]) -> Arc<UOp> {
    UOp::new(
        Op::VConst(ops::VConst { values: values.into_iter().map(ConstValue::Float).collect() }),
        dtype.vec(2).expect("two-lane vector"),
    )
}
fn special(dtype: DType, end: i64) -> Arc<UOp> {
    UOp::new(Op::Special(ops::Special { end: int_const(dtype.clone(), end), name: "lidx0".to_string() }), dtype)
}
fn shaped_stack(dtype: DType) -> Arc<UOp> {
    UOp::stack((0..4).map(|value| UOp::const_(dtype.clone(), ConstValue::Float(value as f64))).collect())
}
fn if_over(dedup_source: Arc<UOp>) -> Arc<UOp> {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    UOp::new(Op::If(ops::If { condition, body: smallvec![dedup_source] }), DType::Void)
}
fn scalar_index() -> Arc<UOp> {
    UOp::index().buffer(global_param(0)).indices(vec![int_const(DType::Int32, 0)]).call().unwrap()
}
/// The `END(END(x, range), backedge)` shape that `split_ends` leaves behind.
fn split_ends_backedge() -> Arc<UOp> {
    let range = UOp::range_axis_dtype(
        int_const(DType::Int32, 4),
        svod_ir::AxisId::Renumbered(0),
        svod_ir::types::AxisType::Loop,
        DType::Int32,
    );
    UOp::noop().end(smallvec![range]).end(smallvec![UOp::const_(DType::Bool, ConstValue::Bool(true))])
}
/// A `16x16x16` WMMA over constants: a program may not carry `DefineVar` operands.
fn const_stack_wmma() -> Arc<UOp> {
    let lanes = || UOp::stack((0..6).map(|value| UOp::native_const(value as f32)).collect());
    UOp::wmma(lanes(), lanes(), lanes(), crate::test::unit::devectorize::helpers::wmma_metadata("acc", None))
}
fn raw_index(index: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Index(ops::Index { buffer: global_param(0), indices: smallvec![index] }), DType::Float32)
}
/// A `Loop` RANGE typed `Int32`, so a fixture built on it reaches the rule under test instead of
/// tripping `rule_no_legacy_index_dtype` on an `Index`-typed extent first.
fn int32_loop_range(end: i64, id: usize) -> Arc<UOp> {
    UOp::range_axis_dtype(
        int_const(DType::Int32, end),
        svod_ir::AxisId::Renumbered(id),
        svod_ir::types::AxisType::Loop,
        DType::Int32,
    )
}
fn tensor_ok(node: Arc<UOp>) -> Arc<UOp> {
    let sink = UOp::sink(vec![node]);
    type_verify(&sink, &spec_tensor()).expect("legal in the tensor graph");
    sink
}
/// Every node the program whitelist accepts, including the shared structural rules for control and wrapper nodes.
#[test_case(structured_buffer(AddrSpace::Local, None); "local buffer")]
#[test_case(structured_buffer(AddrSpace::Reg, None); "reg buffer")]
#[test_case(int_const(DType::Int32, 1); "concrete int const")]
#[test_case(float_const(DType::Float32, 1.0); "concrete float const")]
#[test_case(float_const(DType::Float32, f64::NAN); "canonical nan")]
#[test_case(float_vconst(DType::Float32, [f64::NAN, 1.0]); "canonical nan lane in a vector")]
#[test_case(special(DType::Int32, 8); "int32 special")]
#[test_case(shaped_stack(DType::BFloat16); "devectorized shaped stack")]
#[test_case(shaped_stack(DType::Float32).index_axes(vec![2]); "shaped index")]
#[test_case(UOp::endif(if_over(scalar_index())); "if closed by endif")]
#[test_case(split_ends_backedge(); "backedge end left by split_ends")]
// spec.py:207-208 places the special SHRINK rule before the general movement rejection.
#[test_case(UOp::new(
    Op::Shrink(ops::Shrink { src: global_param(0), offsets: int_const(DType::Int32, 0), sizes: int_const(DType::Int32, 1) }),
    DType::Float32,
); "special shrink wins over the movement rejection")]
#[test_case(UOp::noop().barrier(smallvec![]); "barrier")]
#[test_case(const_stack_wmma(); "wmma")]
#[test_case(codegen_param(0, DType::UInt64, AddrSpace::Global, None).call(smallvec![], CallInfo::default()); "uint64 call target")]
#[test_case(UOp::custom(smallvec![], "code".to_string(), DType::Float32); "custom")]
#[test_case(UOp::new(Op::BitCast(ops::BitCast { src: int_const(DType::Int32, 1), dtype: DType::UInt32 }), DType::UInt32); "matching bitcast")]
fn spec_program_accepts(node: Arc<UOp>) {
    apply_spec_program(&UOp::sink(vec![node])).expect("spec_program should accept");
}
#[test_case(structured_buffer(AddrSpace::Global, Some(DeviceSpec::Cpu)), "structured REG/LOCAL allocation"; "global buffer")]
#[test_case(UOp::native_const(1.0f32).reduce_with_num_axes(smallvec![], ReduceOp::Add, 1), "must be rangeified"; "tensor-form reduce")]
#[test_case(UOp::const_(DType::Index, ConstValue::Int(1)), "legacy Index dtype must be lowered"; "legacy index dtype")]
#[test_case(float_const(DType::Float32, 1.0 + 2f64.powi(-24)), "not canonical for its dtype"; "unrepresentable scalar")]
#[test_case(float_vconst(DType::Float32, [1.0, 1.0 + 2f64.powi(-24)]), "not canonical for its dtype"; "unrepresentable lane")]
#[test_case(float_const(DType::Float32, f64::from_bits(0x7ff8_0000_0000_0001)), "not canonical for its dtype"; "nan with a payload")]
#[test_case(if_over(int_const(DType::Int32, 0)), "CAST/INDEX/SHRINK dedup source"; "if over a non-index dedup source")]
#[test_case(UOp::new(Op::EndIf(ops::EndIf { if_op: int_const(DType::Int32, 0) }), DType::Void), "ENDIF must be void and close an IF"; "endif without an if")]
#[test_case(special(DType::Int64, 8), "must be int32 after index lowering"; "int64 special")]
#[test_case(UOp::new(Op::Multi(ops::Multi { src: int_const(DType::Int32, 0), axis: 0 }), DType::Int32), "no matching rule"; "op outside the whitelist")]
fn spec_program_rejects(node: Arc<UOp>, expected: &str) {
    let err = program_err(&UOp::sink(vec![node]));
    assert!(err.contains(expected), "unexpected error: {err}");
}
/// Forms the tensor graph may still carry but a program may not.
#[test_case(UOp::const_(DType::WeakInt, ConstValue::Int(1)), "weak dtype must be lowered"; "weakint")]
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)), "weak dtype must be lowered"; "weakfloat")]
#[test_case(UOp::new(Op::Reshape(ops::Reshape { src: int_const(DType::Int32, 0), new_shape: int_const(DType::Int32, 1) }), DType::Int32), "movement op must be lowered away"; "movement op")]
#[test_case(UOp::invalid_marker(), "Invalid constant must be folded out"; "invalid marker")]
fn spec_tensor_accepts_what_spec_program_rejects(node: Arc<UOp>, expected: &str) {
    let err = program_err(&tensor_ok(node));
    assert!(err.contains(expected), "unexpected error: {err}");
}
/// Tensor-only forms: the tensor whitelist accepts them and the program whitelist turns them away.
///
/// Each row pins the reason, because a bare `is_err()` here proves nothing about the form. An
/// `Index` or `WeakInt` constant anywhere in the fixture trips `rule_no_legacy_index_dtype` or
/// `rule_no_weak_dtype` — both ahead of everything else in `spec_program` — and the row then
/// passes without the named op ever being reached. So the STAGE axis is an `Int32` `Loop` RANGE,
/// not `range_const`, and the buffer carries an `Int32` shape.
#[test_case(UOp::stage_global(UOp::native_const(1.0f32), vec![int32_loop_range(4, 0)]),
    "op not allowed in this spec (no matching rule)"; "stage")]
#[test_case(codegen_param(0, DType::Int32, AddrSpace::Global, None).bind(int_const(DType::Int32, 3)),
    "op not allowed in this spec (no matching rule)"; "bind")]
#[test_case(UOp::tuple(smallvec![int_const(DType::Int32, 1)]),
    "op not allowed in this spec (no matching rule)"; "tuple")]
// A canonical tensor BUFFER's shape is built from WeakInt constants, so `rule_no_weak_dtype` is
// what turns it away and the buffer rule is never reached. The buffer rule itself is pinned by
// `spec_program_rejects::global_buffer`, whose hand-built Int32 shape gets past the dtype rules.
#[test_case(UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32),
    "weak dtype must be lowered"; "structured global buffer")]
fn spec_tensor_accepts_tensor_only_forms(node: Arc<UOp>, expected: &str) {
    let err = program_err(&tensor_ok(node));
    assert!(err.contains(expected), "unexpected rejection: {err}");
}

/// An AFTER over a movement passthrough is *not* a tensor-only form: `rule_after` (spec.rs:495-514)
/// accepts a movement passthrough outright, and it is the RESHAPE underneath that `rule_no_movement`
/// turns away. Pinning that keeps the two rules from being confused for one another.
#[test]
fn an_after_is_accepted_and_its_movement_passthrough_is_not() {
    let reshape = UOp::new(
        Op::Reshape(ops::Reshape { src: int_const(DType::Int32, 0), new_shape: int_const(DType::Int32, 1) }),
        DType::Int32,
    );
    let err = program_err(&tensor_ok(reshape.after(smallvec![])));
    assert!(err.contains("movement op must be lowered away"), "the RESHAPE child must be the rejection: {err}");
    apply_spec_program(&UOp::sink(vec![scalar_index().after(smallvec![])]))
        .expect("the same AFTER over a program-legal passthrough is accepted");
}
#[test_case(UOp::new(Op::Tuple(ops::Tuple { src: smallvec![int_const(DType::Int32, 1)] }), DType::Int32), "TUPLE must be void"; "non-void tuple")]
#[test_case(UOp::group(vec![int_const(DType::Int32, 1)]), "GROUP must be void and may only hold"; "group of a non-store")]
#[test_case(int_const(DType::Int32, 1).call(smallvec![], CallInfo::default()), "non-void CALL target must be uint64"; "call with a non-uint64 body")]
#[test_case(UOp::new(Op::After(ops::After { passthrough: int_const(DType::Int32, 1), deps: smallvec![] }), DType::Int32), "no matching rule"; "after without a storage or movement passthrough")]
fn spec_tensor_rejects_structural_violations(node: Arc<UOp>, expected: &str) {
    let err = type_verify(&UOp::sink(vec![node]), &spec_tensor()).expect_err("malformed tensor node").to_string();
    assert!(err.contains(expected), "unexpected rejection: {err}");
}
#[test_case(UOp::const_(DType::WeakInt, ConstValue::Int(0)); "weakint")]
#[test_case(int_const(DType::Int32, 1); "int32")]
#[test_case(UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1)], DType::Int32); "int32 vector")]
#[test_case(UOp::invalid_marker(); "invalid marker")]
#[test_case(UOp::vconst(vec![ConstValue::Invalid; 4], DType::Bool); "vector of invalid")]
#[test_case(UOp::stack(smallvec![UOp::invalid_marker(); 4]); "stack of invalid markers")]
fn spec_shared_accepts_index_value(index: Arc<UOp>) {
    type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor()).expect("legal INDEX address operand");
}
#[test_case(UOp::const_(DType::WeakFloat, ConstValue::Float(0.0)), "non-integer value reached a memory INDEX operand"; "weakfloat")]
#[test_case(UOp::const_(DType::Bool, ConstValue::Bool(false)), "non-integer value reached a memory INDEX operand"; "bool")]
#[test_case(UOp::vconst(vec![ConstValue::Bool(false), ConstValue::Bool(true)], DType::Bool), "non-integer value reached a memory INDEX operand"; "bool vector")]
#[test_case(
    UOp::new(Op::VConst(ops::VConst { values: vec![ConstValue::Bool(false), ConstValue::Int(0)] }), DType::Bool.vec(2).unwrap()),
    "VCONST value types do not match its dtype";
    "vector mixing bool and int lanes")]
fn spec_shared_rejects_malformed_index_operands(index: Arc<UOp>, expected: &str) {
    let err = type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor())
        .expect_err("malformed INDEX operand")
        .to_string();
    assert!(err.contains(expected), "unexpected rejection: {err}");
}
fn shift(op: BinaryOp, lhs: &DType, count: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Binary(op, integer_const(lhs.clone(), 8), count), lhs.clone())
}
/// Pinned shift matrix: only a scalar `uint32` count may differ from the left operand's dtype, and a weak count is
/// tensor-only.
#[test_case(DType::Int8; "int8")]
#[test_case(DType::UInt8; "uint8")]
#[test_case(DType::Int16; "int16")]
#[test_case(DType::UInt16; "uint16")]
#[test_case(DType::Int32; "int32")]
#[test_case(DType::UInt32; "uint32")]
#[test_case(DType::Int64; "int64")]
#[test_case(DType::UInt64; "uint64")]
fn spec_shift_dtype_matrix_matches_pinned_uint32_exception(lhs_dtype: DType) {
    for op in [BinaryOp::Shl, BinaryOp::Shr] {
        let same = shift(op, &lhs_dtype, integer_const(lhs_dtype.clone(), 1));
        assert!(apply_spec_program(&UOp::sink(vec![same])).is_ok(), "{op:?} {lhs_dtype:?}");
        let uint32 = shift(op, &lhs_dtype, UOp::native_const(1u32));
        assert!(apply_spec_program(&UOp::sink(vec![uint32])).is_ok(), "{op:?} {lhs_dtype:?} by u32");
        let weak = shift(op, &lhs_dtype, UOp::index_const(1));
        assert!(type_verify(&UOp::sink(vec![weak.clone()]), &spec_tensor()).is_ok(), "{op:?} {lhs_dtype:?} weak count");
        assert!(apply_spec_program(&UOp::sink(vec![weak])).is_err(), "weak count must commit first");
        let unrelated = if lhs_dtype == DType::Int8 { DType::Int16 } else { DType::Int8 };
        assert!(
            apply_spec_program(&UOp::sink(vec![shift(op, &lhs_dtype, integer_const(unrelated, 1))])).is_err(),
            "{op:?} {lhs_dtype:?} by unrelated"
        );
    }
}
/// The guard at `spec.rs:314-319` covers `Shl | Shr`, so both must be pinned: the scalar-only
/// uint32 exception does not extend to a vector count for either op.
#[test_case(BinaryOp::Shl; "shl")]
#[test_case(BinaryOp::Shr; "shr")]
fn a_vector_uint32_shift_count_is_not_the_exception(op: BinaryOp) {
    let lhs = UOp::vconst(vec![ConstValue::Int(8), ConstValue::Int(16)], DType::Int16);
    let count = UOp::vconst(vec![ConstValue::UInt(1), ConstValue::UInt(2)], DType::UInt32);
    let vector = UOp::new(Op::Binary(op, lhs, count), DType::Int16.vec(2).unwrap());
    assert!(apply_spec_program(&UOp::sink(vec![vector])).is_err(), "{op:?} by a vector u32 is not the exception");
}
#[test]
fn spec_hcq_accepts_exact_getaddr_and_rejects_non_storage_source() {
    let address = UOp::new(Op::GetAddr(ops::GetAddr { src: global_param(0), device: DeviceSpec::Cpu }), DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![address]), &spec_hcq()).is_ok());
    let invalid =
        UOp::new(Op::GetAddr(ops::GetAddr { src: UOp::native_const(1u32), device: DeviceSpec::Cpu }), DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![invalid]), &spec_hcq()).is_err());
}
fn kernel_param(slot: usize, dtype: DType) -> Arc<UOp> {
    codegen_param(slot, dtype.clone(), AddrSpace::Global, None)
}
fn kernel_call(formals: Vec<Arc<UOp>>, args: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(formals).call(args.into(), CallInfo::default())
}
fn cpu_buffer(dtype: DType) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 4, dtype)
}
fn device_mstack() -> Arc<UOp> {
    let cuda = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 4, DType::Float32);
    UOp::mstack(smallvec![cpu_buffer(DType::Float32), cuda])
}
/// One CALL feeding two buffers through AFTER.
fn multi_output_call_graph() -> Arc<UOp> {
    let call = kernel_call(vec![kernel_param(0, DType::Float32)], vec![cpu_buffer(DType::Float32)]);
    let out0 = cpu_buffer(DType::Float32).after(smallvec![call.clone()]);
    let out1 = cpu_buffer(DType::Float32).after(smallvec![call]);
    UOp::sink(vec![out0, out1])
}
/// The schedule-level scan loop: `END(CALL, [RANGE])` with the counter bound into the CALL's arguments, replayed by
/// `create_pre_schedule`.
fn scan_loop_graph() -> Arc<UOp> {
    let range = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Loop);
    let bind = UOp::define_var("t".to_string(), 0, 3).bind(range.clone());
    let call = UOp::sink(vec![kernel_param(0, DType::Float32)])
        .call(smallvec![cpu_buffer(DType::Float32), bind], CallInfo::default());
    UOp::sink(vec![cpu_buffer(DType::Float32).after(smallvec![call.end(smallvec![range])])])
}
/// A cross-device COPY is the one non-SINK body a CALL may wrap.
fn cross_device_copy_call() -> Arc<UOp> {
    let copy = kernel_param(0, DType::Float32).copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    UOp::sink(vec![copy.call(smallvec![cpu_buffer(DType::Float32)], CallInfo::default())])
}
#[test_case(multi_output_call_graph(); "one call feeding two outputs")]
#[test_case(UOp::sink(vec![device_mstack().mselect(1)]); "concrete-device mstack layout")]
#[test_case(cross_device_copy_call(); "cross-device copy call body")]
#[test_case(scan_loop_graph(); "schedule-level scan loop")]
fn spec_kernel_graph_accepts(sink: Arc<UOp>) {
    verify_kernel_graph(&sink).expect("valid kernel graph");
}
#[test_case(
    UOp::sink(vec![UOp::native_const(0i32).call(smallvec![], CallInfo::default())]),
    "supported opaque body"; "const call body")]
#[test_case(
    UOp::sink(vec![kernel_call(vec![kernel_param(0, DType::Float32), kernel_param(1, DType::Int32)], vec![cpu_buffer(DType::Int32), cpu_buffer(DType::Float32)])]),
    "positional arguments"; "call arguments swapped against their slots")]
#[test_case(UOp::sink(vec![device_mstack().mselect(2)]), "in-range MSTACK"; "mselect out of range")]
#[test_case(
    UOp::sink(vec![UOp::mstack(smallvec![cpu_buffer(DType::Float32), kernel_param(0, DType::Float32)])]),
    "MSTACK"; "mstack mixing device and device-free sources")]
#[test_case(
    UOp::sink(vec![cpu_buffer(DType::Float32).copy_to_device(DeviceSpec::Cuda { device_id: 0 })]),
    "no matching rule"; "bare copy in the outer graph")]
#[test_case(
    UOp::sink(vec![UOp::native_const(1i32).end(smallvec![UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Loop)])]),
    "END must close at most one RANGE over a CALL"; "end wrapping a non-call")]
#[test_case(UOp::sink(vec![stack([cpu_buffer(DType::Float32)])]), "CONST/BIND/PARAM"; "stack of a non-constant source")]
#[test_case(
    UOp::sink(vec![UOp::buffer(0, 4, DType::Float32, AddrSpace::Local, None)]),
    "must be GLOBAL"; "local buffer in the kernel graph")]
fn spec_kernel_graph_rejects(sink: Arc<UOp>, expected: &str) {
    let err = verify_kernel_graph(&sink).expect_err("invalid kernel graph");
    assert!(err.to_string().contains(expected), "unexpected error: {err}");
}
#[test]
fn verification_errors_locate_the_offending_node() {
    let malformed = cpu_buffer(DType::Float32).after(smallvec![UOp::native_const(1i32)]);
    let SpecError::Verification { boundary, uop_id, source_path, reason, .. } =
        verify_kernel_graph(&UOp::sink(vec![malformed.clone()])).expect_err("AFTER dependency must be callable");
    assert_eq!(boundary, "kernel graph");
    assert_eq!(uop_id, malformed.id);
    assert_eq!(source_path, vec![0]);
    assert!(reason.contains("CALL/END(CALL)/AFTER dependencies"), "unexpected reason: {reason}");
    let stale = UOp::new(Op::Noop, DType::Index);
    let SpecError::Verification { boundary, uop_id, source_path, reason, .. } =
        verify_no_legacy_index_dtype(&UOp::sink(vec![stale.clone()])).expect_err("stale Index dtype");
    assert_eq!(boundary, "post-index-lowering");
    assert_eq!(uop_id, stale.id);
    assert_eq!(source_path, vec![0]);
    assert_eq!(reason, "legacy Index dtype must be lowered before a program");
}
#[test_case(
    UOp::new(Op::Binary(BinaryOp::Add, UOp::native_const(1i32), UOp::const_(DType::Float32, ConstValue::Float(1.0))), DType::Int32),
    "binary operand/result dtype mismatch"; "mixed alu dtype")]
#[test_case(structured_buffer(AddrSpace::Global, Some(DeviceSpec::Cpu)), "tensor BUFFER must be structured GLOBAL storage"; "global buffer with a non-weakint shape")]
#[test_case(UOp::native_const(1i32).mselect(0), "MSELECT requires"; "mselect of a non-multi source")]
#[test_case(
    UOp::new(Op::ReduceAxis(ops::ReduceAxis { src: UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32), reduce_op: ReduceOp::Add, axes: vec![0] }), DType::Float32),
    "no matching rule"; "legacy reduce_axis")]
#[test_case(UOp::multi(UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32), 1), "MULTI must"; "multi axis outside its source shape")]
fn preoptimization_rejects_at_the_tensor_boundary(node: Arc<UOp>, expected: &str) {
    let err = apply_pre_optimization(UOp::sink(vec![node])).expect_err("malformed tensor graph");
    assert!(err.to_string().contains(expected), "unexpected error: {err}");
}
#[test]
fn preoptimization_accepts_a_hand_authored_custom_kernel() {
    let index = UOp::index().buffer(global_param(0)).indices(vec![int_const(DType::Int32, 0)]).call().unwrap();
    let custom = UOp::custom(smallvec![load(index.clone())], "({0} + 1.0f)".to_string(), DType::Float32);
    assert!(apply_pre_optimization(UOp::sink(vec![index.store(custom)])).is_ok());
}
