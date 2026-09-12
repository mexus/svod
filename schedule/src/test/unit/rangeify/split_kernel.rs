//! `split_store`: which STOREs become their own kernel, and what the resulting CALL carries.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec, ScalarDType};
use svod_ir::{AxisType, Op, UOp, ops};
use test_case::test_case;

use super::helpers::{closed_range_count, extract_kernel};
use crate::rangeify::kernel::{KernelGraphError, split_store, try_get_kernel_graph};
use crate::test::support::build::{buffer, buffer_on, index, range, store};
use crate::test::support::prelude::{expect_call, expect_index, expect_sink, expect_store, unwrap_op};
use crate::test::support::vars::index_const;

/// The kernel `split_store` mints for `x`, or `None` when `x` is not a boundary.
fn split_store_of(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    split_store(&mut Vec::new(), x)
}

fn store_at_zero(value: Arc<UOp>) -> Arc<UOp> {
    store(index(buffer(100), 0), value)
}

/// `LOAD(INDEX(buffer, address))`.
fn load_at(buffer: Arc<UOp>, address: Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(indexed(buffer, address)).call()
}

/// `INDEX(buffer, address)`.
fn indexed(buffer: Arc<UOp>, address: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![address]).call().expect("INDEX")
}

/// The cuda device this suite pins copies to.
fn cuda() -> DeviceSpec {
    DeviceSpec::Cuda { device_id: 0 }
}

/// The kernel graph and its distinct CALLs.
fn kernels_of(root: Arc<UOp>) -> (Arc<UOp>, Vec<Arc<UOp>>) {
    let graph = try_get_kernel_graph(root).expect("kernel graph").0;
    let kernels = graph.toposort().into_iter().filter(|node| matches!(node.op(), Op::Call(..))).collect();
    (graph, kernels)
}

/// The single STORE a compute kernel body wraps.
#[track_caller]
fn only_store(body: &Arc<UOp>) -> Arc<UOp> {
    let sources = expect_sink(body);
    let [stored] = sources.as_slice() else { panic!("expected exactly one STORE in the body\n{}", body.tree()) };
    stored.clone()
}

#[track_caller]
fn assert_no_split(x: &Arc<UOp>) {
    assert!(split_store_of(x).is_none(), "expected no split, got a kernel for\n{}", x.tree());
}

/// The computation a kernel body wraps, seeing through nested `END`s.
#[track_caller]
fn under_end(body: Arc<UOp>) -> Arc<UOp> {
    match body.op() {
        Op::End(ops::End { computation, .. }) => under_end(computation.clone()),
        _ => body,
    }
}

/// `BUFFER(slot, size, Float32, addrspace)`; every but `Global` is device-less.
fn alloc(slot: usize, size: usize, addrspace: AddrSpace) -> Arc<UOp> {
    let device = (addrspace == AddrSpace::Global).then_some(DeviceSpec::Cpu);
    UOp::buffer(slot, size, DType::Float32, addrspace, device)
}

/// The whole `ParamArg` behind `var`, asserting its PARAM semantics. The record
/// is the boundary contract: slot, dtype, name, bounds, address space, device and
/// volatility all have to survive the crossing, not just the first two fields.
#[track_caller]
fn param_arg(var: &Arc<UOp>) -> svod_ir::ParamArg {
    (*unwrap_op!(var, Op::Param(p) => p).arg).clone()
}

// ===== what a plain STORE lowers to =====

/// A STORE with no open ranges splits straight into `CALL(SINK(STORE))`: the destination BUFFER becomes a body-local PARAM and the CALL binds it.
#[test_case(DType::Float32, UOp::native_const(1.0f32) ; "float const")]
#[test_case(DType::Int32, UOp::native_const(1i32) ; "int const")]
#[test_case(DType::Bool, UOp::native_const(true) ; "bool const")]
#[test_case(DType::Float32, UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add") ; "arithmetic")]
fn a_closed_store_becomes_a_call_over_a_param(dtype: DType, value: Arc<UOp>) {
    let store = store(index(UOp::new_buffer(DeviceSpec::Cpu, 100, dtype), 0), Arc::clone(&value));
    assert!(store.in_scope_ranges().is_empty(), "the fixture must have no open ranges");
    let kernel = split_store_of(&store).expect("a closed STORE splits");
    let args = unwrap_op!(&kernel, Op::Call(c) => c).args.to_vec();
    assert!(!args.is_empty(), "the CALL must bind the destination buffer");
    let (store_index, stored_value, _) = expect_store(&only_store(&expect_call(&kernel)));
    let (storage, _) = expect_index(&store_index);
    assert_eq!(
        unwrap_op!(&storage, Op::Param(p) => p).arg.device,
        Some(DeviceSpec::Cpu),
        "the body reaches storage through a codegen PARAM"
    );
    assert!(Arc::ptr_eq(&stored_value, &value));
}

/// Two STOREs in one graph become two CALLs over distinct destinations: the split is per STORE, not per graph, and the two kernels do not share a body.
#[test]
fn two_stores_in_one_graph_become_two_distinct_kernels() {
    let root = UOp::sink(vec![store_at_zero(UOp::native_const(1.0f32)), store_at_zero(UOp::native_const(2.0f32))]);
    let (graph, kernels) = kernels_of(root);
    assert_eq!(kernels.len(), 2, "one CALL per STORE:\n{}", graph.tree());
    assert!(!Arc::ptr_eq(&kernels[0], &kernels[1]), "distinct stores must not share one CALL");
    let bodies: Vec<_> = kernels.iter().map(expect_call).collect();
    assert!(
        bodies.iter().all(|body| matches!(only_store(body).op(), Op::Store(..))),
        "each kernel body owns exactly one STORE:\n{}",
        graph.tree()
    );
    assert!(!Arc::ptr_eq(&bodies[0], &bodies[1]), "the two destinations must stay distinct");
}

/// A CALL already produced by the split is a marked SINK: re-running the pass must not descend into it and re-split its body.
#[test]
fn an_already_split_kernel_is_not_split_again() {
    let kernel = split_store_of(&store_at_zero(UOp::native_const(1.0f32))).expect("first split");
    let (graph, kernels) = kernels_of(UOp::sink(vec![kernel.clone()]));
    assert_eq!(kernels.len(), 1, "the kernel must survive unmatched:\n{}", graph.tree());
    assert!(Arc::ptr_eq(&kernels[0], &kernel));
}

// ===== END(STORE): closed ranges make a kernel whatever their axis type =====

/// END closes its ranges, so the STORE under it always splits and the CALL body keeps every closed range — order and axis type do not gate the split.
#[test_case(vec![UOp::range_const(10, 0)]; "one weak range")]
#[test_case(vec![UOp::range_const(4, 0), UOp::range_const(8, 1)]; "two weak ranges")]
#[test_case(vec![range(10, AxisType::Loop, 0)]; "one loop range")]
#[test_case(vec![UOp::range_const(4, 0), range(8, AxisType::Loop, 1)]; "weak before loop")]
#[test_case(vec![range(8, AxisType::Loop, 1), UOp::range_const(4, 0)]; "loop before weak")]
fn an_end_over_a_store_keeps_every_closed_range(ranges: Vec<Arc<UOp>>) {
    let expected = ranges.len();
    let end = store_at_zero(UOp::native_const(1.0f32)).end(ranges.into());
    let kernel = split_store_of(&end).expect("END(STORE) splits");
    let args = unwrap_op!(&kernel, Op::Call(c) => c).args.to_vec();
    assert!(!args.is_empty(), "the CALL must bind the destination buffer");
    assert_eq!(closed_range_count(&expect_call(&kernel)), expected);
}

/// The splitter accepts exactly one END layer over a STORE. `split_all_stores` gates on `END(END(..))` too, but `split_store` itself declines the inner nesting, so a double END is not a kernel here.
#[test]
fn an_end_over_an_end_over_a_store_is_not_a_kernel_root() {
    let inner = store_at_zero(UOp::native_const(1.0f32)).end(smallvec![UOp::range_const(4, 0)]);
    assert_no_split(&inner.end(smallvec![UOp::range_const(8, 1)]));
}

/// Neither a bare constant nor an END over anything but a STORE is a kernel boundary.
#[test]
fn only_a_store_rooted_shape_splits() {
    let end = |value: Arc<UOp>| value.end(smallvec![UOp::range_const(10, 0)]);
    for not_a_root in [
        UOp::native_const(1.0f32),
        UOp::noop().end(smallvec![UOp::range_const(10, 0)]),
        end(UOp::native_const(1.0f32)),
        end(load_at(buffer(100), range(10, AxisType::Global, 0))),
    ] {
        assert_no_split(&not_a_root);
    }
}

// ===== open ranges gate the split =====

/// An open computational loop means the STORE is interior — it belongs to the enclosing kernel. A DEVICE range is a launch lane, not a loop, so it does not block the split.
#[test_case(AxisType::Weak, false ; "open weak range")]
#[test_case(AxisType::Loop, false ; "open loop range")]
#[test_case(AxisType::Device, true ; "open device lane")]
fn an_open_range_blocks_the_split_unless_it_is_a_launch_lane(axis_type: AxisType, splits: bool) {
    let store = store(indexed(buffer(64), range(4, axis_type, 0)), UOp::native_const(1.0f32));
    assert!(!store.in_scope_ranges().is_empty(), "the fixture must have an open range");
    assert_eq!(split_store_of(&store).is_some(), splits);
}

#[test]
fn a_device_lane_crossed_with_a_loop_still_blocks_the_split() {
    let flat = range(2, AxisType::Device, 0).mul(&UOp::index_const(2)).add(&range(2, AxisType::Loop, 1));
    let store = store(indexed(buffer(4), flat), UOp::native_const(1.0f32));
    assert!(split_store_of(&store).is_none());
}

// ===== COPY kernels =====

/// A COPY stored to a buffer becomes the kernel body directly — no SINK wrapper — so mixed-op runtime lowering can still recover it, even under an END.
#[test_case(|copy: Arc<UOp>| index(UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32), 0).store(copy) ; "copy stored directly")]
#[test_case(|copy: Arc<UOp>| index(UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32), 0).store(copy).end(smallvec![UOp::range_const(10, 0)]) ; "copy under an end")]
#[test_case(|copy: Arc<UOp>| store_at_zero(copy.copy_to_device(DeviceSpec::Cpu)) ; "copy of a copy")]
fn a_stored_copy_becomes_the_kernel_body(build: fn(Arc<UOp>) -> Arc<UOp>) {
    let copy = buffer(100).copy_to_device(cuda());
    let body = under_end(expect_call(&split_store_of(&build(copy)).expect("a stored COPY splits")));
    assert!(matches!(body.op(), Op::Copy(..)), "expected a COPY body, got {}", body.tree());
}

/// A cross-device COPY survives an END and the whole kernel graph: the COPY marker is what licenses the two devices, so no SINK-boundary `KernelSplitMixedDevices` may fire for it.
#[test_case(cuda(), |store| store ; "sink-boundary store")]
#[test_case(cuda(), |store| store.end(smallvec![UOp::range_const(10, 0)]) ; "store under an end")]
fn a_cross_device_copy_survives_the_whole_kernel_graph(device: DeviceSpec, wrap: fn(Arc<UOp>) -> Arc<UOp>) {
    let copy = buffer(16).copy_to_device(cuda());
    let store = store(index(buffer_on(16, ScalarDType::Float32, device), 0), copy);
    let (graph, kernels) = kernels_of(UOp::sink(vec![wrap(store)]));
    assert_eq!(kernels.len(), 1, "the COPY kernel must survive:\n{}", graph.tree());
    let body = under_end(expect_call(&kernels[0]));
    assert!(matches!(body.op(), Op::Copy(..)), "expected a direct COPY body, got {}", body.tree());
}

// ===== device validation =====

/// A non-copy kernel reads and writes one device. Reading a CPU buffer and writing an Amd one without a COPY marker must be rejected, not silently compiled for whichever device won.
#[test_case(cpu_to_amd ; "cpu read into an amd store")]
#[test_case(cpu_plus_amd ; "cpu and amd operands in one kernel")]
#[test_case(amd_behind_an_after ; "amd producer behind an AFTER")]
fn a_mixed_device_kernel_is_rejected(build: fn() -> Arc<UOp>) {
    let Err(err) = try_get_kernel_graph(build()) else {
        panic!("mixed devices must not be compiled");
    };
    assert!(
        matches!(err, KernelGraphError::Ir { source: svod_ir::Error::KernelSplitMixedDevices { ref devices } } if devices.len() > 1),
        "expected KernelSplitMixedDevices, got {err:?}"
    );
}

fn amd_buffer(size: usize) -> Arc<UOp> {
    buffer_on(size, ScalarDType::Float32, DeviceSpec::Amd { device_id: 0 })
}

fn cpu_to_amd() -> Arc<UOp> {
    UOp::sink(vec![store(index(amd_buffer(16), 0), load_at(buffer(16), index_const(0)))])
}

fn cpu_plus_amd() -> Arc<UOp> {
    let cpu = buffer(16);
    let value = load_at(cpu.clone(), index_const(0)).try_add(&load_at(amd_buffer(16), index_const(0))).unwrap();
    UOp::sink(vec![store(index(cpu, 0), value)])
}

fn amd_behind_an_after() -> Arc<UOp> {
    let amd = amd_buffer(16);
    let dependency = store(index(amd.clone(), 0), UOp::native_const(1.0f32));
    let value = load_at(amd, index_const(0)).after(smallvec![dependency]);
    UOp::sink(vec![store(index(buffer(16), 0), value)])
}

/// The control: two STOREs to two devices are two independent kernels, each of which is single-device, so the same graph shape is accepted.
#[test]
fn two_single_device_stores_split_into_two_kernels() {
    let (_, kernels) = kernels_of(UOp::sink(vec![
        store(index(buffer(16), 0), UOp::native_const(1.0f32)),
        store(index(amd_buffer(16), 0), UOp::native_const(2.0f32)),
    ]));
    assert_eq!(kernels.len(), 2);
    for call in &kernels {
        let args = unwrap_op!(call, Op::Call(c) => c).args.to_vec();
        let device = args.iter().find_map(|arg| arg.device_spec()).expect("the kernel binds a device buffer");
        assert!(
            args.iter().filter_map(|arg| arg.device_spec()).all(|candidate| candidate == device),
            "each kernel binds exactly one device"
        );
    }
}

/// A BIND argument may carry device-owned storage for its value; that storage is not the kernel's, so it must not enter device validation.
#[test]
fn bind_args_do_not_participate_in_kernel_device_validation() {
    let cuda_param = UOp::param(1, 1, DType::Index, Some(cuda()));
    let bound = UOp::define_var("i".to_string(), 0, 15).bind(cuda_param);
    let root = UOp::sink(vec![store(indexed(buffer(16), bound), UOp::native_const(1.0f32))]);
    let (graph, _) = kernels_of(root);
    assert!(extract_kernel(&graph).is_some(), "BIND args must not fail device validation");
}

// ===== the CALL argument tuple =====

/// Storage identities are sparse by construction. Only globals and scalar bindings become CALL positions — local and register allocations stay inside the body — and the body's PARAM slots are renumbered dense to match.
#[test]
fn globals_and_scalar_bindings_become_dense_call_positions() {
    let output = alloc(41, 4, AddrSpace::Global);
    let local = alloc(700, 4, AddrSpace::Local);
    let local_peer = alloc(701, 4, AddrSpace::Local);
    let reg = alloc(800, 1, AddrSpace::Reg);
    let input = alloc(990, 4, AddrSpace::Global);
    let scalar = UOp::variable("N".to_string(), 1, 4, DType::Float32).bind(UOp::native_const(2.0f32));
    let local_stack = UOp::new(
        Op::MStack(ops::MStack { buffers: smallvec![local.clone(), local_peer] }),
        DType::Float32.ptr(Some(8), AddrSpace::Local).expect("local ptr"),
    )
    .after(smallvec![UOp::noop()]);
    let value = load_at(local_stack, index_const(0))
        .try_add(&load_at(reg.clone().after(smallvec![UOp::noop()]), index_const(0)))
        .expect("add")
        .try_add(&load_at(input.clone(), index_const(0)))
        .expect("add")
        .try_add(&scalar)
        .expect("add");
    let kernel = split_store_of(&store(index(output.clone(), 0), value)).expect("STORE should split");
    let body = expect_call(&kernel);
    let args = unwrap_op!(&kernel, Op::Call(c) => c).args.to_vec();
    assert_eq!(args.len(), 3, "two globals followed by the scalar binding");
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &output)));
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &input)));
    let last_arg = args.last().expect("scalar binding");
    let Op::Bind(ops::Bind { var: call_var, value: call_value }) = last_arg.op() else {
        panic!("the last CALL arg must be the scalar BIND")
    };
    let Op::Bind(ops::Bind { var: body_var, value: body_value }) = scalar.op() else { unreachable!() };
    assert!(Arc::ptr_eq(call_value, body_value));
    assert!(!Arc::ptr_eq(call_var, body_var), "CALL binding must not alias the body-local PARAM");
    assert_eq!(param_arg(call_var), param_arg(body_var), "boundary identity must not change scalar metadata");
    assert_eq!(call_var.dtype(), body_var.dtype());
    let mut global_slots: Vec<usize> = body
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Param(ops::Param { arg, .. }) if arg.addrspace == Some(AddrSpace::Global) => Some(arg.slot),
            _ => None,
        })
        .collect();
    global_slots.sort_unstable();
    global_slots.dedup();
    assert_eq!(global_slots, vec![0, 1], "PARAM slots must be dense CALL positions");
    let program_info = svod_ir::ProgramInfo::from_sink(&body, DeviceSpec::Cpu);
    assert_eq!(program_info.globals, vec![0, 1], "PROGRAM globals are direct CALL tuple positions");
    assert_eq!(program_info.vars.len(), 1, "scalar variables are PROGRAM values, not globals");
}

/// LOCAL and REG allocations never bind an argument: they are body-local.
#[test]
fn local_and_register_allocations_stay_inside_the_body() {
    let output = alloc(41, 4, AddrSpace::Global);
    let local = alloc(700, 4, AddrSpace::Local);
    let reg = alloc(800, 1, AddrSpace::Reg);
    let value = load_at(reg.after(smallvec![UOp::noop()]), index_const(0))
        .try_add(&load_at(local.after(smallvec![UOp::noop()]), index_const(0)))
        .expect("add");
    let kernel = split_store_of(&store(index(output, 0), value)).expect("STORE should split");
    let body = expect_call(&kernel);
    let args = unwrap_op!(&kernel, Op::Call(c) => c).args.to_vec();
    // The slot is part of the claim: globals densify to [0, 1], and a renumbering
    // that swept LOCAL/REG along with them would move these off 700/800.
    for (addrspace, slot) in [(AddrSpace::Local, 700), (AddrSpace::Reg, 800)] {
        let buffer_in = |root: &Arc<UOp>| {
            root.toposort().iter().any(|u| {
                matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(addrspace) && arg.slot == slot)
            })
        };
        assert!(!args.iter().any(buffer_in), "{addrspace:?} slot {slot} must not be a CALL argument");
        assert!(buffer_in(&body), "{addrspace:?} slot {slot} must stay inside the body");
    }
}
