//! Unit tests for the Scheduler (kernel optimization state manager).

use std::sync::{Arc, Mutex};

use proptest::prelude::*;
use svod_ir::AxisType::{Global, GroupReduce, Local, Loop, Reduce, Thread, Unroll, Upcast, Weak};
use svod_ir::{AxisId, AxisType, BinaryOp, DType, Op, ReduceOp, TernaryOp, UOp, ops};
use test_case::test_case;

use crate::optimizer::error::OptError;
use crate::optimizer::heuristics::apply_threading;
use crate::optimizer::{
    KernelInfo, KernelNaming, Opt, OptArg, OptOps, Renderer, Scheduler, apply_opt, finalize_kernel_name,
};
use crate::test::support::prelude::*;

/// `SINK[value, RANGE(extent, type)..]`, one range per axis at its position id;
/// `reduce_at` names the positions `value` reduces with `ADD`.
fn kernel(axes: &[(i64, AxisType)], reduce_at: &[usize], value: Arc<UOp>) -> Arc<UOp> {
    let ranges: Vec<_> = axes.iter().enumerate().map(|(id, &(extent, ty))| kernel_range(extent, ty, id)).collect();
    let value = match reduce_at {
        [] => value,
        at => reduce(value, at.iter().map(|&index| ranges[index].clone()).collect(), ReduceOp::Add),
    };
    UOp::sink(std::iter::once(value).chain(ranges).collect())
}

fn axis_types(scheduler: &Scheduler) -> Vec<AxisType> {
    scheduler.rngs().iter().map(range_axis_type).collect()
}

/// `UOp::range_axis` casts the extent to `WeakInt`, unlike `build::range`, whose
/// `Global`/`Local` extents are `Index`-typed and un-splittable by `shift_to`.
fn kernel_range(extent: i64, ty: AxisType, id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(id), ty)
}

#[track_caller]
fn binary_operands(u: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    match u.op() {
        Op::Binary(_, lhs, rhs) => (lhs.clone(), rhs.clone()),
        other => panic!("expected BINARY, got {other:?}\n{}", u.tree()),
    }
}

#[track_caller]
fn kernel_info(ast: &Arc<UOp>) -> Arc<KernelInfo> {
    ast.metadata::<KernelInfo>().unwrap_or_else(|| panic!("optimized AST carries no KernelInfo\n{}", ast.tree()))
}

#[track_caller]
fn apply(scheduler: &mut Scheduler, opt: &Opt) {
    apply_opt(scheduler, opt, true).unwrap_or_else(|error| panic!("{opt:?} should apply: {error:?}"));
}

#[track_caller]
fn apply_err(scheduler: &mut Scheduler, opt: &Opt) -> OptError {
    match apply_opt(scheduler, opt, false) {
        Ok(()) => panic!("{opt:?} should fail"),
        Err(error) => error,
    }
}

/// A scheduler over `axes` with a constant value, reducing the `reduce_at` axes.
fn scheduled(axes: &[(i64, AxisType)], reduce_at: &[usize], renderer: Renderer) -> Scheduler {
    Scheduler::new(kernel(axes, reduce_at, UOp::native_const(1.0f32)), renderer)
}

/// [`scheduled`] plus its ranges in canonical order.
fn scheduled_rngs(axes: &[(i64, AxisType)], renderer: Renderer) -> (Scheduler, Vec<Arc<UOp>>) {
    let scheduler = scheduled(axes, &[], renderer);
    let rngs = scheduler.rngs().to_vec();
    (scheduler, rngs)
}

/// A CPU renderer with a fixed thread budget, independent of the host.
fn threaded_renderer() -> Renderer {
    let mut renderer = Renderer::cpu();
    renderer.global_max = Some(vec![64]);
    renderer
}

/// Serializes tests that read the process-global kernel-name counter.
static NAME_COUNTER: Mutex<()> = Mutex::new(());

fn name_lock() -> std::sync::MutexGuard<'static, ()> {
    NAME_COUNTER.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn test_scheduler_new() {
    let scheduler = Scheduler::new(UOp::native_const(1.0f32), Renderer::cpu());
    assert!(scheduler.applied_opts.is_empty());
    assert!(!scheduler.dont_use_locals);
    assert_eq!(scheduler.shape_len(), 0);
}

/// Canonical range order: axis-type priority, and size-1 axes are not scheduled.
#[test_case(|| scheduled(&[(16, Global), (8, Local), (32, Reduce), (4, Loop)], &[], Renderer::cpu()), &[Loop, Global, Local, Reduce], 4; "Loop(-1) < Global(0) < Local(2) < Reduce(4)")]
#[test_case(|| scheduled(&[(1, Global), (16, Global), (1, Reduce)], &[2], Renderer::cpu()), &[Global], 1; "size-1 ranges have vmax 0 and must not be scheduled")]
fn test_scheduler_rngs(fixture: fn() -> Scheduler, types: &[AxisType], len: usize) {
    let scheduler = fixture();
    assert_eq!(scheduler.rngs().len(), len);
    assert_eq!(scheduler.shape_len(), len);
    assert_eq!(axis_types(&scheduler), types);
}

#[test]
fn test_scheduler_maxarg() {
    let ranges =
        [(10, Loop, 5), (20, Global, 2), (30, Reduce, 10)].map(|(extent, ty, id)| kernel_range(extent, ty, id));
    let ast = UOp::sink(std::iter::once(UOp::native_const(1.0f32)).chain(ranges).collect());
    assert_eq!(Scheduler::new(ast, Renderer::cpu()).maxarg(), 10);
}

#[test]
fn test_scheduler_helper_properties() {
    let scheduler = scheduled(&[(16, Global), (8, Local), (32, Reduce)], &[2], Renderer::cpu());
    let reduceop = scheduler.reduceop().expect("a reduce axis is scheduled");
    assert_op!(reduceop, Op::Reduce(..));
    assert_eq!(scheduler.reduceops().len(), 1);
    assert_eq!(scheduler.output_shape(), vec![16, 8], "output_shape drops the REDUCE axis");
    assert_eq!(scheduler.full_shape(), vec![16, 8, 32]);
    assert_eq!(scheduler.upcast_size(), 1, "no UPCAST/UNROLL axis means a width of 1");
    assert_eq!(scheduler.group_for_reduces(), 0);
    assert!(scheduler.bufs().is_empty(), "this kernel has no INDEX");
}

#[test]
fn test_scheduler_bufs_lists_index_operations() {
    let access = index_of(buffer(16), kernel_range(16, Global, 0));
    let scheduler = Scheduler::new(kernel(&[(16, Global)], &[], load(access.clone())), Renderer::cpu());
    assert_eq!(scheduler.bufs().len(), 1);
    assert_same!(scheduler.bufs()[0], access);
}

/// The upcast width, and the parity predicate that a lone UNROLL satisfies too.
#[test_case(|| scheduled(&[(4, Upcast), (8, Unroll), (16, Global)], &[], Renderer::cpu()), 32; "UPCAST and UNROLL multiply")]
#[test_case(|| scheduled(&[(4, Unroll)], &[], Renderer::cpu()), 4; "UNROLL alone satisfies the upcast-parity predicate")]
fn test_scheduler_upcast_size(fixture: fn() -> Scheduler, size: usize) {
    let scheduler = fixture();
    assert_eq!(scheduler.upcast_size(), size);
    assert!(scheduler.upcasted());
}

/// `upcastable_dims`, `unrollable_dims` and `group_for_reduces` over three shapes.
#[test_case(|| scheduled(&[(16, Global), (32, Weak), (16, Reduce), (1, Global)], &[], Renderer::cpu()), &[0, 1], &[2], 0; "GLOBAL and WEAK are upcastable; REDUCE and the size-1 axis are not")]
#[test_case(|| scheduled(&[(16, Global), (32, Reduce), (16, Reduce), (1, Reduce)], &[1, 2], Renderer::cpu()), &[0], &[1, 2], 0; "only GLOBAL is upcastable, only REDUCE is unrollable")]
#[test_case(|| scheduled(&[(16, GroupReduce), (32, Reduce)], &[], Renderer::cpu()), &[], &[0, 1], 1; "a GROUP_REDUCE axis is a reduce group")]
fn test_scheduler_dim_queries(fixture: fn() -> Scheduler, upcastable: &[usize], unrollable: &[usize], groups: usize) {
    let scheduler = fixture();
    assert_eq!(scheduler.upcastable_dims(), upcastable);
    assert_eq!(scheduler.unrollable_dims(), unrollable);
    assert_eq!(scheduler.group_for_reduces(), groups);
}

#[test]
fn test_scheduler_axes_of() {
    let scheduler = scheduled(&[(16, Global), (8, Local), (32, Reduce)], &[], Renderer::cpu());
    assert_eq!(scheduler.axes_of(&[Global]), vec![0]);
    assert_eq!(scheduler.axes_of(&[Reduce]), vec![2]);
    assert_eq!(scheduler.axes_of(&[Upcast, Local]), vec![1]);
    let reduce_ranges = scheduler.ranges_of(&[Reduce]);
    assert_eq!(reduce_ranges.len(), 1);
    assert_eq!(range_axis_type(&reduce_ranges[0]), Reduce);
}

#[test]
fn test_scheduler_symbolic_extents_are_unknown() {
    let symbolic = UOp::range_axis(UOp::variable("V".into(), 1, 64, DType::Int32), AxisId::Renumbered(0), Global);
    let fixed = kernel_range(16, Global, 1);
    let sink = UOp::sink(vec![UOp::native_const(1.0f32), symbolic, fixed]);
    let scheduler = Scheduler::new(sink, Renderer::cpu());
    assert_eq!(scheduler.colored_shape(), "g?g16");
    assert_eq!(scheduler.shape_str().join(","), "g?,g16");
    assert_eq!(scheduler.full_shape(), vec![-1, 16], "a symbolic extent is -1");
    assert_eq!(scheduler.output_shape(), vec![16], "a symbolic extent is not an output size");
    assert_eq!(scheduler.upcastable_dims(), vec![1], "a symbolic extent is not upcastable");
}

/// The four axis-mapping families, one row per family.
#[test_case(OptOps::UPCAST, Some(1), 1; "UPCAST indexes rngs directly")]
#[test_case(OptOps::LOCAL, Some(0), 0; "LOCAL indexes rngs directly")]
#[test_case(OptOps::UNROLL, Some(0), 2; "UNROLL indexes unrollable reduction axes")]
#[test_case(OptOps::UNROLL, Some(1), 3; "UNROLL's second logical axis")]
#[test_case(OptOps::GROUP, Some(0), 2; "GROUP indexes REDUCE axes")]
#[test_case(OptOps::GROUPTOP, Some(1), 3; "GROUPTOP maps like GROUP")]
#[test_case(OptOps::TC, None, -1; "TC is axisless")]
#[test_case(OptOps::NOLOCALS, None, -1; "NOLOCALS is axisless")]
fn test_scheduler_real_axis_maps_logical_to_physical(op: OptOps, axis: Option<usize>, expected: isize) {
    let scheduler = scheduled(&[(16, Global), (16, Loop), (32, Reduce), (16, Reduce)], &[2, 3], Renderer::cpu());
    assert_eq!(scheduler.real_axis(op, axis).expect("in range"), expected);
}

#[test_case(OptOps::UPCAST, Some(10); "direct axis beyond the shape")]
#[test_case(OptOps::UNROLL, Some(5); "logical unroll axis beyond the unrollable set")]
#[test_case(OptOps::GROUP, Some(9); "logical group axis beyond the REDUCE set")]
fn test_scheduler_real_axis_rejects_out_of_bounds(op: OptOps, axis: Option<usize>) {
    let scheduler = scheduled(&[(16, Global), (32, Reduce)], &[1], Renderer::cpu());
    assert!(matches!(scheduler.real_axis(op, axis), Err(OptError::AxisOutOfBounds { .. })));
}

#[test_case(OptOps::UPCAST; "UPCAST needs an axis")]
#[test_case(OptOps::UNROLL; "UNROLL needs an axis")]
#[test_case(OptOps::GROUP; "GROUP needs an axis")]
#[test_case(OptOps::GROUPTOP; "GROUPTOP needs an axis")]
#[test_case(OptOps::THREAD; "THREAD needs an axis")]
fn test_scheduler_real_axis_requires_axis(op: OptOps) {
    let scheduler = scheduled(&[(16, Global)], &[], Renderer::cpu());
    assert!(matches!(scheduler.real_axis(op, None), Err(OptError::MissingAxisParameter)));
}

/// Four views of one sorted axis list; a row must agree across all four.
#[test_case(
    &[(16, Global), (8, Local), (32, Reduce), (4, Upcast)],
    &[2], "g16l8u4R32", "r"; "reduce axes sort last"
)]
#[test_case(&[(256, Global), (256, Global)], &[], "g256g256", "E"; "no reduce is elementwise")]
#[test_case(
    &[(2, Loop), (32, Global), (16, Local), (32, Reduce), (4, Upcast), (8, Unroll)],
    &[3, 5], "L2g32l16u4R32r8", "r"; "every axis type in priority order"
)]
fn test_scheduler_colored_shape_and_display(
    axes: &[(i64, AxisType)],
    reduce_axes: &[usize],
    colored: &str,
    kernel_type: &str,
) {
    let scheduler = scheduled(axes, reduce_axes, Renderer::cpu());
    assert_eq!(scheduler.colored_shape(), colored);
    assert_eq!(scheduler.shape_str().concat(), colored, "colored_shape is shape_str concatenated");
    assert_eq!(scheduler.kernel_type(), kernel_type);
    assert_eq!(scheduler.to_string(), format!("{kernel_type}_{colored}"));
}

#[test]
fn test_shift_to_basic_split() {
    let (mut scheduler, rngs) = scheduled_rngs(&[(16, Global)], Renderer::cpu());
    assert_eq!((scheduler.shape_len(), scheduler.maxarg()), (1, 0));
    // Warm rngs/maxarg/shape_len caches before mutating the AST.
    let _warm = (scheduler.rngs().len(), scheduler.maxarg(), scheduler.shape_len());
    let (replaced, new) = scheduler.shift_to(rngs[0].clone(), 4, Upcast, false, None).expect("16 % 4 == 0");
    assert_eq!(range_axis_id(&replaced), AxisId::Renumbered(0), "the reduced axis keeps its id");
    assert_eq!(range_axis_type(&replaced), Global);
    assert_eq!(expect_range_extent(&replaced), 4);
    assert_eq!(range_axis_id(&new), AxisId::Renumbered(1), "the new axis takes maxarg + 1");
    assert_eq!(range_axis_type(&new), Upcast);
    assert_eq!(expect_range_extent(&new), 4);
    assert_eq!((scheduler.shape_len(), scheduler.maxarg()), (2, 1), "every cache was invalidated");
}

/// `top` swaps which axis varies fastest; the stride constant distinguishes them.
#[test_case(false, Global, 8, 2, Local; "bottom: the new axis varies fastest")]
#[test_case(true, Local, 2, 8, Global; "top: the reduced axis varies fastest")]
fn test_shift_to_top_order_selects_the_iteration_order(
    top: bool,
    head_type: AxisType,
    head_extent: i64,
    stride: i64,
    tail_type: AxisType,
) {
    let mut scheduler = Scheduler::new(kernel(&[(16, Global)], &[], kernel_range(16, Global, 0)), Renderer::cpu());
    let rngs = scheduler.rngs().to_vec();
    scheduler.shift_to(rngs[0].clone(), 2, Local, top, None).expect("16 % 2 == 0");
    let substituted = expect_sink(scheduler.ast())[0].clone();
    let (scaled, tail) = binary_operands(&substituted);
    let (head, inner_stride) = binary_operands(&scaled);
    assert_const!(inner_stride, stride);
    assert_eq!(range_axis_type(&head), head_type);
    assert_eq!(expect_range_extent(&head), head_extent);
    assert_eq!(range_axis_type(&tail), tail_type);
}

/// `shift_to` rejects a constant extent it cannot divide, and a symbolic extent
/// whose divisibility it cannot prove.
#[test_case(|| scheduled(&[(15, Global)], &[], Renderer::cpu()), 4, |e| matches!(e, OptError::DivisionError { size: 15, amount: 4 }); "15 % 4 != 0")]
#[test_case(|| symbolic_product(3).0, 2, |e| matches!(e, OptError::SymbolicDivisionError { amount: 2 }); "V*3 is not provably even")]
fn test_shift_to_rejects_indivisible(fixture: fn() -> Scheduler, amount: usize, expected: fn(&OptError) -> bool) {
    let mut scheduler = fixture();
    let rng = scheduler.rngs()[0].clone();
    let error = scheduler.shift_to(rng, amount, Upcast, false, None).expect_err("indivisible");
    assert!(expected(&error), "unexpected error: {error:?}");
}

/// `SINK[1.0, RANGE(V * factor)]` with `V` bounded to `[1, 64]`.
fn symbolic_product(factor: i64) -> (Scheduler, Arc<UOp>) {
    let v = UOp::define_var("V".to_string(), 1, 64);
    let end = v.try_mul(&UOp::index_const(factor)).expect("index mul");
    let rng = UOp::range_axis(end, AxisId::Renumbered(0), Global);
    (Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32), rng]), Renderer::cpu()), v)
}

#[test]
fn test_shift_to_symbolic_exact_division() {
    // end = V * 2 exactly divides by 2, and the quotient must keep depending on V.
    let (mut scheduler, v) = symbolic_product(2);
    let rng = scheduler.rngs()[0].clone();
    let (replaced, _) = scheduler.shift_to(rng, 2, Upcast, false, None).expect("V*2 divides by 2");
    let end = expect_range(&replaced).0;
    assert!(end.any_in_subtree(|node| node.id == v.id), "the quotient must still depend on V");
    assert_eq!(scheduler.shape_len(), 2);
}

#[test]
fn test_shift_to_with_custom_range() {
    let (mut scheduler, rngs) = scheduled_rngs(&[(16, Global)], Renderer::cpu());
    let custom = kernel_range(4, Upcast, 99);
    let (_, new) = scheduler.shift_to(rngs[0].clone(), 4, Upcast, false, Some(custom.clone())).expect("16 % 4 == 0");
    assert_same!(new, custom);
    assert_eq!(range_axis_id(&new), AxisId::Renumbered(99));
}

#[test]
fn test_shift_to_multiple_splits() {
    let (mut scheduler, rngs) = scheduled_rngs(&[(64, Global)], Renderer::cpu());
    let (global_16, _upcast_4) = scheduler.shift_to(rngs[0].clone(), 4, Upcast, false, None).expect("64 % 4 == 0");
    assert_eq!((scheduler.shape_len(), scheduler.maxarg()), (2, 1));
    scheduler.shift_to(global_16, 2, Local, false, None).expect("16 % 2 == 0");
    assert_eq!((scheduler.shape_len(), scheduler.maxarg()), (3, 2));
    assert_eq!(scheduler.rngs().len(), 3);
}

/// Splitting adds one axis and one id; the reduced axis keeps its identity.
#[test]
fn shift_to_split_law() {
    let cases = cheap();
    proptest!(cases, |(quotient in 2i64..1024, amount in 2usize..=16, top in any::<bool>())| {
        let size = quotient * amount as i64;
        let (mut scheduler, rngs) = scheduled_rngs(&[(size, Global)], Renderer::cpu());
        let before = scheduler.maxarg();
        let (replaced, new) = scheduler.shift_to(rngs[0].clone(), amount, Upcast, top, None).expect("divisible");
        prop_assert_eq!(expect_range_extent(&replaced), quotient);
        prop_assert_eq!(expect_range_extent(&new), amount as i64);
        prop_assert_eq!(range_axis_type(&replaced), Global);
        prop_assert_eq!(range_axis_type(&new), Upcast);
        prop_assert_eq!(range_axis_id(&replaced), AxisId::Renumbered(0));
        prop_assert_eq!(range_axis_id(&new), AxisId::Renumbered(before + 1));
        prop_assert_eq!(scheduler.shape_len(), 2);
        prop_assert_eq!(scheduler.maxarg(), before + 1);
    });
}

/// An accepted application: the fixture's scheduler gains exactly these axes and
/// records the opt.
#[test_case(|| scheduled(&[(16, Global)], &[], Renderer::cpu()), Opt::upcast(0, 4), &[Global, Upcast], 2; "UPCAST splits the global axis")]
#[test_case(|| scheduled(&[(64, Global)], &[], Renderer::cuda()), Opt::local(0, 8), &[Global, Local], 2; "LOCAL splits on a GPU renderer")]
#[test_case(|| scheduled(&[(64, Reduce)], &[0], Renderer::cuda()), Opt::group(0, 8), &[GroupReduce, Reduce], 2; "GROUP schedules a grouped reduce")]
#[test_case(|| scheduled(&[(64, Weak)], &[], threaded_renderer()), Opt::thread(0, 8), &[Weak, Thread], 2; "THREAD schedules the outermost axis")]
fn test_apply_opt_lands_the_axes(fixture: fn() -> Scheduler, opt: Opt, axes: &[AxisType], shape_len: usize) {
    let mut scheduler = fixture();
    apply(&mut scheduler, &opt);
    assert_eq!(scheduler.applied_opts, [opt]);
    assert_eq!(axis_types(&scheduler), axes);
    assert_eq!(scheduler.shape_len(), shape_len);
}

/// Every rejected application: the fixture builds the scheduler, then `opt` must
/// fail with the row's predicate.
#[test_case(|| scheduled(&[(32, Reduce)], &[0], Renderer::cpu()), Opt::upcast(0, 4), |e| matches!(e, OptError::ValidationFailed { op: "UPCAST", .. }); "a REDUCE axis needs UNROLL, not UPCAST")]
#[test_case(|| scheduled(&[(256, Global)], &[], Renderer::cpu()), Opt::upcast(0, 32), |e| matches!(e, OptError::DeviceLimitExceeded { limit_type: "upcast", .. }); "UPCAST beyond the device vector width")]
#[test_case(|| scheduled(&[(64, Global)], &[], Renderer::cpu()), Opt::local(0, 8), |e| matches!(e, OptError::UnsupportedFeature { feature: "local memory" }); "LOCAL on a renderer without local memory")]
#[test_case(|| scheduled(&[(32, Reduce)], &[0], Renderer::cuda()), Opt::local(0, 4), |e| matches!(e, OptError::ValidationFailed { op: "LOCAL", .. }); "LOCAL needs a GLOBAL axis")]
#[test_case(|| scheduled(&[(32, Reduce)], &[0], Renderer::cpu()), Opt::unroll(1, 4), |e| matches!(e, OptError::AxisOutOfBounds { axis: 1, .. }); "UNROLL's logical axis beyond the unrollable set")]
#[test_case(|| scheduled(&[(128, Reduce)], &[0], Renderer::cpu()), Opt::unroll(0, 64), |e| matches!(e, OptError::DeviceLimitExceeded { limit_type: "unroll", max: 32, .. }); "UNROLL beyond the device width")]
#[test_case(|| scheduled(&[(64, Reduce)], &[0], Renderer::cpu()), Opt::group(0, 8), |e| matches!(e, OptError::UnsupportedFeature { feature: "local memory" }); "GROUP needs a renderer with local memory")]
#[test_case(|| { let mut r = Renderer::cuda(); r.has_shared = false; scheduled(&[(64, Reduce)], &[0], r) }, Opt::group(0, 8), |e| matches!(e, OptError::UnsupportedFeature { feature: "shared memory" }); "GROUP needs shared memory")]
#[test_case(|| { let mut s = scheduled(&[(64, Reduce)], &[0], Renderer::cuda()); s.applied_opts.push(Opt::new(OptOps::TC, None, OptArg::Int(0))); s }, Opt::group(0, 8), |e| matches!(e, OptError::ValidationFailed { op: "GROUP", reason: "no grouping with tensor cores" }); "GROUP refuses to compose with tensor cores")]
#[test_case(|| Scheduler::new(UOp::sink(vec![reduce(reduce(UOp::native_const(1.0f32), vec![kernel_range(64, Reduce, 1)], ReduceOp::Add), vec![kernel_range(64, Reduce, 0)], ReduceOp::Add)]), Renderer::cuda()), Opt::group(0, 8), |e| matches!(e, OptError::ValidationFailed { op: "GROUP", reason: "cannot apply GROUP inside another reduction" }); "GROUP inside another reduction")]
#[test_case(|| { let mut r = Renderer::cuda(); r.shared_max = 1; scheduled(&[(64, Reduce)], &[0], r) }, Opt::group(0, 8), |e| matches!(e, OptError::DeviceLimitExceeded { limit_type: "shared memory", max: 1, .. }); "GROUP beyond the shared-memory budget")]
#[test_case(|| scheduled(&[(24, Upcast)], &[], Renderer::cpu()), Opt::padto(0, 32), |e| matches!(e, OptError::ValidationFailed { op: "PADTO", reason: "cannot pad vectorized/unrolled/thread axes" }); "PADTO refuses a vectorized axis")]
#[test_case(|| scheduled(&[(8, Global)], &[], Renderer::cpu()), Opt::padto(0, 32), |e| matches!(e, OptError::ValidationFailed { op: "PADTO", reason: "padding would add more than 4x work" }); "PADTO past four times the work")]
#[test_case(|| { let address = kernel_range(24, Global, 0).cast(DType::Int32); let access = UOp::index().buffer(buffer(24)).indices(vec![address.clone(), address]).call().expect("multi-index INDEX"); Scheduler::new(kernel(&[(24, Global)], &[], access), Renderer::cpu()) }, Opt::padto(0, 32), |e| matches!(e, OptError::ValidationFailed { op: "PADTO", reason: "multi-index INDEX is unsupported; Tinygrad PADTO requires one index source" }); "PADTO refuses a multi-index INDEX")]
#[test_case(|| { let mut s = scheduled(&[(64, Global)], &[], Renderer::cuda()); apply(&mut s, &Opt::local(0, 8)); s }, Opt::nolocals(), |e| matches!(e, OptError::ValidationFailed { op: "NOLOCALS", .. }); "NOLOCALS after LOCAL")]
#[test_case(|| scheduled(&[(16, Global), (32, Global)], &[], Renderer::cpu()), Opt::swap(0, 5), |e| matches!(e, OptError::AxisOutOfBounds { axis: 5, .. }); "SWAP beyond the shape")]
#[test_case(|| scheduled(&[(16, Global), (32, Reduce)], &[1], Renderer::cpu()), Opt::swap(0, 1), |e| matches!(e, OptError::ValidationFailed { op: "SWAP", .. }); "SWAP of a REDUCE axis")]
#[test_case(|| scheduled(&[(16, Global)], &[], Renderer::cpu()), Opt::new(OptOps::UPCAST, Some(0), OptArg::TensorCore { tc_select: 0, opt_level: 0, use_tc: 0 }), |e| matches!(e, OptError::InvalidArgType { expected: "Int", .. }); "a non-integer opt argument")]
fn test_apply_opt_rejects(fixture: fn() -> Scheduler, opt: Opt, expected: fn(&OptError) -> bool) {
    let error = apply_err(&mut fixture(), &opt);
    assert!(expected(&error), "unexpected error for {opt:?}: {error:?}");
}

/// `amount = 0` means the whole axis; its reduced size-1 axis leaves the set.
#[test]
fn test_upcast_full_axis_amount_zero_resolves_to_the_axis_size() {
    let mut scheduler = scheduled(&[(8, Local)], &[], Renderer::cuda());
    let opt = Opt::upcast(0, 0);
    apply(&mut scheduler, &opt);
    assert_eq!(scheduler.applied_opts, [opt]);
    assert_eq!(scheduler.shape_len(), 1, "the reduced Local(1) axis is not scheduled");
    assert_eq!(axis_types(&scheduler), [Upcast]);
    assert_eq!(expect_range_extent(&scheduler.rngs()[0]), 8);
}

#[test]
fn test_unroll_basic() {
    let mut scheduler = scheduled(&[(32, Reduce)], &[0], Renderer::cpu());
    assert_eq!(scheduler.unrollable_dims().len(), 1);
    apply(&mut scheduler, &Opt::unroll(0, 4));
    assert_eq!(scheduler.shape_len(), 2);
    assert_eq!(axis_types(&scheduler), [Reduce, Unroll]);
}

#[test]
fn test_apply_opt_multiple_operations() {
    let mut scheduler = scheduled(&[(64, Global), (32, Reduce)], &[0, 1], Renderer::cpu());
    let (upcast, unroll) = (Opt::upcast(0, 4), Opt::unroll(0, 8));
    apply(&mut scheduler, &upcast);
    apply(&mut scheduler, &unroll);
    assert_eq!(scheduler.applied_opts, [upcast, unroll]);
    assert_eq!(scheduler.shape_len(), 4);
}

#[test]
fn test_nolocals_basic() {
    let mut scheduler = scheduled(&[(16, Global)], &[], Renderer::cuda());
    apply(&mut scheduler, &Opt::nolocals());
    assert!(scheduler.dont_use_locals);
    let error = apply_err(&mut scheduler, &Opt::local(0, 4));
    assert!(matches!(error, OptError::ValidationFailed { op: "LOCAL", .. }));
}

#[test]
fn test_swap_basic() {
    let mut scheduler = scheduled(&[(16, Global), (32, Global)], &[], Renderer::cpu());
    apply(&mut scheduler, &Opt::swap(0, 1));
    let ids: Vec<_> = scheduler.rngs().iter().map(range_axis_id).collect();
    let extents: Vec<_> = scheduler.rngs().iter().map(expect_range_extent).collect();
    assert_eq!((ids, extents), (vec![AxisId::Renumbered(0), AxisId::Renumbered(1)], vec![32, 16]));
}

/// Equal extents make a naive fixed-point `substitute` cyclic (`{r0 -> r1, r1 ->
/// r0}`); the single-pass `substitute_walk` lands the swap without tags.
#[test]
fn test_swap_square_axes() {
    let product = kernel_range(16, Global, 0).mul(&UOp::index_const(16)).add(&kernel_range(16, Global, 1));
    let mut scheduler = Scheduler::new(kernel(&[(16, Global), (16, Global)], &[], product), Renderer::cpu());
    apply(&mut scheduler, &Opt::swap(0, 1));
    assert!(scheduler.ast().toposort().iter().all(|node| node.tag().is_none()), "swap leaked a tag");
    assert_eq!(scheduler.rngs().iter().map(expect_range_extent).collect::<Vec<_>>(), [16, 16]);
    assert_eq!(
        scheduler.rngs().iter().map(range_axis_id).collect::<Vec<_>>(),
        [AxisId::Renumbered(0), AxisId::Renumbered(1)]
    );
    // The high digit (the range multiplied by the stride) must now carry axis_id 1.
    let multiply = first_op(scheduler.ast(), |op| matches!(op, Op::Binary(BinaryOp::Mul, ..))).expect("a MUL");
    let (lhs, rhs) = binary_operands(&multiply);
    let high_digit = [lhs, rhs]
        .into_iter()
        .find_map(|side| match side.op() {
            Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.clone()),
            _ => None,
        })
        .expect("a MUL with a RANGE operand");
    assert_eq!(high_digit, AxisId::Renumbered(1), "swap relabels the high-digit axis 0 -> 1");
}

#[test]
fn test_padto_rounds_the_axis_and_gates_the_index() {
    let rng = kernel_range(24, Global, 0);
    let mut scheduler = Scheduler::new(kernel(&[(24, Global)], &[], load(index_of(buffer(24), rng))), Renderer::cpu());
    let opt = Opt::padto(0, 32);
    apply(&mut scheduler, &opt);
    assert_eq!(scheduler.applied_opts, [opt]);
    assert_eq!(scheduler.rngs().iter().map(expect_range_extent).collect::<Vec<_>>(), [32], "24 rounds up to 32");
    let access = scheduler.bufs()[0].clone();
    let (_, indices) = expect_index(&access);
    let (condition, index, invalid) = match indices[0].op() {
        Op::Ternary(TernaryOp::Where, cond, value, invalid) => (cond.clone(), value.clone(), invalid.clone()),
        other => panic!("expected WHERE, got {other:?}\n{}", access.tree()),
    };
    assert!(UOp::is_invalid_marker(&invalid), "padded lanes must read Invalid\n{}", access.tree());
    assert_eq!(expect_range_extent(&index), 32, "the index reads the padded axis");
    let comparison = first_op(&condition, |op| matches!(op, Op::Binary(BinaryOp::Lt, ..))).expect("idx < old_size");
    let (_, old_size) = binary_operands(&comparison);
    assert_const!(old_size, 24);
}

#[test]
fn test_padto_rejects_a_symbolic_axis() {
    let symbolic = UOp::range_axis(UOp::variable("V".into(), 1, 64, DType::Int32), AxisId::Renumbered(0), Global);
    let mut scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32), symbolic]), Renderer::cpu());
    assert_eq!(scheduler.shape_len(), 1, "a bounded symbolic axis is still scheduled");
    let error = apply_err(&mut scheduler, &Opt::padto(0, 32));
    assert!(matches!(error, OptError::ValidationFailed { op: "PADTO", reason: "can only pad constant-sized axes" }));
}

/// `Weak` axes become parallel (`Global`) only where the renderer has local memory.
#[test_case(|| Scheduler::new(kernel(&[(16, Weak), (16, Weak)], &[], UOp::native_const(1.0f32)), Renderer::cuda()), &[Global, Global]; "a GPU renderer globalizes every WEAK axis")]
#[test_case(|| scheduled(&[(16, Weak)], &[], Renderer::cpu()), &[Weak]; "CPU has no local memory, so the axis stays serial")]
fn test_convert_loop_to_global(fixture: fn() -> Scheduler, types: &[AxisType]) {
    let mut scheduler = fixture();
    scheduler.convert_loop_to_global().expect("conversion cannot fail");
    assert_eq!(axis_types(&scheduler), types);
}

/// Only `Weak` is a globalizable axis; a genuine `Loop` range is left alone.
#[test]
fn test_convert_loop_to_global_leaves_genuine_loop_axes() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32), kernel_range(16, Weak, 0), kernel_range(16, Loop, 1)]);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    scheduler.convert_loop_to_global().expect("conversion cannot fail");
    assert_eq!(axis_types(&scheduler), [Loop, Global]);
    assert_eq!(
        scheduler.rngs().iter().map(range_axis_id).collect::<Vec<_>>(),
        [AxisId::Renumbered(1), AxisId::Renumbered(0)]
    );
}

/// A globalizable range must appear in every output, not just one store.
#[test]
fn test_globalizable_rngs_require_every_store() {
    let shared = kernel_range(64, Weak, 0);
    let private = kernel_range(64, Weak, 1);
    let access = index(buffer(64), 0);
    let sink = UOp::sink(vec![access.clone().store(shared.add(&private)), access.store(shared.cast(DType::Int32))]);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    assert_eq!(scheduler.globalizable_rngs().len(), 1);
    assert_same!(scheduler.globalizable_rngs()[0], shared);
    scheduler.convert_loop_to_global().expect("conversion cannot fail");
    assert_eq!(axis_types(&scheduler), [Weak, Global]);
    assert_eq!(range_axis_id(&scheduler.rngs()[1]), AxisId::Renumbered(0), "the shared axis became Global");
}

#[test]
fn test_globalizable_rngs_require_a_shared_store() {
    let (first, second) = (kernel_range(64, Weak, 0), kernel_range(64, Weak, 1));
    let access = index(buffer(64), 0);
    let sink = UOp::sink(vec![access.clone().store(first), access.store(second)]);
    let scheduler = Scheduler::new(sink, threaded_renderer());
    assert!(scheduler.globalizable_rngs().is_empty(), "no axis appears in both stores");
}

#[test_case(&[(16, Global), (8, Local), (32, Reduce), (4, Upcast)], &[2], "r_16_8_4_32"; "reduce kernel lists extents in axis order")]
#[test_case(&[(256, Global)], &[], "E_256"; "elementwise kernel has no reduce part")]
fn test_get_optimized_ast_names_the_kernel(axes: &[(i64, AxisType)], reduce_axes: &[usize], expected: &str) {
    let optimized = scheduled(axes, reduce_axes, Renderer::cuda()).get_optimized_ast(None);
    assert!(kernel_info(&optimized).name.starts_with(expected), "{}", kernel_info(&optimized).name);
}

/// `Scheduler::finish` has a second arm for a root that is not a SINK: it has no
/// structural `Sink::info` to write the name into, so the root is kept as it is
/// and only the `KernelInfo` metadata carries the schedule. Flattening still has
/// to reach the nested REDUCE.
#[test]
fn test_flatten_ranges_store() {
    let range = kernel_range(32, Reduce, 0);
    let value = reduce(UOp::native_const(1.0f32), vec![range.clone()], ReduceOp::Add);
    let optimized = Scheduler::new(index(buffer(32), 0).store(value), Renderer::cuda()).get_optimized_ast(None);

    assert!(matches!(optimized.op(), Op::Store(..)), "the non-SINK root is kept as it is\n{}", optimized.tree());
    assert!(
        !optimized.toposort().iter().any(|node| matches!(node.op(), Op::Sink(..))),
        "the arm wraps nothing in a SINK\n{}",
        optimized.tree()
    );

    // The nested REDUCE was flattened: its axes live in `ranges`, not in a count.
    let stored = unwrap_op!(optimized, Op::Store(ops::Store { value, .. }) => value);
    let reduced = unwrap_op!(stored, Op::Reduce(reduced) => reduced);
    assert_eq!((reduced.num_axes, reduced.ranges.len()), (0, 1));
    assert_same!(reduced.ranges[0], range);

    let info = kernel_info(&optimized);
    assert!(info.name.starts_with("r_32"), "the reduce axis reaches the namer through the STORE root: {}", info.name);
    assert!(info.applied_opts.is_empty());
}

#[test]
fn test_kernel_name_places_special_extents_before_range_extents() {
    let special =
        UOp::new(Op::Special(ops::Special { end: UOp::index_const(8), name: "gidx1".to_string() }), DType::Int32);
    let ast = Scheduler::new(kernel(&[(16, Local)], &[], special), Renderer::cuda()).get_optimized_ast(None);
    let name = kernel_info(&ast).name.clone();
    assert!(name.starts_with("E_8_16"), "{name}");
}

#[test]
fn test_get_optimized_ast_custom_name_bypasses_the_counter() {
    let scheduler = scheduled(&[(16, Global)], &[], Renderer::cuda());
    let named = |scheduler: &Scheduler| {
        kernel_info(&scheduler.get_optimized_ast(Some("custom_kernel".to_string()))).name.clone()
    };
    assert_eq!(named(&scheduler), "custom_kernel");
    assert_eq!(named(&scheduler), "custom_kernel", "an override is not suffixed on reuse");
}

#[test]
fn test_phase7_integration() {
    let (mut scheduler, _) = scheduled_rngs(&[(16, Weak), (16, Weak)], Renderer::cuda());
    scheduler.convert_loop_to_global().expect("conversion cannot fail");
    apply(&mut scheduler, &Opt::upcast(0, 4));
    let info = kernel_info(&scheduler.get_optimized_ast(None));
    assert_eq!(info.applied_opts, [Opt::upcast(0, 4)]);
}

#[test]
fn test_kernel_name_deduplication() {
    let _guard = name_lock();
    let scheduler = scheduled(&[(16, Global)], &[], Renderer::cuda());
    let names: Vec<String> = (0..3).map(|_| kernel_info(&scheduler.get_optimized_ast(None)).name.clone()).collect();
    assert_eq!(names.iter().collect::<std::collections::HashSet<_>>().len(), 3, "{names:?}");
    assert!(names.iter().all(|name| name.starts_with("E_16")), "{names:?}");
}

/// `finalize_kernel_name` draws the suffix on both name channels.
#[test]
fn test_deferred_naming_finalizes_to_the_unique_name() {
    let _guard = name_lock();
    let scheduler = scheduled(&[(4099, Global)], &[], Renderer::cuda());
    fn ordinal(name: &str) -> usize {
        match name.strip_prefix("E_4099").unwrap_or_else(|| panic!("unexpected name {name}")) {
            "" => 0,
            suffix => suffix.strip_prefix('n').expect("suffix form").parse().expect("ordinal"),
        }
    }
    let unique = scheduler.get_optimized_ast(None);
    let deferred = scheduler.get_optimized_ast_with_naming(KernelNaming::Deferred);
    assert_eq!(kernel_info(&deferred).name, "E_4099");
    let finalized = finalize_kernel_name(&deferred);
    let name = kernel_info(&finalized).name.clone();
    assert_eq!(ordinal(&name), ordinal(&kernel_info(&unique).name) + 1, "{name}");
    let structural = unwrap_op!(finalized, Op::Sink(ops::Sink { info: Some(info), .. }) => info);
    assert_eq!(structural.name.as_deref(), Some(name.as_str()));
    assert_eq!(kernel_info(&finalized).applied_opts, kernel_info(&unique).applied_opts);
    let unnamed = UOp::sink(vec![UOp::native_const(2.0f32)]);
    assert_same!(finalize_kernel_name(&unnamed), unnamed);
}

/// A non-SINK root is never rewritten into a SINK, only its metadata renamed.
#[test]
fn test_finalize_kernel_name_leaves_a_non_sink_root_structurally_unchanged() {
    let _guard = name_lock();
    let index = kernel_range(16, Global, 0);
    let root = index.cast(DType::Int32).with_metadata(KernelInfo {
        name: "NON_SINK_ROOT".to_string(),
        applied_opts: Vec::new(),
        dont_use_locals: false,
    });
    let first = finalize_kernel_name(&root);
    let second = finalize_kernel_name(&root);
    assert_op!(first, Op::Cast(..));
    assert_op!(second, Op::Cast(..));
    assert_eq!(kernel_info(&first).name, "NON_SINK_ROOT", "the first draw keeps the name");
    assert_eq!(kernel_info(&second).name, "NON_SINK_ROOTn1", "further draws are suffixed");
    assert_eq!(kernel_info(&root).name, "NON_SINK_ROOT", "the original node keeps its metadata");
}

/// A hand-authored name is kept on first use and suffixed afterwards.
#[test]
fn test_authored_kernel_names_go_through_the_counter() {
    let _guard = name_lock();
    let authored = |value: f32| {
        let structural = svod_ir::KernelInfo {
            name: Some("authored_kernel_test".into()),
            opts_to_apply: Some(vec![]),
            ..Default::default()
        };
        UOp::sink_with_info(vec![UOp::native_const(value)], structural).with_metadata(KernelInfo {
            name: "authored_kernel_test".into(),
            applied_opts: vec![],
            dont_use_locals: false,
        })
    };
    let name_of = |ast: &Arc<UOp>| {
        let (structural, metadata) =
            (unwrap_op!(ast, Op::Sink(ops::Sink { info: Some(info), .. }) => info), kernel_info(ast));
        assert_eq!(structural.name.as_deref(), Some(metadata.name.as_str()));
        metadata.name.clone()
    };
    assert_eq!(name_of(&finalize_kernel_name(&authored(1.0))), "authored_kernel_test");
    assert_eq!(name_of(&finalize_kernel_name(&authored(2.0))), "authored_kernel_testn1");
}

#[test]
fn test_thread_rejects_second_application() {
    let (mut scheduler, _) = scheduled_rngs(&[(64, Weak)], threaded_renderer());
    apply(&mut scheduler, &Opt::thread(0, 8));
    assert!(!scheduler.axes_of(&[Thread]).is_empty());
    let error = apply_err(&mut scheduler, &Opt::thread(0, 2));
    assert!(matches!(error, OptError::ValidationFailed { op: "THREAD", reason: "already threaded" }));
}

#[test]
fn test_thread_rejects_non_globalizable_axis() {
    let (first, second) = (kernel_range(64, Weak, 0), kernel_range(64, Weak, 1));
    let access = index(buffer(64), 0);
    let sink = UOp::sink(vec![access.clone().store(first), access.store(second)]);
    let mut scheduler = Scheduler::new(sink, threaded_renderer());
    let error = apply_err(&mut scheduler, &Opt::thread(0, 2));
    assert!(matches!(error, OptError::ValidationFailed { op: "THREAD", reason: "can't apply range to this dim" }));
    assert!(scheduler.axes_of(&[Thread]).is_empty(), "a rejected THREAD creates no axis");
}

/// The threading heuristic needs `threads * 131072` elements, computed from the
/// symbolic `vmax` when the extent is not constant.
#[test_case(|| scheduled(&[(262144, Weak)], &[], threaded_renderer()), 2; "a large Loop axis threads")]
#[test_case(|| { let v = UOp::variable("V".into(), 1, 131072, DType::WeakInt); let end = v.try_mul(&UOp::index_const(4)).expect("index mul"); Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32), UOp::range_axis(end, AxisId::Renumbered(0), Weak)]), threaded_renderer()) }, 4; "the symbolic vmax bounds the work")]
fn test_apply_threading_heuristic(fixture: fn() -> Scheduler, threads: usize) {
    let mut scheduler = fixture();
    assert!(apply_threading(&mut scheduler, threads), "the work bound makes threading worthwhile");
    assert!(!scheduler.axes_of(&[Thread]).is_empty());
}

/// `apply_opt` never panics: any opt/axis/arg yields `Ok` or an `OptError`.
#[test]
fn apply_opt_is_total() {
    let cases = cheap();
    proptest!(cases, |(op in 0usize..10, axis in prop::option::of(0usize..6), arg in 1usize..64)| {
        let ops = [
            OptOps::TC,
            OptOps::UPCAST,
            OptOps::LOCAL,
            OptOps::UNROLL,
            OptOps::NOLOCALS,
            OptOps::SWAP,
            OptOps::GROUP,
            OptOps::GROUPTOP,
            OptOps::THREAD,
            OptOps::PADTO,
        ];
        let opt = match axis {
            Some(axis) => match op {
                9 => Opt::padto(axis, arg),
                8 => Opt::thread(axis, arg),
                6 | 7 => Opt::group(axis, arg),
                5 => Opt::swap(axis, arg),
                4 => Opt::nolocals(),
                _ => Opt::new(ops[op], Some(axis), OptArg::Int(arg)),
            },
            None => Opt::new(ops[op], None, OptArg::Int(arg)),
        };
        let mut scheduler = scheduled(&[(16, Global), (8, Reduce)], &[1], Renderer::cpu());
        match apply_opt(&mut scheduler, &opt, true) {
            Ok(()) => {}
            Err(error) => match error {
                OptError::Spec { .. }
                | OptError::InvalidArgType { .. }
                | OptError::ValidationFailed { .. }
                | OptError::AxisOutOfBounds { .. }
                | OptError::DivisionError { .. }
                | OptError::SymbolicDivisionError { .. }
                | OptError::ExpectedRangeOperation
                | OptError::MissingAxisParameter
                | OptError::UnsupportedFeature { .. }
                | OptError::MissingRendererCapabilities
                | OptError::DeviceLimitExceeded { .. }
                | OptError::BeamWorker { .. } => {}
            },
        }
    });
}
