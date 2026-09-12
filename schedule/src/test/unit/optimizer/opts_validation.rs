//! Ported from tinygrad's `test_kernel_opts.py`: the structural contract of
//! each `OptOps` application (axis kinds and counts), not its numerics.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, ReduceOp, UOp};
use test_case::test_case;

use crate::optimizer::error::OptError;
use crate::optimizer::{Opt, Renderer, Scheduler, apply_opt};
use crate::test::support::prelude::*;

/// A `WeakInt` RANGE with a constant extent, the typing `Ranged` uses for every
/// axis so the scheduler's own splits keep it.
fn axis(size: i64, id: usize, axis_type: AxisType) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(id), axis_type)
}

/// `SINK[RANGE(sizes)..]` over `sizes` GLOBAL axes — no compute source, so the
/// scheduler's `full_shape` is exactly the axis list.
fn global_sink(sizes: &[i64]) -> Arc<UOp> {
    UOp::sink(sizes.iter().enumerate().map(|(id, &size)| axis(size, id, AxisType::Global)).collect())
}

/// `SINK[REDUCE(const, reduce axes), GLOBAL globals]`.
fn reduce_over(globals: &[i64], reduces: &[i64], op: ReduceOp) -> Arc<UOp> {
    let axes: Vec<_> =
        reduces.iter().enumerate().map(|(id, &size)| axis(size, globals.len() + id, AxisType::Reduce)).collect();
    let compute = UOp::native_const(1.0f32).reduce(axes.into(), op);
    let sources =
        std::iter::once(compute).chain(globals.iter().enumerate().map(|(id, &size)| axis(size, id, AxisType::Global)));
    UOp::sink(sources.collect())
}

/// Apply `opt` to a fresh scheduler over `sink`, panicking on a rejection.
#[track_caller]
fn applied(sink: Arc<UOp>, renderer: Renderer, opt: &Opt) -> Scheduler {
    let mut scheduler = Scheduler::new(sink, renderer);
    apply_opt(&mut scheduler, opt, true).unwrap_or_else(|error| panic!("{opt} must apply: {error:?}"));
    scheduler
}

/// Apply every opt in `opts` to a fresh scheduler and check the resulting axis
/// counts.
fn assert_opts_apply(sink: Arc<UOp>, renderer: Renderer, opts: &[Opt], expected: &[(AxisType, usize)]) {
    let mut scheduler = Scheduler::new(sink, renderer);
    for opt in opts {
        apply_opt(&mut scheduler, opt, true).unwrap_or_else(|error| panic!("{opt} must apply: {error:?}"));
    }
    for &(axis, count) in expected {
        assert_eq!(scheduler.axes_of(&[axis]).len(), count, "{axis:?} after {opts:?}");
    }
}

/// UPCAST splits a Global axis into `(Global, Upcast)`; a split that consumes
/// the full axis leaves no Global behind.
#[test_case(&[16, 16], 2, &[8, 16, 2], 2; "upcast by two")]
#[test_case(&[16, 16], 4, &[4, 16, 4], 2; "upcast by four")]
#[test_case(&[16, 16], 8, &[2, 16, 8], 2; "upcast by eight")]
#[test_case(&[4], 4, &[4], 0; "a full upcast consumes its axis")]
fn upcast_splits_the_global_axis(shape: &[i64], amount: usize, full_shape: &[i64], globals: usize) {
    let scheduler = applied(global_sink(shape), Renderer::cpu(), &Opt::upcast(0, amount));

    assert_eq!(scheduler.full_shape(), full_shape);
    assert_eq!(scheduler.axes_of(&[AxisType::Global]).len(), globals);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), 1);
}

/// LOCAL, GROUPTOP, UPCAST and UNROLL compose, in any order, on a reduce whose
/// split factors keep every axis under its budget.
#[test_case(&[Opt::local(0, 2)], &[(AxisType::Local, 1), (AxisType::Global, 3)]; "LOCAL on the leading axis")]
#[test_case(&[Opt::local(2, 8)], &[(AxisType::Local, 1), (AxisType::Global, 3)]; "LOCAL by eight")]
#[test_case(&[Opt::local(2, 16)], &[(AxisType::Local, 1), (AxisType::Global, 3)]; "LOCAL by sixteen")]
#[test_case(&[Opt::grouptop(0, 2)], &[(AxisType::GroupReduce, 1)]; "GROUPTOP by two")]
#[test_case(&[Opt::grouptop(0, 32)], &[(AxisType::GroupReduce, 1)]; "GROUPTOP by thirty-two")]
#[test_case(&[Opt::grouptop(0, 64)], &[(AxisType::GroupReduce, 1)]; "GROUPTOP by sixty-four")]
#[test_case(&[Opt::local(0, 2), Opt::grouptop(0, 2)], &[(AxisType::Local, 1), (AxisType::GroupReduce, 1)]; "LOCAL then GROUPTOP")]
#[test_case(&[Opt::local(2, 16), Opt::grouptop(0, 16)], &[(AxisType::Local, 1), (AxisType::GroupReduce, 1)]; "LOCAL sixteen then GROUPTOP sixteen")]
#[test_case(
    &[Opt::local(2, 2), Opt::grouptop(0, 2), Opt::upcast(0, 2), Opt::unroll(0, 2)],
    &[(AxisType::Local, 1), (AxisType::Upcast, 1), (AxisType::Unroll, 1), (AxisType::GroupReduce, 0)];
    "the full ladder folds the group into the unroll"
)]
fn local_and_grouped_reduce_compose(opts: &[Opt], expected: &[(AxisType, usize)]) {
    assert_opts_apply(reduce_over(&[4, 4, 128], &[128], ReduceOp::Add), Renderer::cuda(), opts, expected);
}

/// The same ladder on a double reduce `(8, 128, 8, 128) -> (8, 8)`: GROUPTOP
/// folds each reduce axis independently.
#[test_case(&[Opt::grouptop(0, 2)], &[(AxisType::GroupReduce, 1)]; "the first reduce axis")]
#[test_case(&[Opt::grouptop(0, 32)], &[(AxisType::GroupReduce, 1)]; "the first reduce axis, wide")]
#[test_case(&[Opt::grouptop(1, 2)], &[(AxisType::GroupReduce, 1)]; "the second reduce axis")]
#[test_case(&[Opt::grouptop(1, 32)], &[(AxisType::GroupReduce, 1)]; "the second reduce axis, wide")]
#[test_case(&[Opt::grouptop(0, 2), Opt::grouptop(1, 2)], &[(AxisType::GroupReduce, 2)]; "both reduce axes")]
#[test_case(&[Opt::grouptop(0, 4), Opt::grouptop(1, 64)], &[(AxisType::GroupReduce, 2)]; "asymmetric factors")]
#[test_case(&[Opt::grouptop(0, 16), Opt::grouptop(1, 2), Opt::unroll(0, 4)], &[(AxisType::Unroll, 1), (AxisType::GroupReduce, 2), (AxisType::Reduce, 2)]; "GROUPTOP then UNROLL")]
#[test_case(&[Opt::local(0, 4), Opt::local(1, 4), Opt::grouptop(0, 4), Opt::grouptop(1, 4)], &[(AxisType::Local, 2), (AxisType::GroupReduce, 2)]; "both axes local and grouped")]
#[test_case(
    &[Opt::local(0, 2), Opt::local(1, 2), Opt::grouptop(0, 8), Opt::grouptop(1, 4), Opt::upcast(0, 2)],
    &[(AxisType::Local, 2), (AxisType::GroupReduce, 2), (AxisType::Upcast, 1)];
    "with an upcast"
)]
#[test_case(
    &[Opt::local(0, 4), Opt::local(1, 4), Opt::grouptop(0, 4), Opt::grouptop(1, 4), Opt::upcast(0, 2), Opt::upcast(0, 2)],
    &[(AxisType::Local, 2), (AxisType::GroupReduce, 2), (AxisType::Upcast, 2), (AxisType::Global, 0)];
    "no globals left"
)]
// The deep ladders below: `real_axis` remaps a logical axis index onto the
// physical one after every split, so a renumbering regression only shows up once
// enough splits have accumulated ahead of the axis an opt names.
#[test_case(
    &[Opt::grouptop(0, 2), Opt::grouptop(1, 32), Opt::unroll(2, 4)],
    &[(AxisType::Global, 2), (AxisType::GroupReduce, 2), (AxisType::Reduce, 2), (AxisType::Unroll, 1)];
    "UNROLL names an axis both GROUPTOPs pushed along"
)]
#[test_case(
    &[Opt::local(0, 4), Opt::local(1, 4), Opt::grouptop(0, 2), Opt::grouptop(1, 32), Opt::unroll(1, 4)],
    &[
        (AxisType::Global, 2),
        (AxisType::Local, 2),
        (AxisType::GroupReduce, 2),
        (AxisType::Reduce, 2),
        (AxisType::Unroll, 1),
    ];
    "two LOCALs and two GROUPTOPs before the UNROLL"
)]
#[test_case(
    &[
        Opt::local(0, 2),
        Opt::local(1, 2),
        Opt::grouptop(0, 8),
        Opt::grouptop(1, 4),
        Opt::upcast(0, 2),
        Opt::unroll(0, 4),
        Opt::unroll(1, 4),
    ],
    &[
        (AxisType::Global, 2),
        (AxisType::Local, 2),
        (AxisType::GroupReduce, 1),
        (AxisType::Reduce, 2),
        (AxisType::Upcast, 1),
        (AxisType::Unroll, 2),
    ];
    "the seven-opt ladder folds one group into an unroll"
)]
fn double_reduce_handles_each_reduce_axis(opts: &[Opt], expected: &[(AxisType, usize)]) {
    assert_opts_apply(reduce_over(&[8, 8], &[128, 128], ReduceOp::Add), Renderer::cuda(), opts, expected);
}

/// Axis counts alone cannot tell a correct renumbering from one that split the
/// wrong axis by the right factor, so the deepest ladder pins the extents too.
#[test]
fn the_deep_ladder_splits_the_axes_it_names() {
    let opts = [
        Opt::local(0, 2),
        Opt::local(1, 2),
        Opt::grouptop(0, 8),
        Opt::grouptop(1, 4),
        Opt::upcast(0, 2),
        Opt::unroll(0, 4),
        Opt::unroll(1, 4),
    ];
    let mut scheduler = Scheduler::new(reduce_over(&[8, 8], &[128, 128], ReduceOp::Add), Renderer::cuda());
    for opt in &opts {
        apply_opt(&mut scheduler, opt, true).unwrap_or_else(|error| panic!("{opt} must apply: {error:?}"));
    }

    assert_eq!(scheduler.full_shape(), vec![2, 4, 2, 2, 2, 2, 16, 32, 4, 4]);
    assert_eq!(scheduler.applied_opts, opts);
}

/// `Opt::upcast(_, 0)` resolves the full axis size through `vmax`.
#[test]
fn full_axis_upcast_resolves_the_constant_end() {
    let scheduler = applied(global_sink(&[8]), Renderer::cpu(), &Opt::upcast(0, 0));

    assert_eq!(scheduler.full_shape(), vec![8]);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), 1);
    assert_eq!(scheduler.axes_of(&[AxisType::Global]).len(), 0);
}

/// A symbolic `Range` end is no longer pre-rejected by the resolver; the
/// full-axis upcast still fails at `shift_to`.
#[test]
fn full_axis_upcast_rejects_a_symbolic_end_at_shift_to() {
    let range = range_symbolic(UOp::variable("b".into(), 1, 4, DType::Int32), 0);
    let mut scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32), range]), Renderer::cpu());

    let error = apply_opt(&mut scheduler, &Opt::upcast(0, 0), true).expect_err("a symbolic non-divisor fails");
    assert!(matches!(error, OptError::SymbolicDivisionError { .. }), "{error:?}");
}

/// PADTO carries no reduce-op guard: tinygrad #16562 pads with Invalid instead
/// of the reduce op's identity.
#[test_case(ReduceOp::Add; "add")]
#[test_case(ReduceOp::Max; "max")]
#[test_case(ReduceOp::Mul; "mul")]
fn padto_is_reduce_op_agnostic(reduce_op: ReduceOp) {
    let scheduler = applied(reduce_over(&[4, 4], &[17], reduce_op), Renderer::cuda(), &Opt::padto(2, 32));

    assert_eq!(scheduler.full_shape()[2], 32);
    assert!(scheduler.applied_opts.contains(&Opt::padto(2, 32)));
}

/// PADTO's two rejections, each reported rather than raised: a symbolic axis has
/// no constant extent to round up, and a zero alignment has no rounding at all.
/// PADTO is the one `OptOps` arm whose amount does not pass through
/// `resolve_full_axis`, so a zero from an author-supplied `opts_to_apply` reaches
/// `apply_padto` directly and must be turned back there — reaching `div_ceil`
/// would abort the process over a user-authored opt list.
#[test_case(|| range_symbolic(UOp::variable("n".into(), 1, 4, DType::Int32), 0), Opt::padto(0, 32), "can only pad constant-sized axes"; "a symbolic axis")]
#[test_case(|| axis(17, 0, AxisType::Reduce), Opt::padto(0, 0), "alignment must be non-zero"; "a zero alignment")]
fn padto_reports_its_rejections(range: fn() -> Arc<UOp>, opt: Opt, reason: &str) {
    let mut scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1.0f32), range()]), Renderer::cuda());

    let error = apply_opt(&mut scheduler, &opt, true).expect_err("PADTO must be rejected");
    assert!(matches!(error, OptError::ValidationFailed { op: "PADTO", reason: got } if got == reason), "{error:?}");
    assert!(scheduler.applied_opts.is_empty(), "a rejected opt is not recorded");
}
