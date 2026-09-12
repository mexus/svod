use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, ReduceOp, UOp};
use test_case::test_case;

use crate::optimizer::config::{HeuristicsConfig, TcOpt};
use crate::optimizer::heuristics::{
    apply_default_upcast, apply_heuristic_upcasts, apply_image_upcasts, apply_local_dims, apply_matvec_fast_path,
    apply_threading, try_grouped_reduction, try_tensor_cores, try_warp_row_reduction,
};
use crate::optimizer::renderer::TcTilePolicy;
use crate::optimizer::{Opt, OptOps, Renderer, Scheduler};
use crate::test::helpers::{create_conv_like_pattern, create_matmul_pattern_with, create_typed_matmul_pattern};
use svod_ir::ops;

/// Matvec-shaped `sum_k A[k] * B[k]` over `stored` buffers; with `wide`, both
/// loads are cast to it before the product.
fn create_matvec_like_pattern(rows: i64, cols: i64, stored: DType, wide: Option<DType>) -> Arc<UOp> {
    create_row_reduce_pattern(AxisType::Global, rows, cols, stored, wide)
}

/// [`create_matvec_like_pattern`] with the row axis of `row_axis` type.
fn create_row_reduce_pattern(row_axis: AxisType, rows: i64, cols: i64, stored: DType, wide: Option<DType>) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), row_axis);
    let reduce = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Reduce);

    let idx_expr = row.try_add(&reduce).expect("index add should succeed");
    let load = || {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, stored.clone());
        let value = UOp::index().buffer(buffer).indices(vec![idx_expr.clone()]).call().expect("index should build");
        match &wide {
            Some(wide) => value.cast(wide.clone()),
            None => value,
        }
    };
    let (a, b) = (load(), load());

    let mul = a.try_mul(&b).expect("mul should succeed");
    let red = mul.reduce(vec![reduce].into(), ReduceOp::Add);
    UOp::sink(vec![red, row])
}

fn create_tc_retry_pattern() -> Arc<UOp> {
    let m_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let n_good_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);
    let n_bad_range = UOp::range_axis(UOp::index_const(15), AxisId::Renumbered(3), AxisType::Global);

    let a_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);
    let b_buf = UOp::new_buffer(DeviceSpec::Cpu, 4096, DType::Float32);

    let a_idx = m_range.try_add(&k_range).expect("A index should build");
    let b_idx = k_range.try_add(&n_bad_range).and_then(|x| x.try_add(&n_good_range)).expect("B index should build");

    let a_val = UOp::index().buffer(a_buf).indices(vec![a_idx]).call().expect("A load should build");
    let b_val = UOp::index().buffer(b_buf).indices(vec![b_idx]).call().expect("B load should build");

    let mul = a_val.try_mul(&b_val).expect("mul should succeed");
    let red = mul.reduce(vec![k_range].into(), ReduceOp::Add);
    UOp::sink(vec![red, m_range, n_good_range, n_bad_range])
}

/// A widening integer cast on the operands is exact under the int8→int32 WMMA,
/// so it must not hide the tensor core; float casts keep the generic path.
#[test_case(DType::Int8, DType::Int32, true; "int8 operands widened to int32 use the integer wmma")]
#[test_case(DType::Float16, DType::Float32, false; "float16 operands widened to float32 stay scalar")]
fn try_tensor_cores_sees_through_widening_integer_casts(stored: DType, wide: DType, uses_tc: bool) {
    let sink = create_typed_matmul_pattern(16, 16, 16, stored.clone(), Some(wide));
    let mut scheduler = Scheduler::new(sink, Renderer::amd_rdna3());

    assert_eq!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()), uses_tc);

    let wmma = scheduler.ast().toposort().into_iter().find_map(|u| match u.op() {
        Op::Wmma(ops::Wmma { metadata, .. }) => Some(metadata.dtype_in.clone()),
        _ => None,
    });
    assert_eq!(wmma, uses_tc.then_some(stored));
}

/// A fused elementwise producer on a MUL operand (`relu(A) @ B`, a padded
/// conv's `WHERE`) leaves the WMMA legal; tinygrad only checks the dtypes, so
/// the hand-coded path must not demand bare loads.
#[test]
fn try_tensor_cores_accepts_fused_operands() {
    let relu = |value: Arc<UOp>| UOp::alu(svod_ir::BinaryOp::Max, value.clone(), value.const_like(0.0f64));
    let sink = create_matmul_pattern_with(16, 16, 16, DType::Float16, relu);
    let mut scheduler = Scheduler::new(sink, Renderer::amd_rdna3());

    assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert!(scheduler.ast().toposort().iter().any(|u| matches!(u.op(), Op::Wmma(..))));
}

/// `(axis type, constant extent)` of a RANGE.
fn range_axis(range: &Arc<UOp>) -> Option<(AxisType, i64)> {
    let Op::Range(ops::Range { end, axis_type, .. }) = range.op() else { return None };
    match end.op() {
        Op::Const(c) => match c.0 {
            ConstValue::Int(extent) => Some((*axis_type, extent)),
            _ => None,
        },
        _ => None,
    }
}

/// A conv-shaped reduce over (channels, taps) takes the tensor core by default:
/// one divisible reduce axis becomes the WMMA K (highest axis id first, so the
/// taps when both divide) and the other survives as a reduce loop around it.
/// `Strict` keeps tinygrad's single-reduce-axis rule.
#[test_case(64, 5, TcOpt::Relaxed, Some(5); "wide channels with five taps")]
#[test_case(16, 25, TcOpt::Relaxed, Some(25); "narrow channels with many taps")]
#[test_case(64, 16, TcOpt::Relaxed, Some(64); "both reduce axes divisible")]
#[test_case(12, 5, TcOpt::Relaxed, None; "no reduce axis divisible")]
#[test_case(64, 5, TcOpt::Strict, None; "strict declines the second reduce axis")]
fn try_tensor_cores_on_conv_shaped_double_reduce(channels: i64, taps: i64, tc_opt: TcOpt, leftover: Option<i64>) {
    let sink = create_conv_like_pattern(32, 32, channels, taps, DType::Float16);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    let uses_tc = leftover.is_some();
    assert_eq!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().tc_opt(tc_opt).build()), uses_tc);
    assert_eq!(scheduler.ast().toposort().iter().any(|u| matches!(u.op(), Op::Wmma(..))), uses_tc);
    let Some(leftover) = leftover else { return };

    let loops: Vec<_> =
        scheduler.rngs().iter().filter_map(range_axis).filter(|(_, extent)| *extent == leftover).collect();
    assert_eq!(
        loops,
        vec![(AxisType::Reduce, leftover)],
        "the other reduce axis must survive as a loop around the WMMA"
    );
}

/// `C[m,n] = sum_k f32(A[m,k] * B[k,n])` over f16 buffers — the mixed-precision
/// shape every tensor-core renderer offers (CDNA and Intel Xe offer only that).
fn create_mixed_matmul_pattern(m: i64, n: i64, k: i64) -> Arc<UOp> {
    let m_range = UOp::range_axis(UOp::index_const(m), AxisId::Renumbered(0), AxisType::Global);
    let n_range = UOp::range_axis(UOp::index_const(n), AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(UOp::index_const(k), AxisId::Renumbered(2), AxisType::Reduce);
    let load = |numel: i64, row: &Arc<UOp>, stride: i64, col: &Arc<UOp>| {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, numel as usize, DType::Float16);
        let index = row.try_mul(&UOp::index_const(stride)).and_then(|x| x.try_add(col)).expect("index should build");
        UOp::index().buffer(buffer).indices(vec![index]).call().expect("load should build")
    };
    let a = load(m * k, &m_range, k, &k_range);
    let b = load(k * n, &k_range, n, &n_range);
    let product = a.try_mul(&b).expect("mul should succeed").cast(DType::Float32);
    let reduce = product.reduce(smallvec::smallvec![k_range], ReduceOp::Add);
    UOp::sink(vec![reduce, m_range, n_range])
}

/// The post-TC opt sequence a matmul gets on `renderer`, as `(op, axis, arg)`.
fn tc_plan(m: i64, n: i64, k: i64, renderer: Renderer) -> Vec<(OptOps, Option<usize>, svod_ir::OptArg)> {
    let sink = create_mixed_matmul_pattern(m, n, k);
    let mut scheduler = Scheduler::new(sink, renderer);
    assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().build()));
    scheduler
        .applied_opts
        .iter()
        .filter(|opt| opt.op != OptOps::TC)
        .map(|opt| (opt.op, opt.axis, opt.arg.clone()))
        .collect()
}

/// `(op, arg)` shorthand for a post-TC UPCAST/LOCAL on axis `axis`.
fn opt(op: OptOps, axis: usize, arg: usize) -> (OptOps, Option<usize>, svod_ir::OptArg) {
    (op, Some(axis), svod_ir::OptArg::Int(arg))
}

/// The CUDA `m16n8k16` core holds four accumulators per lane, and the lane
/// budget is 128, so a GEMM with work to spare grows its warp tile 32-fold —
/// split 4 (M) by 8 (N) to land on a square 64x64 tile — and stops there. Axis
/// 0 is the leftover M range, axis 1 the leftover N range; a warp already fills
/// a CUDA block, so no LOCAL follows.
#[test_case(8192, 3072, 768, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 8)]; "gigaam 768 to 3072 projection")]
#[test_case(8192, 768, 3072, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 8)]; "gigaam 3072 to 768 projection")]
#[test_case(8192, 48, 768, &[(OptOps::UPCAST, 0, 2), (OptOps::UPCAST, 1, 3)]; "narrow output spends the budget on M")]
#[test_case(8192, 768, 320, &[(OptOps::UPCAST, 0, 4), (OptOps::UPCAST, 1, 4)]; "short reduce cannot amortise a wider tile")]
#[test_case(256, 768, 768, &[(OptOps::UPCAST, 0, 2), (OptOps::UPCAST, 1, 4)]; "small output keeps warps over the tile")]
#[test_case(64, 64, 64, &[]; "an output of 32 warp tiles is not worth growing")]
fn cuda_tensor_core_warp_tile(m: i64, n: i64, k: i64, expected: &[(OptOps, usize, usize)]) {
    let expected: Vec<_> = expected.iter().map(|&(op, axis, arg)| opt(op, axis, arg)).collect();
    assert_eq!(tc_plan(m, n, k, Renderer::cuda()), expected);
}

/// Every target off CUDA keeps [`TcTilePolicy::FixedStep`], tinygrad's step:
/// UPCAST M then N by the first of `[5, 4, 3, 2]` that divides, then LOCAL N by
/// the first of `[4, 2]`. This pins the AMD and Metal codegen, which cannot be
/// measured here, byte for byte against the sequence that shipped.
#[test_case(Renderer::amd_rdna3(); "rdna3 wmma")]
#[test_case(Renderer::amd_rdna4(); "rdna4 wmma")]
#[test_case(Renderer::amd_cdna3(); "cdna3 mfma")]
#[test_case(Renderer::amd_cdna4(); "cdna4 mfma")]
#[test_case(Renderer::metal(); "metal simdgroup")]
#[test_case(Renderer::intel_xe(); "intel xe dpas")]
fn non_cuda_tensor_core_tiling_is_the_fixed_step(renderer: Renderer) {
    assert_eq!(renderer.tc_tile_policy(), TcTilePolicy::FixedStep);
    let expected = [opt(OptOps::UPCAST, 0, 4), opt(OptOps::UPCAST, 1, 4), opt(OptOps::LOCAL, 1, 4)];
    assert_eq!(tc_plan(8192, 3072, 768, renderer), expected);
}

/// The post-TC sequence as it stood before [`TcTilePolicy`] existed,
/// transcribed from that code: UPCAST M then N by the first of `[5, 4, 3, 2]`
/// that divides the leftover tile count, then LOCAL N by the first of
/// `[4, 2]`. Axis numbering follows `rngs()`, which drops an axis the moment it
/// collapses to one tile — so N slides to index 0 once M is fully consumed.
fn fixed_step_reference(
    m: i64,
    n: i64,
    dims: (usize, usize),
    has_local: bool,
) -> Vec<(OptOps, Option<usize>, svod_ir::OptArg)> {
    let first = |extent: usize, ladder: &[usize]| ladder.iter().copied().find(|f| extent.is_multiple_of(*f));
    let (mut m_tiles, mut n_tiles) = (m as usize / dims.1, n as usize / dims.0);
    let mut plan = Vec::new();
    if m_tiles > 1
        && let Some(factor) = first(m_tiles, &[5, 4, 3, 2])
    {
        plan.push(opt(OptOps::UPCAST, 0, factor));
        m_tiles /= factor;
    }
    let n_axis = usize::from(m_tiles > 1);
    if n_tiles > 1
        && let Some(factor) = first(n_tiles, &[5, 4, 3, 2])
    {
        plan.push(opt(OptOps::UPCAST, n_axis, factor));
        n_tiles /= factor;
    }
    if has_local
        && n_tiles > 1
        && let Some(factor) = first(n_tiles, &[4, 2])
    {
        plan.push(opt(OptOps::LOCAL, n_axis, factor));
    }
    plan
}

/// Every non-CUDA renderer reproduces [`fixed_step_reference`] exactly, on
/// every shape — including the ones where the ladder's odd 5 and 3 win and the
/// ones where an axis collapses and renumbers the next. This is the guard that
/// the lane-budget rule left AMD, Metal and Intel codegen untouched.
#[test_case(Renderer::amd_rdna3(), (16, 16); "rdna3 wmma")]
#[test_case(Renderer::amd_rdna4(), (16, 16); "rdna4 wmma")]
#[test_case(Renderer::amd_cdna3(), (16, 16); "cdna3 mfma")]
#[test_case(Renderer::amd_cdna4(), (16, 16); "cdna4 mfma")]
#[test_case(Renderer::metal(), (8, 8); "metal simdgroup")]
#[test_case(Renderer::intel_xe(), (8, 8); "intel xe dpas")]
fn non_cuda_tiling_matches_the_shipped_fixed_step(renderer: Renderer, dims: (usize, usize)) {
    assert_eq!(renderer.tc_tile_policy(), TcTilePolicy::FixedStep);
    for m in [256i64, 1024, 8192] {
        for n in [48i64, 320, 768, 1536, 3072] {
            for k in [320i64, 768, 3072] {
                assert_eq!(
                    tc_plan(m, n, k, renderer.clone()),
                    fixed_step_reference(m, n, dims, renderer.has_local),
                    "{m}x{n}x{k}"
                );
            }
        }
    }
}

/// A wider warp tile must never record an UPCAST the renderer would refuse to
/// replay: beam's cache and `opts_to_apply` both re-apply the recorded list
/// through `apply_opt`, which rejects an amount over `upcast_max`.
///
/// Only [`TcTilePolicy::LaneBudget`] is held to this. The fixed step opens its
/// ladder at 5 without consulting `upcast_max`, so a 320-wide N already records
/// an unreplayable `UPCAST 5` on Metal (`upcast_max` 4); that predates this
/// policy and fixing it would change Metal codegen this machine cannot measure.
#[test_case(8192, 3072, 768; "gigaam projection")]
#[test_case(8192, 768, 3072; "wide reduce")]
#[test_case(1024, 320, 768; "an N the odd factors reach for")]
#[test_case(256, 768, 768; "small output")]
fn lane_budget_warp_tile_stays_replayable(m: i64, n: i64, k: i64) {
    let renderer = Renderer::cuda();
    assert!(matches!(renderer.tc_tile_policy(), TcTilePolicy::LaneBudget { .. }));
    let upcast_max = renderer.upcast_max;
    for (op, _, arg) in tc_plan(m, n, k, renderer) {
        let svod_ir::OptArg::Int(amount) = arg else { panic!("post-TC opts carry Int args") };
        assert!(op != OptOps::UPCAST || amount <= upcast_max, "{m}x{n}x{k}: UPCAST {amount} > {upcast_max}");
    }
}

/// The default level changes nothing for a single-reduce matmul: the same
/// opts land on the same axes (the recorded TC arg carries the level itself).
#[test]
fn try_tensor_cores_default_matches_strict_on_plain_matmul() {
    let plan = |tc_opt: TcOpt| {
        let sink = create_typed_matmul_pattern(64, 64, 64, DType::Float16, None);
        let mut scheduler = Scheduler::new(sink, Renderer::cuda());
        assert!(try_tensor_cores(&mut scheduler, &HeuristicsConfig::builder().tc_opt(tc_opt).build()));
        let opts: Vec<_> = scheduler.applied_opts.iter().map(|opt| (opt.op, opt.axis)).collect();
        let axes: Vec<_> = scheduler.rngs().iter().map(range_axis).collect();
        (opts, axes)
    };
    assert_eq!(HeuristicsConfig::default().tc_opt, TcOpt::Relaxed);
    assert_eq!(plan(TcOpt::default()), plan(TcOpt::Strict));
}

/// The matvec fast path applies GROUP + LOCAL + UPCAST in one shot, unless
/// `matvec_enabled` turns it off.
#[test_case(true; "enabled")]
#[test_case(false; "disabled by config")]
fn test_apply_matvec_fast_path(enabled: bool) {
    let sink = create_matvec_like_pattern(64, 128, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    let config = HeuristicsConfig::builder().matvec_enabled(enabled).build();

    assert_eq!(apply_matvec_fast_path(&mut scheduler, &config), enabled);
    for axis in [AxisType::GroupReduce, AxisType::Local, AxisType::Upcast] {
        assert_eq!(!scheduler.axes_of(&[axis]).is_empty(), enabled, "{axis:?}");
    }
}

/// Widened int8 operands, the shape every integer contraction takes after the
/// early `Cast(Mul)` rewrite, still qualify for the matvec fast path.
#[test]
fn matvec_fast_path_accepts_widened_integer_operands() {
    let sink = create_matvec_like_pattern(64, 128, DType::Int8, Some(DType::Int32));
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    assert!(apply_matvec_fast_path(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert!(!scheduler.axes_of(&[AxisType::GroupReduce]).is_empty());
}

#[test_case(DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] }, true; "image buffer")]
#[test_case(DType::Float32, false; "plain rank three tensor")]
fn test_apply_image_upcasts_non_stub_behavior(dtype: DType, expected: bool) {
    let g = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Global);
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, dtype.clone(), AddrSpace::Global, Some(DeviceSpec::Cpu));
    let img = UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), dtype);
    let indexed = UOp::index().buffer(img).indices(vec![g.clone()]).call().expect("image index should build");
    let sink = UOp::sink(vec![indexed, g]);

    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    assert_eq!(apply_image_upcasts(&mut scheduler), expected);
    assert_eq!(scheduler.axes_of(&[AxisType::Upcast]).len(), usize::from(expected));
}

#[test]
fn test_try_tensor_cores_retries_axis_choices() {
    let sink = create_tc_retry_pattern();
    let mut scheduler = Scheduler::new(sink, Renderer::metal());

    let config = HeuristicsConfig::builder().tc_opt(TcOpt::Relaxed).build();
    let applied = try_tensor_cores(&mut scheduler, &config);
    assert!(applied, "try_tensor_cores should recover with a later axis choice");

    let tc_opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::TC).expect("TC opt should be recorded");
    assert_eq!(tc_opt.axis, Some(1), "retry should commit the passing axis choice");
}

/// Elementwise SINK with one WEAK axis plus an optional extra axis of `extra`
/// type, so `apply_default_upcast`'s gate and axis pick can be exercised.
fn create_default_upcast_pattern(size: i64, extra: Option<(i64, AxisType)>) -> Arc<UOp> {
    let weak = UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(0), AxisType::Weak);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, size as usize * 64, DType::Float32);
    let (idx, mut sink_srcs) = match extra {
        Some((extra_size, axis_type)) => {
            let other = UOp::range_axis(UOp::index_const(extra_size), AxisId::Renumbered(1), axis_type);
            (weak.try_add(&other).expect("index add"), vec![weak.clone(), other])
        }
        None => (weak.clone(), vec![weak.clone()]),
    };
    let val = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build");
    let doubled = val.try_add(&val).expect("add should succeed");
    sink_srcs.insert(0, doubled);
    UOp::sink(sink_srcs)
}

#[test_case(16, None, true; "divisible weak axis upcasts")]
#[test_case(6, None, false; "size not divisible by four")]
#[test_case(1, None, false; "size one axis is not upcastable")]
#[test_case(16, Some((4, AxisType::Unroll)), false; "unrolled kernel skips the fallback")]
#[test_case(16, Some((4, AxisType::Upcast)), false; "already upcast kernel skips the fallback")]
#[test_case(16, Some((8, AxisType::Reduce)), true; "reduce axis does not block the fallback")]
fn default_upcast_follows_tinygrad_gate(size: i64, extra: Option<(i64, AxisType)>, expected: bool) {
    let pre_existing = usize::from(matches!(extra, Some((_, AxisType::Upcast))));
    let mut scheduler = Scheduler::new(create_default_upcast_pattern(size, extra), Renderer::cpu());

    assert_eq!(apply_default_upcast(&mut scheduler), expected);
    assert_eq!(
        scheduler.axes_of(&[AxisType::Upcast]).len(),
        pre_existing + usize::from(expected),
        "UPCAST axis count after the fallback"
    );
}

#[test]
fn default_upcast_picks_the_innermost_upcastable_axis() {
    // Tinygrad takes `k.upcastable_dims[-1]`; both axes qualify here, and only
    // the trailing one must be split.
    let sink = create_default_upcast_pattern(16, Some((8, AxisType::Global)));
    let mut scheduler = Scheduler::new(sink, Renderer::cpu());
    let innermost = *scheduler.upcastable_dims().last().expect("two upcastable dims");

    assert!(apply_default_upcast(&mut scheduler));
    let opt = scheduler.applied_opts.iter().find(|opt| opt.op == OptOps::UPCAST).expect("UPCAST recorded");
    assert_eq!(opt.axis, Some(innermost));
}

/// Elementwise SINK over `axes` GLOBAL axes of extent `size`, summing `axes`
/// row-major buffers; with `stride0`, buffer `i` skips axis `i`.
fn create_stride0_pattern(axes: usize, size: i64, stride0: bool) -> Arc<UOp> {
    let ranges: Vec<Arc<UOp>> =
        (0..axes).map(|i| UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(i), AxisType::Global)).collect();
    let loads: Vec<Arc<UOp>> = (0..axes)
        .map(|skip| {
            let idx = ranges
                .iter()
                .enumerate()
                .filter(|(i, _)| !(stride0 && *i == skip))
                .map(|(i, rng)| rng.try_mul(&UOp::index_const(size.pow((axes - 1 - i) as u32))).expect("index mul"))
                .reduce(|acc, term| acc.try_add(&term).expect("index add"))
                .expect("at least one axis");
            let buf = UOp::new_buffer(DeviceSpec::Cpu, size.pow(axes as u32) as usize, DType::Float32);
            UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build")
        })
        .collect();
    let sum = loads.into_iter().reduce(|acc, load| acc.try_add(&load).expect("add")).expect("one load");
    UOp::sink(std::iter::once(sum).chain(ranges).collect())
}

/// The stride ranking picks, per round, the stride-0 axis with the fewest and
/// smallest strides and the smaller of the amounts 3 and 4 that divide it,
/// until the output shape drops below 1024 elements.
#[test_case(3, 12, true, &[(2, 3)]; "innermost axis by stride sum, amount three first")]
#[test_case(4, 8, true, &[(3, 4), (2, 4)]; "second round after the shape stays large")]
#[test_case(3, 12, false, &[]; "no stride-0 buffer means no candidate")]
fn heuristic_upcasts_rank_by_strides(axes: usize, size: i64, stride0: bool, expected: &[(usize, usize)]) {
    let mut scheduler = Scheduler::new(create_stride0_pattern(axes, size, stride0), Renderer::cpu());

    assert_eq!(apply_heuristic_upcasts(&mut scheduler), !expected.is_empty());
    let expected: Vec<Opt> = expected.iter().map(|&(axis, amount)| Opt::upcast(axis, amount)).collect();
    assert_eq!(scheduler.applied_opts, expected);
}

/// Elementwise SINK over `shape` axes of `axis_type`, loading one row-major
/// buffer, so every axis is a LOCAL/THREAD candidate without a broadcast.
fn create_elementwise_pattern(shape: &[i64], axis_type: AxisType) -> Arc<UOp> {
    let ranges: Vec<Arc<UOp>> = shape
        .iter()
        .enumerate()
        .map(|(i, &size)| UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(i), axis_type))
        .collect();
    let mut stride = 1i64;
    let mut idx = UOp::index_const(0);
    for (rng, &size) in ranges.iter().zip(shape).rev() {
        idx = idx.try_add(&rng.try_mul(&UOp::index_const(stride)).expect("index mul")).expect("index add");
        stride *= size;
    }
    let buf = UOp::new_buffer(DeviceSpec::Cpu, stride as usize, DType::Float32);
    let val = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should build");
    let doubled = val.try_add(&val).expect("add should succeed");
    UOp::sink(std::iter::once(doubled).chain(ranges).collect())
}

/// `out[row] = sum_c x[row * row_stride + c * reduce_stride]`, so the layout a
/// reduce axis is walked with is a parameter rather than a shape.
fn create_laid_out_reduce(rows: i64, cols: i64, row_stride: i64, reduce_stride: i64, dtype: DType) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), AxisType::Global);
    let reduce = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Reduce);
    let term = |rng: &Arc<UOp>, stride: i64| rng.try_mul(&UOp::index_const(stride)).expect("index mul");
    let idx = term(&row, row_stride).try_add(&term(&reduce, reduce_stride)).expect("index add");

    let buffer = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, dtype);
    let value = UOp::index().buffer(buffer).indices(vec![idx]).call().expect("index should build");
    let sum = value.reduce(vec![reduce].into(), ReduceOp::Add);
    UOp::sink(vec![sum, row])
}

/// A row reduce with many rows gets a wave split off its reduce axis, so one
/// wave walks one row together, plus the unroll that widens each lane's burst.
/// The gate is the layout, not the shape: an axis a buffer strides over, or
/// one too short to leave a serial loop, stays a per-thread loop, and few
/// enough rows still take the shared-block path (GROUPTOP 16).
#[test_case(8192, 768, 768, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "many rows split a warp off the contiguous reduce")]
#[test_case(8192, 3072, 3072, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "a longer row keeps the same split")]
#[test_case(131072, 1024, 1024, 1, &[Opt::group(0, 32), Opt::unroll(1, 4)]; "a softmax row reduce splits too")]
#[test_case(8192, 32, 32, 1, &[]; "a reduce shorter than one wave stays serial")]
#[test_case(8192, 768, 1, 8192, &[]; "a strided reduce axis is left alone")]
#[test_case(1024, 768, 768, 1, &[Opt::grouptop(0, 16)]; "few rows keep the shared-block path")]
fn row_reduces_split_a_wave_off_a_contiguous_reduce(
    rows: i64,
    cols: i64,
    row_stride: i64,
    reduce_stride: i64,
    expected: &[Opt],
) {
    let sink = create_laid_out_reduce(rows, cols, row_stride, reduce_stride, DType::Float16);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());
    let config = HeuristicsConfig::builder().build();

    let grouped = try_grouped_reduction(&mut scheduler, &config);
    assert_eq!(grouped || try_warp_row_reduction(&mut scheduler, &config), !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}

/// A CDNA wave is 64 lanes wide, so the split follows the renderer rather than
/// a hard-coded 32.
#[test]
fn the_wave_split_follows_the_renderer_wave_width() {
    let sink = create_laid_out_reduce(8192, 768, 768, 1, DType::Float16);
    let mut scheduler = Scheduler::new(sink, Renderer::amd_cdna3());

    assert!(try_warp_row_reduction(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, vec![Opt::group(0, 64), Opt::unroll(1, 4)]);
}

/// `out[r, c] = x[r, c] * s[r]`: a row-major elementwise kernel over `dtype`
/// with a per-row broadcast operand, the shape of a layer-norm epilogue. The
/// broadcast operand is what lets the upcast heuristic see a stride-0 buffer.
fn create_row_scaled_pattern(rows: i64, cols: i64, dtype: DType) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), AxisType::Global);
    let col = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Global);
    let idx = row.try_mul(&UOp::index_const(cols)).and_then(|r| r.try_add(&col)).expect("index should build");

    let wide = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, dtype);
    let value = UOp::index().buffer(wide).indices(vec![idx]).call().expect("index should build");
    let rowwise = UOp::new_buffer(DeviceSpec::Cpu, rows as usize, DType::Float32);
    let scale = UOp::index().buffer(rowwise).indices(vec![row.clone()]).call().expect("index should build");

    let scaled = value.cast(DType::Float32).try_mul(&scale).expect("mul should succeed");
    UOp::sink(vec![scaled, row, col])
}

/// An elementwise kernel vectorizes along the axis its buffers walk
/// contiguously and hands that axis `lidx0`. The LOCAL *sizes* are the ones
/// this heuristic always picked; what changed is which one is applied first,
/// and therefore which becomes the fastest thread index — here the contiguous
/// column axis rather than the row axis. The upcast width follows the element
/// size: four halves and four floats are both a machine vector, three of
/// either is not, and four doubles are too wide.
#[test_case(DType::Float16, &[Opt::upcast(1, 4), Opt::local(1, 16), Opt::local(0, 8)]; "four halves vectorize")]
#[test_case(DType::Float32, &[Opt::upcast(1, 4), Opt::local(1, 16), Opt::local(0, 8)]; "four floats vectorize")]
#[test_case(DType::Float64, &[Opt::upcast(1, 3), Opt::local(1, 16), Opt::local(0, 8)]; "four doubles keep the ascending width order")]
fn elementwise_kernels_vectorize_and_lane_along_the_contiguous_axis(dtype: DType, expected: &[Opt]) {
    let mut scheduler = Scheduler::new(create_row_scaled_pattern(8192, 768, dtype), Renderer::cuda());
    let config = HeuristicsConfig::builder().build();

    assert!(apply_heuristic_upcasts(&mut scheduler));
    assert!(apply_local_dims(&mut scheduler, &config));
    assert_eq!(scheduler.applied_opts, expected);
}

/// `out[c, r] = x[r, c]`: a transposing copy, contiguous on one side of every
/// axis and strided on the other.
fn create_transpose_pattern(rows: i64, cols: i64) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), AxisType::Global);
    let col = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Global);
    let at = |a: &Arc<UOp>, stride: i64, b: &Arc<UOp>| {
        a.try_mul(&UOp::index_const(stride)).and_then(|a| a.try_add(b)).expect("index should build")
    };
    let load = |idx: Arc<UOp>| {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols) as usize, DType::Float16);
        UOp::index().buffer(buffer).indices(vec![idx]).call().expect("index should build")
    };
    let value = load(at(&row, cols, &col)).try_add(&load(at(&col, rows, &row))).expect("add should succeed");
    UOp::sink(vec![value, row, col])
}

/// No axis of a transposing copy stays inside a sector in every buffer, so
/// there is no lane axis to promote and the mapping is left as it was.
#[test]
fn a_transposing_copy_keeps_the_previous_local_order() {
    let mut scheduler = Scheduler::new(create_transpose_pattern(8192, 768), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, vec![Opt::local(0, 8), Opt::local(1, 16)]);
}

/// `out[r, c] = sum_k x[(r * cols + c) * taps + k]`: a stencil whose column
/// axis would qualify as the lane axis on its stride alone.
fn create_stencil_reduce(rows: i64, cols: i64, taps: i64) -> Arc<UOp> {
    let row = UOp::range_axis(UOp::index_const(rows), AxisId::Renumbered(0), AxisType::Global);
    let col = UOp::range_axis(UOp::index_const(cols), AxisId::Renumbered(1), AxisType::Global);
    let tap = UOp::range_axis(UOp::index_const(taps), AxisId::Renumbered(2), AxisType::Reduce);
    let idx = row
        .try_mul(&UOp::index_const(cols * taps))
        .and_then(|r| r.try_add(&col.try_mul(&UOp::index_const(taps))?))
        .and_then(|r| r.try_add(&tap))
        .expect("index should build");

    let buffer = UOp::new_buffer(DeviceSpec::Cpu, (rows * cols * taps) as usize, DType::Float16);
    let value = UOp::index().buffer(buffer).indices(vec![idx]).call().expect("index should build");
    let sum = value.reduce(vec![tap].into(), ReduceOp::Add);
    UOp::sink(vec![sum, row, col])
}

/// Where a reduce loop sits inside the block, the block shape decides more than
/// the thread mapping — a stencil's halo and the loop's own reuse ride on it
/// too — and promoting the lane axis measured worse on GigaAM's convolution.
/// Such a kernel keeps the order it had, even though its column axis spans only
/// one sector.
#[test]
fn a_reducing_kernel_keeps_the_previous_local_order() {
    let mut scheduler = Scheduler::new(create_stencil_reduce(8192, 768, 5), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, vec![Opt::local(0, 8), Opt::local(1, 16)]);
}

/// The vector-width preference is for lane-parallel backends only: a CPU
/// kernel keeps the ascending amount order, so CPU code generation is
/// untouched.
#[test]
fn the_vector_width_preference_is_gpu_only() {
    let mut scheduler = Scheduler::new(create_row_scaled_pattern(8192, 768, DType::Float16), Renderer::cpu());

    assert!(apply_heuristic_upcasts(&mut scheduler));
    assert_eq!(scheduler.applied_opts, vec![Opt::upcast(1, 3)]);
}

/// A global axis none of the standard LOCAL sizes divides gets the largest
/// divisor within the budget when that fills the warps better, and is
/// padded to a real block size otherwise; divisible axes are unchanged.
#[test_case(51865, &[Opt::padto(0, 32), Opt::local(0, 32)]; "whisper vocabulary pads seven elements to 32")]
#[test_case(10007, &[Opt::padto(0, 32), Opt::local(0, 32)]; "prime extent pads to 32")]
#[test_case(385, &[Opt::local(0, 77)]; "5·7·11 keeps its exact divisor 77")]
#[test_case(12, &[Opt::local(0, 4)]; "candidate list still wins for 12")]
#[test_case(96, &[Opt::local(0, 32)]; "candidate list still wins for 96")]
#[test_case(1024, &[Opt::local(0, 32)]; "candidate list still wins for 1024")]
#[test_case(25, &[Opt::local(0, 25)]; "tie keeps the exact divisor")]
fn local_dims_fall_back_for_undividable_axes(size: i64, expected: &[Opt]) {
    let mut scheduler = Scheduler::new(create_elementwise_pattern(&[size], AxisType::Global), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, expected);
}

/// The lane-efficiency model follows the renderer's wave width: a padded
/// 32-thread block fills a whole warp on CUDA and RDNA but half a wave on
/// CDNA, where the exact divisor 115 (of 5·11·23·41) then wins.
#[test_case(Renderer::cuda(), &[Opt::padto(0, 32), Opt::local(0, 32)]; "warp32 pads")]
#[test_case(Renderer::amd_rdna3(), &[Opt::padto(0, 32), Opt::local(0, 32)]; "wave32 pads")]
#[test_case(Renderer::amd_cdna3(), &[Opt::local(0, 115)]; "wave64 keeps the divisor")]
fn local_fallback_scores_lanes_per_wave(renderer: Renderer, expected: &[Opt]) {
    let mut scheduler = Scheduler::new(create_elementwise_pattern(&[51865], AxisType::Global), renderer);

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, expected);
}

/// The decoder logits shape `[2, 51865]`: the vocabulary axis is padded and
/// localized, and the row axis still folds into the same block.
///
/// The vocabulary axis is the one the buffer walks contiguously, so it leads
/// and lands on `lidx0` — a warp then reads 32 adjacent logits. The row axis
/// costs a whole vocabulary row per lane and follows.
#[test]
fn local_dims_pad_the_vocabulary_axis_beside_the_row_local() {
    let mut scheduler = Scheduler::new(create_elementwise_pattern(&[2, 51865], AxisType::Global), Renderer::cuda());

    assert!(apply_local_dims(&mut scheduler, &HeuristicsConfig::builder().build()));
    assert_eq!(scheduler.applied_opts, vec![Opt::padto(1, 32), Opt::local(1, 32), Opt::local(0, 2)]);
    assert_eq!(scheduler.full_shape(), vec![1621, 32, 2]);
}

/// The matvec fast path pads a row axis the row tile does not divide when
/// the padding is cheap, and declines when it is not.
#[test_case(64, Some(&[Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "divisible rows are unchanged")]
#[test_case(51865, Some(&[Opt::padto(0, 16), Opt::group(0, 8), Opt::local(0, 4), Opt::upcast(0, 4)][..]); "vocabulary rows pad to the tile")]
#[test_case(17, None; "padding almost doubling the rows is declined")]
fn matvec_fast_path_pads_the_row_axis(rows: i64, expected: Option<&[Opt]>) {
    let sink = create_matvec_like_pattern(rows, 128, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, Renderer::cuda());

    assert_eq!(apply_matvec_fast_path(&mut scheduler, &HeuristicsConfig::builder().build()), expected.is_some());
    assert_eq!(scheduler.applied_opts, expected.unwrap_or_default());
}

/// CPU threading pads a loop axis no thread count divides (otherwise it runs
/// on one core); an axis some count divides keeps that count.
#[test_case(10007, 512, &[Opt::padto(0, 32), Opt::thread(0, 32)]; "prime rows pad to 32 threads")]
#[test_case(51865, 512, &[Opt::thread(0, 5)]; "a dividing count is still preferred")]
#[test_case(96, 65536, &[Opt::thread(0, 32)]; "divisible rows are unchanged")]
#[test_case(10007, 4, &[]; "too little work stays single threaded")]
fn threading_pads_undividable_loop_axes(rows: i64, cols: i64, expected: &[Opt]) {
    let mut renderer = Renderer::cpu();
    // Renderer::cpu() caps threads at the host core count; these expectations need 32.
    renderer.global_max = Some(vec![32]);

    let sink = create_row_reduce_pattern(AxisType::Weak, rows, cols, DType::Float32, None);
    let mut scheduler = Scheduler::new(sink, renderer);

    assert_eq!(apply_threading(&mut scheduler, 32), !expected.is_empty());
    assert_eq!(scheduler.applied_opts, expected);
}
