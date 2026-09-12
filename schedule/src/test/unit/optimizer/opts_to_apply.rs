//! Tests for `KernelInfo.opts_to_apply`: the author-supplied opt list that
//! replaces the strategy for one kernel (tinygrad's `opts_to_apply`).

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisType, ConstValue, KernelInfo, Op, Opt, UOp};
use test_case::test_case;

use crate::optimizer::config::OptimizerConfig;
use crate::optimizer::error::OptError;
use crate::optimizer::{Renderer, optimize_kernel_with_config};
use crate::test::support::prelude::*;

/// A hand-ranged `out[i] = in[i] + 1` SINK over PARAM buffers, marked with
/// `opts_to_apply`; `gidx` swaps the manual RANGE for a hand-lowered index.
fn hand_ranged_sink(n: i64, opts_to_apply: Option<Vec<Opt>>, gidx: bool) -> Arc<UOp> {
    let (out, input) = (param(0, n as usize, DType::Float32), param(1, n as usize, DType::Float32));
    let index = if gidx { UOp::special(UOp::index_const(n), "gidx0".into()) } else { range(n, AxisType::Weak, 0) };
    let value = load(index_of(input, index.clone()))
        .try_add(&UOp::const_(DType::Float32, ConstValue::Float(1.0)))
        .expect("add");
    let store = index_of(out, index.clone()).store(value);
    let store = if gidx { store } else { store.end(smallvec![index]) };
    UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply, ..Default::default() })
}

/// The constant extents of every surviving axis of `axis_type`.
fn extents(ast: &Arc<UOp>, axis: AxisType) -> Vec<i64> {
    ast.toposort()
        .iter()
        .filter(|node| matches!(node.op(), Op::Range(..)) && range_axis_type(node) == axis)
        .map(expect_range_extent)
        .collect()
}

/// The widest expanded vector in the lowered AST.
fn vector_width(ast: &Arc<UOp>) -> usize {
    ast.toposort()
        .iter()
        .filter_map(|node| match node.op() {
            Op::Stack(stack) => Some(stack.sources.len()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn optimize(sink: Arc<UOp>, config: &OptimizerConfig) -> Result<Arc<UOp>, OptError> {
    optimize_kernel_with_config(sink, &cpu(), config)
}

fn cpu() -> Renderer {
    Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None)
}

/// `opts_to_apply = Some(vec![])` (the tinygrad `()` analog) applies ZERO opts:
/// the manual Weak range survives and no UPCAST is introduced — including on the
/// hand-lowered `Op::Special` form, whose dedicated bypass is gone.
#[test_case(false; "off a manual RANGE")]
#[test_case(true; "off a hand-lowered gidx")]
fn opts_to_apply_empty_applies_no_opts(gidx: bool) {
    let optimized = optimize(hand_ranged_sink(8, Some(vec![]), gidx), &OptimizerConfig::default()).expect("optimize");

    assert_eq!(vector_width(&optimized), 0, "an empty list must not introduce an UPCAST");
    if gidx {
        assert!(optimized.toposort().iter().any(|node| matches!(node.op(), Op::Special(..))), "{}", optimized.tree());
    } else {
        assert_eq!(extents(&optimized, AxisType::Weak), vec![8], "the manual Weak range must survive untouched");
    }
}

/// A non-empty list is applied verbatim: the UPCAST splits the manual range
/// whether the list comes off the kernel marker or the config.
#[test_case(true; "off the kernel marker")]
#[test_case(false; "off the config")]
fn opts_to_apply_applies_exactly_the_supplied_list(on_marker: bool) {
    let opts = vec![Opt::upcast(0, 4)];
    let (sink, config) = if on_marker {
        (hand_ranged_sink(8, Some(opts), false), OptimizerConfig::default())
    } else {
        (hand_ranged_sink(8, None, false), OptimizerConfig { opts_to_apply: Some(opts), ..Default::default() })
    };

    let optimized = optimize(sink, &config).expect("optimize");
    assert_eq!(extents(&optimized, AxisType::Weak), vec![2]);
    assert_eq!(vector_width(&optimized), 4);
}

/// The kernel marker wins over the config-level list, including when it is
/// explicitly empty.
#[test]
fn kernel_marker_precedes_the_config_level_list() {
    let config = OptimizerConfig { opts_to_apply: Some(vec![Opt::upcast(0, 4)]), ..Default::default() };
    let optimized = optimize(hand_ranged_sink(8, Some(vec![]), false), &config).expect("optimize");

    assert_eq!(extents(&optimized, AxisType::Weak), vec![8]);
    assert_eq!(vector_width(&optimized), 0);
}

/// A failing opt propagates its own error instead of being swallowed.
#[test]
fn opts_to_apply_propagates_a_failure() {
    let sink = hand_ranged_sink(8, Some(vec![Opt::upcast(0, 64)]), false);

    let error = optimize(sink, &OptimizerConfig::default()).expect_err("64 exceeds the CPU upcast cap");
    assert!(matches!(error, OptError::DeviceLimitExceeded { limit_type: "upcast", value: 64, max: 16 }), "{error:?}");
}
