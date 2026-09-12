//! Phi-dominance regression: the kmeans generic baseline (`matmul → min over K`) produced
//! invalid LLVM IR at K≥1024 on gfx1151, because a value derived from an inner-loop counter
//! was used after that loop exited. These are the minimal graphs that reproduce it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, UOp, ops};
use test_case::test_case;

use crate::linearize::linearize_with_cfg;
use crate::optimizer::config::OptStrategy;
use crate::optimizer::tc;
use crate::optimizer::{
    OptimizerConfig, Renderer, Scheduler, apply_post_optimization_with_renderer, optimize_kernel_with_config,
};

/// `C[n,k] = Σ_d A[n,d] · B[d,k]`, optionally followed by `MIN_k` — the kmeans baseline
/// `x @ cᵀ → min(1)`. BFloat16 inputs keep RDNA4 WMMA (bf16→f32) selectable.
fn build_matmul(n: i64, k: i64, d: i64, min_over_k: bool) -> Arc<UOp> {
    let axis = |end, id, ty| UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(id), ty);
    let (n_r, k_r, d_r) = (axis(n, 0, AxisType::Global), axis(k, 1, AxisType::Global), axis(d, 2, AxisType::Reduce));
    let (nf, kf, df) = (n_r.cast(DType::BFloat16), k_r.cast(DType::BFloat16), d_r.cast(DType::BFloat16));
    let matmul =
        nf.try_add(&df).unwrap().try_mul(&df.try_add(&kf).unwrap()).unwrap().reduce(smallvec![d_r], ReduceOp::Add);
    if min_over_k {
        UOp::sink(vec![matmul.reduce(smallvec![k_r], ReduceOp::Min), n_r])
    } else {
        UOp::sink(vec![matmul, n_r, k_r])
    }
}

/// The RANGE/END/After dependencies of one instruction, inherited from its sources.
fn inherited_deps(deps: &HashMap<u64, HashSet<u64>>, uop: &Arc<UOp>) -> HashSet<u64> {
    uop.op().sources().iter().flat_map(|src| deps.get(&src.id).cloned().unwrap_or_default()).collect()
}

/// Every dependency of one instruction on a closed RANGE is a violation.
fn require_open(deps: &HashSet<u64>, open: &HashSet<u64>, idx: usize, label: &str) -> Result<(), String> {
    match deps.iter().find(|range| !open.contains(range)) {
        Some(closed) => Err(format!("{label} at [{idx}] depends on closed range {closed}")),
        None => Ok(()),
    }
}

/// Validate that no instruction in the linearized list references a value from a closed
/// (ended) loop scope without going through AFTER: `open` tracks the RANGEs whose END has not
/// been seen, and AFTER removes the ranges its dependency chain ends (Tinygrad's `ended_ranges`).
fn check_phi_dominance(linear: &[Arc<UOp>]) -> Result<(), String> {
    let (mut deps, mut open) = (HashMap::new(), HashSet::new());
    for (idx, uop) in linear.iter().enumerate() {
        let mut deps_of = inherited_deps(&deps, uop);
        match uop.op() {
            Op::Range(..) => {
                deps_of.insert(uop.id);
                open.insert(uop.id);
            }
            Op::End(ops::End { ranges, .. }) => {
                require_open(&deps_of, &open, idx, "END")?;
                for range in ranges {
                    open.remove(&range.id);
                }
            }
            Op::After(..) => {
                for ended in uop.op().ended_ranges() {
                    match ended.op() {
                        Op::Range(..) => {
                            deps_of.remove(&ended.id);
                        }
                        _ => {
                            for range in deps.get(&ended.id).cloned().unwrap_or_default() {
                                deps_of.remove(&range);
                            }
                        }
                    }
                }
                require_open(&deps_of, &open, idx, "AFTER")?;
            }
            // Build the message only on the failure path: an eager `format!` here
            // Debug-formats every instruction and once dominated the suite's runtime.
            _ => {
                if let Some(closed) = deps_of.iter().find(|range| !open.contains(range)) {
                    return Err(format!(
                        "phi-dominance violation at [{idx}]: {:?} depends on closed range {closed}",
                        uop.op()
                    ));
                }
            }
        }
        deps.insert(uop.id, deps_of);
    }
    Ok(())
}

/// Check the pre-linearization DAG for cross-scope dependencies: node `u` with RANGE `r` in its
/// `InScopeRanges` consumed by `v` that neither has `r` in scope nor ends it — a malformed tree.
fn check_tree_scope(root: &Arc<UOp>) -> Result<(), String> {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;
    let topo = root.toposort();
    for u in &topo {
        let u_scope = InScopeRangesProperty::get(u);
        if u_scope.is_empty() {
            continue;
        }
        for v in &topo {
            if !v.op().sources().iter().any(|s| s.id == u.id) || matches!(v.op(), Op::After(..)) {
                continue;
            }
            let v_scope = InScopeRangesProperty::get(v);
            let v_ended: HashSet<u64> = v.op().ended_ranges().iter().map(|r| r.id).collect();
            for range in u_scope {
                if !v_scope.contains(range) && !v_ended.contains(range) {
                    return Err(format!(
                        "tree-scope violation: {:?} (scope={u_scope:?}) -> {:?} (scope={v_scope:?}) does not end range {range}",
                        u.op(),
                        v.op()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn all_capabilities(renderer: Renderer) -> Renderer {
    renderer.with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None)
}

#[test_case(Renderer::amd_rdna4(), 1024, false; "rdna4 matmul only")]
#[test_case(Renderer::amd_cdna3(), 1024, false; "cdna3 matmul only")]
#[test_case(Renderer::amd_rdna4(), 256, true; "rdna4 matmul+min small k")]
#[test_case(Renderer::amd_rdna4(), 1024, true; "rdna4 matmul+min large k")]
#[test_case(Renderer::amd_cdna3(), 1024, true; "cdna3 matmul+min large k")]
fn heuristic_optimizer_keeps_phi_dominance(renderer: Renderer, k: i64, min_over_k: bool) {
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let renderer = all_capabilities(renderer);
    let optimized =
        optimize_kernel_with_config(build_matmul(64, k, 64, min_over_k), &renderer, &config).expect("optimizer");
    check_phi_dominance(&linearize_with_cfg(optimized)).unwrap();
}

/// The heuristic optimizer does not always pick TC for these hand-built graphs, so apply it
/// explicitly before the post-optimization + linearize pipeline.
#[test_case(false; "matmul only")]
#[test_case(true; "matmul+min")]
fn tensor_cores_keep_phi_dominance_on_rdna4(min_over_k: bool) {
    let renderer = all_capabilities(Renderer::amd_rdna4());
    let mut scheduler = Scheduler::new(build_matmul(64, 1024, 64, min_over_k), renderer.clone());
    tc::apply(&mut scheduler, -1, 0, 1).expect("TC apply");
    let ast = scheduler.get_optimized_ast(None);
    assert!(ast.toposort().iter().any(|u| matches!(u.op(), Op::Wmma(..))), "TC apply did not produce WMMA");
    let post = apply_post_optimization_with_renderer(ast, &renderer).expect("post optimization");
    check_tree_scope(&post).unwrap();
    check_phi_dominance(&linearize_with_cfg(post)).unwrap();
}

/// The oracle has to report, not silently pass: using a loop's value after its END is exactly
/// the phi-dominance violation the pass prevents, while an AFTER carrying the range is legal.
#[test_case(false ; "a use after END is reported")]
#[test_case(true ; "an AFTER that ends the loop is accepted")]
fn the_oracle_reports_closed_loop_uses(thread_through_after: bool) {
    let range = UOp::range_const(4, 0);
    let inside = range.cast(DType::Float32);
    let end = inside.clone().end(smallvec![range.clone()]);
    let tail = if thread_through_after {
        UOp::new(Op::After(ops::After { passthrough: inside.clone(), deps: smallvec![end.clone()] }), DType::Float32)
    } else {
        inside.add(&UOp::native_const(2.0f32))
    };
    if thread_through_after {
        check_phi_dominance(&[range, inside, end, tail]).expect("the AFTER drops the closed range");
    } else {
        let error = check_phi_dominance(&[range, inside, end, tail]).expect_err("the use after END must fail");
        assert!(error.contains("closed range"), "{error}");
    }
}
