//! Property invariants for the rangeify passes: laws that must hold for every
//! generated input, not just the hand-written rows in the unit tables.
//!
//! The graph shapes come from `test::property::generators` wherever a generator
//! already builds what a pass consumes; the case budgets come from the shared
//! `cheap()`/`equivalence()` presets.

use std::sync::Arc;

use proptest::prelude::*;
use smallvec::{SmallVec, smallvec};
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, CallInfo, ConstValue, Op, ReduceOp, SInt, UOp, UOpKey, ops};

use super::helpers::{any_op, count_stages, loop_range, reduce_range};
use crate::rangeify::indexing::IndexingContext;
use crate::rangeify::patterns::{buffer_limit_patterns, pm_remove_bufferize};
use crate::rangeify::transforms::{rangeify, resolve_calls};
use crate::rangeify::{SplitRangesContext, pm_split_ranges};
use crate::rewrite::graph_rewrite;
use crate::test::property::checks::{fingerprint, operands};
use crate::test::property::generators::{arb_movement_graph, arb_movement_sink};
use crate::test::support::build::buffer;
use crate::test::support::eval::{Bindings, eval_typed, fold_at, range_points};
use crate::test::support::proptest::equivalence;

/// `where(range < cut, 1.0, 0.0)` over a reduce range of `extent` steps.
fn gated_body(extent: i64, cut: i64) -> (Arc<UOp>, Arc<UOp>) {
    let range = reduce_range(extent, 0);
    let gate = range.try_cmplt(&UOp::index_const(cut)).expect("cmplt");
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    (UOp::try_where(gate, UOp::native_const(1.0f32), zero).expect("where"), range)
}

/// The value that gated sum has over `[0, extent)`.
fn counted_extent(extent: i64, cut: i64) -> f32 {
    (0..extent).filter(|step| *step < cut).count() as f32
}

/// Every integer value `expr` takes over the whole box of its operands' ranges, sorted.
///
/// The cap is the box size, which [`range_points`] enumerates as a bijection, so
/// this is the exhaustive sweep and not a sample.
fn swept_values(expr: &Arc<UOp>) -> Vec<i64> {
    let mut values: Vec<i64> = range_points(&operands(&[expr]), usize::MAX)
        .filter_map(|bindings| match eval_typed(expr, &bindings) {
            Some(ConstValue::Int(value)) => Some(value),
            _ => None,
        })
        .collect();
    values.sort_unstable();
    values
}

/// A chain of `leaves` buffer reads on one axis, plus one more read beside it:
/// `leaves + 1` distinct buffers, with an elementwise node at every join.
fn buffer_chain(leaves: usize, ctx: &mut IndexingContext) -> Arc<UOp> {
    let range = ctx.new_range(&SInt::Const(16), AxisType::Loop);
    let read = |slot: usize| UOp::index().buffer(buffer(16 + slot)).indices(vec![range.clone()]).call().expect("INDEX");
    let chain = (0..leaves).map(read).reduce(|acc, term| acc.try_add(&term).expect("add")).expect("at least one leaf");
    chain.try_add(&read(leaves)).expect("add")
}

/// The sum of `leaves`, built over whichever operands are handed in.
fn sum_of(leaves: &[Arc<UOp>]) -> Arc<UOp> {
    leaves.iter().cloned().reduce(|acc, term| acc.try_add(&term).expect("add")).expect("at least one leaf")
}

proptest! {
    #![proptest_config(equivalence())]

    /// The collapse of a gated ADD must equal the brute-force sum. This is the
    /// oracle the `load_collapse` table rows cannot provide: they only check the
    /// few bounds a human thought of. A trip-1 extent is in range because
    /// `dead_loop_patterns` folds that axis to a constant before the collapse
    /// sees it, which `reduce_unparented` has to survive.
    #[test]
    fn a_gated_reduce_collapses_to_the_brute_force_count(extent in 1i64..24, cut in 0i64..24) {
        let (body, range) = gated_body(extent, cut);
        let collapsed = crate::rangeify::reduce_load_collapse(&body, &[range])
            .expect("a single-range gated ADD must collapse");
        prop_assert!(!any_op(&collapsed, |op| matches!(op, Op::Range(..))), "no RANGE survives: {}", collapsed.tree());
        let value = match collapsed.op() {
            Op::Const(constant) => constant.0.try_float().expect("a float constant") as f32,
            other => return Err(TestCaseError::fail(format!("expected a folded constant, got {other:?}"))),
        };
        prop_assert_eq!(value, counted_extent(extent, cut), "extent {} cut {}", extent, cut);
    }

    /// An unparented ADD folds to `source * extent`, for any signed source.
    #[test]
    fn an_unparented_add_scales_by_the_extent(source in -20i32..20, extent in 1i64..16) {
        let reduce = UOp::native_const(source).reduce(smallvec![reduce_range(extent, 0)], ReduceOp::Add);
        let folded = crate::rangeify::pm_reduce_simplify().rewrite(&reduce, &mut ());
        let svod_ir::RewriteResult::Rewritten(folded) = folded else {
            return Err(TestCaseError::fail("an unparented ADD must fold"));
        };
        let expected = ConstValue::Int(source as i64 * extent);
        prop_assert_eq!(fold_at(&folded, &Bindings::none()), Some(expected), "{}", folded.tree());
    }

    /// Split divisibility, the law `pm_split_ranges` rests on: `r % c` splits the
    /// axis into `outer * c + inner`, and that pair must walk exactly the indices
    /// `r` walked, once each. The modulo is only the marker; the bare axis beside
    /// it is what the substitution has to reconstruct.
    #[test]
    fn splitting_an_axis_reconstructs_every_index_exactly_once(outer in 1i64..8, factor in 2i64..9) {
        let extent = outer * factor;
        let axis = UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(0), AxisType::Loop);
        let sink = UOp::sink(vec![axis.mod_(&axis.const_like(factor)), axis.clone()]);

        let split = graph_rewrite(&pm_split_ranges(), sink.clone(), &mut SplitRangesContext::new());
        let Op::Sink(ops::Sink { sources, .. }) = split.op() else {
            return Err(TestCaseError::fail(format!("expected a SINK, got {}", split.tree())));
        };
        let reconstructed = sources[1].clone();

        prop_assert!(!Arc::ptr_eq(&reconstructed, &axis), "the axis must be rewritten: {}", split.tree());
        prop_assert_eq!(swept_values(&reconstructed), (0..extent).collect::<Vec<_>>(), "{}", split.tree());
    }

    /// Buffer-limit enforcement is exactly a threshold: a graph that reads more
    /// distinct buffers than one kernel may take gets a STAGE forced into it, and
    /// one that stays inside the budget comes back as the identical node.
    ///
    /// `check_buffer_limit` compares against `max_buffers - 1` because the store
    /// takes the last slot, and the widest node here is the root, which reads
    /// `leaves + 1` buffers.
    #[test]
    fn buffer_limits_materialise_exactly_above_the_limit(limit in 2usize..10, leaves in 2usize..6) {
        let mut ctx = IndexingContext::new();
        let root = buffer_chain(leaves, &mut ctx);
        prop_assert_eq!(count_stages(&root), 0, "the input is plain INDEX arithmetic");

        let result = graph_rewrite(&buffer_limit_patterns(limit), root.clone(), &mut ctx);

        if leaves + 1 > limit - 1 {
            prop_assert!(count_stages(&result) > 0, "an over-limit graph must materialise: {}", result.tree());
        } else {
            prop_assert!(Arc::ptr_eq(&result, &root), "a graph inside the budget is untouched: {}", result.tree());
        }
        prop_assert_eq!(result.dtype(), DType::Float32, "the value keeps its dtype");
        prop_assert!(
            !any_op(&result, |op| matches!(op, Op::Store(..) | Op::After(..))),
            "materialisation introduces STAGEs, never stores or ordering edges: {}",
            result.tree()
        );
    }

    /// `pm_remove_bufferize` declines any graph with no `INDEX(STAGE)`.
    #[test]
    fn remove_bufferize_leaves_bare_arithmetic_alone(lhs in -50i32..50, rhs in -50i32..50) {
        let root = UOp::native_const(lhs).try_add(&UOp::native_const(rhs)).expect("add");
        let result = pm_remove_bufferize().rewrite(&root, &mut ());
        prop_assert!(matches!(result, svod_ir::RewriteResult::NoMatch), "got {result:?}");
        prop_assert_eq!(fold_at(&root, &Bindings::none()), Some(ConstValue::Int((lhs as i64) + (rhs as i64))));
    }

    /// Inlining `INDEX(STAGE(compute, [r]), [r'])` is exactly the *gated*
    /// substitution the STAGE describes (`patterns.rs:577` runs
    /// `substitute_gated`, not `substitute`): the result equals the compute with
    /// `r -> r'`, and the STAGE is gone. The range-free summand is what makes the
    /// two substitutions distinguishable — it is a subtree the gate skips, so a
    /// gate that dropped or rebuilt it would show up here.
    #[test]
    fn an_inlined_stage_is_the_compute_with_the_buffer_range_substituted(
        scale in 1.0f32..8.0,
        step in 0.0f32..4.0,
    ) {
        let buffer_range = crate::test::support::build::global_range(8, 0);
        let index_range = crate::test::support::build::global_range(8, 1);
        let value = UOp::load()
            .index(UOp::index().buffer(buffer(8)).indices(vec![buffer_range.clone()]).call().expect("INDEX"))
            .call();
        let range_free = crate::test::support::build::load(crate::test::support::build::index(buffer(8), 0));
        let compute = value
            .try_mul(&UOp::native_const(scale)).expect("mul")
            .try_add(&range_free.try_add(&UOp::native_const(step)).expect("add")).expect("add");
        let staged = crate::test::support::build::stage(compute.clone(), vec![buffer_range.clone()]);
        let root = UOp::index().buffer(staged).indices(vec![index_range.clone()]).call().expect("INDEX");

        let inlined = match pm_remove_bufferize().rewrite(&root, &mut ()) {
            svod_ir::RewriteResult::Rewritten(inlined) => inlined,
            other => return Err(TestCaseError::fail(format!("a one-buffer STAGE must inline, got {other:?}"))),
        };
        prop_assert!(!any_op(&inlined, |op| matches!(op, Op::Stage(..))), "the STAGE is gone: {}", inlined.tree());

        let expected_map: std::collections::HashMap<_, _> =
            [(UOpKey(buffer_range.clone()), index_range.clone())].into_iter().collect();
        let expected = compute.substitute_gated(&expected_map);
        prop_assert!(
            Arc::ptr_eq(&inlined, &expected),
            "inlining must be the gated substitution itself\n got: {}\nwant: {}",
            inlined.tree(),
            expected.tree()
        );
    }

    /// `resolve_calls` *is* the inlining: the FUNCTION disappears and its body
    /// comes back with parameter `slot` replaced by argument `slot`. (Its
    /// fixpoint half belongs to `test::property::passes::rangeify_is_idempotent`,
    /// which makes that claim for the whole pipeline.)
    #[test]
    fn resolve_calls_inlines_the_body_over_its_arguments(terms in 1usize..5) {
        let params: Vec<Arc<UOp>> = (0..terms).map(|slot| UOp::param(slot, 8, DType::Float32, None)).collect();
        let args: SmallVec<[Arc<UOp>; 4]> = (0..terms).map(|_| buffer(8)).collect();
        let function = sum_of(&params).function(args.clone(), CallInfo::default());

        let resolved = resolve_calls(function).expect("a plain FUNCTION resolves");
        prop_assert!(!any_op(&resolved, |op| matches!(op, Op::Function(..) | Op::Param(..))), "{}", resolved.tree());
        // A FUNCTION body is always a TUPLE (`hardware.rs:124`), and resolving keeps it.
        let expected = sum_of(&args).maketuple();
        prop_assert!(
            Arc::ptr_eq(&resolved, &expected),
            "inlining must be the argument substitution itself\n got: {}\nwant: {}",
            resolved.tree(),
            expected.tree()
        );
    }

    /// The whole pipeline consumes every FUNCTION, whatever its arity.
    #[test]
    fn rangeify_leaves_no_function_behind(terms in 1usize..5) {
        let body = sum_of(&(0..terms).map(|slot| UOp::param(slot, 8, DType::Float32, None)).collect::<Vec<_>>());
        let args: SmallVec<[Arc<UOp>; 4]> = (0..terms).map(|_| buffer(8)).collect();
        let function = body.function(args, CallInfo::default());

        let (resolved, _) = rangeify(function).expect("rangeify");
        prop_assert!(!any_op(&resolved, |op| matches!(op, Op::Function(..))), "{}", resolved.tree());
    }

    /// Rangeify's job is to turn views into index arithmetic: a materialised view
    /// comes back as a program that keeps the root dtype and holds no movement op.
    #[test]
    fn rangeify_lowers_every_movement_op_to_index_arithmetic(graph in arb_movement_sink()) {
        let (resolved, _) = rangeify(graph.clone()).expect("rangeify accepts a movement sink");
        prop_assert_eq!(resolved.dtype(), graph.dtype(), "the root dtype survives");
        prop_assert!(!any_op(&resolved, is_movement), "a movement op survived: {}", resolved.tree());
    }

    /// Permuting a view and then undoing that permutation is the view itself, and
    /// rangeify must not be able to tell the two apart: the index arithmetic it
    /// builds for a round trip is the same program, node for node.
    #[test]
    fn a_permutation_and_its_inverse_lower_to_the_same_program(case in arb_movement_graph()) {
        let (permuted, dims, axes) = case;
        let mut inverse = vec![0usize; axes.len()];
        for (position, &axis) in axes.iter().enumerate() {
            inverse[axis] = position;
        }
        let shape: svod_ir::shape::Shape = dims.iter().map(|&dim| SInt::Const(dim as usize)).collect();
        let numel: i64 = dims.iter().product();
        let flat = buffer(numel as usize).try_reshape(&shape).expect("RESHAPE");
        let round_trip = permuted.try_permute(inverse).expect("the inverse permutation");

        let lowered = |view: Arc<UOp>| {
            rangeify(UOp::sink(vec![view.contiguous()])).expect("rangeify accepts a movement sink").0
        };
        prop_assert_eq!(fingerprint(&lowered(round_trip)), fingerprint(&lowered(flat)));
    }

    /// A forced STAGE keeps the value's dtype, whatever axis it was staged on.
    #[test]
    fn rangeify_preserves_the_value_dtype(extent in 1i64..8, terms in 1usize..4) {
        let value = (0..terms)
            .map(|_| UOp::native_const(2.0f32).try_mul(&loop_range(extent, 0).cast(DType::Float32)).expect("mul"))
            .reduce(|acc, term| acc.try_mul(&term).expect("mul"))
            .expect("at least one term");
        let (resolved, _) = rangeify(value.clone()).expect("rangeify");
        prop_assert_eq!(resolved.dtype(), value.dtype());
    }
}

/// The view ops rangeify exists to erase.
fn is_movement(op: &Op) -> bool {
    matches!(op, Op::Reshape(..) | Op::Permute(..) | Op::Expand(..) | Op::Pad(..) | Op::Shrink(..) | Op::Flip(..))
}
