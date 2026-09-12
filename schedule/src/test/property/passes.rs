//! Pass-level invariants: every rewrite pass must be a fixpoint, and the passes that only
//! reorganise a program must preserve what it evaluates to. These are `symbolic_meta`'s
//! claims for the passes after the symbolic tier, plus the lowering chain's metamorphic relation.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::DType;
use svod_ir::UOp;

use crate::devectorize::{ReduceContext, bool_storage_patterns, no_vectorized_alu, pm_reduce};
use crate::optimizer::{Renderer, apply_pre_optimization};
use crate::rangeify::{SimplifyRangesContext, pm_simplify_ranges, rangeify};
use crate::rewrite::graph_rewrite;
use crate::symbolic::{pm_fold_cast_const, symbolic, symbolic_simple};
use crate::test::property::checks::{fingerprint, same_value, samples};
use crate::test::property::generators::{
    arb_bool_memory_graph, arb_gated_range_graph, arb_kernel_graph, arb_movement_sink, arb_op_tree_up_to,
    arb_reduce_graph, arb_shaped_alu_graph, arb_strength_reducible_graph,
};
use crate::test::support::prelude::*;

/// A pass runs to a fixed point: applying it to its own output changes nothing,
/// keeps the dtype, and leaves a DAG that toposorts (it panics on a cycle).
///
/// The dtype is compared against the *input* graph's. Comparing the two outputs instead is
/// a tautology: they are already pointer-equal by the assertion above it.
fn fixpoint(pass: impl Fn(Arc<UOp>) -> Arc<UOp>, graph: Arc<UOp>) -> Result<(), TestCaseError> {
    let dtype = graph.dtype();
    let once = pass(graph);
    let twice = pass(once.clone());
    prop_assert!(Arc::ptr_eq(&once, &twice), "pass is not a fixpoint\nonce:  {}\ntwice: {}", once.tree(), twice.tree());
    prop_assert_eq!(once.dtype(), dtype, "a pass must preserve the root dtype");
    once.toposort();
    Ok(())
}

/// Draws per anti-vacuity sample, and the share of them a pass must rewrite; see
/// [`every_generator_reaches_its_pass`]. Four of the five generators reach their pass on
/// every draw; `arb_gated_range_graph` draws its bound independently of the range extent,
/// so roughly one draw in fourteen carries a gate that does not narrow anything and has
/// nothing for `pm_simplify_ranges` to do. Three quarters is the bar that shape clears with
/// room to spare while still catching a generator that has drifted away from its pass.
const GUARD_DRAWS: usize = 64;
const GUARD_MIN_HITS: usize = GUARD_DRAWS * 3 / 4;

/// A generator must reach its pass on at least [`GUARD_MIN_HITS`] of its draws.
#[track_caller]
fn reaches(what: &str, graphs: Vec<Arc<UOp>>, pass: impl Fn(Arc<UOp>) -> Arc<UOp>) {
    let (hits, missed): (Vec<&Arc<UOp>>, Vec<&Arc<UOp>>) =
        graphs.iter().partition(|graph| !Arc::ptr_eq(graph, &pass((*graph).clone())));
    assert!(
        hits.len() >= GUARD_MIN_HITS,
        "{what} fired on only {} of {} draws, e.g. not on\n{}",
        hits.len(),
        graphs.len(),
        missed[0].tree()
    );
}

proptest! {
    // Four single-pass fixpoints merged into one pass-parameterised property.
    #![proptest_config(ProptestConfig::with_cases(4 * CHEAP))]

    /// Each single-pass lowering runs to a fixed point on the graphs it is written
    /// for; `which` picks the pass. `pm_reduce`'s accumulator loop must hold no
    /// REDUCE for a second pass to lower, Bool storage leaves a CAST that must not
    /// match again, devectorised ALU has no shape left, and range simplification
    /// narrows a gated range at most once.
    #[test]
    fn single_pass_lowering_is_a_fixpoint(case in prop_oneof![
        arb_reduce_graph().prop_map(|graph| (0usize, graph)),
        arb_bool_memory_graph().prop_map(|graph| (1, graph)),
        arb_shaped_alu_graph().prop_map(|graph| (2, graph)),
        arb_gated_range_graph().prop_map(|graph| (3, graph)),
    ]) {
        let (which, graph) = case;
        fixpoint(
            |u| match which {
                0 => graph_rewrite(&pm_reduce(), u, &mut ReduceContext::default()),
                1 => graph_rewrite(bool_storage_patterns(), u, &mut ()),
                2 => graph_rewrite(no_vectorized_alu(), u, &mut ()),
                _ => graph_rewrite(&pm_simplify_ranges(), u, &mut SimplifyRangesContext::default()),
            },
            graph,
        )?;
    }

    /// The late strength-reduction matchers rewrite a power-of-two operand once.
    #[test]
    fn late_decompositions_are_a_fixpoint(graph in prop_oneof![
        arb_strength_reducible_graph(),
        arb_op_tree_up_to(DType::Int32, 3),
    ]) {
        use crate::rangeify::patterns::{
            pm_comparison_negations, pm_div_to_shr, pm_fdiv_to_mul, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul,
        };
        let matcher = symbolic_simple()
            + pm_fold_cast_const()
            + pm_mul_to_shl()
            + pm_mod_to_and()
            + pm_div_to_shr()
            + pm_fdiv_to_mul()
            + pm_neg_from_mul()
            + pm_comparison_negations();
        fixpoint(|u| graph_rewrite(&matcher, u, &mut ()), graph)?;
    }

    /// Per-kernel pre-optimization is a fixpoint over every kernel it accepts.
    #[test]
    fn pre_optimization_is_a_fixpoint(graph in prop_oneof![arb_kernel_graph(), arb_movement_sink()]) {
        fixpoint(|u| apply_pre_optimization(u).expect("pre-optimization accepts a plain graph"), graph)?;
    }
}

proptest! {
    // The two symbolic tiers merged into one `full: bool` property.
    #![proptest_config(ProptestConfig::with_cases(2 * EQUIVALENCE))]

    /// `rangeify` lowers movement ops to STAGE/INDEX; a second pass over its own
    /// output must produce the same program. Node *identity* cannot be compared:
    /// `add_tags` re-tags every node each run, so the claim is the structural one.
    #[test]
    fn rangeify_is_idempotent(graph in arb_movement_sink()) {
        let once = rangeify(graph).expect("rangeify accepts a supported sink").0;
        let twice = rangeify(once.clone()).expect("rangeify accepts its own output").0;
        prop_assert_eq!(
            fingerprint(&once),
            fingerprint(&twice),
            "rangeify is not idempotent\nonce:  {}\ntwice: {}",
            once.tree(),
            twice.tree()
        );
    }

    /// `devectorize` scalarizes every shaped op in one call; a second call must find
    /// nothing left to scalarize.
    #[test]
    fn devectorize_is_idempotent(graph in arb_movement_sink()) {
        let once = crate::devectorize::devectorize(&rangeify(graph).expect("rangeify").0, &Renderer::cpu());
        let twice = crate::devectorize::devectorize(&once, &Renderer::cpu());
        prop_assert!(Arc::ptr_eq(&once, &twice), "devectorize is not idempotent\nonce:  {}\ntwice: {}", once.tree(), twice.tree());
    }

    /// Every symbolic rewrite must compute the same value at every point of the
    /// operands' ranges; `full` selects the tier-2 matcher, which adds the range-based
    /// rules the simple tier does not run.
    #[test]
    fn symbolic_rewrites_preserve_evaluated_values(graph in arb_op_tree_up_to(DType::Int32, 3), full in any::<bool>()) {
        let matcher = if full { symbolic() } else { symbolic_simple() };
        same_value(&graph, &graph_rewrite(matcher, graph.clone(), &mut ()))?;
    }
}

/// Each generator must produce graphs its pass actually rewrites. A single draw is not
/// enough evidence: it is satisfied by a generator that reaches its pass once in a hundred
/// draws, while `single_pass_lowering_is_a_fixpoint` stays vacuous on the other ninety-nine.
/// So the guard samples a batch and requires a hit *rate*.
#[test]
fn every_generator_reaches_its_pass() {
    reaches("pm_reduce", samples(arb_reduce_graph(), GUARD_DRAWS), |graph| {
        graph_rewrite(&pm_reduce(), graph, &mut ReduceContext::default())
    });
    reaches("bool_storage_patterns", samples(arb_bool_memory_graph(), GUARD_DRAWS), |graph| {
        graph_rewrite(bool_storage_patterns(), graph, &mut ())
    });
    reaches("no_vectorized_alu", samples(arb_shaped_alu_graph(), GUARD_DRAWS), |graph| {
        graph_rewrite(no_vectorized_alu(), graph, &mut ())
    });
    reaches("pm_simplify_ranges", samples(arb_gated_range_graph(), GUARD_DRAWS), |graph| {
        graph_rewrite(&pm_simplify_ranges(), graph, &mut SimplifyRangesContext::default())
    });
    reaches("pre-optimization", samples(arb_kernel_graph(), GUARD_DRAWS), |graph| {
        apply_pre_optimization(graph).expect("pre-optimization accepts a plain graph")
    });
}
