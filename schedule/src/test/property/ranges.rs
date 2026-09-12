//! Range and cache properties: a derived range must contain every value the graph can
//! take, and the memoised range/shape views must agree with the graph. `vmin`/`vmax`
//! guards every value-sensitive rule, so a wrong bound silently miscompiles.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::DType;
use svod_ir::Op;
use svod_ir::UOp;
use svod_ir::types::ConstValue;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::{SoundVminVmaxProperty, VminVmaxProperty};

use crate::symbolic::{symbolic, symbolic_simple};
use crate::test::property::checks::{SAMPLES, operands};
use crate::test::property::generators::{arb_int_property_dtype, arb_movement_graph, arb_op_tree_up_to};
use crate::test::support::prelude::*;

/// The integer width of a constant, if it has one.
fn integer(value: ConstValue) -> Option<i128> {
    match value {
        ConstValue::Int(value) => Some(value as i128),
        ConstValue::UInt(value) => Some(value as i128),
        ConstValue::Bool(value) => Some(value as i128),
        _ => None,
    }
}

/// `vmin <= value <= vmax` at every sampled point of the declared ranges.
fn bounds_every_point(graph: &Arc<UOp>, bounds: (ConstValue, ConstValue)) -> Result<(), TestCaseError> {
    let (Some(low), Some(high)) = (integer(bounds.0), integer(bounds.1)) else {
        prop_assert!(false, "integer graph has non-integer bounds {:?}", bounds);
        return Ok(());
    };
    prop_assert!(low <= high, "inverted range [{low}, {high}] for\n{}", graph.tree());
    // Points are folded in as dtype-correct constants ([`fold_at`]), the same way
    // [`crate::test::property::checks::same_value`] does and for the same reason: half the
    // dtypes this property sweeps are unsigned, a point the evaluator cannot reach is
    // skipped silently, and a property that compared nothing would still pass — which is
    // precisely how a missing unsigned zero bound goes unnoticed.
    for bindings in range_points(&operands(&[graph]), SAMPLES) {
        if let Some(value) = fold_at(graph, &bindings).and_then(integer) {
            prop_assert!(
                low <= value && value <= high,
                "value {value} outside [{low}, {high}] at {bindings:?}\n{}",
                graph.tree()
            );
        }
    }
    Ok(())
}

/// The RANGE ids below `node` and the ids it memoises, both sorted and deduped.
fn cached_ranges(node: &Arc<UOp>) -> (Vec<u64>, Vec<u64>) {
    let mut memoised: Vec<u64> = node.ranges().iter().map(|range| range.id).collect();
    let mut collected: Vec<u64> =
        node.toposort().iter().filter(|n| matches!(n.op(), Op::Range(..))).map(|n| n.id).collect();
    for ids in [&mut memoised, &mut collected] {
        ids.sort_unstable();
        ids.dedup();
    }
    (memoised, collected)
}

proptest! {
    // Two per-dtype bound properties merged into the dtype-parameterised one below.
    #![proptest_config(ProptestConfig::with_cases(2 * CHEAP))]

    /// The cached range analysis must contain every value the graph evaluates to, at
    /// every integer dtype: a missing unsigned zero bound is the usual bug.
    #[test]
    fn derived_value_range_bounds_every_evaluated_point(graph in arb_int_property_dtype().prop_flat_map(|dtype| arb_op_tree_up_to(dtype, 3))) {
        bounds_every_point(&graph, *VminVmaxProperty::get(&graph))?;
    }

    /// [`SoundVminVmaxProperty`] may decline, but when it answers it must be sound —
    /// this is the oracle the constant-folding guards read.
    #[test]
    fn sound_value_range_bounds_every_evaluated_point(graph in arb_op_tree_up_to(DType::Int32, 3)) {
        if let Some(bounds) = *SoundVminVmaxProperty::get(&graph) {
            bounds_every_point(&graph, bounds)?;
        }
    }

    /// Rewriting must not widen the declared range: a pass that loses a bound
    /// silently disables the value-sensitive rules downstream.
    #[test]
    fn rewriting_never_widens_the_derived_range(graph in arb_op_tree_up_to(DType::Int32, 3)) {
        let bounds = |u: &Arc<UOp>| (integer(VminVmaxProperty::get(u).0), integer(VminVmaxProperty::get(u).1));
        let (before, after) = (bounds(&graph), bounds(&rewrite(Matchers::simple(), graph.clone())));
        if let (Some(before), Some(after)) = (before.0, after.0) {
            prop_assert!(after >= before, "rewriting raised vmin {before} -> {after}");
        }
        if let (Some(before), Some(after)) = (before.1, after.1) {
            prop_assert!(after <= before, "rewriting lowered vmax {before} -> {after}");
        }
    }

    /// Every node's memoised `ranges` is exactly the set of RANGE nodes below it, and
    /// its in-scope set is a subset that always contains its own RANGE. The inputs
    /// cover the raw graph and each rewriting tier, so a stale cache after a real
    /// rewrite (a rewrite-order bug) is caught too.
    #[test]
    fn range_caches_agree_with_the_graph(graph in arb_op_tree_up_to(DType::Int32, 3)) {
        let tiers = [graph.clone(), rewrite(Matchers::simple(), graph.clone()), rewrite(Matchers::full(), graph.clone()), crate::rewrite::graph_rewrite(symbolic(), graph, &mut ())];
        for rewritten in tiers {
            for node in rewritten.toposort() {
                let (memoised, collected) = cached_ranges(&node);
                prop_assert_eq!(&memoised, &collected, "cached ranges disagree at\n{}", node.tree());
                for in_scope in node.in_scope_ranges() {
                    prop_assert!(memoised.binary_search(in_scope).is_ok(), "in-scope range {in_scope} is not in the graph at\n{}", node.tree());
                }
                if matches!(node.op(), Op::Range(..)) {
                    prop_assert!(node.in_scope_ranges().contains(&node.id), "a RANGE must be in scope at itself\n{}", node.tree());
                }
            }
        }
    }


    /// `PERMUTE(PERMUTE(x, p), p^-1)` is `x` again, and every movement step reports
    /// the shape the algebra says it has.
    #[test]
    fn reshape_and_permute_round_trip((graph, dims, axes) in arb_movement_graph()) {
        prop_assert_eq!(graph.shape().expect("movement shape").expect("a shaped node").len(), dims.len());
        let mut inverse = vec![0usize; axes.len()];
        for (position, &axis) in axes.iter().enumerate() {
            inverse[axis] = position;
        }
        let permuted_back = graph.try_permute(inverse).expect("the inverse is a permutation");
        prop_assert_eq!(permuted_back.shape().expect("movement shape"), graph.shape().expect("movement shape"), "a double permutation must restore the shape\n{}", permuted_back.tree());
        let flat: svod_ir::shape::Shape = [svod_ir::SInt::Const(dims.iter().product::<i64>() as usize)].into_iter().collect();
        let reshaped_back = graph.try_reshape(&flat).expect("RESHAPE keeps the element count");
        prop_assert_eq!(reshaped_back.shape().expect("movement shape"), Some(&flat), "reshaping to the flat shape must give the flat shape\n{}", reshaped_back.tree());
    }
}

/// The same symbolic tier reached through the raw matcher, to cover `symbolic_simple`
/// on movement graphs as well.
#[test]
fn movement_shapes_are_stable_across_rewrites() {
    for graph in [
        elementwise(&[4, 6], svod_ir::AxisType::Global),
        matmul(4, 6, 8, DType::Float32, None),
        reduce_sink(&[4], &[8], svod_ir::ReduceOp::Add),
    ] {
        let rewritten = crate::rewrite::graph_rewrite(symbolic_simple(), graph.clone(), &mut ());
        assert_eq!(
            rewritten.shape().expect("shape"),
            graph.shape().expect("shape"),
            "symbolic_simple changed the shape of\n{}",
            graph.tree()
        );
    }
}
