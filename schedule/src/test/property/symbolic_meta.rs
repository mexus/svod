//! Structural invariants every symbolic rewrite must hold: a fixpoint, dtype-preserving,
//! acyclic, and not meaningfully larger. The generator covers every op and dtype the tables
//! rewrite, so a rule that only misbehaves at Int8 or on a Bool `And` is reached.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::DType;
use svod_ir::{Op, TypedPatternMatcher, UOp};

use crate::symbolic::symbolic;
use crate::test::property::generators::{arb_op_tree_up_to, arb_property_dtype};
use crate::test::support::prelude::*;

use svod_ir::test::property::generators::*;
use svod_ir::test::property::shrinking::{uop_depth, uop_op_count};

/// Every constant must stay in the same type family as the node holding it.
fn prop_assert_constant_dtypes(uop: &Arc<UOp>) -> Result<(), TestCaseError> {
    let is_int = |dtype: &DType| dtype.scalar().is_some_and(|scalar| scalar.is_int());
    for node in uop.toposort() {
        let Op::Const(value) = node.op() else { continue };
        prop_assert_eq!(
            is_int(&value.0.dtype()),
            is_int(&node.dtype()),
            "constant dtype family mismatch: {:?} in a {:?} node",
            value.0.dtype(),
            node.dtype()
        );
    }
    Ok(())
}

/// A tree over the full op surface at every dtype family the tables claim, plus the
/// algebraic shapes only `known_property_graph` builds.
fn arb_graph() -> impl Strategy<Value = Arc<UOp>> {
    let typed = arb_property_dtype;
    prop_oneof![
        4 => typed().prop_flat_map(|dtype| arb_op_tree_up_to(dtype.clone(), 4)),
        1 => typed().prop_flat_map(|dtype| arb_arithmetic_tree_up_to(dtype, 4)),
        1 => arb_known_property_graph().prop_map(|graph| graph.build()),
    ]
}

/// Rewriting twice must equal rewriting once, the dtype must survive, every constant
/// must keep its type family, and the result must toposort (a cycle panics).
fn rewriting_is_stable(matcher: &TypedPatternMatcher, graph: Arc<UOp>) -> Result<Arc<UOp>, TestCaseError> {
    let dtype = graph.dtype();
    let once = rewrite(matcher, graph);
    let twice = rewrite(matcher, once.clone());
    prop_assert!(
        Arc::ptr_eq(&once, &twice),
        "rewriting twice must equal rewriting once\nonce:  {}\ntwice: {}",
        once.tree(),
        twice.tree()
    );
    prop_assert_eq!(once.dtype(), dtype, "rewriting must preserve the dtype");
    prop_assert_constant_dtypes(&once)?;
    once.toposort();
    Ok(once)
}

proptest! {
    // The two tier-1 tables merged into one `full: bool` property, so double the budget.
    #![proptest_config(ProptestConfig::with_cases(2 * CHEAP))]

    /// Rewriting is a structure-preserving fixpoint under both tier-1 tables: `full`
    /// selects `Matchers::full`, which is `symbolic_simple() + pm_fold_cast_const()` — the
    /// table the DCE tests fold with, *not* the tier-2 one. Tier 2 is `symbolic()` and has
    /// its own property below. Distribution rules may add a couple of nodes before a later
    /// fold removes them, hence the slack on the size bounds; dtype, acyclicity and
    /// constant families are absolute.
    #[test]
    fn rewriting_is_a_structure_preserving_fixpoint(graph in arb_graph(), full in any::<bool>()) {
        let (ops, depth) = (uop_op_count(&graph), uop_depth(&graph));
        let matcher = if full { Matchers::full() } else { Matchers::simple() };
        let once = rewriting_is_stable(matcher, graph)?;
        prop_assert!(uop_op_count(&once) <= ops + 2, "op count grew {} -> {}", ops, uop_op_count(&once));
        prop_assert!(uop_depth(&once) <= depth + 1, "depth grew {} -> {}", depth, uop_depth(&once));
    }

    /// The tier-2 table — `symbolic()`, which adds the term-combining, range-based and
    /// affine-congruence rules the tier-1 tables never run — must reach a fixpoint too.
    /// No other property in the suite asserts that; the size bounds are deliberately not
    /// claimed here, because those rules legitimately restructure a division into a wider
    /// affine form on the way to folding it.
    #[test]
    fn the_tier_two_table_is_a_structure_preserving_fixpoint(graph in arb_graph()) {
        rewriting_is_stable(symbolic(), graph)?;
    }
}

proptest! {
    // These two shapes keep their own budget rather than a sixth of `arb_graph`'s mixture,
    // which is what merging them into its `prop_oneof` had cost them.
    #![proptest_config(ProptestConfig::with_cases(2 * CHEAP))]

    /// Deep Int32 arithmetic: the shape the size bounds above were written against.
    #[test]
    fn arithmetic_trees_rewrite_to_a_fixpoint(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        rewriting_is_stable(Matchers::simple(), graph)?;
    }

    /// Graphs built from known algebraic identities, which reach far more rules than the
    /// random arithmetic trees do.
    #[test]
    fn known_property_graphs_rewrite_to_a_fixpoint(kpg in arb_known_property_graph()) {
        rewriting_is_stable(Matchers::simple(), kpg.build())?;
    }
}
