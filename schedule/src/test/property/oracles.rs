//! Z3-backed oracles (`--features z3`): a formal equivalence check for the rewrites the
//! structural and algebraic properties only sample. Timeouts and conversion failures are
//! not counterexamples, but `Found` never is.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::DType;
use svod_ir::UOp;
use svod_ir::types::ConstValue;

use crate::symbolic::symbolic;
use crate::test::property::checks::{same_value, shape};
use crate::test::support::prelude::*;
use crate::z3::{CounterExample, verify_equivalence};

use svod_ir::test::property::generators::*;

/// Whether Z3 *proved* the rewrite equivalent. An unsupported operation or a timeout is
/// not a failure — the converter does not model the whole IR — but a counterexample is,
/// and it fails here rather than being folded into the verdict.
fn z3_proves_equivalent(original: &Arc<UOp>, rewritten: &Arc<UOp>) -> Result<bool, TestCaseError> {
    match verify_equivalence(original, rewritten) {
        Ok(()) => Ok(true),
        Err(CounterExample::ConversionFailed { .. } | CounterExample::Unknown { .. }) => Ok(false),
        Err(CounterExample::Found { .. }) => Err(TestCaseError::fail(format!(
            "Z3 found a counterexample:\noriginal:  {}\nrewritten: {}",
            original.tree(),
            rewritten.tree()
        ))),
    }
}

/// [`z3_proves_equivalent`], discarding the verdict: a proof is welcome, a declined one is
/// tolerated, a counterexample fails.
fn assert_z3_equivalent(original: &Arc<UOp>, rewritten: &Arc<UOp>) -> Result<(), TestCaseError> {
    z3_proves_equivalent(original, rewritten).map(drop)
}

proptest! {
    // The Z3 oracles are slow per case but each one is a formal proof; these are the
    // budgets they were written with, which `cheap()` had cut by half.
    #![proptest_config(proptest_config(500))]

    /// Every known-property graph must reduce to the known answer — in value, so that
    /// an equivalent-but-unfolded form is reported as a missed fold and not as a wrong
    /// answer — and Z3 must prove the rewrite equivalent. Z3 sees the graph rebuilt at
    /// Int32: at UInt8 the IR's `x - x` is `x + x*255`, which wraps and looks like a
    /// counterexample to an unbounded solver.
    #[test]
    fn known_property_graphs_reduce_to_their_expected_value(kpg in arb_known_property_graph()) {
        let graph = kpg.build();
        let simplified = rewrite(symbolic(), graph.clone());
        if let Some(expected) = kpg.expected_result() {
            same_value(&graph, &expected)?;
            same_value(&simplified, &expected)?;
        }
        let (signed, _) = known_property_at(&kpg, DType::Int32);
        let signed_rewritten = rewrite(symbolic(), signed.clone());
        same_value(&signed, &signed_rewritten)?;
        assert_z3_equivalent(&signed, &signed_rewritten)?;
    }

    /// The identity elimination `x + 0 = x` is pointer-identical and Z3-proven.
    #[test]
    fn z3_verify_identity_add_zero(x in arb_var_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_add(&zero).expect("ADD accepts matching dtypes");
        let simplified = rewrite(Matchers::simple(), expr.clone());
        prop_assert!(Arc::ptr_eq(&simplified, &x));
        verify_equivalence(&expr, &simplified).expect("Z3 should verify x + 0 = x");
    }

    /// Zero propagation `x * 0 = 0` is pointer-identical and Z3-proven.
    #[test]
    fn z3_verify_zero_mul(x in arb_var_uop(DType::Int32)) {
        let zero = UOp::native_const(0i32);
        let expr = x.try_mul(&zero).expect("MUL accepts matching dtypes");
        let simplified = rewrite(Matchers::simple(), expr.clone());
        prop_assert!(Arc::ptr_eq(&simplified, &zero));
        verify_equivalence(&expr, &simplified).expect("Z3 should verify x * 0 = 0");
    }

    /// Self-division is `1` whenever the declared range excludes zero, and Z3 proves
    /// the rule is sound.
    #[test]
    fn z3_verify_self_div(name in "[a-z]", min_val in 1i64..100, range_size in 1i64..100) {
        let x = UOp::var(&name, DType::Int32, min_val, min_val + range_size);
        let expr = x.try_div(&x).expect("a nonzero variable is a legal divisor");
        let simplified = rewrite(Matchers::simple(), expr.clone());
        assert_const!(simplified, 1);
        verify_equivalence(&expr, &simplified).expect("Z3 should verify x / x = 1 for x != 0");
    }

    /// Z3 over whole arithmetic trees, where the structural properties cannot
    /// enumerate the input space. Bounded constants keep Z3's unbounded integers
    /// inside the IR's wrapping semantics.
    #[test]
    fn z3_verify_arithmetic_optimization(graph in arb_arithmetic_tree_bounded_up_to(DType::Int32, 3)) {
        let optimized = rewrite(Matchers::simple(), graph.clone());
        assert_z3_equivalent(&graph, &optimized)?;
    }

}

proptest! {
    #![proptest_config(proptest_config(300))]

    /// The dtype-widening oracle: a rewrite that fires at the narrowest member of a
    /// family must fire identically at the widest, and both must be Z3-provable.
    ///
    /// Pointer equality with the expected form is claimed only for the rules with a unique
    /// form and a non-float family: `x - x` is narrowed through CASTs the term combiner does
    /// not see through at the widest member, and `x + 0.0` is deliberately *not* the identity
    /// under IEEE 754.
    #[test]
    fn dtype_widening_keeps_the_rewrite_and_its_proof(kpg in arb_known_property_graph(), family in arb_dtype_family()) {
        let (narrow, narrow_expected) = known_property_at(&kpg, family.narrowest());
        let (wide, wide_expected) = known_property_at(&kpg, family.widest());
        let (narrow_rewritten, wide_rewritten) = (rewrite(symbolic(), narrow.clone()), rewrite(symbolic(), wide.clone()));
        same_value(&narrow, &narrow_rewritten)?;
        same_value(&wide, &wide_rewritten)?;
        if matches!(family, DTypeFamily::SignedInt) {
            // Only the signed family has an exact unbounded-integer encoding. Beyond "no
            // counterexample either way", the two verdicts are related: the narrow and the
            // wide member differ only in a dtype the converter does not even encode, so a
            // proof at the narrow width that the wide one cannot reproduce — a conversion
            // failure included — means the rewrite itself changed shape with the width.
            let narrow_proven = z3_proves_equivalent(&narrow, &narrow_rewritten)?;
            let wide_proven = z3_proves_equivalent(&wide, &wide_rewritten)?;
            prop_assert!(
                !narrow_proven || wide_proven,
                "{:?} verified at {:?} but not at {:?}\nnarrow: {}\nwide:   {}",
                kpg,
                family.narrowest(),
                family.widest(),
                narrow_rewritten.tree(),
                wide_rewritten.tree()
            );
        }
        // The structural shape of a rewriting *result*, ignoring the dtype that distinguishes a
        // family's members.
        fn bare(uop: &Arc<UOp>) -> Vec<(svod_ir::op::OpMask, Vec<usize>)> {
            shape(uop, |_| ()).into_iter().map(|(mask, (), children)| (mask, children)).collect()
        }
        if !matches!(family, DTypeFamily::Float) {
            let (Some(narrow_expected), Some(wide_expected)) = (narrow_expected, wide_expected) else { return Ok(()) };
            prop_assert_eq!(bare(&narrow_rewritten), bare(&narrow_expected), "narrow");
            prop_assert_eq!(bare(&wide_rewritten), bare(&wide_expected), "wide");
        }
    }
}

/// `kpg` rebuilt at `dtype`, preserving the operation the generator chose, together with
/// the structural shape of the form it must reduce to (`None` when there is no unique form).
fn known_property_at(kpg: &KnownPropertyGraph, dtype: DType) -> (Arc<UOp>, Option<Arc<UOp>>) {
    let x = UOp::var("x", dtype.clone(), 0, 100);
    let constant = |value: i64| UOp::const_(dtype.clone(), ConstValue::Int(value));
    let expected = match kpg {
        KnownPropertyGraph::AddZero { .. } | KnownPropertyGraph::MulOne { .. } | KnownPropertyGraph::SubZero { .. } => {
            Some(x.clone())
        }
        KnownPropertyGraph::MulZero { .. } => Some(UOp::const_(dtype.clone(), ConstValue::Int(0))),
        KnownPropertyGraph::SubSelf { .. } | KnownPropertyGraph::AddSelf { .. } => None,
    };
    let graph = match kpg {
        KnownPropertyGraph::AddZero { .. } => x.try_add(&constant(0)).expect("ADD"),
        KnownPropertyGraph::MulOne { .. } => x.try_mul(&constant(1)).expect("MUL"),
        KnownPropertyGraph::SubZero { .. } => x.try_sub(&constant(0)).expect("SUB"),
        KnownPropertyGraph::MulZero { .. } => x.try_mul(&constant(0)).expect("MUL"),
        KnownPropertyGraph::SubSelf { .. } => x.try_sub(&x).expect("SUB"),
        KnownPropertyGraph::AddSelf { .. } => x.try_add(&x).expect("ADD"),
    };
    (graph, expected)
}
