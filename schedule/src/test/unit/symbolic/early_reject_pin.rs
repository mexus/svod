//! Equivalence pin for the pattern matcher's early reject on the production symbolic tiers.
//!
//! An early reject only skips entries whose fixed-position sources demand an op kind the
//! node's children do not have, so those entries could not have matched. Rewriting with the
//! rejects cleared must therefore produce the pointer-identical graph — hash consing makes
//! `Arc::ptr_eq` an exact structural check. The mechanism itself (derivation, `src_ops`,
//! wildcards) is covered by `svod_ir::test::unit::pattern::early_reject`; what is pinned
//! here is that the real tiers stay equivalent under it.
//!
//! Tinygrad equivalent: `if not early_reject.issubset(ler): continue` (uop/ops.py:1482).

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use svod_dtype::DType;
use svod_ir::op::pattern_derived::OpKey;
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::test::property::generators::arb_arithmetic_tree_up_to;
use svod_ir::{BinaryOp, ConstValue, Op, UOp, UnaryOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::{symbolic, symbolic_simple};
use crate::test::support::proptest::equivalence;

/// The tiers under test, with the op keys whose early-reject sets must be non-empty.
fn tiers() -> [(&'static TypedPatternMatcher, &'static str); 2] {
    [(symbolic(), "symbolic"), (symbolic_simple(), "symbolic_simple")]
}

/// Without a non-trivial requirement somewhere, every equivalence below would be vacuous.
const REQUIREMENT_KEYS: [OpKey; 3] =
    [OpKey::Binary(BinaryOp::Add), OpKey::Binary(BinaryOp::Mul), OpKey::Binary(BinaryOp::FloorMod)];

#[test]
fn the_production_tiers_derive_early_rejects() {
    for (matcher, label) in tiers() {
        let requirements = REQUIREMENT_KEYS
            .iter()
            .flat_map(|key| matcher.early_rejects(key))
            .filter(|reject| !reject.is_empty())
            .count();
        assert!(requirements > 0, "{label} derived no early rejects");
    }
}

fn idx(value: i64) -> Arc<UOp> {
    UOp::index_const(value)
}

/// `op(lhs, rhs)` as authored: a comparison takes the Bool it really has, everything else
/// inherits the left operand's dtype so mixed-width pairs still build.
fn bin(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Arc<UOp> {
    let dtype = if matches!(op, BinaryOp::Lt | BinaryOp::Ne) { DType::Bool } else { lhs.dtype() };
    UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), dtype)
}

/// A deterministic pool of arithmetic, comparison, select, cast and gated-index shapes over
/// the leaves the symbolic tiers reason about, built so each rule family sees both matching
/// and non-matching nodes.
///
/// The leaf set is the point: the pinned tier-2 entries key on node kinds no arithmetic-tree
/// generator emits — `range_based_mod_div_patterns` on a `RANGE`, `vmin_vmax_collapse_patterns`
/// on a `PARAM`/`SPECIAL`, `long_to_int_narrowing_patterns` on a weak-integer operand — and
/// `REQUIREMENT_KEYS` asserts a non-empty reject set for `FloorMod`, which needs a division
/// to reach at all.
fn graphs() -> Vec<Arc<UOp>> {
    let leaves = vec![
        UOp::range(idx(64), 0),
        UOp::range(idx(16), 1),
        UOp::special(idx(32), "gidx0".to_string()),
        UOp::var("n", DType::WeakInt, 1, 128),
        idx(0),
        idx(1),
        idx(4),
        idx(-3),
    ];
    let ops = [
        BinaryOp::Add,
        BinaryOp::Mul,
        BinaryOp::Sub,
        BinaryOp::Max,
        BinaryOp::FloorDiv,
        BinaryOp::FloorMod,
        BinaryOp::Lt,
        BinaryOp::Ne,
        BinaryOp::And,
        BinaryOp::Or,
    ];

    let mut pool: Vec<Arc<UOp>> = leaves.clone();
    for (i, lhs) in leaves.iter().enumerate() {
        for (j, rhs) in leaves.iter().enumerate() {
            let op = ops[(i * leaves.len() + j) % ops.len()];
            // Division by a literal zero is not a legal node to build.
            if matches!(op, BinaryOp::FloorDiv | BinaryOp::FloorMod)
                && matches!(rhs.op(), Op::Const(c) if c.0 == ConstValue::Int(0))
            {
                continue;
            }
            pool.push(bin(op, lhs, rhs));
        }
    }

    let (conditions, values): (Vec<Arc<UOp>>, Vec<Arc<UOp>>) =
        pool.iter().cloned().partition(|u| u.dtype() == DType::Bool);
    assert!(!conditions.is_empty(), "the WHERE shapes below need a boolean to gate on");
    let mut nested = Vec::new();
    for (i, value) in values.iter().enumerate() {
        nested.push(bin(BinaryOp::Add, value, &values[(i + 1) % values.len()]));
        nested.push(bin(BinaryOp::Mul, value, &values[(i + 3) % values.len()]));
        nested.push(UOp::new(Op::Unary(UnaryOp::Neg, value.clone()), value.dtype()));
        nested.push(value.cast(DType::Int32));
        let condition = conditions[i % conditions.len()].clone();
        // A WHERE needs both branches at one dtype; the pool deliberately mixes widths.
        if let Ok(selected) = UOp::try_where(condition, value.clone(), values[(i + 2) % values.len()].clone()) {
            nested.push(selected);
        }
    }
    pool.extend(nested);
    pool
}

fn assert_equivalent(matcher: &TypedPatternMatcher, graph: Arc<UOp>) -> Result<(), TestCaseError> {
    let permissive = matcher.without_early_reject();
    let with_reject = graph_rewrite(matcher, graph.clone(), &mut ());
    let without_reject = graph_rewrite(&permissive, graph, &mut ());
    prop_assert!(
        Arc::ptr_eq(&with_reject, &without_reject),
        "early reject diverged:\nwith:    {}\nwithout: {}",
        with_reject.tree(),
        without_reject.tree()
    );
    Ok(())
}

/// The deterministic half: every graph in the pool, at both tiers, every run.
#[test]
fn early_reject_preserves_symbolic_rewrites() {
    for (matcher, label) in tiers() {
        for graph in graphs() {
            assert_equivalent(matcher, graph.clone()).unwrap_or_else(|error| panic!("{label}: {error}"));
        }
    }
}

/// Wrap a pair of arithmetic trees into one of the shapes the symbolic rules branch on.
fn shape_graph(lhs: Arc<UOp>, rhs: Arc<UOp>, shape: usize) -> Arc<UOp> {
    let condition = lhs.clone().lt(&rhs.clone());
    match shape {
        0 => lhs,
        1 => lhs.try_add(&rhs).expect("add builds"),
        2 => lhs.try_mul(&rhs).expect("mul builds"),
        3 => condition,
        4 => UOp::try_where(condition, lhs, rhs).expect("where builds"),
        5 => lhs.cast(DType::Int64),
        _ => lhs.try_mod(&rhs.or_(&UOp::native_const(1i32))).expect("mod builds"),
    }
}

proptest! {
    #![proptest_config(equivalence())]

    /// The randomised supplement: deeper Int32 arithmetic than the pool enumerates, where
    /// clearing the early-reject derivation must still not change the rewritten graph.
    #[test]
    fn early_reject_is_semantics_preserving(
        lhs in arb_arithmetic_tree_up_to(DType::Int32, 3),
        rhs in arb_arithmetic_tree_up_to(DType::Int32, 3),
        shape in 0usize..7,
    ) {
        let graph = shape_graph(lhs, rhs, shape);
        for (matcher, _) in tiers() {
            assert_equivalent(matcher, graph.clone())?;
        }
    }
}
