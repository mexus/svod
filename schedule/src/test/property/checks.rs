//! Shared property checks: structural comparison, value agreement and collapse drivers.

use std::collections::HashMap;
use std::sync::Arc;

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use svod_dtype::DType;
use svod_ir::op::OpMask;
use svod_ir::types::BinaryOp;
use svod_ir::{Op, TypedPatternMatcher, UOp};

use crate::test::property::generators::build_binary;
use crate::test::support::prelude::*;

/// Points sampled per semantic check.
pub const SAMPLES: usize = 64;

/// A graph's structure in topological order: each node's op mask, a caller-chosen
/// `label`, and its children by position. Ids, tags and origins are absent, so two
/// runs that re-intern the same graph compare equal.
pub fn shape<L>(uop: &Arc<UOp>, label: impl Fn(&UOp) -> L) -> Vec<(OpMask, L, Vec<usize>)> {
    let nodes = uop.toposort();
    let positions: HashMap<u64, usize> = nodes.iter().enumerate().map(|(i, node)| (node.id, i)).collect();
    nodes
        .iter()
        .map(|node| {
            let children = node.op().children().iter().map(|child| positions[&child.id]).collect();
            (OpMask::of_op(node.op()), label(node), children)
        })
        .collect()
}

/// The structural signature of one graph: op kind, dtype and child positions.
pub fn fingerprint(uop: &Arc<UOp>) -> Vec<(OpMask, DType, Vec<usize>)> {
    shape(uop, |node| node.dtype())
}

/// The distinct symbolic operands reachable from `graphs`.
pub fn operands(graphs: &[&Arc<UOp>]) -> Vec<Arc<UOp>> {
    let mut vars: Vec<Arc<UOp>> = Vec::new();
    for node in graphs.iter().flat_map(|graph| graph.toposort()) {
        if var_range(&node).is_some() && !vars.iter().any(|seen| Arc::ptr_eq(seen, &node)) {
            vars.push(node);
        }
    }
    vars
}

/// `left` and `right` must evaluate identically at every sampled point of the operands'
/// ranges. Points are folded as dtype-correct constants ([`fold_at`]), because a raw
/// `Bindings` entry leaves an unsigned operand unevaluated and the comparison silently passes.
pub fn same_value(left: &Arc<UOp>, right: &Arc<UOp>) -> Result<(), TestCaseError> {
    same_value_over(left, right, SAMPLES)
}

/// [`same_value`] with an explicit sweep budget. `range_points` enumerates the whole
/// cartesian product of the operands' ranges only when `cap` reaches its size, so a caller
/// that knows its box is bigger than [`SAMPLES`] — and wants every point of it — says so.
pub fn same_value_over(left: &Arc<UOp>, right: &Arc<UOp>, cap: usize) -> Result<(), TestCaseError> {
    for bindings in range_points(&operands(&[left, right]), cap) {
        if let (Some(a), Some(b)) = (fold_at(left, &bindings), fold_at(right, &bindings)) {
            prop_assert_eq!(a, b, "value mismatch at {:?}\nleft:  {}\nright: {}", bindings, left.tree(), right.tree());
        }
    }
    Ok(())
}

/// Rewrite `expr` with `matcher`, require a single `op` node, and hand its operands
/// back for the caller to pin.
pub fn folded_binary(
    matcher: &TypedPatternMatcher,
    expr: Arc<UOp>,
    op: BinaryOp,
) -> Result<(Arc<UOp>, Arc<UOp>), TestCaseError> {
    let simplified = rewrite(matcher, expr);
    match simplified.op() {
        Op::Binary(actual, lhs, rhs) => {
            prop_assert_eq!(*actual, op, "wrong op in\n{}", simplified.tree());
            Ok((lhs.clone(), rhs.clone()))
        }
        other => Err(TestCaseError::fail(format!("expected {op:?}, got {other:?}\n{}", simplified.tree()))),
    }
}

/// `(forward, backward)`: `op(lhs, rhs)` and `op(rhs, lhs)`, or `None` outside `op`'s domain.
pub fn swapped(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Option<(Arc<UOp>, Arc<UOp>)> {
    Some((build_binary(op, lhs.clone(), rhs.clone())?, build_binary(op, rhs, lhs)?))
}

/// `count` samples from a strategy, all from one fixed seed, so a "the generator reaches
/// this shape" guard neither depends on the ambient RNG nor rests on a single lucky draw:
/// one sample says only that *some* draw reaches the shape, which stays true even when
/// almost every draw misses and the property it guards is almost entirely vacuous.
pub fn samples<S: Strategy>(strategy: S, count: usize) -> Vec<S::Value> {
    let mut runner = TestRunner::deterministic();
    (0..count).map(|_| strategy.new_tree(&mut runner).expect("a generated value").current()).collect()
}
