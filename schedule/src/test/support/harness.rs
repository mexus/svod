//! Rewrite-harness assertions: a [`Term`] pair in, a semantic or structural
//! verdict out.

use std::sync::Arc;

use svod_ir::{ConstValue, Op, TypedPatternMatcher, UOp};

use super::eval::{eval_typed, range_points};
use super::matcher::rewrite;
use super::vars::{Term, TestVars, var_range};

/// Points sampled per semantic check.
const SAMPLES: usize = 64;

/// Rewrite `input` and build `expected` over the same variable set.
#[track_caller]
fn rewrite_pair(matcher: &TypedPatternMatcher, input: Term, expected: Term) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>) {
    let vars = TestVars::new();
    let original = input(&vars);
    (original.clone(), rewrite(matcher, original), expected(&vars))
}

/// `input` must rewrite to `expected`, node for node.
#[track_caller]
pub fn assert_rewrites_to(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    let (_, folded, want) = rewrite_pair(matcher, input, expected);
    assert!(Arc::ptr_eq(&folded, &want), "assert_rewrites_to failed\n got: {}\nwant: {}", folded.tree(), want.tree());
}

/// `input` must rewrite to `expected` and keep the evaluator's value at every sample.
#[track_caller]
pub fn assert_rewrites_to_and_evaluates(matcher: &TypedPatternMatcher, input: Term, expected: Term) {
    let (original, folded, want) = rewrite_pair(matcher, input, expected);
    assert!(Arc::ptr_eq(&folded, &want), "wrong rewrite\n got: {}\nwant: {}", folded.tree(), want.tree());
    assert_agrees(&original, &folded);
}

/// `input` must come back untouched.
#[track_caller]
pub fn assert_unchanged(matcher: &TypedPatternMatcher, input: Term) {
    let vars = TestVars::new();
    let original = input(&vars);
    let folded = rewrite(matcher, original.clone());
    assert!(Arc::ptr_eq(&folded, &original), "assert_unchanged failed: rewrote to {}", folded.tree());
}

/// `pass` must preserve the evaluator's value over the sampled range points.
#[track_caller]
pub fn assert_pass_preserves(pass: impl Fn(Arc<UOp>) -> Arc<UOp>, input: Term) {
    let vars = TestVars::new();
    let original = input(&vars);
    let transformed = pass(original.clone());
    assert_agrees(&original, &transformed);
}

/// `u` must be a `Const` holding `expected`.
#[track_caller]
pub fn assert_const_value(u: &Arc<UOp>, expected: ConstValue) {
    match u.op() {
        Op::Const(value) => assert_eq!(value.0, expected, "assert_const_value failed\n{}", u.tree()),
        other => panic!("assert_const_value expected Const({expected:?}), got {other:?}\n{}", u.tree()),
    }
}

/// Compare the evaluator's value for `a` and `b` at every sampled point where both
/// are defined, and require at least one such point.
///
/// The guard is the whole point: the evaluator returns `None` outside its domain,
/// so without it a pair that never evaluates agrees vacuously and the check reads
/// as semantic while asserting nothing. A rewrite over operands the evaluator
/// cannot reach — a LOAD, a SPECIAL — belongs in [`assert_rewrites_to`], which
/// makes the structural-only claim at the call site.
#[track_caller]
fn assert_agrees(a: &Arc<UOp>, b: &Arc<UOp>) {
    let mut vars: Vec<Arc<UOp>> = Vec::new();
    for node in a.toposort().into_iter().chain(b.toposort()) {
        if var_range(&node).is_some() && !vars.iter().any(|seen| Arc::ptr_eq(seen, &node)) {
            vars.push(node);
        }
    }
    let mut compared = 0usize;
    for bindings in range_points(&vars, SAMPLES) {
        if let (Some(left), Some(right)) = (eval_typed(a, &bindings), eval_typed(b, &bindings)) {
            assert_eq!(left, right, "value mismatch at {:?}\nleft:  {}\nright: {}", bindings, a.tree(), b.tree());
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no sampled point evaluated, so nothing was compared; use assert_rewrites_to for a \
         structural-only claim\nleft:  {}\nright: {}",
        a.tree(),
        b.tree()
    );
}
