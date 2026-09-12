//! Z3 test helpers.
//!
//! `check() == Sat` after asserting an equality only proves it has a solution, while
//! refuting the negation (`check() == Unsat`) proves it holds for every input: every
//! rewrite and converter test should use the latter.

use std::sync::Arc;

use svod_ir::{TypedPatternMatcher, UOp};
use z3::SatResult;
use z3::ast::{Bool, Int};

use crate::rewrite::graph_rewrite;
use crate::z3::CounterExample;
use crate::z3::verify::verify_equivalence;

/// Rewrite `expr` with `matcher` and prove the result equivalent to the input.
#[track_caller]
pub fn verify_roundtrip(matcher: &TypedPatternMatcher, expr: Arc<UOp>) -> Arc<UOp> {
    let simplified = graph_rewrite(matcher, expr.clone(), &mut ());
    verify_equivalence(&expr, &simplified).expect("the rewrite must preserve semantics");
    simplified
}

/// Prove `lhs == rhs` for every input, in a fresh solver.
#[track_caller]
pub fn assert_valid(lhs: &Int, rhs: &Int) {
    assert_valid_in(&z3::Solver::new(), lhs, rhs);
}

/// Prove `lhs == rhs` in `solver`, whose asserted constraints are the assumptions.
#[track_caller]
pub fn assert_valid_in(solver: &z3::Solver, lhs: &Int, rhs: &Int) {
    solver.push();
    solver.assert(lhs.eq(rhs).not());
    let result = solver.check();
    solver.pop(1);
    assert_eq!(result, SatResult::Unsat, "expected {lhs} == {rhs} to hold for every input");
}

/// Prove `lhs == rhs` for booleans.
#[track_caller]
pub fn assert_valid_bool(lhs: &Bool, rhs: &Bool) {
    let solver = z3::Solver::new();
    solver.assert(lhs.eq(rhs).not());
    assert_eq!(solver.check(), SatResult::Unsat, "expected {lhs} == {rhs} to hold for every input");
}

/// Disprove an equivalence and return the exact failure mode.
#[track_caller]
pub fn assert_not_equivalent(lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> CounterExample {
    match verify_equivalence(lhs, rhs) {
        Err(error) => error,
        Ok(()) => panic!("expected a counterexample, but the expressions were proven equivalent"),
    }
}
