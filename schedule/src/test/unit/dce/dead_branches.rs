//! Dead branch elimination in `WHERE`: a condition the declared ranges decide selects its
//! branch outright, an undecided one keeps the selection, and the `INVALID` marker — the
//! gate that says "this access does not happen" — is canonicalized into the false branch
//! and propagated rather than folded away.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::TernaryOp;
use svod_ir::{Op, UOp};
use test_case::test_case;

use crate::test::support::prelude::*;

/// A condition no analysis can decide: `x < 50` over `x ∈ [0, 100]`.
fn undecided(v: &TestVars) -> Arc<UOp> {
    v.x.lt(&v.c(50))
}

/// A condition the declared range decides: `b < 20` over `b ∈ [0, 10]`.
fn decided(_: &TestVars) -> Arc<UOp> {
    UOp::var("b10", DType::Int32, 0, 10).lt(&UOp::native_const(20i32))
}

fn select(condition: &Arc<UOp>, then_branch: &Arc<UOp>, else_branch: &Arc<UOp>) -> Arc<UOp> {
    UOp::try_where(condition.clone(), then_branch.clone(), else_branch.clone()).unwrap()
}

/// A range-decided condition selects its branch; an undecided one keeps the WHERE intact.
#[test_case(|v| v.b(true), Some(true) ; "constant true")]
#[test_case(|v| v.b(false), Some(false) ; "constant false")]
#[test_case(decided, Some(true) ; "comparison the declared range decides")]
#[test_case(undecided, None ; "comparison the range cannot decide")]
fn where_selects_a_branch_or_survives(condition: Term, takes_true: Option<bool>) {
    let vars = TestVars::new();
    let condition = condition(&vars);
    let (then_branch, else_branch) = (vars.c(42), vars.c(0));
    let folded = rewrite(Matchers::simple(), select(&condition, &then_branch, &else_branch));
    match takes_true {
        Some(true) => assert_same!(folded, then_branch),
        Some(false) => assert_same!(folded, else_branch),
        None => match folded.op() {
            Op::Ternary(TernaryOp::Where, cond, then_, else_) => {
                assert_same!(Arc::clone(cond), condition);
                assert_same!(Arc::clone(then_), then_branch);
                assert_same!(Arc::clone(else_), else_branch);
            }
            other => panic!("expected the WHERE to survive, got {other:?}"),
        },
    }
}

/// `WHERE(cond, INVALID, INVALID)` must collapse to `INVALID` instead of ping-ponging through
/// the INVALID canonicalization, which flips the gate forever, and an invalid gate poisons
/// the whole selection.
#[test_case(|v| select(&undecided(v), &UOp::invalid_marker(), &UOp::invalid_marker()) ; "two invalid branches")]
#[test_case(|v| select(&undecided(v).not(), &UOp::invalid_marker(), &UOp::invalid_marker()) ; "two invalid branches under a negated condition")]
#[test_case(|v| select(&UOp::invalid_marker(), &v.c(1), &v.c(2)) ; "an invalid condition poisons the selection")]
fn invalid_wheres_collapse_to_invalid(case: Term) {
    let folded = rewrite(Matchers::simple(), case(&TestVars::new()));
    assert!(UOp::is_invalid_marker(&folded), "expected bare INVALID, got {:?}", folded.op());
}

/// `WHERE(cond, INVALID, x)` is canonicalized to `WHERE(!cond, x, INVALID)`, and that form is
/// a fixpoint: re-running the matcher must not flip it back.
#[test]
fn an_invalid_true_branch_is_canonicalized_once() {
    let vars = TestVars::new();
    let (condition, value) = (undecided(&vars), vars.c(7));
    let folded =
        rewrite(Matchers::simple(), UOp::try_where(condition.clone(), UOp::invalid_marker(), value.clone()).unwrap());
    let Op::Ternary(TernaryOp::Where, cond, taken, marker) = folded.op() else {
        panic!("expected the canonicalized WHERE, got {:?}", folded.op())
    };
    assert_same!(Arc::clone(cond), condition.not());
    assert_same!(Arc::clone(taken), value);
    assert!(UOp::is_invalid_marker(marker), "the marker must move to the false branch");
    assert_same!(rewrite(Matchers::simple(), folded.clone()), folded);
}

/// A nested WHERE normalizes against its parent: an inner and outer false branch that agree
/// (including on the INVALID marker) merge, while a nested INVALID false branch lifts into the
/// condition and keeps the live outer branch.
#[test_case(|v| {
    let (outer, inner, value, invalid) = (v.p.clone(), v.q.clone(), v.c(5), UOp::invalid_marker());
    let input = select(&outer, &select(&inner, &value, &invalid), &invalid);
    (input, select(&outer.and_(&inner), &value, &invalid))
} ; "a shared invalid false branch merges")]
#[test_case(|v| {
    let (outer, inner, live, value, invalid) = (v.p.clone(), v.q.clone(), v.c(3), v.c(5), UOp::invalid_marker());
    let input = select(&outer, &live, &select(&inner, &value, &invalid));
    (input, select(&outer.or_(&inner), &select(&outer, &live, &value), &invalid))
} ; "a nested invalid false branch lifts into the condition")]
fn nested_wheres_with_invalid_branches_normalize(case: fn(&TestVars) -> (Arc<UOp>, Arc<UOp>)) {
    let (input, expected) = case(&TestVars::new());
    let folded = rewrite(Matchers::simple(), input);
    assert_same!(folded, expected);
}
