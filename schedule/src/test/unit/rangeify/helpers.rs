//! The one fixture surface for the rangeify tests.

use std::sync::Arc;

use svod_ir::{BinaryOp, ConstValue, DType, Matcher, Op, RewriteResult, UOp};
use test_case::test_case;

use crate::rangeify::indexing::{get_const_value, is_const, is_identity_value, is_zero_value};
use crate::test::support::count::count_kinds;

pub(crate) use crate::test::support::build::{has_op, reduce_range};

/// Distinct `CALL`s, i.e. the kernel count.
pub(crate) use crate::test::support::count::kernels as count_kernels;

pub(crate) fn count_stores(uop: &Arc<UOp>) -> usize {
    count_kinds(uop).stores
}

pub(crate) fn count_stages(uop: &Arc<UOp>) -> usize {
    count_kinds(uop).stages
}

pub(crate) fn any_op(uop: &Arc<UOp>, pred: impl Fn(&Op) -> bool) -> bool {
    uop.toposort().iter().any(|node| pred(node.op()))
}

#[track_caller]
pub(crate) fn assert_same_ptr(a: &Arc<UOp>, b: &Arc<UOp>) {
    assert!(Arc::ptr_eq(a, b), "expected the same node\n got: {}\nwant: {}", a.tree(), b.tree());
}

/// The rewritten node, panicking on `NoMatch`/`Gate`: `if let Rewritten { .. }` with no
/// `else` would pass when the pattern never fires.
#[track_caller]
pub(crate) fn rewritten<C>(matcher: &(impl Matcher<C> + ?Sized), uop: &Arc<UOp>, ctx: &mut C) -> Arc<UOp> {
    match matcher.rewrite(uop, ctx) {
        RewriteResult::Rewritten(out) => out,
        other => panic!("expected Rewritten, got {other:?}\nfor {}", uop.tree()),
    }
}

#[track_caller]
pub(crate) fn assert_no_match<C>(matcher: &(impl Matcher<C> + ?Sized), uop: &Arc<UOp>, ctx: &mut C) {
    if let result @ (RewriteResult::Rewritten(_) | RewriteResult::Gate(_)) = matcher.rewrite(uop, ctx) {
        panic!("expected NoMatch, got {result:?}\nfor {}", uop.tree());
    }
}

#[track_caller]
pub(crate) fn const_value(uop: &Arc<UOp>) -> ConstValue {
    match uop.op() {
        Op::Const(value) => value.0,
        other => panic!("expected Const, got {other:?}\n{}", uop.tree()),
    }
}

#[track_caller]
pub(crate) fn assert_const_float(uop: &Arc<UOp>, expected: f32) {
    let value = const_value(uop).try_float().unwrap_or_else(|| panic!("expected a float const\n{}", uop.tree()));
    assert_eq!(value as f32, expected, "expected {expected}, got\n{}", uop.tree());
}

/// The first CALL of a pipeline result, which may be CALL, `AFTER(_, [END(CALL)])`,
/// `SINK([.., CALL, ..])` or `END(CALL)`.
pub(crate) fn extract_kernel(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    match uop.op() {
        Op::Call(..) => Some(uop.clone()),
        Op::After(svod_ir::ops::After { deps, .. }) => deps.iter().find_map(extract_kernel),
        Op::Sink(svod_ir::ops::Sink { sources, .. }) => sources.iter().find_map(extract_kernel),
        Op::End(svod_ir::ops::End { computation, .. }) => extract_kernel(computation),
        _ => None,
    }
}

pub(crate) fn loop_range(end: i64, axis_id: usize) -> Arc<UOp> {
    crate::test::support::build::range(end, svod_ir::AxisType::Loop, axis_id)
}

/// The `RANGE`s an `END`/`SINK` subtree closes.
pub(crate) fn closed_range_count(uop: &Arc<UOp>) -> usize {
    match uop.op() {
        Op::End(svod_ir::ops::End { ranges, .. }) => ranges.len(),
        Op::Sink(svod_ir::ops::Sink { sources, .. }) => sources.iter().map(closed_range_count).sum(),
        _ => 0,
    }
}

/// `is_identity_value` is per-operator and side-aware: `-` and `//` have a right
/// identity only; the bitwise operators use the all-ones mask.
#[test_case(ConstValue::Int(0), BinaryOp::Add, false, true ; "zero is a left add identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Add, true, true ; "zero is a right add identity")]
#[test_case(ConstValue::Float(0.0), BinaryOp::Add, false, true ; "float zero is an add identity")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, false, true ; "one is a left mul identity")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, true, true ; "one is a right mul identity")]
#[test_case(ConstValue::Float(1.0), BinaryOp::Mul, false, true ; "float one is a mul identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Sub, false, false ; "sub has no left identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Sub, true, true ; "sub has a right identity")]
#[test_case(ConstValue::Int(1), BinaryOp::FloorDiv, false, false ; "div has no left identity")]
#[test_case(ConstValue::Int(1), BinaryOp::FloorDiv, true, true ; "div has a right identity")]
#[test_case(ConstValue::Float(1.0), BinaryOp::Fdiv, true, true ; "float div has a right identity")]
#[test_case(ConstValue::Float(2.0), BinaryOp::Fdiv, true, false ; "float div right identity is one")]
#[test_case(ConstValue::Int(0), BinaryOp::Or, false, true ; "zero is an or identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Xor, true, true ; "zero is a xor identity")]
#[test_case(ConstValue::Int(-1), BinaryOp::And, false, true ; "all ones is an and identity")]
#[test_case(ConstValue::Int(0), BinaryOp::And, true, false ; "zero is not an and identity")]
#[test_case(ConstValue::Int(2), BinaryOp::Add, false, false ; "two is not an add identity")]
#[test_case(ConstValue::Int(0), BinaryOp::Mul, false, false ; "zero is not a mul identity")]
fn identity_values(value: ConstValue, op: BinaryOp, right: bool, expected: bool) {
    assert_eq!(is_identity_value(&value, &op, right), expected);
}

/// `is_zero_value` is the absorbing element, not the literal zero.
#[test_case(ConstValue::Int(0), BinaryOp::Mul, true ; "zero absorbs mul")]
#[test_case(ConstValue::Float(0.0), BinaryOp::Mul, true ; "float zero absorbs mul")]
#[test_case(ConstValue::Float(-0.0), BinaryOp::Mul, true ; "signed zero absorbs mul too")]
#[test_case(ConstValue::Int(0), BinaryOp::And, true ; "zero absorbs and")]
#[test_case(ConstValue::UInt(0), BinaryOp::And, false ; "only Int zero is in the absorbing table")]
#[test_case(ConstValue::Int(1), BinaryOp::Mul, false ; "one does not absorb mul")]
#[test_case(ConstValue::Int(0), BinaryOp::Add, false ; "zero does not absorb add")]
fn zero_values(value: ConstValue, op: BinaryOp, expected: bool) {
    assert_eq!(is_zero_value(&value, &op), expected);
}

#[test]
fn only_constants_have_a_const_value() {
    let c = UOp::native_const(42i32);
    assert_eq!(get_const_value(&c), Some(ConstValue::Int(42)));
    assert!(is_const(&c, &ConstValue::Int(42)));
    assert!(!is_const(&c, &ConstValue::Int(0)));

    assert_eq!(get_const_value(&UOp::param(0, 1, DType::Float32, None)), None);
}
