//! Parity tests for the cached RANGE/LOAD match guards.
//!
//! `no_range` / `no_load` used to run an uncached `any_in_subtree` DFS on every
//! pattern attempt. They now read the cached `RangesProperty` /
//! `has_index_in_sources` flags; these rows pin the truth table against the
//! original DFS implementations, kept here as oracles.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisType, Op, ReduceOp, UOp};
use test_case::test_case;

use crate::rangeify::indexing::no_range;
use crate::rangeify::patterns::no_load;
use crate::test::support::prelude::*;

fn no_range_oracle(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Range(..)))
}

fn no_load_oracle(u: &Arc<UOp>) -> bool {
    !u.any_in_subtree(|x| matches!(x.op(), Op::Index(..)))
}

fn reduce_range() -> Arc<UOp> {
    range(10, AxisType::Reduce, 0)
}

fn load_at(idx: Arc<UOp>) -> Arc<UOp> {
    load(index_of(buffer(100), idx))
}

/// REDUCE ends the range, so `in_scope_ranges` is empty here — but the RANGE is
/// still in the backward slice and the guard must keep reporting "has range".
#[test_case(|| UOp::native_const(42i32), true, true ; "bare const")]
#[test_case(|| UOp::native_const(10i32).try_add(&UOp::native_const(20i32)).expect("add"), true, true ; "const arithmetic")]
#[test_case(reduce_range, false, true ; "bare range")]
#[test_case(|| reduce_range().cast(DType::Int32).try_add(&UOp::native_const(5i32)).expect("add"), false, true ; "range arithmetic")]
#[test_case(|| { let r = reduce_range(); r.cast(DType::Int32).reduce(smallvec::smallvec![r], ReduceOp::Add) }, false, true ; "range consumed by reduce")]
#[test_case(|| load_at(UOp::index_const(0)), true, false ; "load at const index")]
#[test_case(|| load_at(reduce_range()), false, false ; "load at range index")]
#[test_case(|| { let cond = reduce_range().try_cmplt(&UOp::index_const(5)).expect("cmplt"); UOp::try_where(cond, load_at(UOp::index_const(0)), UOp::native_const(0.0f32)).expect("where") }, false, false ; "load under a where")]
#[test_case(|| { let cond = UOp::index_const(1).try_cmplt(&UOp::index_const(5)).expect("cmplt"); UOp::try_where(cond, UOp::native_const(1.0f32), UOp::native_const(0.0f32)).expect("where") }, true, true ; "range-free load-free where")]
fn cached_guards_match_dfs_oracle(build: fn() -> Arc<UOp>, expect_no_range: bool, expect_no_load: bool) {
    let u = build();
    assert_eq!(no_range_oracle(&u), expect_no_range, "oracle no_range");
    assert_eq!(no_load_oracle(&u), expect_no_load, "oracle no_load");
    assert_eq!(no_range(&u), no_range_oracle(&u), "cached no_range diverged from DFS oracle");
    assert_eq!(no_load(&u), no_load_oracle(&u), "cached no_load diverged from DFS oracle");
}
