//! `in_scope_ranges` is what gpudims store masking uses to find the LOCAL ranges that are still open at a store
//! (`gpudims.rs::compute_store_masks`). A `toposort().filter(Range)` would instead return every range the graph ever
//! opened, including ended ones.

use std::collections::HashSet;
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisType, Op, UOp};

use crate::test::support::prelude::*;

fn local_range(end: i64, id: usize) -> Arc<UOp> {
    range(end, AxisType::Local, id)
}

fn in_scope_ids(uop: &Arc<UOp>) -> HashSet<u64> {
    uop.in_scope_ranges().iter().copied().collect()
}

/// An END closes only the ranges it names: a sibling range that the same computation still
/// depends on stays in scope, while the ended one leaves it *without* leaving the graph.
#[test]
fn an_end_closes_only_the_ranges_it_names() {
    let (ended_range, open_range) = (local_range(16, 0), local_range(32, 1));
    let ended = ended_range.add(&open_range).end(smallvec![ended_range.clone()]);
    // AFTER sequences against the END, which is Void and cannot feed an ALU.
    let downstream = open_range.add(&index_const(5)).after(smallvec![ended]);

    let in_scope = in_scope_ids(&downstream);
    assert!(!in_scope.contains(&ended_range.id), "the ended range must leave scope");
    assert!(in_scope.contains(&open_range.id), "the sibling range was never ended and must stay in scope");
    assert!(
        downstream.toposort().iter().any(|uop| matches!(uop.op(), Op::Range(..)) && uop.id == ended_range.id),
        "the ended range is still reachable by toposort — that is why the store mask cannot use it",
    );
}

/// An INDEX holds only the ranges it addresses with, so a local range the store never
/// indexes by is exactly what `compute_store_masks` must mask against.
#[test]
fn an_index_only_holds_the_ranges_it_addresses_with() {
    let (addressed, unused) = (local_range(16, 0), local_range(16, 1));
    let index = index_of(param(0, 1024, DType::Float32), addressed.clone());

    let in_scope = in_scope_ids(&index);
    assert!(in_scope.contains(&addressed.id), "the addressing range is in scope at the INDEX");
    assert!(!in_scope.contains(&unused.id), "a range the INDEX never addresses with is not in scope");
}
