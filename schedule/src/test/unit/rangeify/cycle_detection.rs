//! `find_bufs` panics when one buffer is reached through two different INDEX
//! source ops. Tinygrad keys on the discriminant of the INDEX *source* — not on
//! LOAD-versus-STORE, and not on buffer identity — so every shape below is legal
//! even when it reads and writes the same buffer.

use std::sync::Arc;

use svod_ir::UOp;
use test_case::test_case;

use crate::rangeify::transforms::find_bufs;
use crate::test::support::prelude::*;

fn distinct_buffers() -> Arc<UOp> {
    index(buffer(100), 0).store(load(index(buffer(100), 0)))
}

/// One BUFFER reaches both the LOAD and the STORE through INDEX(BUFFER, ..):
/// same discriminant, so the shape carries no cycle.
fn one_buffer_read_and_written() -> Arc<UOp> {
    let storage = buffer(100);
    let loaded = load(index(storage.clone(), 0));
    index(storage, 0).store(loaded)
}

/// Gate on the LOAD/STORE (post-gater), not on the address.
fn gated_load_and_store() -> Arc<UOp> {
    let gate = UOp::native_const(true);
    let loaded = UOp::load().index(index(buffer(100), 0)).alt(UOp::native_const(0.0f32)).gate(gate.clone()).call();
    index(buffer(100), 0).store_gated(loaded, gate)
}

#[test_case(super::distinct_buffers ; "distinct load and store buffers")]
#[test_case(super::one_buffer_read_and_written ; "one buffer through one index source")]
#[test_case(super::gated_load_and_store ; "post-gater load and store")]
#[test_case(|| { let valid = UOp::index_const(0).valid(UOp::native_const(true)); index(buffer(100), 0).store(load(index_of(buffer(100), valid))) } ; "pre-gater valid index")]
fn accepted(build: fn() -> Arc<UOp>) {
    find_bufs(&build());
}

/// The same buffer reached directly and through a movement wrapper is two
/// different INDEX source op kinds — the case the discriminant guard exists for.
#[test]
#[should_panic(expected = "cycle detected while indexing")]
fn distinct_index_sources_for_one_buffer_are_a_cycle() {
    let storage = buffer(100);
    let direct = index(storage.clone(), 0);
    let selected = index(storage.mselect(0), 0);

    find_bufs(&selected.store(load(direct)));
}
