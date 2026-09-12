//! `STAGE(compute, ranges)` becomes
//! `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, ..), compute), ranges)])`, and a
//! STAGE over an AFTER reuses the buffer that AFTER passes through.
//!
//! The BUFFER → PARAM conversion happens later, in `split_store`.

use std::sync::Arc;

use svod_dtype::AddrSpace;
use svod_ir::{BufferizeOpts, Op, UOp};

use crate::rangeify::{RangeifyBufferContext, bufferize_to_store};
use crate::test::support::prelude::*;

fn scalar_stage(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    let opts = BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true };
    stage_with(compute, ranges, opts)
}

#[test]
fn a_staged_compute_becomes_a_buffer_backed_store() {
    let mut ctx = RangeifyBufferContext::new();
    let compute = UOp::native_const(42.0f32);
    let range = global_range(10, 0);
    let staged = stage(Arc::clone(&compute), vec![Arc::clone(&range)]);

    let result = bufferize_to_store(&staged, &mut ctx).expect("a global STAGE converts");

    let (passthrough, deps) = expect_after(&result);
    assert!(matches!(passthrough.op(), Op::Buffer(..)), "the passthrough is the allocated BUFFER");
    let [dep] = deps.as_slice() else { panic!("expected exactly one dep") };

    let (computation, ranges) = expect_end(dep);
    assert_eq!(ranges.len(), 1);
    assert_same!(ranges[0], range);

    let (index, value, gate) = expect_store(&computation);
    assert!(gate.is_none());
    assert_same!(value, compute);
    let (buffer, _) = expect_index(&index);
    assert!(Arc::ptr_eq(&buffer, &passthrough), "the STORE writes the buffer the AFTER passes through");

    let tracked = ctx.get_buffer(&staged).expect("the STAGE is tracked");
    assert_same!(tracked, result);
    assert_eq!(ctx.local_counter, 0, "a global BUFFER does not consume a local slot");
}

/// Case 0: a STAGE whose compute is an AFTER does not allocate — it reuses the
/// buffer the AFTER already passes through. With no STORE deps to re-close, the
/// underlying buffer is handed back unchanged, and a STORE whose value *is* the
/// index it would write is the identity, so it is skipped.
#[test]
fn a_stage_over_an_after_reuses_the_underlying_buffer() {
    let storage = buffer(8);
    let after = storage.clone().after(smallvec::smallvec![UOp::noop()]);
    let staged = scalar_stage(after, vec![global_range(8, 0)]);
    let mut ctx = RangeifyBufferContext::new();
    let result = bufferize_to_store(&staged, &mut ctx).expect("a STAGE over an AFTER reuses its buffer");

    assert_same!(result, storage);
    assert_same!(ctx.get_buffer(&staged).expect("tracked"), result);

    let storage = buffer(8);
    let target = index(storage.clone(), 0);
    let after = storage.clone().after(smallvec::smallvec![store(target.clone(), target)]);
    let mut ctx = RangeifyBufferContext::new();
    let result = bufferize_to_store(&scalar_stage(after, vec![global_range(8, 0)]), &mut ctx).expect("Case 0");

    assert_same!(result, storage);
    assert!(!has_op(&result, |op| matches!(op, Op::Store(..))), "{}", result.tree());
}

/// Multi-range STAGEs are lowered upstream; reaching one here is a bug, not a
/// case to linearise.
#[test]
#[should_panic(expected = "unexpected multi-range")]
fn a_multi_range_stage_is_rejected() {
    let ranges = vec![global_range(4, 0), global_range(8, 1)];
    bufferize_to_store(&stage(UOp::native_const(100i32), ranges), &mut RangeifyBufferContext::new());
}

#[test]
fn only_a_stage_converts() {
    let mut ctx = RangeifyBufferContext::new();
    assert!(bufferize_to_store(&UOp::native_const(1.0f32), &mut ctx).is_none());
}
