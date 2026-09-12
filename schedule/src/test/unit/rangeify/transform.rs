//! `transform_sources_with_bufferize` / `transform_single_source`: how a
//! consumer's ranges are pushed into each of its sources.

use std::sync::Arc;

use svod_ir::{Op, UOp};

use crate::rangeify::IndexingContext;
use crate::rangeify::transforms::transform_sources_with_bufferize;
use crate::test::support::prelude::*;

fn consumer_with_ranges(consumer: &Arc<UOp>, ranges: &[Arc<UOp>]) -> IndexingContext {
    let mut ctx = IndexingContext::new();
    ctx.set_ranges(consumer, ranges.to_vec(), ranges.to_vec());
    ctx
}

/// A consumer's BUFFER sources are wrapped in an INDEX over its ranges, and a
/// compute the context marked realized is materialised through a STAGE first.
#[test]
fn buffer_sources_are_indexed_and_realized_computes_are_staged() {
    let consumer = buffer(40).try_add(&buffer(40)).expect("add");
    let range = range(10, svod_ir::AxisType::Loop, 0);
    let mut ctx = consumer_with_ranges(&consumer, std::slice::from_ref(&range));

    let sources = transform_sources_with_bufferize(&consumer, &mut ctx).expect("buffers transform");

    assert_eq!(sources.len(), 2);
    for source in sources {
        let (storage, indices) = expect_index(&source);
        assert!(matches!(storage.op(), Op::Buffer(..)));
        assert_same!(indices[0], range);
    }

    let x = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let consumer = x.try_sqrt().expect("sqrt");
    let mut ctx = consumer_with_ranges(&consumer, std::slice::from_ref(&range));
    ctx.set_ranges(&x, vec![range.clone()], vec![range.clone()]);
    ctx.mark_realize(&x, vec![0]);

    let source =
        crate::rangeify::transforms::transform_single_source(&consumer, &x, std::slice::from_ref(&range), &mut ctx);

    let (storage, indices) = expect_index(&source);
    assert!(matches!(storage.op(), Op::Stage(..)), "a realized compute is staged: {}", source.tree());
    assert_same!(indices[0], range);
}

/// A consumer with no assigned ranges has nothing to push down, and a movement
/// chain over a buffer is deferred to the BPM movement rewrite, which needs the
/// index context that only the full pass has.
#[test]
fn rangeless_and_movement_consumers_are_left_alone() {
    let consumer = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    assert!(transform_sources_with_bufferize(&consumer, &mut IndexingContext::new()).is_none());

    let view = || {
        buffer(12).try_reshape(&smallvec::smallvec![svod_ir::SInt::Const(3), svod_ir::SInt::Const(4)]).expect("reshape")
    };
    let consumer = view().try_add(&view()).expect("add");
    let ranges = [range(3, svod_ir::AxisType::Loop, 0), range(4, svod_ir::AxisType::Loop, 1)];
    let mut ctx = consumer_with_ranges(&consumer, &ranges);

    assert!(transform_sources_with_bufferize(&consumer, &mut ctx).is_none());
    assert!(has_op(&consumer, |op| op.is_movement()), "the fixture must contain a movement op");
}
