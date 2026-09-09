//! Tests for `pm_generate_realize_map` rows ported from tinygrad's
//! `tinygrad/schedule/indexing.py:37-56`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, CallInfo, Op, SInt, UOp};
use test_case::test_case;

use crate::rangeify::IndexingContext;
use crate::rangeify::indexing::pm_generate_realize_map;
use svod_ir::ops;

fn run(root: Arc<UOp>, ctx: &mut IndexingContext) {
    crate::rewrite::graph_rewrite_bottom_up_preserve_calls(pm_generate_realize_map(), root, ctx);
}

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

/// `CALL(SINK, args...)` — a hand-written kernel over `args`.
fn call_with(args: Vec<Arc<UOp>>) -> Arc<UOp> {
    let body = UOp::sink(vec![UOp::param(0, 4, DType::Float32, Some(DeviceSpec::Cpu))]);
    UOp::new(
        Op::Call(ops::Call { body, args: args.into_iter().collect(), info: CallInfo::default().into() }),
        DType::Void,
    )
}

/// `STORE(INDEX(dest, r), value)` — `dest` is indexed without any movement op.
fn store_of(value: Arc<UOp>) -> Arc<UOp> {
    let r = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index().buffer(buffer(4)).indices(vec![r]).call().expect("INDEX");
    index.store(value)
}

#[test]
fn custom_kernel_source_is_realized_and_pinned() {
    let arg = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&arg)]), &mut ctx);

    assert!(ctx.should_realize(&arg), "a CALL input that is not already a buffer must be realized");
    assert!(ctx.is_non_removable_realize(&arg), "the kernel reads it through a PARAM slot, so it must stay a buffer");
}

#[test]
fn custom_kernel_source_realizes_through_reshapes() {
    // `while s.op is Ops.RESHAPE: s = s.src[0]` — the compute is realized, not
    // the view of it.
    let compute = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let view = compute.try_reshape(&smallvec![1usize.into()]).expect("reshape");
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&view)]), &mut ctx);

    assert!(ctx.should_realize(&compute), "the reshaped-through source is the one realized");
    assert!(!ctx.should_realize(&view), "the RESHAPE view itself needs no buffer");
}

#[test]
fn custom_kernel_buffer_source_is_left_alone() {
    let buf = buffer(4);
    let mut ctx = IndexingContext::new();

    run(call_with(vec![Arc::clone(&buf)]), &mut ctx);

    assert!(!ctx.should_realize(&buf), "an ALWAYS_CONTIGUOUS CALL input is already a buffer");
}

#[test]
fn slice_source_of_store_loses_its_realize_entry() {
    let slice = buffer(8).contiguous_slice(4, 0, DType::Float32);
    let store = store_of(Arc::clone(&slice));

    let mut ctx = IndexingContext::new();
    ctx.mark_realize_pending(&slice);
    run(store, &mut ctx);

    assert!(!ctx.should_realize(&slice), "the store target already is the output buffer");
}

#[test]
fn slice_source_keeps_its_realize_entry_behind_a_movement_op() {
    let slice = buffer(8).contiguous_slice(4, 0, DType::Float32);
    let r = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
    let dest = buffer(4)
        .try_reshape(&smallvec![2usize.into(), 2usize.into()])
        .expect("reshape")
        .try_permute(vec![1, 0])
        .expect("permute");
    assert!(matches!(dest.op(), Op::Permute(..)), "the test needs a real PERMUTE on the destination");
    let index = UOp::index().buffer(dest).indices(vec![r]).call().expect("INDEX");
    let store = index.store(Arc::clone(&slice));

    let mut ctx = IndexingContext::new();
    ctx.mark_realize_pending(&slice);
    run(store, &mut ctx);

    assert!(ctx.should_realize(&slice), "a moved destination does not line up with the SLICE");
}

/// The scan counter: a loop variable over `[0, 100]` used as a window offset.
fn t() -> Arc<UOp> {
    UOp::define_var("t".to_string(), 0, 100)
}

/// `src[offset : offset + size]`, the way `narrow` windows a scan slot.
fn window(src: &Arc<UOp>, offset: Arc<UOp>, size: SInt) -> Arc<UOp> {
    let begin = SInt::Symbolic(offset);
    let end = &begin + &size;
    src.try_shrink(&[(begin, end)]).expect("shrink")
}

/// `src[offset : offset + 1, 0 : 1]` over a `[n, 1]` view.
fn column_window(src: &Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    let begin = SInt::Symbolic(offset);
    let end = &begin + 1;
    src.try_shrink(&[(begin, end), (0.into(), 1.into())]).expect("shrink")
}

/// `STORE(target, read * read)`: whether the WAR temp on the value survives.
fn self_assign_keeps_temp(target: Arc<UOp>, read: Arc<UOp>) -> bool {
    let value = read.try_mul(&read).expect("mul");
    let mut ctx = IndexingContext::new();
    run(target.store(Arc::clone(&value)), &mut ctx);
    ctx.is_non_removable_realize(&value)
}

/// Windows of one buffer that are provably disjoint drop the WAR temp; every
/// unproven pair keeps it. Offsets are in units of the scan counter `t`.
#[test_case(|t| t.add(&t.const_like(1)), 1, |t| t.clone(), 1, false ; "the next slot misses the current one")]
#[test_case(|t| t.clone(), 1, |t| t.add(&t.const_like(1)), 1, false ; "the current slot misses the next one")]
#[test_case(|t| t.const_like(2).mul(&t.add(&t.const_like(1))), 2, |t| t.const_like(2).mul(t), 2, false ; "a stride distributed over the offset")]
#[test_case(|t| t.add(&t.const_like(1)).mul(&t.const_like(3)), 3, |t| t.mul(&t.const_like(3)), 3, false ; "a stride applied after the offset")]
#[test_case(|t| t.add(&t.const_like(1)), 2, |t| t.clone(), 1, false ; "a wider write past a narrower read")]
#[test_case(|t| t.clone(), 1, |t| t.add(&t.const_like(1)), 2, false ; "a narrower write before a wider read")]
#[test_case(|t| t.clone(), 1, |t| t.clone(), 1, true ; "the same slot overlaps")]
#[test_case(|t| t.const_like(2).mul(&t.add(&t.const_like(1))), 2, |t| t.const_like(2).mul(t).add(&t.const_like(2)), 2, true ; "equal offsets spelled differently overlap")]
#[test_case(|t| t.add(&t.const_like(2)), 1, |t| t.const_like(1).add(&t.add(&t.const_like(1))), 1, true ; "equal offsets nested differently overlap")]
#[test_case(|t| t.add(&t.const_like(1)), 1, |t| t.clone(), 2, true ; "a wider read reaches into the write")]
#[test_case(|t| t.clone(), 2, |t| t.add(&t.const_like(1)), 1, true ; "a wider write reaches into the read")]
#[test_case(|t| t.add(&t.const_like(200)).cast(DType::Int8), 100, |t| t.cast(DType::Int8), 100, true ; "a narrowing cast may wrap the gap away")]
fn self_assign_windows(
    write: fn(&Arc<UOp>) -> Arc<UOp>,
    write_size: usize,
    read: fn(&Arc<UOp>) -> Arc<UOp>,
    read_size: usize,
    keeps_temp: bool,
) {
    let (base, t) = (buffer(512), t());
    let target = window(&base, write(&t), write_size.into());
    let read = window(&base, read(&t), read_size.into());
    assert_eq!(self_assign_keeps_temp(target, read), keeps_temp);
}

#[test]
fn self_assign_keeps_temp_for_a_symbolically_sized_window() {
    let (base, t) = (buffer(8), t());
    let size = SInt::Symbolic(UOp::define_var("s".to_string(), 1, 4));
    let target = window(&base, t.add(&t.const_like(1)), size.clone());
    assert!(self_assign_keeps_temp(target, window(&base, t, size)), "the size bound exceeds the gap");
}

#[test]
fn self_assign_keeps_temp_for_a_read_that_is_not_a_shrink() {
    let (base, t) = (buffer(8), t());
    let target = window(&base, t.add(&t.const_like(1)), 1.into());
    assert!(self_assign_keeps_temp(target, base), "reading the whole buffer covers the written slot");
}

#[test]
fn self_assign_keeps_temp_behind_a_movement_op_other_than_reshape() {
    let (base, t) = (buffer(8), t());
    let target = window(&base, t.add(&t.const_like(1)), 1.into());
    let read = window(&base.try_flip(vec![true]).expect("flip"), t, 1.into());
    assert!(self_assign_keeps_temp(target, read), "a flipped window is not addressed like the target");
}

#[test]
fn self_assign_keeps_temp_for_windows_viewed_through_different_shapes() {
    let (base, t) = (buffer(8), t());
    let target = window(&base, t.add(&t.const_like(1)), 1.into());
    let column = base.try_reshape(&smallvec![8usize.into(), 1usize.into()]).expect("reshape");
    let read = column_window(&column, t);
    assert!(self_assign_keeps_temp(target, read), "windows over different views are not compared");
}

#[test]
fn self_assign_drops_temp_for_windows_viewed_through_the_same_reshape() {
    let (base, t) = (buffer(8), t());
    let column = base.try_reshape(&smallvec![8usize.into(), 1usize.into()]).expect("reshape");
    let target = column_window(&column, t.add(&t.const_like(1)));
    let read = column_window(&column, t);
    assert!(!self_assign_keeps_temp(target, read), "the reshape is transparent to the window comparison");
}

#[test]
fn self_assign_keeps_temp_when_the_target_is_not_a_shrink() {
    let (base, t) = (buffer(8), t());
    assert!(
        self_assign_keeps_temp(Arc::clone(&base), window(&base, t, 1.into())),
        "an unwindowed target covers every read"
    );
}
