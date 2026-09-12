//! Tests for `pm_generate_realize_map` rows ported from tinygrad's
//! `tinygrad/schedule/indexing.py:37-56`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisType, CallInfo, Op, SInt, UOp};
use test_case::test_case;

use crate::rangeify::IndexingContext;
use crate::rangeify::indexing::pm_generate_realize_map;
use crate::test::support::prelude::*;

fn run(root: Arc<UOp>, ctx: &mut IndexingContext) {
    crate::rewrite::graph_rewrite_bottom_up_preserve_calls(pm_generate_realize_map(), root, ctx);
}

/// `CALL(SINK, args...)` — a hand-written kernel over `args`, the shape a custom
/// kernel source reaches the realize map in.
fn run_kernel(args: Vec<Arc<UOp>>, ctx: &mut IndexingContext) {
    run(UOp::sink(vec![param(0, 4, DType::Float32)]).call(args.into_iter().collect(), CallInfo::default()), ctx);
}

/// `STORE(INDEX(dest, r), value)` — `dest` is indexed without any movement op.
fn store_of(value: Arc<UOp>) -> Arc<UOp> {
    index_of(buffer(4), range(4, AxisType::Loop, 0)).store(value)
}

fn scan_counter() -> Arc<UOp> {
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

#[test]
fn custom_kernel_source_is_realized_and_pinned() {
    let arg = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let mut ctx = IndexingContext::new();

    run_kernel(vec![Arc::clone(&arg)], &mut ctx);

    assert!(ctx.should_realize(&arg), "a STAGE input that is not already a buffer must be realized");
    assert!(ctx.is_non_removable_realize(&arg), "the kernel reads it through a PARAM slot, so it must stay a buffer");
}

#[test]
fn custom_kernel_source_realizes_through_reshapes() {
    // `while s.op is Ops.RESHAPE: s = s.src[0]` — the compute is realized, not
    // the view of it.
    let compute = UOp::native_const(1.0f32).try_add(&UOp::native_const(2.0f32)).expect("add");
    let view = compute.try_reshape(&smallvec::smallvec![1usize.into()]).expect("reshape");
    let mut ctx = IndexingContext::new();

    run_kernel(vec![Arc::clone(&view)], &mut ctx);

    assert!(ctx.should_realize(&compute), "the reshaped-through source is the one realized");
    assert!(!ctx.should_realize(&view), "the RESHAPE view itself needs no buffer");
}

/// The `!is_always_contiguous(src)` guard: a CALL input that is already a buffer
/// needs no realize entry. Marking every CALL argument would cost a redundant
/// buffer and a copy on every launch.
#[test]
fn custom_kernel_buffer_source_is_left_alone() {
    let buf = buffer(4);
    let mut ctx = IndexingContext::new();

    run_kernel(vec![Arc::clone(&buf)], &mut ctx);

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
    let r = range(4, AxisType::Loop, 0);
    let dest = buffer(4)
        .try_reshape(&smallvec::smallvec![2usize.into(), 2usize.into()])
        .expect("reshape")
        .try_permute(vec![1, 0])
        .expect("permute");
    assert!(matches!(dest.op(), Op::Permute(..)), "the test needs a real PERMUTE on the destination");
    let store = index_of(dest, r).store(Arc::clone(&slice));

    let mut ctx = IndexingContext::new();
    ctx.mark_realize_pending(&slice);
    run(store, &mut ctx);

    assert!(ctx.should_realize(&slice), "a moved destination does not line up with the SLICE");
}

/// Windows of one buffer that are provably disjoint drop the WAR temp; every
/// unproven pair keeps it. The first rows vary the offset arithmetic, the rest
/// remove the proof one step at a time (symbolic extent, whole-buffer read, other
/// movement op, different view, unwindowed target).
#[test_case(|t| t.clone(), |t| t.add(&t.const_like(1)), 1, 1, false ; "the current slot misses the next one")]
#[test_case(|t| t.add(&t.const_like(1)), |t| t.clone(), 1, 1, false ; "the next slot misses the current one")]
#[test_case(|t| t.const_like(2).mul(&t.add(&t.const_like(1))), |t| t.const_like(2).mul(t), 2, 2, false ; "a stride distributed over the offset")]
#[test_case(|t| t.add(&t.const_like(1)).mul(&t.const_like(3)), |t| t.mul(&t.const_like(3)), 3, 3, false ; "a stride applied after the offset")]
#[test_case(|t| t.add(&t.const_like(1)), |t| t.clone(), 2, 1, false ; "a wider write past a narrower read")]
#[test_case(|t| t.clone(), |t| t.add(&t.const_like(1)), 1, 2, false ; "a narrower write before a wider read")]
#[test_case(|t| t.clone(), |t| t.clone(), 1, 1, true ; "the same slot overlaps")]
#[test_case(|t| t.const_like(2).mul(&t.add(&t.const_like(1))), |t| t.const_like(2).mul(t).add(&t.const_like(2)), 2, 2, true ; "equal offsets spelled differently overlap")]
#[test_case(|t| t.add(&t.const_like(2)), |t| t.const_like(1).add(&t.add(&t.const_like(1))), 1, 1, true ; "equal offsets nested differently overlap")]
#[test_case(|t| t.add(&t.const_like(1)), |t| t.clone(), 1, 2, true ; "a wider read reaches into the write")]
#[test_case(|t| t.clone(), |t| t.add(&t.const_like(1)), 2, 1, true ; "a wider write reaches into the read")]
#[test_case(|t| t.add(&t.const_like(200)).cast(DType::Int8), |t| t.cast(DType::Int8), 100, 100, true ; "a narrowing cast may wrap the gap away")]
fn self_assign_windows(
    write: fn(&Arc<UOp>) -> Arc<UOp>,
    read: fn(&Arc<UOp>) -> Arc<UOp>,
    write_size: usize,
    read_size: usize,
    keeps: bool,
) {
    let (base, t) = (buffer(512), scan_counter());
    assert_eq!(
        self_assign_keeps_temp(window(&base, write(&t), write_size.into()), window(&base, read(&t), read_size.into())),
        keeps
    );
}

/// The proven relationship under a shared reshape drops the temp; moving the
/// read or the write off that view (or off the window) restores it.
#[test_case(|base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.add(&t.const_like(1)), SInt::Symbolic(UOp::define_var("s".into(), 1, 4))),
    |base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.clone(), SInt::Symbolic(UOp::define_var("s".into(), 1, 4))), true ; "a symbolic size exceeds the gap")]
#[test_case(|base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.add(&t.const_like(1)), 1.into()), |base: &Arc<UOp>, _| Arc::clone(base), true ; "reading the whole buffer covers the written slot")]
#[test_case(
    |base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.add(&t.const_like(1)), 1.into()),
    |base: &Arc<UOp>, t: &Arc<UOp>| window(&base.try_flip(vec![true]).expect("flip"), t.clone(), 1.into()),
    true ; "a flipped window is not addressed like the target")]
#[test_case(
    |base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.add(&t.const_like(1)), 1.into()),
    |base: &Arc<UOp>, t: &Arc<UOp>| column_window(&column_of(base), t.clone()),
    true ; "windows over different views are not compared")]
#[test_case(
    |base: &Arc<UOp>, t: &Arc<UOp>| column_window(&column_of(base), t.add(&t.const_like(1))),
    |base: &Arc<UOp>, t: &Arc<UOp>| column_window(&column_of(base), t.clone()),
    false ; "the same reshape is transparent to the comparison")]
#[test_case(|base: &Arc<UOp>, _| Arc::clone(base), |base: &Arc<UOp>, t: &Arc<UOp>| window(base, t.clone(), 1.into()), true ; "an unwindowed target covers every read")]
fn self_assign_keeps_the_temp_unless_disjointness_is_proven(
    write: fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    read: fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    keeps: bool,
) {
    let (base, t) = (buffer(8), scan_counter());
    assert_eq!(self_assign_keeps_temp(write(&base, &t), read(&base, &t)), keeps);
}

/// `[8, 1]`-shaped view of an 8-element buffer.
fn column_of(base: &Arc<UOp>) -> Arc<UOp> {
    base.try_reshape(&smallvec::smallvec![8usize.into(), 1usize.into()]).expect("reshape")
}
