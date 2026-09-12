use smallvec::smallvec;
use svod_ir::{CallInfo, DType, DeviceSpec, Error, KernelInfo, UOp};

use super::fix_assign;
use svod_ir::ops;

fn buffer() -> std::sync::Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32)
}

fn sink_call(args: Vec<std::sync::Arc<UOp>>) -> std::sync::Arc<UOp> {
    UOp::sink_with_info(vec![], KernelInfo::default()).call(args.into_iter().collect(), CallInfo::default())
}

#[test]
fn test_fix_assign_cycle_returns_typed_error() {
    let (b1, b2) = (buffer(), buffer());

    let call_for_b2 = sink_call(vec![b2.clone()]);
    let after_b2 = b2.after(smallvec![call_for_b2]);

    let call_for_b1 = UOp::sink_with_info(vec![], KernelInfo::default()).call(
        smallvec![b2.clone()],
        CallInfo { grad_tag: None, name: Some("writer".to_string()), ..CallInfo::default() },
    );
    let after_b1 = b1.after(smallvec![after_b2, call_for_b1]);

    let err = fix_assign(&UOp::sink(vec![after_b1])).expect_err("expected typed cycle error");

    assert!(matches!(
        err,
        Error::KernelSplitDependencyCycle { writer_buffer, read_buffer }
            if writer_buffer == b1.id && read_buffer == b2.id
    ));
}

fn find_after_for_buffer(root: &std::sync::Arc<UOp>, buffer_id: u64) -> std::sync::Arc<UOp> {
    root.toposort()
        .into_iter()
        .find(|u| matches!(u.op(), svod_ir::Op::After(..)) && u.buf_uop().id == buffer_id)
        .expect("expected AFTER for buffer")
}

fn after_deps(uop: &std::sync::Arc<UOp>) -> smallvec::SmallVec<[std::sync::Arc<UOp>; 4]> {
    let svod_ir::Op::After(ops::After { deps, .. }) = uop.op() else { panic!("expected AFTER op") };
    deps.clone()
}

/// A WAR dep is added when the callable differs even though the two AFTERs share
/// a dependency; the shared dep is still there.
#[test]
fn test_fix_assign_adds_war_dep_when_callable_differs_even_with_shared_dep() {
    let (read_write_buf, output_buf) = (buffer(), buffer());
    let shared_dep = UOp::noop();

    let writer_after = read_write_buf.after(smallvec![shared_dep.clone(), sink_call(vec![read_write_buf.clone()])]);
    let reader_after =
        output_buf.after(smallvec![shared_dep, sink_call(vec![read_write_buf.clone(), output_buf.clone()])]);

    let fixed =
        fix_assign(&UOp::sink(vec![writer_after.clone(), reader_after.clone()])).expect("fix_assign should succeed");

    let deps = after_deps(&find_after_for_buffer(&fixed, read_write_buf.id));
    assert!(
        deps.iter().any(|d| matches!(d.op(), svod_ir::Op::After(..)) && d.buf_uop().id == output_buf.id),
        "writer AFTER should depend on reader AFTER when callable differs"
    );
}

#[test]
fn test_fix_assign_skips_war_dep_for_same_callable_multi_output() {
    let (shared_buf, output_buf) = (buffer(), buffer());

    let shared_callable = sink_call(vec![shared_buf.clone(), output_buf.clone()]);
    let writer_after = shared_buf.after(smallvec![shared_callable.clone()]);
    let reader_after = output_buf.after(smallvec![shared_callable]);

    let fixed = fix_assign(&UOp::sink(vec![writer_after.clone(), reader_after])).expect("fix_assign should succeed");

    let deps = after_deps(&find_after_for_buffer(&fixed, shared_buf.id));
    assert_eq!(deps.len(), 1, "same-callable outputs should not receive extra WAR deps");
}
