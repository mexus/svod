//! Origin propagation from the tensor graph to the kernel cut, and the harvest, strip and stamp that `split_store` performs there.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::origin::{self, OriginId, OriginScope};
use svod_ir::{CallInfo, Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;

use crate::rangeify::{kernel_graph_pre_cut, rangeify, try_get_kernel_graph};
use crate::test::support::build::{buffer, expect_call};

/// Build a sink under a fresh module scope, returning it with that scope's id.
fn under_module(name: &str, build: impl FnOnce() -> Arc<UOp>) -> (Arc<UOp>, OriginId) {
    let _scope = OriginScope::module(name);
    (UOp::sink(vec![build()]), origin::current().expect("module scope while capture is on"))
}

/// The kernel graph the cut produces from `sink`.
fn kernel_graph_of(sink: Arc<UOp>) -> Arc<UOp> {
    try_get_kernel_graph(rangeify(sink).expect("rangeify").0).expect("kernel graph").0
}

/// The CALLs in topological order.
fn kernels_of(graph: &Arc<UOp>) -> Vec<Arc<UOp>> {
    graph.toposort().into_iter().filter(|node| matches!(node.op(), Op::Call(..))).collect()
}

fn stores_of(graph: &Arc<UOp>) -> Vec<Arc<UOp>> {
    graph.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..))).collect()
}

fn info_of(call: &Arc<UOp>) -> &CallInfo {
    match call.op() {
        Op::Call(ops::Call { info, .. }) => info,
        op => panic!("expected CALL, got {op:?}"),
    }
}

/// A kernel is charged to `id` alone, and its body is origin-free so that identical kernels hash-cons to one program.
fn assert_scoped_kernel(call: &Arc<UOp>, id: OriginId) {
    let info = info_of(call);
    assert_eq!(info.origin, Some(id), "kernel is charged to the scope it was built under");
    assert_eq!(info.origins.iter().copied().collect::<Vec<_>>(), [id], "single-scope kernel carries one origin");
    assert!(
        expect_call(call).toposort().iter().all(|node| node.origin().is_none()),
        "the kernel body must be origin-free so identical kernels share one program"
    );
}

fn matrix(rows: usize, cols: usize) -> Arc<UOp> {
    buffer(rows * cols).try_reshape(&smallvec![SInt::Const(rows), SInt::Const(cols)]).expect("reshape")
}

// =========================================================================
// Propagation up to the cut
// =========================================================================

/// The `kind`-th graph shape: one row per op family the cut sees.
#[track_caller]
fn graph(kind: usize) -> Arc<UOp> {
    match kind {
        0 => buffer(8).try_mul(&buffer(8)).expect("mul").contiguous(),
        1 => matrix(2, 3).try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce").contiguous(),
        2 => matrix(2, 3).try_permute(vec![1, 0]).expect("permute").contiguous(),
        // A reshape of a realized buffer is a pure view; it needs compute under it.
        3 => buffer(6)
            .try_mul(&buffer(6))
            .expect("mul")
            .try_reshape(&smallvec![SInt::Const(3), SInt::Const(2)])
            .expect("reshape")
            .contiguous(),
        4 => matrix(3, 1).try_expand(&smallvec![SInt::Const(3), SInt::Const(4)]).expect("expand").contiguous(),
        5 => matrix(2, 3).try_pad(&[(1.into(), 1.into()), (0.into(), 0.into())]).expect("pad").contiguous(),
        6 => matrix(4, 4)
            .try_shrink(&[(SInt::Const(1), SInt::Const(3)), (SInt::Const(0), SInt::Const(4))])
            .expect("shrink")
            .contiguous(),
        7 => buffer(8).cast(DType::Float16).contiguous(),
        8 => matrix(2, 3)
            .try_mul(&matrix(2, 3))
            .expect("mul")
            .try_reduce_axis(ReduceOp::Add, vec![1])
            .expect("reduce")
            .contiguous(),
        // A permuted view forces a CONTIGUOUS materialisation ahead of the transfer.
        9 => graph(2).copy_to_device(DeviceSpec::Amd { device_id: 0 }),
        other => panic!("no graph shape {other}"),
    }
}

/// One row per graph shape: the pre-cut STORE chain keeps the module origin, and every kernel the cut produces is charged to that same scope with an origin-free body.
#[test_case(0; "elementwise")]
#[test_case(1; "reduce")]
#[test_case(2; "permute")]
#[test_case(3; "reshape")]
#[test_case(4; "expand")]
#[test_case(5; "pad")]
#[test_case(6; "shrink")]
#[test_case(7; "cast")]
#[test_case(8; "fused reduce")]
#[test_case(9; "permuted cross-device copy")]
fn every_store_and_kernel_carries_the_scope_origin(kind: usize) {
    let _capture = origin::capture_for_thread(true);
    let (sink, id) = under_module("scoped", || graph(kind));
    let rangeified = rangeify(sink).expect("rangeify").0;
    let pre_cut = kernel_graph_pre_cut(rangeified.clone()).0;
    let stores = stores_of(&pre_cut);
    assert!(!stores.is_empty(), "the pipeline must produce at least one STORE:\n{}", pre_cut.tree());
    assert!(
        stores.iter().all(|store| store.origin() == Some(id)),
        "every STORE in the chain must keep the module origin:\n{}",
        pre_cut.tree()
    );
    let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");
    let kernels = kernels_of(&graph);
    assert!(!kernels.is_empty(), "expected at least one kernel:\n{}", graph.tree());
    for call in &kernels {
        assert_scoped_kernel(call, id);
    }
}

// =========================================================================
// Harvest, strip, stamp
// =========================================================================

/// The load-bearing property: same computation, two scopes ⇒ two CALLs with distinct attribution over one shared body.
#[test]
fn identical_kernels_in_two_scopes_share_a_body_and_differ_in_origin() {
    let _capture = origin::capture_for_thread(true);
    let kernel_of = |name: &str| {
        let (sink, id) = under_module(name, || graph(0));
        let call = kernels_of(&kernel_graph_of(sink)).first().cloned().expect("one kernel");
        (call, id)
    };
    let (left, left_id) = kernel_of("a");
    let (right, right_id) = kernel_of("b");
    assert_ne!(left_id, right_id);
    assert_eq!(info_of(&left).origin, Some(left_id));
    assert_eq!(info_of(&right).origin, Some(right_id));
    assert!(!Arc::ptr_eq(&left, &right), "distinct origins must not collapse the two dispatches");
    assert!(
        Arc::ptr_eq(&expect_call(&left), &expect_call(&right)),
        "stripped bodies hash-cons to one node, so the optimizer and every kernel cache see one kernel"
    );
}

/// The strip must be an identity on structure: the stripped body is the very node an origin-free build produces.
#[test]
fn a_stripped_body_hash_conses_to_the_origin_free_build() {
    let bodies = |capture: bool| {
        let _capture = origin::capture_for_thread(capture);
        let sink = if capture { under_module("stripped", || graph(8)).0 } else { UOp::sink(vec![graph(8)]) };
        let graph = kernel_graph_of(sink);
        kernels_of(&graph).iter().map(expect_call).collect::<Vec<_>>()
    };
    let scoped = bodies(true);
    let plain = bodies(false);
    assert_eq!(scoped.len(), plain.len());
    for (with_origin, without) in scoped.iter().zip(&plain) {
        assert!(
            Arc::ptr_eq(with_origin, without),
            "stripping must reproduce the origin-free node, not merely an equal one"
        );
    }
}

#[test]
fn a_kernel_fusing_two_scopes_carries_both_origins() {
    let _capture = origin::capture_for_thread(true);
    let (left, left_id) = {
        let _scope = OriginScope::module("left");
        (buffer(8).try_add(&buffer(8)).expect("add"), origin::current().expect("scope"))
    };
    let (right, right_id) = {
        let _scope = OriginScope::module("right");
        // The multiplication is the stored value, so `right` is the primary; the
        // addition fuses into the same kernel and joins the set.
        (left.try_mul(&buffer(8)).expect("mul").contiguous(), origin::current().expect("scope"))
    };
    let graph = kernel_graph_of(UOp::sink(vec![right]));
    let call = kernels_of(&graph).first().cloned().expect("one kernel");
    let info = info_of(&call);
    assert_eq!(info.origin, Some(right_id), "the stored value's scope is the primary");
    assert_eq!(info.origins.len(), 2, "a fused kernel lists every scope it consumed: {:?}", info.origins);
    assert!(
        info.origins.contains(&right_id) && info.origins.contains(&left_id),
        "both the primary and the fused scope must be listed: {:?}",
        info.origins
    );
}

/// A module inside a module nests: the compute is charged to the innermost scope it was built under, while the sink belongs to the outer one.
#[test]
fn a_nested_module_scope_is_attributed_to_the_innermost_module() {
    let _capture = origin::capture_for_thread(true);
    let _outer = OriginScope::module("outer");
    let outer_id = origin::current().expect("outer scope");
    let (compute, inner_id) = {
        let _inner = OriginScope::module("inner");
        (graph(0), origin::current().expect("inner scope"))
    };
    assert_ne!(inner_id, outer_id, "each module scope gets its own id");
    let graph = kernel_graph_of(UOp::sink(vec![compute]));
    let kernels = kernels_of(&graph);
    assert!(!kernels.is_empty(), "expected a kernel:\n{}", graph.tree());
    for call in &kernels {
        assert_scoped_kernel(call, inner_id);
    }
}

#[test]
fn a_copy_only_kernel_is_attributed() {
    let _capture = origin::capture_for_thread(true);
    // Cross-device so the copy is not elided; nothing is executed here.
    let (sink, id) = under_module("copy", || buffer(8).copy(DeviceSpec::Amd { device_id: 0 }));
    let graph = kernel_graph_of(sink);
    let kernels = kernels_of(&graph);
    assert!(!kernels.is_empty(), "a copy is still a kernel:\n{}", graph.tree());
    assert!(
        kernels.iter().all(|call| info_of(call).origin == Some(id)),
        "copy-only kernels lose their tag at bufferize; the origin must survive it"
    );
    assert!(
        kernels.iter().all(|call| expect_call(call).toposort().iter().all(|node| node.origin().is_none())),
        "a direct COPY body is stripped like any other kernel body"
    );
}

#[test]
fn capture_off_leaves_kernels_unattributed() {
    let _capture = origin::capture_for_thread(false);
    // The scope is a no-op while capture is off, exactly as in a default build.
    let _scope = OriginScope::module("ignored");
    let sink = UOp::sink(vec![graph(0)]);
    let (graph, _) = try_get_kernel_graph(rangeify(sink).expect("rangeify").0).expect("kernel graph");
    let kernels = kernels_of(&graph);
    assert!(!kernels.is_empty(), "the graph is unattributed, not empty:\n{}", graph.tree());
    for call in &kernels {
        assert_eq!(info_of(call).origin, None);
        assert!(info_of(call).origins.is_empty());
    }
}
