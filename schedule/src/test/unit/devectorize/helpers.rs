//! The one fixture surface for the devectorizer and its neighbours: pass entry points, shaped/WMMA builders, and the
//! toposort assertions they share.
use crate::devectorize::{bool_storage_patterns, devectorize, devectorize_patterns, no_vectorized_alu};
use crate::optimizer::Renderer;
use crate::rewrite::graph_rewrite;
use crate::spec::SpecError;
pub use crate::test::support::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{Op, ParamArg, RendererDevice, SInt, UOp, WmmaMetadata, WmmaUpcastAxes, ops};
/// A structured codegen PARAM, the form `spec.rs` and the shaped-INDEX rules use. Distinct from the prelude's
/// `param`, which takes a size instead of an address space and a device.
pub fn codegen_param(slot: usize, dtype: DType, addrspace: AddrSpace, device: Option<DeviceSpec>) -> Arc<UOp> {
    let arg = ParamArg::buffer(slot, dtype.clone(), addrspace, device);
    UOp::new(Op::Param(ops::Param { shape: UOp::stack(SmallVec::new()), arg: arg.into() }), dtype)
}
/// A PARAM with no address space: `INDEX` over it is what the devectorizer splits.
pub fn buffer_to_define(buffer: &Arc<UOp>) -> Arc<UOp> {
    UOp::param(buffer.id as usize, buffer.buffer_size().unwrap_or(1024), buffer.dtype(), None)
}
/// `INDEX(PARAM(buffer), STACK(offsets))` — the shaped address the pass splits.
pub fn shaped_addr(buffer: &Arc<UOp>, offsets: impl IntoIterator<Item = i64>) -> Arc<UOp> {
    let lanes: SmallVec<[Arc<UOp>; 4]> = offsets.into_iter().map(index_const).collect();
    UOp::new(
        Op::Index(ops::Index { buffer: buffer_to_define(buffer), indices: smallvec![UOp::stack(lanes)] }),
        DType::Scalar(buffer.dtype().base()),
    )
}
pub fn iota_addr(buffer: &Arc<UOp>, count: usize) -> Arc<UOp> {
    shaped_addr(buffer, 0..count as i64)
}
/// A `count`-lane Float32 STACK reshaped to `shape`; `tag` keeps operands distinct.
pub fn shaped_f32(tag: &str, count: usize, shape: &[usize]) -> Arc<UOp> {
    let values = UOp::stack((0..count).map(|i| UOp::var(format!("{tag}_{i}"), DType::Float32, -100, 100)).collect());
    values.try_reshape(&shape.iter().copied().map(SInt::Const).collect()).expect("reshape shape must match the source")
}
/// `RESHAPE(src, shape)`.
pub fn reshape_to(src: &Arc<UOp>, shape: &[usize]) -> Arc<UOp> {
    src.try_reshape(&shape.iter().copied().map(SInt::Const).collect()).expect("reshape shape must match the source")
}
/// `16x16x16` Float32 CPU WMMA metadata.
pub fn wmma_metadata(name: &str, upcast_axes: Option<WmmaUpcastAxes>) -> WmmaMetadata {
    WmmaMetadata {
        name: name.into(),
        dims: (16, 16, 16),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: RendererDevice::Cpu,
        threads: 32,
        upcast_axes,
        reduce_axes: vec![],
    }
}
/// A WMMA with the `[6]`-lane test operand on both inputs.
pub fn wmma_default(c: Arc<UOp>) -> Arc<UOp> {
    let operand = shaped_f32("operand", 6, &[6]);
    UOp::wmma(operand.clone(), operand, c, wmma_metadata("test", None))
}
/// `Bool` LOAD/STORE -> uint8 storage, BitCast(Bool) -> CAST.
pub fn apply_bool_storage(uop: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(bool_storage_patterns(), uop.clone(), &mut ())
}
pub fn apply_devectorize(uop: &Arc<UOp>) -> Arc<UOp> {
    devectorize(uop, &Renderer::cpu())
}
pub fn apply_devectorize_patterns(uop: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(devectorize_patterns(), uop, &mut ())
}
pub fn apply_no_vectorized_alu(uop: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(no_vectorized_alu(), uop.clone(), &mut ())
}
/// REDUCE -> accumulator (`reduce_to_acc`).
pub fn apply_pm_reduce(uop: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&crate::devectorize::pm_reduce(), uop.clone(), &mut crate::devectorize::ReduceContext::default())
}
pub fn apply_gater(root: &Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(&crate::late::pm_move_gates_from_index(), root.clone(), &mut ())
}
pub fn apply_final_rewrite(root: Arc<UOp>) -> Arc<UOp> {
    graph_rewrite(crate::optimizer::final_rewrite_patterns(), root, &mut ())
}
pub fn apply_spec_program(root: &Arc<UOp>) -> Result<(), SpecError> {
    crate::spec::type_verify(root, &crate::spec::spec_program())
}
pub fn assert_no_invalid(root: &Arc<UOp>) {
    assert!(!root.toposort().iter().any(UOp::is_invalid_marker), "unexpected Invalid marker:\n{}", root.tree());
}
/// The scalar count `uop` carries: its shape product, else its mechanical vector width.
pub fn assert_vcount(uop: &Arc<UOp>, expected: usize) {
    let count = uop
        .shape()
        .ok()
        .flatten()
        .and_then(|shape| shape.iter().try_fold(1usize, |product, dim| Some(product * dim.as_const()?)))
        .unwrap_or_else(|| uop.dtype().vcount());
    assert_eq!(count, expected, "element count mismatch: expected {expected}, got {count}");
}
pub fn loads(uop: &Arc<UOp>) -> usize {
    count(uop, |node| matches!(node.op(), Op::Load(..)))
}
pub fn stores(uop: &Arc<UOp>) -> usize {
    count(uop, |node| matches!(node.op(), Op::Store(..)))
}
pub fn ends(uop: &Arc<UOp>) -> usize {
    count(uop, |node| matches!(node.op(), Op::End(..)))
}
pub fn regs(uop: &Arc<UOp>) -> usize {
    count(uop, |node| matches!(node.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(AddrSpace::Reg)))
}
