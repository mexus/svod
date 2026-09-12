//! Constructors for the shapes the schedulers and passes consume, plus typed
//! accessors that panic with the rendered tree instead of a bare discriminant.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{AxisId, AxisType, BufferizeOpts, ConstValue, Op, ReduceOp, UOp, ops};

use super::vars::index_const;

pub fn buffer(size: usize) -> Arc<UOp> {
    buffer_of(size, ScalarDType::Float32)
}

pub fn buffer_of(size: usize, scalar: ScalarDType) -> Arc<UOp> {
    buffer_on(size, scalar, DeviceSpec::Cpu)
}

pub fn buffer_on(size: usize, scalar: ScalarDType, device: DeviceSpec) -> Arc<UOp> {
    UOp::new_buffer(device, size, DType::Scalar(scalar))
}

pub fn param(slot: usize, size: usize, dtype: DType) -> Arc<UOp> {
    UOp::param(slot, size, dtype, None)
}

#[track_caller]
fn indexed(buffer: Arc<UOp>, indices: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(indices).call().unwrap_or_else(|error| panic!("INDEX should build: {error:?}"))
}

pub fn index(buffer: Arc<UOp>, idx: i64) -> Arc<UOp> {
    index_of(buffer, index_const(idx))
}

#[track_caller]
pub fn index_of(buffer: Arc<UOp>, idx: Arc<UOp>) -> Arc<UOp> {
    indexed(buffer, vec![idx])
}

#[track_caller]
pub fn shaped_index(buffer: Arc<UOp>, offsets: impl IntoIterator<Item = i64>) -> Arc<UOp> {
    indexed(buffer, vec![stack(offsets.into_iter().map(index_const))])
}

pub fn load(index: Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(index).call()
}

pub fn store(index: Arc<UOp>, value: Arc<UOp>) -> Arc<UOp> {
    index.store(value)
}

/// `Global`/`Local` are Index-typed parallel axes; every other axis type is a `WeakInt` one.
pub fn range(end: i64, axis: AxisType, id: usize) -> Arc<UOp> {
    let dtype = match axis {
        AxisType::Global | AxisType::Local => DType::Index,
        _ => DType::WeakInt,
    };
    UOp::range_axis_dtype(UOp::const_(dtype.clone(), ConstValue::Int(end)), AxisId::Renumbered(id), axis, dtype)
}

/// A `Global`, Index-typed axis.
///
/// Named for its axis type on purpose: `UOp::range_const` builds a `Weak`,
/// WeakInt one, and the two were interchanged by name in this suite until the
/// axis type under test started drifting silently.
pub fn global_range(end: i64, id: usize) -> Arc<UOp> {
    range(end, AxisType::Global, id)
}

pub fn range_symbolic(end: Arc<UOp>, id: usize) -> Arc<UOp> {
    UOp::range(end, id)
}

pub fn reduce_range(end: i64, id: usize) -> Arc<UOp> {
    range(end, AxisType::Reduce, id)
}

pub fn stage(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::stage_global(compute, ranges)
}

pub fn stage_with(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>, opts: BufferizeOpts) -> Arc<UOp> {
    UOp::stage(compute, ranges, opts)
}

pub fn stack(values: impl IntoIterator<Item = Arc<UOp>>) -> Arc<UOp> {
    UOp::stack(values.into_iter().collect())
}

pub fn float_values(values: impl IntoIterator<Item = f64>) -> Arc<UOp> {
    stack(values.into_iter().map(|value| UOp::const_(DType::Float32, ConstValue::Float(value))))
}

pub fn bool_values(values: impl IntoIterator<Item = bool>) -> Arc<UOp> {
    stack(values.into_iter().map(|value| UOp::const_(DType::Bool, ConstValue::Bool(value))))
}

pub fn reduce(src: Arc<UOp>, ranges: Vec<Arc<UOp>>, op: ReduceOp) -> Arc<UOp> {
    src.reduce(ranges.into_iter().collect(), op)
}

pub fn elementwise(sizes: &[i64], axis: AxisType) -> Arc<UOp> {
    let mut sources = vec![UOp::native_const(1.0f32)];
    sources.extend(sizes.iter().enumerate().map(|(id, &size)| range(size, axis, id)));
    UOp::sink(sources)
}

pub fn reduce_sink(global_sizes: &[i64], reduce_sizes: &[i64], op: ReduceOp) -> Arc<UOp> {
    let globals = global_sizes.iter().enumerate().map(|(id, &size)| range(size, AxisType::Global, id));
    let reduces =
        reduce_sizes.iter().enumerate().map(|(id, &size)| range(size, AxisType::Reduce, global_sizes.len() + id));
    let mut sources = vec![UOp::native_const(1.0f32).reduce(reduces.collect(), op)];
    sources.extend(globals);
    UOp::sink(sources)
}

/// A per-load operand transform, e.g. a widening cast or an activation.
pub type OperandFn = Box<dyn Fn(Arc<UOp>) -> Arc<UOp>>;

/// `INDEX(buffer(numel))` at `Σ term * stride`, loaded from `stored` through `operand`.
fn load_sum(
    stored: &DType,
    operand: &impl Fn(Arc<UOp>) -> Arc<UOp>,
    numel: i64,
    terms: &[(&Arc<UOp>, i64)],
) -> Arc<UOp> {
    let index = terms
        .iter()
        .map(|(term, stride)| term.try_mul(&index_const(*stride)).expect("index should build"))
        .reduce(|acc, term| acc.try_add(&term).expect("index should build"))
        .expect("a load has at least one term");
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, numel as usize, stored.clone());
    operand(UOp::index().buffer(buffer).indices(vec![index]).call().expect("load should build"))
}

/// Row-major `C[m,n] = sum_k A[m,k] * B[k,n]` over `stored` buffers, each operand through `operand`.
pub fn matmul(m: i64, n: i64, k: i64, stored: DType, operand: Option<OperandFn>) -> Arc<UOp> {
    let apply = |value: Arc<UOp>| match &operand {
        Some(operand) => operand(value),
        None => value,
    };
    let (m_range, n_range, k_range) =
        (range(m, AxisType::Global, 0), range(n, AxisType::Global, 1), range(k, AxisType::Reduce, 2));
    // Address arithmetic is Index-typed throughout; the reduce axes stay WeakInt
    // and enter the index through a cast, as they do in scheduler-built kernels.
    let (m_idx, n_idx, k_idx) = (m_range.cast(DType::Index), n_range.cast(DType::Index), k_range.cast(DType::Index));
    let a = load_sum(&stored, &apply, m * k, &[(&m_idx, k), (&k_idx, 1)]);
    let b = load_sum(&stored, &apply, k * n, &[(&k_idx, n), (&n_idx, 1)]);
    let product = a.try_mul(&b).expect("mul should succeed");
    UOp::sink(vec![product.reduce(std::iter::once(k_range).collect(), ReduceOp::Add), m_range, n_range])
}

#[track_caller]
fn expected(want: &str, u: &Arc<UOp>) -> ! {
    panic!("expected {want}, got {:?}\n{}", u.op(), u.tree())
}

#[track_caller]
pub fn expect_sink(u: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let Op::Sink(ops::Sink { sources, .. }) = u.op() else { expected("SINK", u) };
    sources.to_vec()
}

#[track_caller]
pub fn expect_end(u: &Arc<UOp>) -> (Arc<UOp>, Vec<Arc<UOp>>) {
    let Op::End(ops::End { computation, ranges }) = u.op() else { expected("END", u) };
    (computation.clone(), ranges.to_vec())
}

#[track_caller]
pub fn expect_store(u: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>, Option<Arc<UOp>>) {
    let Op::Store(ops::Store { index, value, gate }) = u.op() else { expected("STORE", u) };
    (index.clone(), value.clone(), gate.clone())
}

#[track_caller]
pub fn expect_index(u: &Arc<UOp>) -> (Arc<UOp>, Vec<Arc<UOp>>) {
    let Op::Index(ops::Index { buffer, indices }) = u.op() else { expected("INDEX", u) };
    (buffer.clone(), indices.to_vec())
}

#[track_caller]
pub fn expect_call(u: &Arc<UOp>) -> Arc<UOp> {
    let Op::Call(ops::Call { body, .. }) = u.op() else { expected("CALL", u) };
    body.clone()
}

#[track_caller]
pub fn expect_after(u: &Arc<UOp>) -> (Arc<UOp>, Vec<Arc<UOp>>) {
    let Op::After(ops::After { passthrough, deps }) = u.op() else { expected("AFTER", u) };
    (passthrough.clone(), deps.to_vec())
}

#[track_caller]
pub fn expect_buffer(u: &Arc<UOp>) -> (usize, DType) {
    let Op::Buffer(ops::Buffer { arg, .. }) = u.op() else { expected("BUFFER", u) };
    let size = u.buffer_size().unwrap_or_else(|| panic!("BUFFER has no static size\n{}", u.tree()));
    (size, arg.dtype.clone())
}

#[track_caller]
pub fn expect_range(u: &Arc<UOp>) -> (Arc<UOp>, AxisId, AxisType) {
    let Op::Range(ops::Range { end, axis_id, axis_type, .. }) = u.op() else { expected("RANGE", u) };
    (end.clone(), axis_id.clone(), *axis_type)
}

#[track_caller]
pub fn expect_range_extent(u: &Arc<UOp>) -> i64 {
    let (end, ..) = expect_range(u);
    let Op::Const(value) = end.op() else { expected("constant RANGE extent", u) };
    value.0.try_int().unwrap_or_else(|| panic!("RANGE extent is not an integer\n{}", u.tree()))
}

#[track_caller]
pub fn range_axis_type(u: &Arc<UOp>) -> AxisType {
    expect_range(u).2
}

#[track_caller]
pub fn range_axis_id(u: &Arc<UOp>) -> AxisId {
    expect_range(u).1
}

/// The first distinct node matching `pred`, in topological order.
pub fn first_op(u: &Arc<UOp>, pred: impl Fn(&Op) -> bool) -> Option<Arc<UOp>> {
    u.toposort().into_iter().find(|node| pred(node.op()))
}

pub fn has_op(u: &Arc<UOp>, pred: impl Fn(&Op) -> bool) -> bool {
    u.toposort().iter().any(|node| pred(node.op()))
}
