use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{ConstValue, Op, ReduceOp, UOp, ops};

use crate::late::demote_unsupported_floats;
use crate::optimizer::Renderer;
use crate::test::support::prelude::*;

fn f64_const(value: f64) -> Arc<UOp> {
    UOp::const_(DType::Float64, ConstValue::Float(value))
}

fn has_dtype(root: &Arc<UOp>, scalar: ScalarDType) -> bool {
    root.toposort().iter().any(|node| node.dtype().base() == scalar)
}

/// `out = f32((f64(in) * 0.5) + 1.25)` — the linspace shape.
fn internal_f64_sink() -> Arc<UOp> {
    let input = load(index(param(1, 4, DType::Float32), 0));
    let scaled = input.cast(DType::Float64).try_mul(&f64_const(0.5)).unwrap().try_add(&f64_const(1.25)).unwrap();
    UOp::sink(vec![store(index(param(0, 4, DType::Float32), 0), scaled.cast(DType::Float32))])
}

/// A renderer that cannot hold internal Float64 computes the chain in Float32.
///
/// One renderer is enough: `demote_unsupported_floats` handles Float64 alone (late/dtype.rs:23-28),
/// so every renderer lacking it — metal, webgpu — drives the identical `DemoteFloat` rewrite, and a
/// second row would exercise no new branch. A Float16-unsupported renderer is likewise no different,
/// because the pass never inspects Float16.
#[test]
fn internal_f64_computes_in_f32_when_unsupported() {
    let sink = internal_f64_sink();
    assert!(has_dtype(&sink, ScalarDType::Float64));

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());

    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    let float_consts: Vec<_> = demoted
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Const(value) if matches!(value.0, ConstValue::Float(_))))
        .collect();
    // Exactly the two the chain started with: a demotion that rebuilds a constant instead of
    // reusing the hash-consed node would leave duplicates behind, and `is_empty()` would miss it.
    assert_eq!(float_consts.len(), 2, "{}", demoted.tree());
    assert!(float_consts.iter().all(|node| node.dtype() == DType::Float32), "{}", demoted.tree());
}

#[test]
fn supported_renderers_are_untouched() {
    let sink = internal_f64_sink();
    let same = demote_unsupported_floats(sink.clone(), &Renderer::cpu());
    assert_same!(same, sink);
}

#[test]
fn external_f64_storage_keeps_its_dtype() {
    let input = param(1, 4, DType::Float64);
    let loaded = load(index(input.clone(), 0));
    let doubled = loaded.try_mul(&f64_const(2.0)).unwrap();
    let sink = UOp::sink(vec![store(index(param(0, 4, DType::Float32), 0), doubled.cast(DType::Float32))]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    let nodes = demoted.toposort();
    let param = nodes.iter().find(|node| matches!(node.op(), Op::Param(..)) && node.dtype() == DType::Float64);
    assert!(param.is_some(), "external storage must keep Float64:\n{}", demoted.tree());
    let loaded = first_op(&demoted, |op| matches!(op, Op::Load(..))).expect("load survives");
    assert_eq!(loaded.dtype(), DType::Float64);
    // The arithmetic itself runs in f32 on a converted load.
    let mul = first_op(&demoted, |op| matches!(op, Op::Binary(svod_ir::BinaryOp::Mul, ..))).expect("mul");
    assert_eq!(mul.dtype(), DType::Float32, "{}", demoted.tree());
}

/// A gated load from **global** f64 storage keeps its `alt` — the same constant still
/// feeds internal f32 math — while a local one demotes the `alt` too.
#[test]
fn gated_load_alts_survive_for_external_storage_only() {
    let gated = |buffer: Arc<UOp>, alt: Arc<UOp>| {
        UOp::new(
            Op::Load(ops::Load { index: index(buffer, 0), alt: Some(alt), gate: Some(UOp::native_const(true)) }),
            DType::Float64,
        )
    };

    let zero = f64_const(0.0);
    let internal = zero.try_add(&f64_const(1.0)).unwrap();
    let value = gated(param(1, 4, DType::Float64), zero).try_add(&internal).unwrap().cast(DType::Float32);
    let sink = UOp::sink(vec![store(index(param(0, 4, DType::Float32), 0), value)]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());
    let load = first_op(&demoted, |op| matches!(op, Op::Load(..))).expect("load survives");
    let alt = unwrap_op!(load, Op::Load(l) => l).alt.clone().expect("alt survives");
    assert_eq!((load.dtype(), alt.dtype()), (DType::Float64, DType::Float64), "{}", demoted.tree());
    let add = demoted
        .toposort()
        .into_iter()
        .find(|node| matches!(node.op(), Op::Binary(svod_ir::BinaryOp::Add, ..)) && node.dtype() == DType::Float32)
        .expect("the internal add runs in f32");
    assert!(add.op().sources().iter().all(|source| source.dtype() == DType::Float32), "{}", demoted.tree());

    let local = UOp::sink(vec![gated(UOp::buffer(3, 4, DType::Float64, AddrSpace::Local, None), f64_const(0.0))]);
    let demoted = demote_unsupported_floats(local, &Renderer::metal());
    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    let load = first_op(&demoted, |op| matches!(op, Op::Load(..))).unwrap();
    let alt = unwrap_op!(load, Op::Load(l) => l).alt.clone().expect("alt survives");
    assert_eq!(alt.dtype(), DType::Float32);
}

/// `Op::Param` and `Op::BitCast` are never rewritten: a PARAM's slot belongs to the
#[test]
fn bitcasts_and_params_are_never_demoted() {
    let bits = UOp::const_(DType::Int64, ConstValue::Int(0x3FF0000000000000)).bitcast(DType::Float64);
    let param = UOp::scalar_param(0, None, DType::Float64, 0, 1);
    let sum = bits.try_add(&param).unwrap();
    let sink = UOp::sink(vec![bits, param, sum]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());

    assert!(first_op(&demoted, |op| matches!(op, Op::BitCast(..))).is_some(), "the bitcast survives");
    assert!(first_op(&demoted, |op| matches!(op, Op::Param(p) if p.arg.dtype == DType::Float64)).is_some());
    let sum = first_op(&demoted, |op| matches!(op, Op::Binary(svod_ir::BinaryOp::Add, ..))).expect("the add survives");
    assert_eq!(sum.dtype(), DType::Float32, "the arithmetic around them still runs in f32");
}

#[test]
fn scratch_buffers_and_reductions_are_demoted() {
    let local = UOp::buffer(3, 16, DType::Float64, AddrSpace::Local, None);
    let value = load(index(param(1, 16, DType::Float32), 0)).cast(DType::Float64);
    let fill = store(index(local.clone(), 0), value);
    let range = reduce_range(16, 0);
    let partial = load(index_of(local, range.clone()));
    let sum = partial.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let out = store(index(param(0, 1, DType::Float32), 0), sum.cast(DType::Float32));
    let sink = UOp::sink(vec![fill, sum.end(smallvec::smallvec![range]), out]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());

    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    let local = first_op(&demoted, |op| matches!(op, Op::Buffer(b) if b.arg.addrspace == Some(AddrSpace::Local)))
        .expect("local buffer");
    let arg = unwrap_op!(local, Op::Buffer(b) => b).arg.clone();
    assert_eq!(arg.dtype, DType::Float32);
}

#[test]
fn vector_f64_becomes_vector_f32() {
    let lanes = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(2.0)], DType::Float64);
    let sink = UOp::sink(vec![lanes.try_add(&lanes).unwrap()]);

    let demoted = demote_unsupported_floats(sink, &Renderer::metal());

    assert!(!has_dtype(&demoted, ScalarDType::Float64), "{}", demoted.tree());
    assert!(demoted.toposort().iter().any(|node| node.dtype() == DType::Float32.vec(2).unwrap()), "{}", demoted.tree());
}
