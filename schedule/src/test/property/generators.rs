//! Schedule-local generators for the shapes the `ir` arithmetic trees never build:
//! bitwise/division/comparison ops, and the shaped, gated and reduced graphs the passes consume.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::{DType, ScalarDType};
use svod_ir::types::{BinaryOp, ConstValue, ReduceOp, UnaryOp};
use svod_ir::{AxisId, AxisType, SInt, UOp};

use crate::test::support::prelude::*;

use svod_ir::test::property::generators::*;

/// `op(lhs, rhs)` for every op the symbolic tiers rewrite, or `None` outside its domain.
pub fn build_binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Option<Arc<UOp>> {
    match op {
        BinaryOp::Add => lhs.try_add(&rhs).ok(),
        BinaryOp::Sub => lhs.try_sub(&rhs).ok(),
        BinaryOp::Mul => lhs.try_mul(&rhs).ok(),
        BinaryOp::Max => lhs.try_max(&rhs).ok(),
        BinaryOp::FloorDiv => lhs.try_div(&rhs).ok(),
        BinaryOp::FloorMod => lhs.try_mod(&rhs).ok(),
        BinaryOp::And => lhs.try_and_op(&rhs).ok(),
        BinaryOp::Or => lhs.try_or_op(&rhs).ok(),
        BinaryOp::Xor => lhs.try_xor_op(&rhs).ok(),
        BinaryOp::Eq => lhs.try_cmpeq(&rhs).ok(),
        BinaryOp::Ne => lhs.try_cmpne(&rhs).ok(),
        _ => None,
    }
}

/// The integer dtypes whose constants `arb_const_uop` can build (`Index`/`Void` excluded).
pub fn arb_int_property_dtype() -> impl Strategy<Value = DType> {
    arb_int_dtype().prop_filter("has a constant generator", |dtype| {
        !matches!(dtype.scalar(), Some(ScalarDType::Index) | Some(ScalarDType::Void))
    })
}

/// Every dtype family the properties sweep.
pub fn arb_property_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![arb_int_property_dtype(), arb_float_dtype()]
}

/// A depth-`depth` tree over the full binary/unary op surface at `dtype`: a constant or a
/// variable whose *name* encodes its range, since the frozen `Bindings` matches by name.
pub fn arb_op_tree(dtype: DType, depth: usize) -> impl Strategy<Value = Arc<UOp>> {
    let mut binary_ops = vec![BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Max, BinaryOp::FloorDiv];
    if !dtype.is_float() {
        binary_ops.extend([BinaryOp::And, BinaryOp::Or, BinaryOp::Xor, BinaryOp::FloorMod]);
    }
    let unary_ops = if dtype.is_float() { vec![UnaryOp::Neg] } else { vec![UnaryOp::Neg, UnaryOp::Not] };
    let leaf = prop_oneof![
        arb_const_uop(dtype.clone()),
        (1i64..100).prop_map(move |max| UOp::var(format!("v{max}"), dtype.clone(), 0, max))
    ];
    leaf.prop_recursive(depth as u32, depth as u32 * 4, 3, move |inner| {
        let binary = (prop::sample::select(binary_ops.clone()), inner.clone(), inner.clone())
            .prop_filter_map("binary op is defined on its operands", |(op, lhs, rhs)| build_binary(op, lhs, rhs));
        let unary = (prop::sample::select(unary_ops.clone()), inner).prop_map(|(op, src)| match op {
            UnaryOp::Neg => src.neg(),
            UnaryOp::Not => src.not(),
            _ => unreachable!("the strategy only samples Neg/Not"),
        });
        prop_oneof![3 => binary, 1 => unary]
    })
}

/// [`arb_op_tree`] at every depth up to `max_depth`.
pub fn arb_op_tree_up_to(dtype: DType, max_depth: usize) -> impl Strategy<Value = Arc<UOp>> {
    (0..=max_depth).prop_flat_map(move |depth| arb_op_tree(dtype.clone(), depth))
}

/// A Float32 buffer of `rows * cols`, shifted by `bias`, reshaped to `[rows, cols]`.
fn shaped_view(rows: i64, cols: i64, bias: f64) -> Arc<UOp> {
    let shape: svod_ir::shape::Shape = [SInt::Const(rows as usize), SInt::Const(cols as usize)].into_iter().collect();
    buffer_of((rows * cols) as usize, ScalarDType::Float32)
        .try_add(&UOp::const_(DType::Float32, ConstValue::Float(bias)))
        .expect("ADD accepts matching scalar dtypes")
        .try_reshape(&shape)
        .expect("RESHAPE keeps the element count")
}

/// `REDUCE(tree, [RANGE(extent)])` — the shape `pm_reduce` lowers.
pub fn arb_reduce_graph() -> impl Strategy<Value = Arc<UOp>> {
    let reduce_op = prop_oneof![Just(ReduceOp::Add), Just(ReduceOp::Mul), Just(ReduceOp::Max), Just(ReduceOp::Min)];
    (arb_op_tree_up_to(DType::Float32, 2), 2i64..9, reduce_op)
        .prop_map(|(src, extent, op)| reduce(src, vec![reduce_range(extent, 0)], op))
}

/// A Bool value stored to and loaded from a Bool buffer (`bool_storage_patterns`).
pub fn arb_bool_memory_graph() -> impl Strategy<Value = Arc<UOp>> {
    (arb_op_tree_up_to(DType::Bool, 2), 1i64..8).prop_map(|(value, at)| {
        let cell = index(buffer_of(8, ScalarDType::Bool), at);
        UOp::sink(vec![store(cell.clone(), value), load(cell)])
    })
}

/// An ALU tree over a two-axis shaped view of a buffer, for `no_vectorized_alu`.
pub fn arb_shaped_alu_graph() -> impl Strategy<Value = Arc<UOp>> {
    let arith = vec![BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Max, BinaryOp::FloorDiv];
    (2i64..6, 2i64..6, prop::sample::select(arith), 0i64..2).prop_map(|(rows, cols, op, which)| {
        let (lhs, rhs) = (shaped_view(rows, cols, which as f64), shaped_view(rows, cols, (which + 1) as f64));
        let combined = build_binary(op, lhs.clone(), rhs).unwrap_or(lhs);
        UOp::sink(vec![store(index(buffer_of(64, ScalarDType::Float32), 0), combined)])
    })
}

/// `INDEX(buffer, range.valid(range < bound))` inside a LOOP, for `pm_simplify_ranges`.
pub fn arb_gated_range_graph() -> impl Strategy<Value = Arc<UOp>> {
    (2i64..16, 1i64..16, 0usize..4).prop_map(|(extent, bound, axis)| {
        let range = UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(axis), AxisType::Loop);
        let gate = range.try_cmplt(&UOp::index_const(bound)).expect("CMPLT over index arithmetic");
        let address = UOp::index()
            .buffer(buffer_of(16, ScalarDType::Float32))
            .indices(vec![range.valid(gate)])
            .call()
            .expect("INDEX accepts one index");
        UOp::sink(vec![load(address)])
    })
}

/// `x * 2^k`, `x % 2^k` or `x.cdiv(2^k)` over an Int32 variable (late strength reduction).
pub fn arb_strength_reducible_graph() -> impl Strategy<Value = Arc<UOp>> {
    (0usize..3, 1u32..6, 1i64..100).prop_map(|(which, shift, max)| {
        let x = UOp::var("late_x", DType::Int32, 0, max);
        let factor = UOp::const_(DType::Int32, ConstValue::Int(1i64 << shift));
        match which {
            0 => x.try_mul(&factor).expect("MUL"),
            1 => x.try_mod(&factor).expect("MOD"),
            _ => x.try_cdiv(&factor).expect("CDIV"),
        }
    })
}

/// `STORE(cell, LOAD(cell) * 1.0)` with no DEFINE_VAR the spec would reject.
pub fn arb_kernel_graph() -> impl Strategy<Value = Arc<UOp>> {
    (1i64..8).prop_map(|at| {
        let cell = index(buffer_of(8, ScalarDType::Float32), at);
        let scaled = load(cell.clone()).try_mul(&UOp::native_const(1.0f32)).expect("MUL accepts matching dtypes");
        UOp::sink(vec![store(cell, scaled)])
    })
}

/// A transposed, materialised view of a buffer, for `rangeify`/`devectorize`.
pub fn arb_movement_sink() -> impl Strategy<Value = Arc<UOp>> {
    (2i64..6, 2i64..6, Just(vec![1, 0])).prop_map(|(rows, cols, axes)| {
        let source = shaped_view(rows, cols, 0.0);
        UOp::sink(vec![source.try_permute(axes).expect("a permutation of the axes").contiguous()])
    })
}

/// `(PERMUTE(RESHAPE(buffer, dims), axes), dims, axes)`: a movement graph with a known inverse.
pub fn arb_movement_graph() -> impl Strategy<Value = (Arc<UOp>, Vec<i64>, Vec<usize>)> {
    (1usize..=3).prop_flat_map(|rank| {
        prop::collection::vec(1i64..5, rank).prop_flat_map(move |dims| {
            let perm = prop::sample::subsequence((0..rank).collect::<Vec<_>>(), rank)
                .prop_filter("a permutation", move |axes| axes.len() == rank);
            let dims_for_map = dims.clone();
            perm.prop_map(move |axes| {
                let numel: i64 = dims_for_map.iter().product();
                let shape: svod_ir::shape::Shape = dims_for_map.iter().map(|&dim| SInt::Const(dim as usize)).collect();
                let reshaped = buffer_of(numel as usize, ScalarDType::Float32).try_reshape(&shape).expect("RESHAPE");
                let permuted = reshaped.try_permute(axes.clone()).expect("a permutation of the axes");
                (permuted, dims_for_map.clone(), axes)
            })
        })
    })
}
