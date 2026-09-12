//! Direct tests for the boundary between pre-gater valid indices and post-gater memory ops.
use super::helpers::*;
use std::sync::Arc;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{ConstValue, Op, UOp, ops};
use test_case::test_case;
/// `INDEX(buffer, [index.valid(gate)])` — the pre-gater shape.
fn valid_index(buffer: Arc<UOp>, index: Arc<UOp>, gate: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![index.valid(gate)]).call().unwrap()
}
fn bool_var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::Bool, 0, 1)
}
fn fold(root: Arc<UOp>) -> Arc<UOp> {
    rewrite(Matchers::simple(), root)
}
#[test_case(false; "load")]
#[test_case(true; "store")]
fn a_valid_index_moves_its_gate_onto_the_access(gated_store: bool) {
    let gate = bool_var("gate");
    let address = UOp::index_const(3);
    let indexed = valid_index(buffer(16), address.clone(), gate.clone());
    let result = apply_gater(&if gated_store { indexed.store(UOp::native_const(2.0f32)) } else { load(indexed) });
    let (index, got_gate) = match result.op() {
        Op::Load(ops::Load { index, alt: Some(alt), gate: Some(got) }) => {
            assert!(matches!(alt.op(), Op::Const(v) if v.0 == ConstValue::Float(0.0)), "{}", result.tree());
            (index.clone(), got.clone())
        }
        Op::Store(ops::Store { index, gate: Some(got), .. }) => (index.clone(), got.clone()),
        other => panic!("expected a gated access, got {other:?}\n{}", result.tree()),
    };
    assert!(Arc::ptr_eq(&got_gate, &gate));
    assert!(
        matches!(index.op(), Op::Index(ops::Index { indices, .. }) if Arc::ptr_eq(&indices[0], &address)),
        "{}",
        result.tree()
    );
}
#[test]
fn shaped_load_gate_uses_post_movement_stack_alt() {
    let indices = UOp::stack((0..4).map(UOp::index_const).collect());
    let rooted = UOp::index().buffer(buffer(16)).indices(vec![indices.valid(bool_values([true; 4]))]).call().unwrap();
    let result = apply_gater(&load(rooted));
    let Op::Load(ops::Load { alt: Some(alt), .. }) = result.op() else { panic!("expected a gated LOAD") };
    assert!(matches!(alt.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4));
    assert!(!alt.toposort().iter().any(|node| node.op().is_movement()));
}
#[test]
fn two_index_gate_needs_an_image_buffer() {
    let gate = bool_var("gate");
    let indices = vec![UOp::index_const(1).valid(gate.clone()), UOp::index_const(2).valid(gate)];
    let index = UOp::index().buffer(buffer_of(16, ScalarDType::Int32)).indices(indices).call().unwrap();
    // The image rules hard-code the two-coordinate form; a plain Int32 two-index access must fall through to the
    // generic rule and keep its own dtype.
    let result = apply_gater(&load(index));
    let Op::Load(ops::Load { index, alt: Some(_), gate: Some(_) }) = result.op() else {
        panic!("expected a gated LOAD")
    };
    assert_eq!(index.dtype(), DType::Int32);
    let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("expected INDEX") };
    assert_eq!(indices.len(), 2);
    assert!(
        matches!(indices[1].op(), Op::Ternary(svod_ir::TernaryOp::Where, ..)),
        "the generic rule only lifts the first index's gate; the image rule would lift both"
    );
}
#[test]
fn where_after_gated_load_becomes_load_alt() {
    let gate = bool_var("gate");
    let alt = UOp::native_const(7.0f32);
    let where_ =
        UOp::try_where(gate.clone(), load(valid_index(buffer(16), UOp::index_const(3), gate.clone())), alt.clone())
            .unwrap();
    let result = apply_gater(&where_);
    let Op::Load(ops::Load { alt: Some(result_alt), gate: Some(result_gate), .. }) = result.op() else {
        panic!("WHERE around a matching gated LOAD must become its alt")
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(Arc::ptr_eq(result_alt, &alt));
}
#[derive(Clone, Copy, Debug)]
enum InvalidFold {
    ScalarLoad,
    ScalarStore,
    ShapedLoad,
    GatedLoadAlt,
}
/// tinygrad folds a fully-invalid memory access away: the LOAD becomes its alt (zero when it has none) with the shape
/// preserved, and the STORE becomes a noop.
#[test_case(InvalidFold::ScalarLoad; "scalar load folds to zero")]
#[test_case(InvalidFold::ScalarStore; "scalar store folds to noop")]
#[test_case(InvalidFold::ShapedLoad; "shaped load folds lane by lane")]
#[test_case(InvalidFold::GatedLoadAlt; "gated load folds to its existing alt")]
fn symbolic_simple_folds_fully_invalid_memory_accesses(case: InvalidFold) {
    let buffer = buffer(16);
    let invalid_index = |lanes: Vec<Arc<UOp>>| {
        let lanes = if lanes.len() == 1 { lanes[0].clone() } else { UOp::stack(lanes.into()) };
        UOp::index().buffer(buffer.clone()).indices(vec![lanes]).call().expect("legal before late lowering")
    };
    let scalar = invalid_index(vec![UOp::invalid_marker()]);
    let result = match case {
        InvalidFold::ScalarLoad => {
            let result = fold(load(scalar));
            assert_const!(result, 0.0);
            result
        }
        InvalidFold::ScalarStore => {
            let result = fold(scalar.store(UOp::native_const(2.0f32)));
            assert!(matches!(result.op(), Op::Noop), "{}", result.tree());
            result
        }
        InvalidFold::ShapedLoad => {
            let shaped = invalid_index(vec![UOp::invalid_marker(), UOp::invalid_marker()]);
            let result = fold(load(shaped));
            assert_eq!(result.dtype(), DType::Float32);
            assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
            assert!(matches!(result.op(), Op::Stack(ops::Stack { sources })
                if sources.iter().all(|lane| matches!(lane.op(), Op::Const(v) if v.0 == ConstValue::Float(0.0)))));
            result
        }
        InvalidFold::GatedLoadAlt => {
            let shaped = invalid_index(vec![UOp::invalid_marker(), UOp::invalid_marker()]);
            let alt = float_values([7.0, 7.0]);
            let gated = UOp::load().index(shaped).alt(alt.clone()).gate(UOp::native_const(true)).call();
            let result = fold(gated);
            assert_same!(result, alt);
            result
        }
    };
    assert_no_invalid(&result);
}
#[derive(Clone, Copy, Debug)]
enum NotFolded {
    MixedLanes,
    SecondDimension,
    GatedStore,
}
#[test_case(NotFolded::MixedLanes; "partially invalid vector index survives for lane lowering")]
#[test_case(NotFolded::SecondDimension; "only the first validity index is folded")]
#[test_case(NotFolded::GatedStore; "the invalid fold excludes gated stores")]
fn invalid_memory_fold_is_not_overbroad(case: NotFolded) {
    let buffer = buffer(16);
    let root = match case {
        NotFolded::MixedLanes => {
            let lanes = UOp::stack(vec![UOp::invalid_marker(), UOp::index_const(1)].into());
            // A mixed vector index only exists transiently during expansion, so construct that IR directly rather
            // than through the INDEX validator.
            let mixed = UOp::new(
                Op::Index(ops::Index { buffer: buffer.broadcast(2), indices: vec![lanes].into() }),
                DType::Float32.vec(2).unwrap(),
            );
            load(mixed)
        }
        NotFolded::SecondDimension => {
            let index =
                UOp::index().buffer(buffer).indices(vec![UOp::index_const(0), UOp::invalid_marker()]).call().unwrap();
            load(index)
        }
        NotFolded::GatedStore => {
            let index = UOp::index().buffer(buffer).indices(vec![UOp::invalid_marker()]).call().unwrap();
            index.store_gated(UOp::native_const(2.0f32), UOp::native_const(true))
        }
    };
    let result = fold(root);
    match case {
        NotFolded::GatedStore => {
            assert!(matches!(result.op(), Op::Store(ops::Store { gate: Some(_), .. })), "{}", result.tree())
        }
        _ => assert!(matches!(result.op(), Op::Load(..)), "{}", result.tree()),
    }
}
/// Final decomposition removes data Invalid before rendering, and a gated memory access keeps its gate, alt and clean
/// address.
#[test]
fn final_rewrite_removes_data_invalid_and_keeps_gated_memory() {
    let gate = bool_var("gate");
    let invalid_data = UOp::try_where(gate.clone(), UOp::native_const(3.0f32), UOp::invalid_marker()).unwrap();
    let value = UOp::stack(
        vec![invalid_data, UOp::native_const(4.0f32), UOp::native_const(5.0f32), UOp::native_const(6.0f32)].into(),
    );
    let store = UOp::new(Op::Store(ops::Store { index: index(buffer(16), 0), value, gate: None }), DType::Void);
    assert_no_invalid(&apply_final_rewrite(store));
    let address = UOp::index_const(3);
    let gated = apply_gater(&load(valid_index(buffer(16), address.clone(), gate.clone())));
    assert_no_invalid(&gated);
    let result = apply_final_rewrite(gated);
    assert_no_invalid(&result);
    let Op::Load(ops::Load { index, alt: Some(_), gate: Some(result_gate) }) = result.op() else {
        panic!("final rewrite must preserve the gated LOAD")
    };
    assert!(Arc::ptr_eq(result_gate, &gate));
    assert!(matches!(index.op(), Op::Index(ops::Index { indices, .. }) if Arc::ptr_eq(&indices[0], &address)));
}
