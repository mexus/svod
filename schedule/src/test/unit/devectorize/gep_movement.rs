//! Target shaped INDEX and movement cleanup tests for `devectorizer2`.
use super::helpers::*;
use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::{BinaryOp, ConstValue, Op, UOp, ops};
use test_case::test_case;
/// `INDEX(STACK, [lane])` built directly: the pass is what folds it, not the constructor.
fn select(buffer: Arc<UOp>, lane: i64) -> Arc<UOp> {
    UOp::new(Op::Index(ops::Index { buffer, indices: smallvec::smallvec![UOp::index_const(lane)] }), DType::Float32)
}
#[test]
fn stacked_index_becomes_stack_of_scalar_indices() {
    let result = apply_devectorize_patterns(shaped_addr(&buffer(8), [1, 3]));
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected STACK: {}", result.tree()) };
    assert_eq!(sources.len(), 2);
    assert!(sources.iter().all(|source| source.dtype() == DType::Float32));
    assert!(
        sources.iter().all(|source| matches!(source.op(), Op::Index(ops::Index { indices, .. }) if indices.len() == 1))
    );
}
#[test]
fn reshaped_index_is_fully_consumed_by_shaped_indexing() {
    let indices = reshape_to(&stack((0i64..2).map(index_const)), &[1, 2]);
    let shaped = UOp::index().buffer(buffer_to_define(&buffer(8))).indices(vec![indices]).call().unwrap();
    let result = apply_devectorize_patterns(
        UOp::index().buffer(shaped).indices(vec![UOp::index_const(0), UOp::index_const(1)]).call().unwrap(),
    );
    assert!(matches!(result.op(), Op::Index(ops::Index { indices, .. }) if indices.len() == 1));
    assert!(!result.toposort().iter().any(|node| node.op().is_movement()));
}
#[test]
fn scalar_expand_becomes_stack() {
    let expand = UOp::new(
        Op::Expand(ops::Expand {
            src: UOp::native_const(1.0f32),
            new_shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![4usize.into()]),
        }),
        DType::Float32,
    );
    let result = apply_devectorize_patterns(expand);
    assert!(matches!(result.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4));
    assert_eq!(result.dtype(), DType::Float32);
}
#[test]
fn singleton_reshape_to_scalar_becomes_index() {
    let result = apply_devectorize_patterns(reshape_to(&float_values([2.0]), &[]));
    assert!(matches!(result.op(), Op::Const(_)), "constant INDEX into STACK should clean up: {}", result.tree());
}
#[test]
fn void_reshape_is_removed() {
    let store = UOp::noop().with_dtype(DType::Void);
    let reshape = UOp::new(
        Op::Reshape(ops::Reshape {
            src: store.clone(),
            new_shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![]),
        }),
        DType::Void,
    );
    let result = apply_devectorize_patterns(reshape);
    assert_same!(result, store);
}
#[test]
fn scalar_index_into_stack_folds_without_reconstructing_storage() {
    let values = float_values([1.0, 2.0]);
    let rebuilt = UOp::stack(smallvec::smallvec![select(values.clone(), 0), select(values.clone(), 1)]);
    let folded = apply_devectorize_patterns(rebuilt);
    assert_op!(folded, Op::Stack(..));
    let selected = apply_devectorize_patterns(select(values, 1));
    assert_const!(selected, 2.0);
}
#[derive(Clone, Copy, Debug)]
enum NestedMovement {
    Reshape,
    Permute,
}
/// Adjacent movement ops are cleaned in upstream order, never left nested.
#[test_case(NestedMovement::Reshape; "nested reshape")]
#[test_case(NestedMovement::Permute; "nested permute")]
fn adjacent_movement_ops_are_cleaned_in_upstream_order(movement: NestedMovement) {
    let values = float_values([1.0, 2.0]);
    let nested = match movement {
        NestedMovement::Reshape => reshape_to(&reshape_to(&values, &[1, 2]), &[2]),
        NestedMovement::Permute => UOp::new(
            Op::Permute(ops::Permute {
                src: UOp::new(Op::Permute(ops::Permute { src: values, axes: vec![0] }), DType::Float32),
                axes: vec![0],
            }),
            DType::Float32,
        ),
    };
    let result = apply_devectorize_patterns(nested);
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
    assert!(!result.toposort().iter().any(|node| node.op().is_movement()));
}
#[test]
fn child_singleton_reshape_is_visible_to_parent_expand() {
    let values = float_values([1.0, 2.0]);
    let expanded = reshape_to(&values, &[1, 2]).try_expand(&smallvec::smallvec![3usize.into(), 2usize.into()]).unwrap();
    let result = apply_devectorize_patterns(expanded);
    let Op::Stack(ops::Stack { sources }) = result.op() else { panic!("expected outer STACK: {}", result.tree()) };
    assert_eq!(sources.len(), 3);
    assert!(sources.iter().all(|source| Arc::ptr_eq(source, &values)));
    assert!(!result.toposort().iter().any(|node| matches!(node.op(), Op::Reshape(..) | Op::Expand(..))));
}
#[test]
fn child_stack_broadcast_is_visible_to_mixed_alu_parent() {
    let broadcast = stack([UOp::native_const(2.0f32)]).try_expand(&smallvec::smallvec![4usize.into()]).unwrap();
    let vector = UOp::vconst(vec![ConstValue::Float(1.0); 4], DType::Float32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, broadcast, vector), DType::Float32.vec(4).unwrap());
    let result = apply_devectorize_patterns(add);
    assert!(
        matches!(result.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 4),
        "expected scalar STACK ALU: {}",
        result.tree()
    );
    assert!(!result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Binary(..) | Op::Ternary(..))
            && node.op().sources().iter().any(|source| matches!(source.op(), Op::Stack(..)))
            && (node.dtype().vcount() > 1 || node.op().sources().iter().any(|source| source.dtype().vcount() > 1))
    }));
}
