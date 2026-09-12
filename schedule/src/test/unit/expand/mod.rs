use crate::devectorize::pm_expand_broadcast;
use crate::expand::{build_range_map, pm_group_for_reduce, pre_expand};
use crate::rewrite::graph_rewrite;
use crate::test::support::prelude::*;
use crate::test::unit::devectorize::helpers::{shaped_f32, wmma_metadata};
use smallvec::smallvec;
use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, SInt, UOp, WmmaUpcastAxes, ops};
use test_case::test_case;
/// A STACK of `count` lanes reshaped to `shape`.
fn shaped(count: usize, shape: &[usize]) -> Arc<UOp> {
    shaped_f32("lanes", count, shape)
}
fn upcast(end: i64, id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(id), AxisType::Upcast)
}
#[test]
fn upcast_and_unroll_ranges_become_shaped_coordinates() {
    let sink =
        UOp::sink(vec![upcast(2, 7), UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(9), AxisType::Unroll)]);
    let map = build_range_map(&sink);
    assert_eq!(map[&AxisId::Renumbered(7)], 0);
    assert_eq!(map[&AxisId::Renumbered(9)], 1);
    let result = pre_expand(&sink);
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert_eq!(sources[0].shape().unwrap().unwrap().as_slice(), &[SInt::Const(2), SInt::Const(1)]);
    assert_eq!(sources[1].shape().unwrap().unwrap().as_slice(), &[SInt::Const(1), SInt::Const(3)]);
}
#[test]
fn expansion_runs_movement_cleanup_in_the_same_fixpoint() {
    let result = pre_expand(&UOp::sink(vec![upcast(4, 7)]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert!(matches!(sources[0].op(), Op::Stack(..)), "{}", sources[0].tree());
    let buffer = UOp::param(0, 4, DType::Float32, None);
    let indexed = (0..4).map(|index| index_of(buffer.clone(), UOp::index_const(index))).collect();
    let result = pre_expand(&UOp::sink(vec![UOp::stack(indexed)]));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    assert_same!(sources[0], buffer);
}
fn wmma_upcast_axes(axes: WmmaUpcastAxes) -> Arc<UOp> {
    let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 2]);
    UOp::wmma(shaped(2, &[2, 1]), shaped(3, &[1, 3]), accumulator, wmma_metadata("test", Some(axes)))
}
/// Operand shapes are expanded independently, and the output coordinates are reconstructed from the upcast axes; a
/// nested split axis keeps its identity.
#[test_case(
    false,
    &[SInt::Const(2), SInt::Const(1)],
    &[SInt::Const(1), SInt::Const(2)],
    &[SInt::Const(1), SInt::Const(3)];
    "independent operand upcasts"
)]
#[test_case(
    true,
    &[SInt::Const(2), SInt::Const(3)],
    &[SInt::Const(3), SInt::Const(2)],
    &[SInt::Const(3), SInt::Const(2)];
    "nested split axis survives contract and unroll"
)]
fn wmma_shapes_operands_independently_and_reconstructs_output(
    nested: bool,
    output_shape: &[SInt],
    a_shape: &[SInt],
    b_shape: &[SInt],
) {
    let (ranges, wmma) = if nested {
        let nested_axis = AxisId::Renumbered(7).child(1).child(0);
        let ranges = vec![UOp::range_axis(UOp::index_const(2), nested_axis.clone(), AxisType::Upcast), upcast(3, 7)];
        let lanes = shaped(6, &[2, 3]);
        let axes = WmmaUpcastAxes {
            a: vec![(nested_axis.clone(), 2)],
            b: vec![(nested_axis.clone(), 2)],
            c: vec![(nested_axis, 2)],
        };
        let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 2]);
        (ranges, UOp::wmma(lanes.clone(), lanes, accumulator, wmma_metadata("nested-test", Some(axes))))
    } else {
        let axes = WmmaUpcastAxes {
            a: vec![(AxisId::Renumbered(7), 2)],
            b: vec![(AxisId::Renumbered(9), 3)],
            c: vec![(AxisId::Renumbered(7), 2)],
        };
        (vec![upcast(2, 7), upcast(3, 9)], wmma_upcast_axes(axes))
    };
    let mut sources = ranges;
    sources.push(wmma);
    let result = pre_expand(&UOp::sink(sources));
    let Op::Sink(ops::Sink { sources, .. }) = result.op() else { panic!("expected SINK") };
    let expanded = &sources[2];
    assert_eq!(expanded.shape().unwrap().unwrap().as_slice(), output_shape);
    let inner = first_op(expanded, |op| matches!(op, Op::Wmma(..))).expect("expanded WMMA");
    let Op::Wmma(ops::Wmma { a, b, metadata, .. }) = inner.op() else { unreachable!() };
    assert_eq!(a.shape().unwrap().unwrap().as_slice(), a_shape);
    assert_eq!(b.shape().unwrap().unwrap().as_slice(), b_shape);
    assert!(metadata.upcast_axes.is_none(), "expansion consumes the metadata");
}
#[test]
fn wmma_broadcast_stacks_fragments_before_reshape() {
    let wmma = UOp::wmma(
        shaped(64, &[4, 1, 16]),
        shaped(64, &[1, 4, 16]),
        shaped(8, &[8]),
        wmma_metadata("broadcast-test", None),
    );
    let result = graph_rewrite(pm_expand_broadcast(), wmma, &mut ());
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(4), SInt::Const(4), SInt::Const(8)]);
    let Op::Reshape(ops::Reshape { src, .. }) = result.op() else { panic!("expected STACK reshape") };
    assert!(matches!(src.op(), Op::Stack(ops::Stack { sources }) if sources.len() == 16));
}
#[test]
fn pre_expansion_wmma_accepts_scalar_inputs() {
    let accumulator = UOp::stack(smallvec![UOp::native_const(0.0f32); 8]);
    let wmma = UOp::wmma(
        UOp::native_const(1.0f32),
        UOp::native_const(2.0f32),
        accumulator,
        wmma_metadata("scalar-input-test", None),
    );
    assert_eq!(wmma.shape().unwrap().unwrap().as_slice(), &[SInt::Const(8)]);
}
#[test]
fn shaped_reduce_axes_are_expanded_before_reduction_lowering() {
    let unroll = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(2), AxisType::Unroll);
    let loop_range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(3), AxisType::Reduce);
    let reduce = unroll.cast(DType::Float32).reduce(smallvec![loop_range, unroll], ReduceOp::Add);
    let result = pre_expand(&reduce);
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Const(1)]);
    assert!(has_op(&result, |op| matches!(op, Op::Reduce(ops::Reduce { num_axes: 1, .. }))));
}
/// An expansion with no concrete extent or no matching coordinate is left alone rather than producing a malformed
/// shape.
#[derive(Clone, Copy, Debug)]
enum Decline {
    SymbolicRange,
    WmmaWithoutAxes,
    WmmaExtentMismatch,
    HorizontalReduce,
}
#[test_case(Decline::SymbolicRange; "symbolic upcast extent")]
#[test_case(Decline::WmmaWithoutAxes; "wmma without upcast axes")]
#[test_case(Decline::WmmaExtentMismatch; "wmma axis extent does not match the operand")]
#[test_case(Decline::HorizontalReduce; "reduce has no range to expand")]
fn incomplete_expansion_is_left_alone(case: Decline) {
    let wmma_operand = |axes: WmmaUpcastAxes| {
        UOp::wmma(
            shaped(2, &[2, 1]),
            shaped(3, &[1, 3]),
            UOp::stack(smallvec![UOp::native_const(0.0f32); 2]),
            wmma_metadata("decline", Some(axes)),
        )
    };
    let sink = match case {
        Decline::SymbolicRange => {
            let n = UOp::variable("n".to_string(), 0, 16, DType::WeakInt);
            UOp::sink(vec![UOp::range_axis(n, AxisId::Renumbered(7), AxisType::Upcast)])
        }
        Decline::WmmaWithoutAxes => UOp::sink(vec![UOp::wmma(
            shaped(2, &[2, 1]),
            shaped(3, &[1, 3]),
            shaped(2, &[2]),
            wmma_metadata("no-axes", None),
        )]),
        // The upcast ranges are what put axes 7 and 9 in the range map; without them `contract_axis` would
        // short-circuit on the missing coordinate instead of reaching the extent guard. Only `a` disagrees with
        // its operand (extent 4 over a 2-wide coordinate), so `contract_axis` is what declines.
        Decline::WmmaExtentMismatch => UOp::sink(vec![
            upcast(2, 7),
            upcast(3, 9),
            wmma_operand(WmmaUpcastAxes {
                a: vec![(AxisId::Renumbered(7), 4)],
                b: vec![(AxisId::Renumbered(9), 3)],
                c: vec![(AxisId::Renumbered(7), 2)],
            }),
        ]),
        Decline::HorizontalReduce => {
            let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(3), AxisType::Reduce);
            UOp::sink(vec![range.cast(DType::Float32).reduce(smallvec![range], ReduceOp::Add)])
        }
    };
    let result = pre_expand(&sink);
    let left_alone = match case {
        Decline::SymbolicRange => matches!(expect_sink(&result)[0].op(), Op::Range(..)),
        Decline::WmmaWithoutAxes => {
            has_op(&result, |op| matches!(op, Op::Wmma(ops::Wmma { metadata, .. }) if metadata.upcast_axes.is_none()))
        }
        Decline::WmmaExtentMismatch => {
            has_op(&result, |op| matches!(op, Op::Wmma(ops::Wmma { metadata, .. }) if metadata.upcast_axes.is_some()))
        }
        Decline::HorizontalReduce => has_op(&result, |op| matches!(op, Op::Reduce(..))),
    };
    assert!(left_alone, "the incomplete expansion must be left alone:\n{}", result.tree());
}
#[test]
fn grouped_reduce_loop_keeps_nested_axis_identity_and_range_dependencies() {
    let ordering = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(3), AxisType::Loop);
    let ended = UOp::new(Op::Noop, DType::Void).end(smallvec![ordering]);
    let after = UOp::index_const(1).after(smallvec![ended]);
    let grouped_axis = AxisId::Renumbered(7).child(1).child(0);
    let grouped = UOp::new(
        Op::Range(ops::Range {
            end: UOp::index_const(4),
            axis_id: grouped_axis.clone(),
            axis_type: AxisType::GroupReduce,
            deps: smallvec![after.clone()],
        }),
        DType::WeakInt,
    );
    let reduce = grouped.cast(DType::Float32).reduce(smallvec![grouped], ReduceOp::Add);
    let lowered = graph_rewrite(pm_group_for_reduce(), reduce, &mut ());
    assert!(has_op(
        &lowered,
        |op| matches!(op, Op::Stage(ops::Stage { opts, .. }) if opts.local_axis.as_ref() == Some(&grouped_axis))
    ));
    let loop_range = first_op(&lowered, |op| {
        matches!(op, Op::Range(ops::Range { axis_id, axis_type: AxisType::Reduce, .. })
        if axis_id == &grouped_axis.group_reduce_loop())
    })
    .expect("derived grouped-reduce loop");
    let Op::Range(ops::Range { deps, .. }) = loop_range.op() else { unreachable!() };
    assert_eq!(deps.len(), 1);
    assert!(Arc::ptr_eq(&deps[0], &after));
    assert_eq!(grouped_axis.group_reduce_loop().path(), &[7, 1, 0, 2]);
}
#[test]
fn grouped_reduce_loop_does_not_collide_with_offset_axis_or_range_map_parent() {
    let grouped_axis = AxisId::Renumbered(0);
    let grouped = UOp::range_axis(UOp::index_const(4), grouped_axis.clone(), AxisType::GroupReduce);
    let old_offset_collision = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(100), AxisType::Upcast);
    let split_outer = UOp::range_axis(UOp::index_const(4), grouped_axis.child(0), AxisType::Upcast);
    let split_inner = UOp::range_axis(UOp::index_const(4), grouped_axis.child(1), AxisType::Upcast);
    let derived = UOp::range_axis(UOp::index_const(4), grouped_axis.group_reduce_loop(), AxisType::Upcast);
    let map = build_range_map(&UOp::sink(vec![old_offset_collision, split_outer, split_inner, derived]));
    assert_eq!(map.len(), 4);
    assert!(map.contains_key(&AxisId::Renumbered(100)));
    assert!(map.contains_key(&grouped_axis.child(0)));
    assert!(map.contains_key(&grouped_axis.child(1)));
    assert!(map.contains_key(&grouped_axis.group_reduce_loop()));
    let reduce = grouped.cast(DType::Float32).reduce(smallvec![grouped], ReduceOp::Add);
    let lowered = graph_rewrite(pm_group_for_reduce(), reduce, &mut ());
    assert!(has_op(&lowered, |op| matches!(op, Op::Range(ops::Range { axis_id, axis_type: AxisType::Reduce, .. })
        if axis_id == &grouped_axis.group_reduce_loop())));
    assert!(!has_op(&lowered, |op| matches!(
        op,
        Op::Range(ops::Range { axis_id: AxisId::Renumbered(100), axis_type: AxisType::Reduce, .. })
    )));
}
