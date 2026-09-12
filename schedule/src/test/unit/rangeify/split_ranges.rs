//! `pm_split_ranges`: a `r % k` range expression becomes an outer/inner range
//! pair, with the axis-id path carried onto both children.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, BinaryOp, Op, ParamArg, UOp};
use test_case::test_case;

use crate::rangeify::{SplitRangesContext, pm_flatten_range, pm_split_ranges};
use crate::test::support::prelude::*;

fn split(sink: Arc<UOp>) -> Arc<UOp> {
    rewrite_with(&pm_split_ranges(), &mut SplitRangesContext::default(), sink)
}

/// `SINK(END(r % k, r))` over an 8-extent axis of `axis_type`.
fn modulo_sink(axis_type: AxisType, divisor: i64, id: usize) -> Arc<UOp> {
    let r = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(id), axis_type);
    UOp::sink(vec![r.mod_(&r.const_like(divisor)).end(smallvec::smallvec![r])])
}

/// The axis ids of every RANGE in the rewritten graph, sorted.
fn range_ids(root: &Arc<UOp>) -> Vec<AxisId> {
    let mut ids: Vec<_> = root.ranges().into_iter().map(|r| range_axis_id(&r)).collect();
    ids.sort();
    ids
}

/// `r % k` splits `r` into an outer and an inner range of the same axis type.
/// WARP and DEVICE are launch lanes with fixed extents and are never split.
#[test_case(AxisType::Global, true ; "global")]
#[test_case(AxisType::Local, true ; "local")]
#[test_case(AxisType::Weak, true ; "weak")]
#[test_case(AxisType::Loop, true ; "loop axis")]
#[test_case(AxisType::Reduce, true ; "reduce")]
#[test_case(AxisType::GroupReduce, true ; "group reduce")]
#[test_case(AxisType::Upcast, true ; "upcast")]
#[test_case(AxisType::Warp, false ; "warp")]
#[test_case(AxisType::Device, false ; "device")]
fn modulo_splits_every_axis_type_but_the_launch_lanes(axis_type: AxisType, splits: bool) {
    let sink = modulo_sink(axis_type, 2, 0);
    let result = split(sink.clone());

    if !splits {
        assert!(Arc::ptr_eq(&result, &sink), "a launch lane must not split");
        return;
    }
    assert_eq!(range_ids(&result), vec![AxisId::Renumbered(0).child(0), AxisId::Renumbered(0).child(1)]);
    assert!(
        result.ranges().iter().all(|r| range_axis_type(r) == axis_type),
        "the split children inherit the axis type"
    );
}

/// A graph the pass cannot split is handed back untouched: a symbolic extent has
/// no divisibility proof, a zero divisor is rejected by the checked constructor
/// upstream, and a non-divisor leaves the expression alone.
#[test]
fn unsplittable_inputs_are_untouched() {
    let r = range_symbolic(UOp::define_var("n".to_string(), 0, 1024), 0);
    let symbolic = UOp::sink(vec![r.mod_(&UOp::index_const(2)).end(smallvec::smallvec![r])]);
    assert_same!(split(symbolic.clone()), symbolic);

    for divisor in [0, 3] {
        let r = range(8, AxisType::Loop, 0);
        let modulo = UOp::new(Op::Binary(BinaryOp::FloorMod, r, UOp::index_const(divisor)), DType::Index);
        let sink = UOp::sink(vec![modulo]);

        assert!(Arc::ptr_eq(&split(sink.clone()), &sink), "divisor {divisor} must not split");
    }
}

/// `mark_range_mod` keeps the first divisor it sees for a range: the second
/// modulo on the same axis does not create a second split.
#[test]
fn the_first_modulo_on_an_axis_wins() {
    let r = range(12, AxisType::Loop, 0);
    let sink = UOp::sink(vec![
        r.mod_(&UOp::index_const(4)).end(smallvec::smallvec![r.clone()]),
        r.mod_(&UOp::index_const(3)).end(smallvec::smallvec![r]),
    ]);

    assert_eq!(range_ids(&split(sink)).len(), 2, "one split, not two");
}

/// An image INDEX pins every range it addresses: `dont_split_ranges_for_image`
/// records the `None` marker, so image coordinates survive range splitting.
#[test]
fn an_image_dtype_index_is_never_split() {
    let r = range(8, AxisType::Loop, 0);
    let image = DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![4, 2, 4] };
    let address = UOp::new(
        Op::Index(svod_ir::ops::Index {
            buffer: UOp::new(Op::Noop, image.clone()),
            indices: smallvec::smallvec![r.clone()],
        }),
        image,
    );
    let sink = UOp::sink(vec![
        r.mod_(&UOp::index_const(2)).end(smallvec::smallvec![r]),
        address.store(UOp::native_const(1.0f32)),
    ]);

    assert!(Arc::ptr_eq(&split(sink.clone()), &sink), "image coordinates must survive range splitting");
}

/// The substituted parent `(outer * k + inner) % k` simplifies to the inner range
/// during the preopt composition with `pm_flatten_range`, and a negative divisor
/// folds the whole expression to 0 (tinygrad's `%` is a floor-mod, and `r` is in
/// `[0, 8)` here), leaving one substituted range.
#[test]
fn split_simplifies_the_substituted_parent_for_every_divisor() {
    let r = range(8, AxisType::Loop, 0);
    let matcher = pm_split_ranges() + pm_flatten_range().with_context::<SplitRangesContext>();

    let result = rewrite_with(
        &matcher,
        &mut SplitRangesContext::default(),
        UOp::sink(vec![r.mod_(&UOp::index_const(2)).end(smallvec::smallvec![r])]),
    );

    let source = expect_sink(&result)[0].clone();
    let (computation, _) = expect_end(&source);
    assert_eq!(expect_range_extent(&computation), 2, "the substituted parent must simplify: {}", result.tree());
    assert_eq!(result.ranges().len(), 2, "the END dependency must retain both split ranges");

    let r = range(8, AxisType::Loop, 7);
    let sink = UOp::sink(vec![r.mod_(&UOp::index_const(-2)).end(smallvec::smallvec![r])]);
    let source = expect_sink(&split(sink))[0].clone();
    let (computation, ranges) = expect_end(&source);
    assert_const!(computation, 0);
    assert_eq!(ranges.len(), 1);
    assert_const!(ranges[0], 0);
}

/// An image STORE still splits — the pin only protects image *index* access — and
/// the split coordinates keep their outer/inner order.
#[test]
fn image_store_modulo_split_preserves_structural_coordinates() {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 1usize.into(), 4usize.into()]);
    let image = UOp::new(
        Op::Param(svod_ir::ops::Param {
            shape,
            arg: ParamArg::buffer(0, DType::Float32, AddrSpace::Global, None).into(),
        }),
        DType::Float32,
    );
    let r = range(8, AxisType::Loop, 0);
    let four = UOp::index_const(4);
    let index =
        UOp::index().buffer(image).indices(vec![r.floor_div(&four), r.mod_(&four)]).call().expect("image INDEX");
    let sink = UOp::sink(vec![index.store(UOp::const_(DType::Float32, 1.0.into()))]);

    let result = split(sink);

    assert_eq!(range_ids(&result).len(), 2, "image stores must not suppress the split: {}", result.tree());
    let ranges = result.ranges();
    let outer = ranges.iter().find(|r| expect_range_extent(r) == 2).expect("outer range").clone();
    let inner = ranges.iter().find(|r| expect_range_extent(r) == 4).expect("inner range").clone();
    let store = first_op(&result, |op| matches!(op, Op::Store(..))).expect("image store");
    let (_, indices) = expect_index(&expect_store(&store).0);
    assert_eq!(indices.len(), 2);
    assert!(Arc::ptr_eq(&indices[0], &outer), "image y coordinate");
    assert!(Arc::ptr_eq(&indices[1], &inner), "image x coordinate");
}

/// Repeated splits append children to each original axis, and a nested split
/// appends to the axis path that is already there.
#[test]
fn splits_append_to_each_original_axis_path() {
    let (r0, r1) = (range(12, AxisType::Loop, 3), range(10, AxisType::Reduce, 7));
    let sink = UOp::sink(vec![
        r0.mod_(&UOp::index_const(4)).end(smallvec::smallvec![r0]),
        r1.mod_(&UOp::index_const(5)).end(smallvec::smallvec![r1]),
    ]);
    assert_eq!(
        range_ids(&split(sink)),
        vec![
            AxisId::Renumbered(3).child(0),
            AxisId::Renumbered(3).child(1),
            AxisId::Renumbered(7).child(0),
            AxisId::Renumbered(7).child(1),
        ]
    );

    let parent = AxisId::Renumbered(5).child(1);
    let r = UOp::range_axis_dtype(UOp::index_const(12), parent.clone(), AxisType::Upcast, DType::WeakInt);
    let sink = UOp::sink(vec![r.mod_(&UOp::index_const(3)).end(smallvec::smallvec![r])]);
    assert_eq!(range_ids(&split(sink)), vec![parent.child(0), parent.child(1)]);
}
