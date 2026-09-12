//! Regressions and corner cases of the devectorize pass.
use super::helpers::*;
use proptest::prelude::*;
use std::sync::Arc;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{AxisType, Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;
/// Register reads sharing a range collapse into one END; an unrelated END over the same range is left alone.
#[test]
fn register_reads_merge_only_their_shared_range_ends() {
    let range = UOp::range_const(4, 0);
    let make_register_end = |slot, value| {
        let buffer = UOp::buffer(slot, 1, DType::Int32, AddrSpace::Reg, None);
        let index = UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();
        let ended = store(index, UOp::native_const(value)).end(smallvec::smallvec![range.clone()]);
        let read = UOp::index()
            .buffer(buffer.after(smallvec::smallvec![ended.clone()]))
            .indices(vec![UOp::index_const(0)])
            .call()
            .unwrap();
        (load(read), ended)
    };
    let (left, left_end) = make_register_end(0, 1i32);
    let (right, right_end) = make_register_end(1, 2i32);
    let unrelated = store(index(buffer_of(1, ScalarDType::Int32), 0), UOp::native_const(3i32))
        .end(smallvec::smallvec![range.clone()]);
    let result = crate::devectorize::merge_register_read_ends(UOp::sink(vec![left, right, unrelated.clone()]));
    let matching: Vec<_> = result
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::End(ops::End { ranges, .. }) if ranges.len() == 1 && Arc::ptr_eq(&ranges[0], &range)))
        .collect();
    assert_eq!(matching.len(), 2);
    assert!(matching.iter().any(|node| Arc::ptr_eq(node, &unrelated)));
    assert!(matching.iter().any(|node| matches!(node.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Group(ops::Group { sources }) if sources.len() == 2))));
    assert!(!result.toposort().iter().any(|node| Arc::ptr_eq(node, &left_end) || Arc::ptr_eq(node, &right_end)));
}
/// `index_axes` records the selected positions verbatim and in order: the shaped INDEX the devectorizer then splits
/// lane by lane, so a reorder or a dedup would permute every upcast access.
#[test]
fn shaped_index_keeps_its_selected_positions() {
    let indexed = float_values((0..8).map(|value| value as f64)).index_axes(vec![1, 3, 5, 7]);
    let (_, indices) = expect_index(&indexed);
    let sources = unwrap_op!(indices[0], Op::Stack(ops::Stack { sources }) => sources);
    let positions: Vec<i64> = sources
        .iter()
        .map(|source| unwrap_op!(source, Op::Const(value) => value).0.try_int().expect("integer position"))
        .collect();
    assert_eq!(positions, vec![1, 3, 5, 7], "{}", indexed.tree());
}
/// A shape is chunkable exactly when it has no zero dimension and its dims multiply to the element count.
#[test_case(4, &[2, 2], true; "square")]
#[test_case(6, &[1, 6], true; "singleton leading dim")]
#[test_case(1, &[], true; "scalar shape holds one element")]
#[test_case(0, &[4, 0], false; "trailing zero dim")]
#[test_case(4, &[4, 0], false; "zero dim with elements")]
#[test_case(3, &[2, 2], false; "not divisible")]
fn stack_with_shape_accepts_iff_the_product_matches(count: usize, dims: &[usize], expected: bool) {
    let elements: Vec<Arc<UOp>> = (0..count).map(|i| UOp::native_const(i as i32)).collect();
    let shape: Vec<SInt> = dims.iter().copied().map(SInt::Const).collect();
    assert_eq!(crate::devectorize::stack_with_shape(elements, &shape).is_some(), expected);
}
proptest! {
    #![proptest_config(cheap())]
    /// The same rule over generated shapes. `count` is drawn around the product rather than independently, so both
    /// branches are reached instead of only the rejecting one.
    #[test]
    fn stack_with_shape_agrees_with_the_product_rule(
        dims in prop::collection::vec(0usize..5, 0..4),
        offset in -2i64..3,
    ) {
        let count = (dims.iter().product::<usize>() as i64 + offset).max(0) as usize;
        let elements: Vec<Arc<UOp>> = (0..count).map(|i| UOp::native_const(i as i32)).collect();
        let shape: Vec<SInt> = dims.iter().copied().map(SInt::Const).collect();
        let expected = dims.iter().all(|&dim| dim > 0) && dims.iter().product::<usize>() == count;
        prop_assert_eq!(crate::devectorize::stack_with_shape(elements, &shape).is_some(), expected, "{:?} over {} elements", dims, count);
    }
}
/// Two reductions that share their reduce ranges merge into one `END(GROUP(..))`, and the pass runs
/// `clean_up_group_sink` over the rebuilt sink.
#[test]
fn reductions_sharing_ranges_merge_into_one_group_end() {
    let shared = reduce_range(16, 0);
    let reduce_with = |value: f32| reduce(UOp::native_const(value), vec![shared.clone()], ReduceOp::Add);
    let result = apply_pm_reduce(&UOp::sink(vec![reduce_with(1.0), reduce_with(2.0), shared.clone()]));
    let merged = result
        .toposort()
        .into_iter()
        .find(|node| matches!(node.op(), Op::End(ops::End { ranges, .. }) if ranges.len() == 1 && Arc::ptr_eq(&ranges[0], &shared)))
        .expect("the shared reduce range must still be closed");
    let Op::End(ops::End { computation, .. }) = merged.op() else { unreachable!() };
    assert!(matches!(computation.op(), Op::Group(ops::Group { sources }) if sources.len() == 2), "{}", result.tree());
    assert_eq!(count(&result, |node| matches!(node.op(), Op::End(..))), 1, "one merged END per reduce-range set");
}
/// Equal reduce ranges at different nesting depths are kept apart: the inner context gets cloned RANGEs, so every
/// RANGE is closed by exactly one END.
#[test]
fn reductions_in_different_contexts_get_their_own_ranges() {
    let shared = reduce_range(4, 0);
    let enclosing = range(8, AxisType::Loop, 1);
    let nested_source = enclosing.cast(DType::Float32).add(&UOp::native_const(2.0f32));
    let top = reduce(UOp::native_const(1.0f32), vec![shared.clone()], ReduceOp::Add);
    let nested = reduce(nested_source, vec![shared.clone()], ReduceOp::Add);
    let result = apply_pm_reduce(&UOp::sink(vec![top, nested, shared.clone(), enclosing]));
    let closed: Vec<Arc<UOp>> = result
        .toposort()
        .into_iter()
        .filter_map(|node| match node.op() {
            Op::End(ops::End { ranges, .. }) => Some(ranges[0].clone()),
            _ => None,
        })
        .collect();
    assert_eq!(closed.len(), 2, "{}", result.tree());
    assert!(!Arc::ptr_eq(&closed[0], &closed[1]), "each context owns its RANGE node");
    assert!(closed.iter().all(|range| expect_range_extent(range) == 4));
    assert!(
        closed.iter().any(|range| range_axis_id(range) == range_axis_id(&shared)),
        "the first context keeps the original axis"
    );
}
/// `clean_up_group_sink` drops structural wrappers the accumulator rewrite leaves.
#[test_case(UOp::sink(vec![UOp::noop(), UOp::native_const(1.0f32)]); "sink drops a noop source")]
#[test_case(UOp::group(vec![UOp::native_const(1.0f32)]); "single-source group collapses")]
fn clean_up_group_sink_removes_structural_wrappers(root: Arc<UOp>) {
    let result = apply_pm_reduce(&root);
    match result.op() {
        Op::Const(..) => assert_const!(result, 1.0),
        Op::Sink(ops::Sink { sources, .. }) => {
            assert_eq!(sources.len(), 1, "{}", result.tree());
            assert_const!(sources[0], 1.0);
        }
        other => panic!("unexpected {other:?}\n{}", result.tree()),
    }
}
