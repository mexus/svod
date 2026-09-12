pub mod heuristics;
pub mod implicit_barriers;
pub mod kernels;
pub mod mod_internal;
pub mod opts_to_apply;
pub mod opts_validation;
pub mod scheduler;
pub mod tc;

#[cfg(test)]
mod pipeline_composition {
    use crate::linearize::pm_split_ends;
    use crate::optimizer::apply_pre_optimization;
    use crate::rewrite::graph_rewrite;
    use crate::test::support::prelude::*;
    use smallvec::smallvec;
    use std::sync::Arc;
    use svod_ir::AxisType::Loop;
    use svod_ir::{AxisId, CanonicalGraph, ConstValue, DType, Op, ReduceOp, UOp, ops};
    use test_case::test_case;
    fn postopt_symbolic(root: Arc<UOp>) -> Arc<UOp> {
        graph_rewrite(&*crate::optimizer::POST_OPT_SYM, root, &mut ())
    }
    /// A `Loop` RANGE with a constant extent, as `rangeify` builds them.
    fn loop_range(extent: i64, id: usize) -> Arc<UOp> {
        UOp::range_axis(UOp::index_const(extent), AxisId::Renumbered(id), Loop)
    }
    #[test]
    fn test_postopt_symbolic_removes_range_unparented_after_zero_fold() {
        let range = reduce_range(7, 0);
        let src = range.cast(DType::Int32).mul(&UOp::native_const(0i32));
        let reduce = reduce(src, vec![range], ReduceOp::Add);
        let result = postopt_symbolic(reduce);
        assert!(!has_op(&result, |op| matches!(op, Op::Reduce(..))));
        assert_const!(result, 0);
    }
    #[test]
    fn test_postopt_symbolic_keeps_parented_reduction() {
        let range = reduce_range(7, 0);
        let reduce = reduce(range.cast(DType::Int32), vec![range], ReduceOp::Add);
        let result = postopt_symbolic(reduce);
        let ranges = unwrap_op!(result, Op::Reduce(ops::Reduce { ranges, .. }) => ranges);
        assert_eq!(ranges.len(), 1);
    }
    /// A condition on the load address itself cannot move into the index; one
    /// that does not depend on it moves onto the address's validity.
    #[test_case(true, true; "an index-dependent condition stays a WHERE")]
    #[test_case(false, false; "an index-independent condition moves onto the index")]
    fn test_postopt_where_load_condition(index_dependent: bool, expect_ternary: bool) {
        let address = index(param(0, 8, DType::Float32), 0);
        let condition = if index_dependent { index(param(1, 8, DType::Bool), 0) } else { param(1, 1, DType::Bool) };
        let masked = UOp::try_where(condition.clone(), address, UOp::native_const(0.0f32)).expect("WHERE");
        let result = postopt_symbolic(masked);
        assert_eq!(matches!(result.op(), Op::Ternary(..)), expect_ternary, "{}", result.tree());
        if !expect_ternary {
            let (_, indices) = expect_index(&result);
            let validity = indices[0].get_valid();
            assert!(
                validity.any_in_subtree(|node| node.id == condition.id),
                "the moved condition must be the index validity\n{}",
                result.tree()
            );
        }
    }
    #[test]
    fn test_pm_split_ends_reattaches_bool_and_void_backedges_outermost() {
        let computation = UOp::native_const(1.0f32);
        let outer_range = loop_range(4, 0);
        let inner_range = loop_range(8, 1);
        let bool_backedge = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let void_backedge = UOp::noop();
        let original = computation
            .end(smallvec![bool_backedge.clone(), outer_range.clone(), void_backedge.clone(), inner_range.clone()])
            .with_tag(smallvec![17, 23]);
        let result = graph_rewrite(pm_split_ends(), original, &mut ());
        assert_eq!(result.tag().as_deref(), Some(&[17, 23][..]));
        let (range_ends, backedges) = expect_end(&result);
        assert_eq!(backedges.len(), 2);
        assert_same!(backedges[0], bool_backedge);
        assert_same!(backedges[1], void_backedge);
        assert_eq!(range_ends.tag(), &None);
        let (inner_end, outer_ranges) = expect_end(&range_ends);
        assert_eq!(outer_ranges.len(), 1);
        assert_same!(outer_ranges[0], outer_range);
        let (leaf, inner_ranges) = expect_end(&inner_end);
        assert_same!(leaf, computation);
        assert_eq!(inner_ranges.len(), 1);
        assert_same!(inner_ranges[0], inner_range);
    }
    #[test]
    fn test_pm_split_ends_empty_target_ranges_preserves_backedges_and_identity() {
        let computation = UOp::native_const(2.0f32);
        let original = computation
            .end(smallvec![UOp::const_(DType::Bool, ConstValue::Bool(false)), UOp::noop()])
            .with_tag(smallvec![99]);
        let result = graph_rewrite(pm_split_ends(), original.clone(), &mut ());
        assert_same!(result, original);
        assert_eq!(result.tag().as_deref(), Some(&[99][..]));
    }
    #[test]
    fn test_pm_split_ends_sorts_nested_axis_ids_and_preserves_range_dependencies() {
        let computation = UOp::native_const(3.0f32);
        let dependency = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let parent = AxisId::Renumbered(2);
        let parent_range = UOp::range_axis(UOp::index_const(2), parent.clone(), Loop);
        let child_zero = UOp::range_axis(UOp::index_const(3), parent.child(0), Loop);
        let child_one = UOp::range_axis(UOp::index_const(5), parent.child(1), Loop)
            .with_sources(vec![UOp::index_const(5), dependency.clone()]);
        let original = computation.end(smallvec![parent_range.clone(), child_zero.clone(), child_one.clone()]);
        let result = graph_rewrite(pm_split_ends(), original, &mut ());
        let mut cursor = result.clone();
        for expected in [&parent_range, &child_zero, &child_one] {
            let (next, ranges) = expect_end(&cursor);
            assert_eq!(ranges.len(), 1);
            assert!(Arc::ptr_eq(&ranges[0], expected));
            cursor = next;
        }
        assert!(Arc::ptr_eq(&cursor, &computation));
        let deps = unwrap_op!(child_one, Op::Range(ops::Range { deps, .. }) => deps);
        assert_eq!(deps.len(), 1);
        assert_same!(deps[0], dependency);
        let expected = computation.end(smallvec![child_one]).end(smallvec![child_zero]).end(smallvec![parent_range]);
        assert_same!(result, expected);
        assert_eq!(
            CanonicalGraph::from_root("split_end", &result).unwrap(),
            CanonicalGraph::from_root("split_end", &expected).unwrap()
        );
    }
    #[test]
    fn test_preopt_split_exposes_range_flattening_in_same_rewrite() {
        let range = loop_range(12, 0);
        let sink = UOp::sink(vec![range.mod_(&UOp::index_const(4)).end(smallvec![range])]);
        let result = apply_pre_optimization(sink).expect("pre-optimization");
        let (_, ranges) = expect_end(&expect_sink(&result)[0]);
        assert_eq!(ranges.len(), 2, "split RANGE dependencies must be flattened into the END");
        assert!(ranges.iter().all(|range| matches!(range.op(), Op::Range(..))));
    }
    #[test]
    fn test_preopt_cast_const_fold_enables_range_end_arithmetic() {
        let cast = UOp::const_(DType::Int32, ConstValue::Int(3)).cast(DType::Index);
        let end = cast.add(&UOp::const_(DType::Index, ConstValue::Int(1)));
        let range = UOp::range_axis(end, AxisId::Renumbered(0), Loop);
        let sink = UOp::sink(vec![range.clone().end(smallvec![range])]);
        let result = apply_pre_optimization(sink).expect("pre-optimization");
        let range = first_op(&result, |op| matches!(op, Op::Range(..))).expect("the range survives pre-optimization");
        let (end, _, _) = expect_range(&range);
        assert_const!(end, 4);
    }
}
