//! White-box tests over `crate::optimizer` internals: local-buffer staging and
//! the index-lowering stage matchers. They reach private helpers, so they live
//! beside the rest of the optimizer unit tests rather than in the source file.

mod stage_local_tests {
    use crate::optimizer::{LocalBufferContext, add_local_buffer};
    use crate::test::support::prelude::*;
    use std::sync::Arc;
    use svod_dtype::{AddrSpace, DType};
    use svod_ir::{AxisId, BufferizeOpts, Op, UOp, ops};
    /// The slot of the buffer an `add_local_buffer` result stores into.
    #[track_caller]
    fn buffer_slot(u: &Arc<UOp>) -> usize {
        match u.buf_uop().op() {
            Op::Buffer(ops::Buffer { arg, .. }) => arg.slot,
            other => panic!("expected BUFFER, got {other:?}\n{}", u.tree()),
        }
    }
    fn local_stage(compute: Arc<UOp>, axis: Option<AxisId>) -> Arc<UOp> {
        let opts = match axis {
            Some(axis) => BufferizeOpts::local_for_axis(axis),
            None => BufferizeOpts::local(),
        };
        stage_with(compute, vec![], opts)
    }
    /// Lower each stage against one shared context and return the assigned slots.
    fn slots(stages: impl IntoIterator<Item = Arc<UOp>>) -> Vec<usize> {
        let mut ctx = LocalBufferContext::default();
        stages.into_iter().map(|stage| buffer_slot(&add_local_buffer(&stage, &mut ctx).expect("lower"))).collect()
    }
    #[test]
    fn add_local_buffer_matches_stage_mapping_and_numbering() {
        let (r0, r1) = (global_range(2, 0), global_range(3, 1));
        let compute = UOp::native_const(7.0f32);
        let stage = stage_with(compute.clone(), vec![r0.clone(), r1.clone()], BufferizeOpts::local());
        let dims: Vec<_> = stage.shape().unwrap().unwrap().iter().map(|dim| dim.as_const()).collect();
        assert_eq!(dims, [Some(2), Some(3)]);
        let mut ctx = LocalBufferContext::default();
        let lowered = add_local_buffer(&stage, &mut ctx).expect("a local STAGE lowers");
        assert_eq!(lowered.dtype(), DType::Float32);
        let (passthrough, deps) = expect_after(&lowered);
        let storage = passthrough.base();
        let arg = unwrap_op!(storage, Op::Buffer(ops::Buffer { arg, .. }) => arg);
        assert_eq!((arg.slot, arg.dtype.clone(), arg.addrspace), (0, DType::Float32, Some(AddrSpace::Local)));
        assert_eq!(deps.len(), 1, "the AFTER carries one END dependency");
        let (computation, ranges) = expect_end(&deps[0]);
        assert!(ranges.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));
        let (index, value, gate) = expect_store(&computation);
        assert_same!(value, compute);
        assert!(gate.is_none());
        let (buffer, indices) = expect_index(&index);
        assert_same!(buffer, passthrough);
        assert!(indices.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));
        let second = add_local_buffer(&local_stage(UOp::native_const(8.0f32), None), &mut ctx).expect("a second STAGE");
        assert_eq!(buffer_slot(&second), 1, "an axisless STAGE takes the next fallback slot");
    }
    #[test]
    fn grouped_local_axis_drives_slot_without_colliding_with_nested_axes() {
        let scalar_axis = AxisId::Renumbered(7);
        let scalar = local_stage(UOp::native_const(1.0f32), Some(scalar_axis.clone()));
        let nested = local_stage(UOp::native_const(2.0f32), Some(scalar_axis.child(0)));
        let found = slots([scalar, nested]);
        assert_eq!(found[0], 7, "a scalar axis keeps its numeric slot");
        assert_ne!(found[1], found[0], "a nested axis must not collide with its scalar parent");
        assert!(found[1] >= 1 << (usize::BITS - 1), "nested slots live in the reserved high namespace");
    }
    #[test]
    fn grouped_local_slots_repeat_across_kernel_rewrites() {
        let lower = || {
            [AxisId::Renumbered(3), AxisId::Renumbered(3).child(1)]
                .into_iter()
                .enumerate()
                .map(|(value, axis)| local_stage(UOp::native_const(value as f32), Some(axis)))
        };
        let found = slots(lower());
        assert_eq!(found, slots(lower()), "slot assignment is deterministic across rewrites");
        assert_eq!(found[0], 3, "the scalar axis keeps its numeric slot");
        assert!(found[1] >= 1 << (usize::BITS - 1), "the nested axis uses the high namespace");
    }
    #[test]
    fn allocate_wraps_on_a_slot_collision() {
        let stage = || local_stage(UOp::native_const(1.0f32), Some(AxisId::Renumbered(3)));
        assert_eq!(slots([stage(), stage()]), [3, 4], "a colliding slot wraps to the next free one");
    }
    #[test]
    fn add_local_buffer_ignores_non_stage_roots() {
        let mut ctx = LocalBufferContext::default();
        assert!(add_local_buffer(&UOp::native_const(1.0f32), &mut ctx).is_none());
    }
    #[test]
    fn add_local_buffer_honours_the_stage_addrspace() {
        let stage = stage(UOp::native_const(1.0f32), vec![]);
        let mut ctx = LocalBufferContext::default();
        let lowered = add_local_buffer(&stage, &mut ctx).expect("a global STAGE still lowers");
        let (passthrough, _) = expect_after(&lowered);
        let storage = passthrough.base();
        let arg = unwrap_op!(storage, Op::Param(ops::Param { arg, .. }) => arg);
        assert_eq!((arg.slot, arg.addrspace), (0, Some(AddrSpace::Global)), "a global STAGE lowers to a PARAM");
    }
}

mod lower_index_stage_tests {
    use crate::optimizer::error::OptError;
    use crate::optimizer::{
        Renderer, apply_post_optimization_with_renderer, extra_symbolic_patterns, lower_index_patterns,
    };
    use crate::rewrite::graph_rewrite;
    use crate::spec::SpecError;
    use crate::symbolic::WeakMemo;
    use crate::symbolic::patterns::symbolic;
    use crate::test::support::prelude::*;
    use std::sync::Arc;
    use svod_dtype::DType;
    use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, ops};
    use test_case::test_case;
    fn weak(value: i64) -> Arc<UOp> {
        UOp::const_(DType::WeakInt, ConstValue::Int(value))
    }
    fn weak_float(value: f64) -> Arc<UOp> {
        UOp::const_(DType::WeakFloat, ConstValue::Float(value))
    }
    /// The f32 midpoint: two weak floats a `f64` tells apart but an `f32` does not.
    /// Folding must happen after the commitment to f32, or the two disagree.
    fn midpoint() -> f64 {
        1.0 + 2f64.powi(-24)
    }
    fn production_value(value: Arc<UOp>) -> Arc<UOp> {
        let root = graph_rewrite(extra_symbolic_patterns(), UOp::sink(vec![value]), &mut ());
        let root = graph_rewrite(lower_index_patterns(), root, &mut WeakMemo::default());
        expect_sink(&graph_rewrite(symbolic(), root, &mut ()))[0].clone()
    }
    #[track_caller]
    fn binary_operands(u: &Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
        match u.op() {
            Op::Binary(_, lhs, rhs) => (lhs.clone(), rhs.clone()),
            other => panic!("expected BINARY, got {other:?}\n{}", u.tree()),
        }
    }
    /// The committed `VCONST` values of `u`, whose dtype must be `dtype`.
    #[track_caller]
    fn vconst_values(u: &Arc<UOp>, dtype: DType) -> Vec<ConstValue> {
        assert_eq!(u.dtype(), dtype, "{}", u.tree());
        unwrap_op!(u, Op::VConst(ops::VConst { values }) => values).clone()
    }
    #[test]
    fn lower_index_composition_pushes_long_cast_through_invalid() {
        let x = UOp::variable("x".into(), 0, 15, DType::WeakInt);
        let valid = x.lt(&weak(8));
        let index = UOp::index()
            .buffer(param(0, 16, DType::Float32))
            .indices(vec![x.valid(valid).cast(DType::Int64)])
            .call()
            .expect("INDEX");
        let lowered = graph_rewrite(lower_index_patterns(), index, &mut WeakMemo::default());
        let indexed = expect_index(&lowered).1[0].clone();
        let (value, invalid) = match indexed.op() {
            Op::Ternary(TernaryOp::Where, _, value, invalid) => (value.clone(), invalid.clone()),
            other => panic!("expected WHERE, got {other:?}\n{}", lowered.tree()),
        };
        assert_eq!(value.dtype(), DType::Int32, "{}", lowered.tree());
        assert!(UOp::is_invalid_marker(&invalid));
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }
    #[test]
    fn post_optimization_propagates_stale_index_before_decomposition() {
        let stale = param(0, 1, DType::Index);
        let renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
        let error = apply_post_optimization_with_renderer(UOp::sink(vec![stale]), &renderer)
            .expect_err("legacy Index must fail at the post-index-lowering invariant");
        assert!(
            matches!(
                error,
                OptError::Spec {
                    source: SpecError::Verification {
                        boundary: "post-index-lowering",
                        reason: "legacy Index dtype must be lowered before a program",
                        ..
                    }
                }
            ),
            "unexpected error: {error:?}"
        );
    }
    #[test]
    fn extra_symbolic_distributes_weak_index_before_lowering() {
        let x = UOp::variable("x".into(), 0, 7, DType::WeakInt);
        let index = UOp::index()
            .buffer(param(0, 64, DType::Float32))
            .indices(vec![x.add(&weak(2)).mul(&weak(4))])
            .call()
            .expect("INDEX");
        let distributed = graph_rewrite(extra_symbolic_patterns(), index, &mut ());
        assert_op!(expect_index(&distributed).1[0], Op::Binary(BinaryOp::Add, ..));
        let lowered = graph_rewrite(lower_index_patterns(), distributed, &mut WeakMemo::default());
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }
    /// Folding a weak-float consumer before its operands commit to f32 would make
    /// the two midpoint neighbors agree. Each sub-case pins one consumer shape.
    #[test]
    fn lower_index_commits_weak_floats_before_folding_their_consumer() {
        let neighbor = 1.0 + 2f64.powi(-23);
        let vconst = |values: [f64; 2]| {
            UOp::vconst(
                vec![ConstValue::Float(values[0]), ConstValue::Float(values[1]), ConstValue::Invalid],
                DType::WeakFloat,
            )
        };
        let comparison = production_value(vconst([midpoint(), neighbor]).try_cmpeq(&vconst([1.0, 1.0])).expect("cmp"));
        assert_eq!(
            vconst_values(&comparison, DType::Bool.vec(3).unwrap()),
            [ConstValue::Bool(true), ConstValue::Bool(false), ConstValue::Invalid]
        );
        let sum = production_value(vconst([midpoint(), neighbor]).try_add(&vconst([midpoint(); 2])).expect("add"));
        assert_eq!(
            vconst_values(&sum, DType::Float32.vec(3).unwrap()),
            [ConstValue::Float(2.0), ConstValue::Float(2.0), ConstValue::Invalid]
        );
        let scalar_comparison = production_value(weak_float(midpoint()).try_cmpeq(&weak_float(1.0)).expect("cmp"));
        assert_eq!(scalar_comparison.dtype(), DType::Bool);
        assert_const!(scalar_comparison, true);
        let scalar_sum = production_value(weak_float(midpoint()).try_add(&weak_float(midpoint())).expect("add"));
        assert_eq!(scalar_sum.dtype(), DType::Float32);
        assert_const!(scalar_sum, 2.0);
    }
    #[test]
    fn lower_index_commits_constant_stack_lanes_before_their_consumer() {
        let lane = |value: f64| UOp::stack(vec![weak_float(value), UOp::invalid_marker()].into());
        let comparison = lane(midpoint()).try_cmpeq(&lane(1.0)).expect("cmp");
        let lowered = graph_rewrite(lower_index_patterns(), UOp::sink(vec![comparison]), &mut WeakMemo::default());
        let lowered = expect_sink(&lowered)[0].clone();
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
        assert_op!(lowered, Op::Binary(BinaryOp::Eq, ..));
        let (lhs, rhs) = binary_operands(&lowered);
        for stack in [lhs, rhs] {
            assert_eq!(stack.dtype(), DType::Float32);
            let sources = unwrap_op!(stack, Op::Stack(ops::Stack { sources }) => sources);
            assert_const!(sources[0], 1.0);
            assert!(UOp::is_invalid_marker(&sources[1]));
        }
    }
    #[test]
    fn production_commits_weak_coefficients_before_term_combining() {
        let x = UOp::variable("x".into(), -10, 10, DType::Float32);
        let expression = x
            .try_mul(&weak_float(midpoint()))
            .expect("mul")
            .try_add(&x.try_mul(&weak_float(-1.0)).expect("mul"))
            .expect("add");
        let lowered = production_value(expression);
        assert_op!(lowered, Op::Binary(BinaryOp::Mul, ..));
        let (value, zero) = binary_operands(&lowered);
        assert_same!(value, x);
        assert_const!(zero, 0.0);
    }
    /// A weak operand must commit to f32 before its consumer folds.
    #[test_case(|| UOp::try_where(weak_float(1.0).try_cmplt(&weak_float(midpoint())).expect("cmp"), UOp::native_const(7i32), UOp::native_const(9i32)).expect("WHERE"), ConstValue::Int(9); "a weak comparison commits before WHERE bounds")]
    #[test_case(|| weak_float(midpoint()).try_pow(&weak_float(1.0).try_add(&weak_float(1.0)).expect("add")).expect("pow"), ConstValue::Float(1.0); "a weak base commits before power decomposition")]
    fn production_commits_weak_operands(expression: fn() -> Arc<UOp>, expected: ConstValue) {
        assert_const!(production_value(expression()), expected);
    }
    #[test_case(f64::from_bits(midpoint().to_bits() - 1), true; "one f64 step below the midpoint still rounds to it")]
    #[test_case(midpoint(), true; "the midpoint itself commits to 1.0f32")]
    #[test_case(f64::from_bits(midpoint().to_bits() + 1), false; "one f64 step above the midpoint rounds away")]
    fn production_scalar_midpoint_neighbors_match_f32_commitment(value: f64, expected: bool) {
        let comparison = weak_float(value).try_cmpeq(&weak_float(1.0)).expect("cmp");
        let lowered = production_value(comparison);
        assert_const!(lowered, expected);
    }
}
