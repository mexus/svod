//! Shared test-support layer for `svod-schedule`: the frozen vocabulary that
//! subsystem tests reach through [`prelude`](crate::test::support::prelude).

pub mod assert;
pub mod build;
pub mod count;
pub mod eval;
pub mod harness;
pub mod matcher;
pub mod prelude;
pub mod proptest;
pub mod vars;

pub use prelude::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smallvec::smallvec;
    use svod_dtype::{AddrSpace, DType, DeviceSpec, ScalarDType};
    use svod_ir::{AxisType, BufferizeOpts, CallInfo, ConstValue, Op, ReduceOp, UOp};
    use test_case::test_case;

    use super::prelude::*;
    use crate::optimizer::{Renderer, Scheduler};

    fn kernel() -> Arc<UOp> {
        elementwise(&[4], AxisType::Global).call(smallvec![], CallInfo::default())
    }

    #[test]
    fn variable_vocabulary() {
        assert_eq!(RangeSpec::NON_NEG, RangeSpec { lo: 0, hi: 100 });
        assert_eq!(RangeSpec::SIGNED, RangeSpec { lo: -100, hi: 100 });
        assert_eq!(RangeSpec::NONZERO, RangeSpec { lo: 1, hi: 100 });
        assert_eq!(RangeSpec::INDEX, RangeSpec { lo: 0, hi: 1024 });

        let vars = TestVars::new();
        let dtypes = [
            (vars.x.dtype(), DType::Int32),
            (vars.a.dtype(), DType::Int32),
            (vars.n.dtype(), DType::Int32),
            (vars.p.dtype(), DType::Bool),
            (vars.i.dtype(), DType::Index),
            (vars.bounded.dtype(), DType::Float32),
            (TestVars::weak().x.dtype(), DType::WeakInt),
            (TestVars::typed(DType::Int8).y.dtype(), DType::Int8),
            (TestVars::typed(DType::Int8).bounded.dtype(), DType::Float32),
            (TestVars::typed(DType::Float64).bounded.dtype(), DType::Float64),
            (TestVars::default().x.dtype(), DType::Int32),
        ];
        for (got, want) in dtypes {
            assert_eq!(got, want);
        }
        assert_eq!(var_name(&vars.x).as_deref(), Some("x"));
        assert_eq!(var_range(&vars.x), Some((0, 100)));
        assert!(matches!(vars.unknown.op(), Op::Load(..)));
        assert!(matches!(unboundable(DType::Float32).op(), Op::Load(..)));

        let pinned = vars.at(&[("x", 5), ("p", 1)]);
        assert!(matches!(pinned.x.op(), Op::Const(v) if v.0 == ConstValue::Int(5)));
        assert!(matches!(pinned.p.op(), Op::Const(v) if v.0 == ConstValue::Bool(true)));
        assert!(matches!(vars.at(&[("unknown", 3)]).unknown.op(), Op::Const(..)));
    }

    /// Every accessor and macro failure must panic with the rendered tree.
    #[test_case(|| assert_op!(UOp::native_const(1i32), Op::Load(..)), "assert_op!" ; "panic assert_op")]
    #[test_case(|| assert_same!(elementwise(&[4], AxisType::Global), elementwise(&[5], AxisType::Global)), "assert_same!" ; "panic assert_same")]
    #[test_case(|| assert_const!(UOp::native_const(1i32), 2), "assert_const!" ; "panic assert_const")]
    #[test_case(|| drop(expect_sink(&UOp::native_const(1i32))), "expected SINK" ; "panic sink accessor")]
    #[test_case(|| drop(expect_call(&UOp::native_const(1i32))), "expected CALL" ; "panic call accessor")]
    #[test_case(|| drop(TestVars::new().at(&[("nope", 1)])), "unknown test variable" ; "panic unknown variable")]
    fn failures_panic_with_the_tree(fail: fn(), needle: &str) {
        let payload = std::panic::catch_unwind(fail).expect_err("must panic");
        let message = payload.downcast_ref::<String>().expect("a formatted panic message");
        assert!(message.contains(needle), "panic did not mention {needle:?}: {message}");
    }

    #[test_case(|| TestVars::new().c(3), DType::Int32 ; "dtype vars c")]
    #[test_case(|| TestVars::new().u(3), DType::Int32 ; "dtype vars u")]
    #[test_case(|| TestVars::new().f(1.5), DType::Float32 ; "dtype vars f")]
    #[test_case(|| TestVars::new().b(true), DType::Bool ; "dtype vars b")]
    #[test_case(|| TestVars::new().ic(3), DType::Index ; "dtype vars ic")]
    #[test_case(|| index_const(3), DType::Index ; "dtype index const")]
    #[test_case(|| UOp::var("e", DType::Int32, 0, 4).c(3), DType::Int32 ; "dtype expr c")]
    #[test_case(|| UOp::var("e", DType::Int32, 0, 4).u(3), DType::Int32 ; "dtype expr u")]
    #[test_case(|| UOp::var("e", DType::Int32, 0, 4).f(1.5), DType::Int32 ; "dtype expr f")]
    #[test_case(|| UOp::var("e", DType::Int32, 0, 4).b(true), DType::Int32 ; "dtype expr b")]
    #[test_case(|| UOp::var("e", DType::Int32, 0, 4).ic(3), DType::Index ; "dtype expr ic")]
    fn consts_vocabulary_covers_both_receivers(build: fn() -> Arc<UOp>, want: DType) {
        assert_eq!(build().dtype(), want);
    }

    #[test_case(|| buffer(4), |op| matches!(op, Op::Buffer(..)) ; "ctor buffer")]
    #[test_case(|| buffer_of(4, ScalarDType::Int8), |op| matches!(op, Op::Buffer(..)) ; "ctor buffer_of")]
    #[test_case(|| buffer_on(2, ScalarDType::Bool, DeviceSpec::Cuda { device_id: 0 }), |op| matches!(op, Op::Buffer(..)) ; "ctor buffer_on")]
    #[test_case(|| param(1, 8, DType::Float32), |op| matches!(op, Op::Param(..)) ; "ctor param")]
    #[test_case(|| index(buffer(4), 2), |op| matches!(op, Op::Index(..)) ; "ctor index")]
    #[test_case(|| shaped_index(buffer(4), [0, 1, 2]), |op| matches!(op, Op::Index(..)) ; "ctor shaped_index")]
    #[test_case(|| load(index(buffer(4), 2)), |op| matches!(op, Op::Load(..)) ; "ctor load")]
    #[test_case(|| store(index(buffer(4), 2), UOp::native_const(1.0f32)), |op| matches!(op, Op::Store(..)) ; "ctor store")]
    #[test_case(|| range(8, AxisType::Global, 3), |op| matches!(op, Op::Range(..)) ; "ctor range")]
    #[test_case(|| global_range(8, 0), |op| matches!(op, Op::Range(..)) ; "ctor range_const")]
    #[test_case(|| reduce_range(8, 0), |op| matches!(op, Op::Range(..)) ; "ctor reduce_range")]
    #[test_case(|| range_symbolic(index_const(8), 0), |op| matches!(op, Op::Range(..)) ; "ctor range_symbolic")]
    #[test_case(|| stage(elementwise(&[4], AxisType::Global), vec![global_range(4, 0)]), |op| matches!(op, Op::Stage(..)) ; "ctor stage")]
    #[test_case(|| stage_with(elementwise(&[4], AxisType::Global), vec![global_range(4, 0)], BufferizeOpts::local()), |op| matches!(op, Op::Stage(..)) ; "ctor stage_with")]
    #[test_case(|| reduce(elementwise(&[4], AxisType::Global), vec![reduce_range(4, 0)], ReduceOp::Add), |op| matches!(op, Op::Reduce(..)) ; "ctor reduce")]
    #[test_case(|| stack([UOp::native_const(1i32), UOp::native_const(2i32)]), |op| matches!(op, Op::Stack(..)) ; "ctor stack")]
    #[test_case(|| float_values([1.0, 2.0]), |op| matches!(op, Op::Stack(..)) ; "ctor float_values")]
    #[test_case(|| bool_values([true, false]), |op| matches!(op, Op::Stack(..)) ; "ctor bool_values")]
    #[test_case(|| elementwise(&[4], AxisType::Global), |op| matches!(op, Op::Sink(..)) ; "ctor elementwise")]
    #[test_case(|| reduce_sink(&[2], &[3, 4], ReduceOp::Add), |op| matches!(op, Op::Sink(..)) ; "ctor reduce_sink")]
    #[test_case(|| matmul(4, 4, 4, DType::Float32, None), |op| matches!(op, Op::Sink(..)) ; "ctor matmul")]
    fn constructors_build_their_op(build: fn() -> Arc<UOp>, want: fn(&Op) -> bool) {
        let built = build();
        assert!(want(built.op()), "unexpected {:?}\n{}", built.op(), built.tree());
    }

    #[test_case(|| buffer(4), 4, DType::Float32 ; "sized buffer")]
    #[test_case(|| buffer_of(4, ScalarDType::Int8), 4, DType::Int8 ; "sized buffer_of")]
    #[test_case(|| buffer_on(2, ScalarDType::Bool, DeviceSpec::Cuda { device_id: 0 }), 2, DType::Bool ; "sized buffer_on")]
    fn buffers_carry_size_and_dtype(build: fn() -> Arc<UOp>, size: usize, dtype: DType) {
        assert_eq!(expect_buffer(&build()), (size, dtype));
    }

    #[test_case(|| range(8, AxisType::Global, 0), DType::Index ; "dtype of global")]
    #[test_case(|| range(8, AxisType::Local, 0), DType::Index ; "dtype of local")]
    #[test_case(|| range(8, AxisType::Reduce, 0), DType::WeakInt ; "dtype of reduce")]
    #[test_case(|| global_range(8, 0), DType::Index ; "dtype of range_const")]
    #[test_case(|| reduce_range(8, 0), DType::WeakInt ; "dtype of reduce_range")]
    #[test_case(|| range_symbolic(index_const(8), 0), DType::WeakInt ; "dtype of range_symbolic")]
    fn range_dtypes(build: fn() -> Arc<UOp>, want: DType) {
        assert_eq!(build().dtype(), want);
    }

    #[test]
    fn range_constructors_expose_end_id_and_axis() {
        let (end, id, axis) = expect_range(&range(8, AxisType::Global, 3));
        assert!(matches!(end.op(), Op::Const(..)));
        assert_eq!(id, svod_ir::AxisId::Renumbered(3));
        assert_eq!(axis, AxisType::Global);
        assert_eq!(expect_range_extent(&range(8, AxisType::Global, 3)), 8);
        assert_eq!(range_axis_type(&range(8, AxisType::Local, 0)), AxisType::Local);
        assert_eq!(range_axis_id(&range(8, AxisType::Global, 1)), svod_ir::AxisId::Renumbered(1));
        assert_eq!(range_axis_type(&global_range(8, 0)), AxisType::Global);
    }

    #[test]
    fn accessors_extract_payloads() {
        let (buf, indices) = expect_index(&index(buffer(4), 2));
        assert!(matches!(buf.op(), Op::Buffer(..)));
        assert_eq!(indices.len(), 1);

        let (_, shaped) = expect_index(&shaped_index(buffer(4), [0, 1, 2]));
        assert_eq!(shaped.len(), 1);
        assert!(matches!(shaped[0].op(), Op::Stack(..)));

        let idx = index_of(buffer(4), index_const(1));
        assert!(matches!(load(idx.clone()).op(), Op::Load(..)));
        let (store_index, value, gate) = expect_store(&store(idx, UOp::native_const(1.0f32)));
        assert!(matches!(store_index.op(), Op::Index(..)));
        assert!(matches!(value.op(), Op::Const(..)));
        assert!(gate.is_none());

        let (computation, ranges) = expect_end(&load(index(buffer(4), 0)).end(smallvec![global_range(4, 0)]));
        assert!(matches!(computation.op(), Op::Load(..)));
        assert_eq!(ranges.len(), 1);

        let dep = UOp::native_const(1.0f32);
        let (passthrough, deps) = expect_after(&load(index(buffer(4), 0)).after(smallvec![dep]));
        assert!(matches!(passthrough.op(), Op::Load(..)));
        assert_eq!(deps.len(), 1);

        assert!(matches!(expect_call(&kernel()).op(), Op::Sink(..)));

        let sink = elementwise(&[4, 5], AxisType::Global);
        assert_eq!(expect_sink(&sink).len(), 3);
        assert_eq!(range_axis_type(&expect_sink(&sink)[1]), AxisType::Global);

        let sources = expect_sink(&reduce_sink(&[2], &[3, 4], ReduceOp::Add));
        assert!(matches!(sources[0].op(), Op::Reduce(..)));
        assert!(matches!(sources[1].op(), Op::Range(..)));
        assert_eq!(expect_range_extent(&sources[1]), 2);
    }

    #[test]
    fn op_search_and_matmul_counts() {
        let call = kernel();
        assert!(has_op(&call, |op| matches!(op, Op::Sink(..))));
        assert!(!has_op(&call, |op| matches!(op, Op::Load(..))));
        assert!(first_op(&call, |op| matches!(op, Op::Range(..))).is_some());
        assert!(first_op(&call, |op| matches!(op, Op::Load(..))).is_none());

        let plain = matmul(4, 4, 4, DType::Float32, None);
        let wide = matmul(4, 4, 4, DType::Float32, Some(Box::new(|v: Arc<UOp>| v.mul(&UOp::native_const(2.0f32)))));
        assert_eq!(count(&plain, |node| matches!(node.op(), Op::Index(..))), 2);
        assert!(wide.node_count() > plain.node_count());
    }

    #[test_case(0 ; "the simple matcher")]
    #[test_case(1 ; "the full matcher")]
    #[test_case(2 ; "the dce matcher")]
    fn matchers_fold_constant_sums(which: usize) {
        let matcher = [Matchers::simple(), Matchers::full(), Matchers::dce()][which];
        let folded = rewrite(matcher, TestVars::new().ic(2).add(&TestVars::new().ic(3)));
        assert_const!(folded, 5);
    }

    #[test]
    fn harness_assertions() {
        assert_rewrites_to(Matchers::simple(), |v| v.ic(2).add(&v.ic(3)), |v| v.ic(5));
        assert_rewrites_to_and_evaluates(Matchers::simple(), |v| v.x.add(&v.c(0)), |v| v.x.clone());
        assert_unchanged(Matchers::simple(), |v| v.x.mul(&v.y));
        assert_pass_preserves(|u| rewrite(Matchers::simple(), u), |v| v.x.add(&v.c(0)));
        assert_const_value(&UOp::const_(DType::Int32, ConstValue::Int(7)), ConstValue::Int(7));
        let folded = rewrite_with(Matchers::simple(), &mut (), TestVars::new().ic(2).add(&TestVars::new().ic(3)));
        assert_const!(folded, 5);
    }

    #[test]
    fn assert_macros_and_bindings() {
        let sink = elementwise(&[4], AxisType::Global);
        assert_op!(sink, Op::Sink(..));
        assert_same!(sink, sink.clone());

        let constant = UOp::native_const(5i32);
        let value = assert_op!(constant, Op::Const(v) => v);
        assert_eq!(value.0, ConstValue::Int(5));
        assert_const!(constant, 5);

        let call = kernel();
        assert!(matches!(unwrap_op!(call, Op::Call(c) => c).body.op(), Op::Sink(..)));

        assert_const!(UOp::native_const(1.5f32), 1.5);
        assert_const!(UOp::native_const(true), true);
        assert_const!(UOp::native_const(5i32), ConstValue::Int(5));

        let vars = TestVars::new();
        assert_eq!(eval_typed(&vars.c(3), &Bindings::none()), Some(ConstValue::Int(3)));
        assert_eq!(eval_typed(&vars.x, &Bindings::at("x", 5)), Some(ConstValue::Int(5)));
        assert_eq!(eval_typed(&vars.x, &Bindings::none().with("x", 6)), Some(ConstValue::Int(6)));
        assert_eq!(eval_typed(&vars.x, &Bindings::none()), None);
        assert_eq!(fold_at(&vars.x.add(&vars.c(1)), &Bindings::at("x", 4)), Some(ConstValue::Int(5)));
        assert_eq!(fold_at(&vars.c(2).add(&vars.c(3)), &Bindings::none()), Some(ConstValue::Int(5)));

        // Wrapping at the node's own width is observable.
        let int8 = |v: i64| UOp::const_(DType::Int8, ConstValue::Int(v));
        assert_eq!(eval_typed(&int8(100).mul(&int8(2)), &Bindings::none()), Some(ConstValue::Int(-56)));
        assert_eq!(
            eval_typed(&UOp::const_(DType::Int32, ConstValue::Int(-1)).bitcast(DType::UInt32), &Bindings::none()),
            Some(ConstValue::UInt(u32::MAX as u64))
        );
        assert_eq!(
            eval_typed(&UOp::const_(DType::Int8, ConstValue::Int(200)).cast(DType::Int8), &Bindings::none()),
            Some(ConstValue::Int(-56))
        );

        let bound = UOp::var("fixed", DType::Int32, 7, 7);
        assert_eq!(eval_typed(&bound, &Bindings::none()), Some(ConstValue::Int(7)));
        assert_eq!(eval_typed(&bound.bind(UOp::native_const(9i32)), &Bindings::none()), Some(ConstValue::Int(9)));
    }

    /// Every operand must vary across a capped sweep. Walking the product in
    /// order pins all but the first at its `lo`, which silently turns a
    /// two-variable law into a one-variable one.
    #[test]
    fn a_capped_sweep_varies_every_operand() {
        let vars = TestVars::new();
        let operands = [vars.x.clone(), vars.y.clone()];
        let mut seen: [std::collections::BTreeSet<i64>; 2] = Default::default();
        for bindings in range_points(&operands, 64) {
            for (slot, operand) in operands.iter().enumerate() {
                let Some(ConstValue::Int(value)) = eval_typed(operand, &bindings) else {
                    panic!("every swept operand evaluates")
                };
                seen[slot].insert(value);
            }
        }
        assert!(seen[0].len() > 1 && seen[1].len() > 1, "both operands must vary, saw {seen:?}");
    }

    /// A cap at or above the product still enumerates the whole box, so the
    /// exhaustive sweeps keep their meaning.
    #[test]
    fn an_uncapped_sweep_is_still_a_bijection() {
        let small = UOp::var("small", DType::Int32, 0, 4);
        let tiny = UOp::var("tiny", DType::Int32, 0, 2);
        let points: std::collections::BTreeSet<(i64, i64)> = range_points(&[small.clone(), tiny.clone()], 4096)
            .map(|at| match (eval_typed(&small, &at), eval_typed(&tiny, &at)) {
                (Some(ConstValue::Int(a)), Some(ConstValue::Int(b))) => (a, b),
                other => panic!("both operands evaluate, got {other:?}"),
            })
            .collect();
        assert_eq!(points.len(), 15, "5 x 3 points, each exactly once");
    }

    #[test]
    fn range_points_sweep_and_cap() {
        let vars = TestVars::new();
        assert_eq!(range_points(&[], 4).count(), 1);
        assert_eq!(range_points(std::slice::from_ref(&vars.x), 4).count(), 4);
        let points: Vec<_> = range_points(std::slice::from_ref(&vars.p), 4).collect();
        assert_eq!(points.len(), 2);
        let first = range_points(std::slice::from_ref(&vars.x), 1).next().expect("one point");
        assert_eq!(eval_typed(&vars.x, &first), Some(ConstValue::Int(0)));
    }

    #[test]
    fn counts_are_topological() {
        let rng = global_range(4, 0);
        let idx = index(param(0, 16, DType::Float32), 0);
        let end = store(idx.clone(), load(idx)).end(smallvec![rng]);
        let call = end.call(smallvec![], CallInfo::default());

        assert_eq!(count(&call, |node| matches!(node.op(), Op::Load(..))), 1);
        let kinds = count_kinds(&call);
        assert_eq!(
            kinds,
            OpCounts { loads: 1, stores: 1, calls: 1, ends: 1, ranges: 1, params: 1, ..Default::default() }
        );
        assert_eq!(kernels(&call), 1);
        assert!(first_call(&call).is_some());
        assert_eq!(count_kinds(&elementwise(&[4], AxisType::Global)).ranges, 1);
        assert_eq!(count_kinds(&kernel()).calls, 1);

        let locals = UOp::buffer(0, 4, DType::Float32, AddrSpace::Local, None);
        let regs = UOp::buffer(1, 4, DType::Float32, AddrSpace::Reg, None);
        let kinds = count_kinds(&UOp::sink(vec![locals, regs]));
        assert_eq!((kinds.locals, kinds.regs, kinds.params), (1, 1, 0));

        let sched = Scheduler::new(elementwise(&[4], AxisType::Global), Renderer::cpu());
        assert_eq!(axis_count(&sched, AxisType::Global), 1);
        assert_eq!(axis_count(&sched, AxisType::Reduce), 0);
        assert_axis!(sched, Global: 1);
    }

    #[test]
    fn proptest_presets() {
        let cases = (CHEAP, EQUIVALENCE, proptest_config(7).cases, cheap().cases, equivalence().cases);
        assert_eq!(cases, (256, 512, 7, 256, 512));
    }

    /// A sibling module sees every frozen name through the public prelude path;
    /// the outer module's own `use` covers the exports not repeated here.
    mod prelude_glob {
        use svod_dtype::DType;
        use svod_ir::{ConstValue, Op};

        use crate::test::support::prelude::*;

        #[test]
        fn every_frozen_name_arrives_through_the_prelude() {
            let vars = TestVars::new();
            assert_eq!(fold_at(&vars.x.add(&vars.c(2)), &Bindings::at("x", 3)), Some(ConstValue::Int(5)));
            assert_eq!(range_points(std::slice::from_ref(&vars.x), 2).count(), 2);
            assert_eq!(eval_typed(&vars.c(1), &Bindings::none()), Some(ConstValue::Int(1)));
            assert_eq!(expect_range_extent(&global_range(4, 0)), 4);
            assert_eq!(count(&vars.x, |node| matches!(node.op(), Op::DefineVar(..))), 1);
            assert_eq!(count(&matmul(2, 2, 2, DType::Float32, None), |node| matches!(node.op(), Op::Sink(..))), 1);
        }
    }
}
