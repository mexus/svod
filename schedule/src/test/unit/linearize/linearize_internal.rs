use super::*;
use proptest::prelude::*;
use smallvec::smallvec;
use std::collections::HashMap;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::types::{AxisId, AxisType, ConstValue, InsArg, RendererDevice, WmmaMetadata};
use svod_ir::{WmmaUpcastAxes, ops, test::property::generators::arb_arithmetic_tree_up_to};

use crate::linearize::line_rewrite_cleanups;
use test_case::test_case;

use crate::test::support::prelude::*;

fn position_of(nodes: &[Arc<UOp>], node: &Arc<UOp>) -> u32 {
    nodes.iter().position(|candidate| Arc::ptr_eq(candidate, node)).expect("node is part of the graph") as u32
}

fn range_key(path: &[usize]) -> ArgKey {
    ArgKey::Range(path.to_vec(), axis_type_value(AxisType::Weak))
}

fn reduce_key(axes: &[usize]) -> ArgKey {
    ArgKey::Reduce(reduce_value(svod_ir::ReduceOp::Add), axes.to_vec(), None)
}

/// `partial_arg_cmp` orders only the prefix it can fully compare and answers `None` as
/// soon as a pair is undecided, which keeps the sort stable but never wrong.
#[test_case(ArgKey::None, ArgKey::None, Some(Ordering::Equal) ; "none equals none")]
#[test_case(ArgKey::Text("a".into()), ArgKey::Text("z".into()), Some(Ordering::Less) ; "text compares lexicographically")]
#[test_case(ArgKey::Const(const_key(ConstValue::Int(1))), ArgKey::Const(const_key(ConstValue::Int(2))), Some(Ordering::Less) ; "constants order by value")]
#[test_case(reduce_key(&[0]), reduce_key(&[0, 1]), Some(Ordering::Less) ; "reduce axes extend the key")]
#[test_case(range_key(&[0, 1]), range_key(&[0, 1]), Some(Ordering::Equal) ; "equal range paths compare equal")]
#[test_case(range_key(&[0]), range_key(&[0, 1]), None ; "a shorter path is not a prefix order")]
#[test_case(range_key(&[0, 1]), range_key(&[0]), None ; "the reversed prefix is not an order either")]
#[test_case(ArgKey::Range(vec![0, 1], 2), range_key(&[0, 1]), None ; "equal paths with different axis types have no order")]
#[test_case(ArgKey::Const(const_key(ConstValue::Float(f64::NAN))), ArgKey::Const(const_key(ConstValue::Float(f64::NAN))), None ; "two NaN constants have no order")]
#[test_case(ArgKey::None, ArgKey::Text("a".into()), None ; "different variants have no order")]
fn partial_arg_comparison_orders_only_what_it_can(left: ArgKey, right: ArgKey, expected: Option<Ordering>) {
    assert_eq!(partial_arg_cmp(&left, &right), expected);
}

#[test]
fn tinygrad_float_keys_coalesce_signed_zero_and_nan_payloads() {
    assert_eq!(const_key(ConstValue::Float(-0.0)), const_key(ConstValue::Float(0.0)));
    let left = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0001)));
    let right = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0002)));
    assert_eq!(left, right);
    assert_eq!(partial_const_cmp(&left, &right), None);
}

/// A parameter projects to a concrete `addrspace` (`Some(4)` is the default), and the
/// comparison stops at the first field it cannot decide.
#[test]
fn partial_param_comparison_orders_by_name_and_stops_at_an_unknown_field() {
    let key = |name: &str| param_key(&ParamArg::variable(name.to_string(), DType::WeakInt, 0, 8));
    assert_eq!(partial_param_cmp(&key("a"), &key("z")), Some(Ordering::Less));
    assert_eq!(partial_param_cmp(&key("z"), &key("a")), Some(Ordering::Greater));
    assert_eq!(partial_param_cmp(&key("a"), &key("a")), Some(Ordering::Equal));
    let projected = arg_key(&Op::Param(ops::Param {
        shape: UOp::index_const(1),
        arg: ParamArg::variable("projected".to_string(), DType::WeakInt, 0, 8).into(),
    }));
    assert!(matches!(projected, ArgKey::Param(ParamKey { addrspace: Some(4), .. })));
    // `name` is compared before `addrspace`, so a decided name short-circuits an undecidable
    // addrspace that follows it.
    let (mut left, mut right) = (key("a"), key("b"));
    left.addrspace = None;
    right.addrspace = Some(1);
    assert_eq!(partial_param_cmp(&left, &right), Some(Ordering::Less), "a decided name beats a later unset field");
    assert_eq!(partial_param_cmp(&right, &left), Some(Ordering::Greater));
    // With the name tied, the undecidable field is the first difference and the order is partial.
    let (mut left, mut right) = (key("a"), key("a"));
    left.addrspace = None;
    right.addrspace = Some(1);
    assert_eq!(partial_param_cmp(&left, &right), None, "an unset field makes the order partial");
}

/// A VCONST's tuplize key is the `STACK(CONST...)` the pinned Tinygrad builds: head `op 16`
/// with no argument and one `op 61` lane per value, sorting after a plain `STACK`.
#[test]
fn tinygrad_vconst_linearizer_key_is_stack_of_constants() {
    let vconst = UOp::vconst(vec![ConstValue::Int(1), ConstValue::Int(2)], DType::WeakInt);
    let topo = vconst.toposort();
    let keys = compute_tuplize(&topo, &node_index(&topo));
    let key = position_of(&topo, &vconst);
    let head = keys.head_of(key);
    assert_eq!((head.op, head.arg.clone(), head.dtype.clone()), (16, ArgKey::None, dtype_key(&DType::WeakInt)));
    let lanes: Vec<_> = keys.sources(key).iter().map(|&lane| keys.head_of(lane)).collect();
    assert_eq!(lanes.len(), 2);
    assert!(lanes.iter().all(|lane| lane.op == 61 && lane.dtype == dtype_key(&DType::WeakInt)));
    let lane_args: Vec<_> = lanes.iter().map(|lane| lane.arg.clone()).collect();
    assert_eq!(lane_args, [ArgKey::Const(ConstKey::Int(1)), ArgKey::Const(ConstKey::Int(2))]);
    let stack = UOp::stack(smallvec![UOp::range_const(8, 0), UOp::special(UOp::index_const(8), "gidx0".to_string())]);
    assert_eq!(tinygrad_tuplize_cmp(&stack, &vconst), Some(Ordering::Less));
}

/// Equal heads are interned once and ranked by one integer; `Index` and `Int64` share a
/// dtype key, so their constants must land on one rank.
#[test]
fn compute_tuplize_interns_equal_heads_onto_one_rank() {
    let sink = tied_keys_graph();
    let topo = sink.toposort();
    let keys = compute_tuplize(&topo, &node_index(&topo));
    assert!(keys.heads.windows(2).all(|pair| pair[0] <= pair[1]), "the head rank table must be sorted");
    let index = UOp::const_(DType::Index, ConstValue::Int(1));
    let long = UOp::const_(DType::Int64, ConstValue::Int(1));
    let rank = |node: &Arc<UOp>| keys.head[position_of(&topo, node) as usize];
    assert_eq!(rank(&index), rank(&long), "Index and Int64 share a dtype key");
}

fn wmma_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "svod_name".to_string(),
        dims: (16, 16, 16),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: RendererDevice::CudaSm80,
        threads: 32,
        upcast_axes: Some(WmmaUpcastAxes { a: vec![(AxisId::Renumbered(3), 2)], b: vec![], c: vec![] }),
        reduce_axes: vec![AxisId::Renumbered(4)],
    }
}

fn wmma_op(metadata: &WmmaMetadata) -> Op {
    let value = UOp::native_const(0.0f32);
    Op::Wmma(ops::Wmma { a: value.clone(), b: value.clone(), c: value, metadata: Box::new(metadata.clone()) })
}

/// The WMMA key carries only `dims`, `dtype_in`, `device` and `threads`; the Svod-only
/// `name`, `dtype_out`, `reduce_axes` and the spelling of an upcast path must not move it.
#[test_case(|_| (), Some(Ordering::Equal) ; "an identical metadata compares equal")]
#[test_case(|m| m.name = "a_svod_name".into(), Some(Ordering::Equal) ; "the Svod-only name is dropped")]
#[test_case(|m| m.dtype_out = DType::Int32, Some(Ordering::Equal) ; "the output dtype is dropped")]
#[test_case(|m| m.reduce_axes.clear(), Some(Ordering::Equal) ; "the reduce axes are dropped")]
#[test_case(|m| m.upcast_axes = Some(WmmaUpcastAxes { a: vec![(AxisId::RenumberedPath(smallvec![3, 1]), 2)], b: vec![], c: vec![] }), Some(Ordering::Equal) ; "an upcast path is spelled the same")]
#[test_case(|m| m.upcast_axes = None, None ; "omitting the upcast axes leaves the order partial")]
#[test_case(|m| m.dims = (8, 16, 16), Some(Ordering::Greater) ; "smaller dims sort earlier")]
#[test_case(|m| m.threads = 16, Some(Ordering::Greater) ; "fewer threads sort earlier")]
#[test_case(|m| m.dtype_in = DType::BFloat16, Some(Ordering::Less) ; "the input dtype is load-bearing")]
#[test_case(|m| m.device = RendererDevice::CudaSm89, Some(Ordering::Equal) ; "every CUDA variant collapses to one device name")]
#[test_case(|m| m.device = RendererDevice::Cpu, Some(Ordering::Greater) ; "another backend is load-bearing")]
fn wmma_key_keeps_only_the_load_bearing_metadata(change: fn(&mut WmmaMetadata), expected: Option<Ordering>) {
    let (mut changed, base) = (wmma_metadata(), wmma_metadata());
    change(&mut changed);
    let agrees = arg_key(&wmma_op(&base)) == arg_key(&wmma_op(&changed));
    assert_eq!(agrees, expected == Some(Ordering::Equal), "arg_key must agree with the tuplize order");
    let (base, changed) = (UOp::new(wmma_op(&base), DType::Float32), UOp::new(wmma_op(&changed), DType::Float32));
    assert_eq!(tinygrad_tuplize_cmp(&base, &changed), expected);
}

/// Upstream dropped the CONST and DEFINE_VAR arms ("don't place consts early"): a symbolic
/// variable is placed as a PARAM.
#[test_case(UOp::param(3, 1, DType::Float32, Some(DeviceSpec::Cpu)), (-20, Some(3)); "param carries its slot")]
#[test_case(UOp::variable("n".to_string(), 0, 8, DType::Int32), (-20, Some(-1)); "define var is a param")]
#[test_case(UOp::buffer(1, 1, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu)), (-18, None); "global buffer")]
#[test_case(UOp::buffer(2, 1, DType::Float32, AddrSpace::Reg, None), (-18, None); "register buffer")]
#[test_case(UOp::buffer(0, 1, DType::Float32, AddrSpace::Local, None), (-17, None); "local buffer")]
#[test_case(UOp::const_(DType::Int32, ConstValue::Int(7)), (0, None); "const is not placed early")]
#[test_case(UOp::range_const(10, 0), (5, None); "range is placed late")]
fn tinygrad_placement_priorities(node: Arc<UOp>, expected: (i32, Option<i64>)) {
    assert_eq!(priority(&node), expected);
}

fn assert_linearize_is_topological(sink: &Arc<UOp>) {
    let order = linearize(sink.clone());
    let at =
        |node: &Arc<UOp>| order.iter().position(|e| Arc::ptr_eq(e, node)).expect("every reachable node is emitted");
    for node in sink.toposort() {
        for source in node.op().sources() {
            assert!(at(&source) < at(&node), "{:?} emitted before its source {:?}", node.op(), source.op());
        }
    }
    assert_same!(order.last().expect("a non-empty linearization").clone(), sink.clone());
}

/// `run_count` is `prod(vmax + 1)` over the in-scope RANGEs; an AFTER whose dependency closed
/// the loop is back outside it, which is what lets the linearizer sink it.
#[test]
fn run_count_is_the_product_of_the_in_scope_trip_counts() {
    let (outer, inner) = (global_range(10, 0), global_range(4, 1));
    let nested = UOp::native_const(1.0f32).add(&inner.cast(DType::Float32)).add(&outer.cast(DType::Float32));
    for (node, count) in [(&UOp::native_const(1.0f32), 1), (&outer, 10), (&inner, 4), (&nested, 40)] {
        assert_eq!(run_count(node), count);
    }
    let end = nested.end(smallvec![outer, inner]);
    let after = UOp::new(Op::After(ops::After { passthrough: nested, deps: smallvec![end] }), DType::Float32);
    assert_eq!(run_count(&after), 1, "the AFTER is outside the loops its dependency closed");
}

/// The ranks are only half the story: `linearize` must actually emit the lower-ranked chain
/// first, all 140 levels of it.
#[test]
fn deep_precast_chain_linearizes_in_tuplize_order() {
    let (mut low, mut high) =
        (UOp::const_(DType::Int32, ConstValue::Int(1)), UOp::const_(DType::Int32, ConstValue::Int(2)));
    for _ in 0..140 {
        low = UOp::new(Op::Precast(ops::Precast { src: low }), DType::Int32);
        high = UOp::new(Op::Precast(ops::Precast { src: high }), DType::Int32);
    }
    let sink = UOp::sink(vec![high.clone(), low.clone()]);
    let topo = sink.toposort();
    let ranks = tuplize_ranks(&topo, &node_index(&topo));
    assert!(ranks[position_of(&topo, &low) as usize] < ranks[position_of(&topo, &high) as usize]);

    let order = linearize(sink);
    let at = |node: &Arc<UOp>| order.iter().position(|u| Arc::ptr_eq(u, node)).expect("the chain tip is emitted");
    assert!(at(&low) < at(&high), "the lower-ranked chain must be emitted first");
}

#[test]
fn equal_dependency_side_effects_use_full_arg_order() {
    let dependency = UOp::native_const(0i32);
    let effect = |code: &str| {
        UOp::new(Op::CustomI(ops::CustomI { deps: smallvec![dependency.clone()], code: code.into() }), DType::Void)
    };
    let (earlier, later) = (effect("a"), effect("z"));
    let order = linearize(UOp::sink(vec![later.clone(), earlier.clone()]));
    assert!(order.iter().position(|u| Arc::ptr_eq(u, &earlier)) < order.iter().position(|u| Arc::ptr_eq(u, &later)));
}

/// Nested axis paths and INS attributes both participate in the key, and the Svod-only
/// INS op value is pinned.
#[test]
fn axis_paths_ins_attributes_and_op_values_are_pinned() {
    let end = UOp::index_const(4);
    let outer = UOp::range_axis(end.clone(), AxisId::RenumberedPath(smallvec![0, 1]), AxisType::Loop);
    let inner = UOp::range_axis(end, AxisId::RenumberedPath(smallvec![0, 2]), AxisType::Loop);
    assert!(arg_key(outer.op()) < arg_key(inner.op()));
    let source = UOp::native_const(1i32);
    let ins = |axis: &str| {
        let arg = InsArg::with_attributes("v_add", vec![("axis".into(), axis.into())]);
        UOp::new(Op::Ins(ops::Ins { sources: smallvec![source.clone()], arg }), DType::Int32)
    };
    assert!(arg_key(ins("1").op()) < arg_key(ins("2").op()));
    assert_eq!(op_value(ins("1").op()), 64);
}

#[test]
fn line_rewrite_expands_only_gated_stores_over_addressable_indices() {
    let address = index(param(0, 16, DType::Float32), 0);
    let value = UOp::native_const(1.0f32);
    let gate = |value: bool| UOp::native_const(value);
    // A gated STORE over an INDEX becomes IF / ungated STORE / ENDIF.
    let gated = address.clone().store_gated(value.clone(), gate(true));
    let expanded = line_rewrite_cleanups(vec![gated.clone()]);
    assert_eq!(expanded.len(), 3);
    let Op::If(ops::If { condition, body }) = expanded[0].op() else { panic!("expected IF") };
    assert_same!(condition.clone(), gate(true));
    assert_eq!(body.len(), 1);
    assert_op!(expanded[1], Op::Store(ops::Store { gate: None, .. }));
    let Op::EndIf(ops::EndIf { if_op }) = expanded[2].op() else { panic!("expected ENDIF") };
    assert_same!(if_op.clone(), expanded[0].clone());
    // Two gated stores expand independently and keep their order.
    let expanded = line_rewrite_cleanups(vec![gated, address.clone().store_gated(value.clone(), gate(false))]);
    assert_eq!(expanded.len(), 6);
    assert_op!(expanded[0], Op::If(..));
    assert_op!(expanded[2], Op::EndIf(..));
    let Op::If(ops::If { condition, .. }) = expanded[3].op() else { panic!("expected the second IF") };
    assert_same!(condition.clone(), gate(false));
    assert_op!(expanded[5], Op::EndIf(..));
    // A cast address is unwrapped before the INDEX test.
    let cast = address.cast(DType::Int64).store_gated(value.clone(), gate(true));
    assert_eq!(line_rewrite_cleanups(vec![cast]).len(), 3);
    // Ungated stores, non-Bool gates and non-index addresses pass through.
    let store = |index: Arc<UOp>, value: Arc<UOp>, gate: Option<Arc<UOp>>| {
        UOp::new(Op::Store(ops::Store { index, value, gate }), DType::Void)
    };
    for passthrough in [
        address.store(value.clone()),
        store(address.clone(), value.clone(), Some(UOp::native_const(1i32))),
        store(UOp::native_const(0i32), value, Some(gate(true))),
    ] {
        assert_same!(line_rewrite_cleanups(vec![passthrough.clone()])[0].clone(), passthrough);
    }
}

#[test]
fn tuplize_comparison_survives_a_forty_thousand_deep_chain() {
    // 2 MiB is a typical non-main thread stack; the recursive comparison overflowed even
    // the 8 MiB main stack somewhere past 20k levels.
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let mut low = UOp::const_(DType::Int32, ConstValue::Int(1));
            let mut high = UOp::const_(DType::Int32, ConstValue::Int(2));
            for _ in 0..40_000 {
                low = UOp::new(Op::Precast(ops::Precast { src: low }), DType::Int32);
                high = UOp::new(Op::Precast(ops::Precast { src: high }), DType::Int32);
            }
            let topo = UOp::sink(vec![high.clone(), low.clone()]).toposort();
            let index = node_index(&topo);
            let keys = compute_tuplize(&topo, &index);
            let (low, high) = (position_of(&topo, &low), position_of(&topo, &high));
            assert_eq!(keys.cmp(low, high, &mut FxHashMap::default(), &mut Vec::new()), Ordering::Less);
            let ranks = tuplize_ranks(&topo, &index);
            assert!(ranks[low as usize] < ranks[high as usize], "the ranks must agree with the pairwise verdict");
            // Releasing a 40k-deep Arc chain recurses in drop glue; hold the graph alive.
            std::mem::forget(topo);
        })
        .expect("spawn comparison thread")
        .join()
        .expect("deep tuplize comparison must not overflow the stack");
}

/// The tuplize key as the previous implementation built it: a recursive tree whose derived
/// `Ord` is the lexicographic `(op, arg, dtype, *src)` order.
#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct OracleKey(u16, ArgKey, DTypeKey, Vec<Arc<OracleKey>>);

/// One `op 61` lane of a VCONST key, as the pinned Tinygrad builds it.
fn oracle_lane(value: &ConstValue, dtype: &DType) -> Arc<OracleKey> {
    Arc::new(OracleKey(61, ArgKey::Const(const_key(*value)), dtype_key(dtype), vec![]))
}

fn oracle_keys(nodes: &[Arc<UOp>]) -> HashMap<u64, Arc<OracleKey>> {
    let mut keys: HashMap<u64, Arc<OracleKey>> = HashMap::new();
    for node in nodes {
        let (arg, dtype, src) = match node.op() {
            Op::VConst(ops::VConst { values }) => {
                let dtype = DType::Scalar(node.dtype().base());
                (ArgKey::None, dtype_key(&dtype), values.iter().map(|value| oracle_lane(value, &dtype)).collect())
            }
            _ => (
                arg_key(node.op()),
                dtype_key(&node.dtype()),
                node.op().sources().iter().map(|child| keys[&child.id].clone()).collect(),
            ),
        };
        keys.insert(node.id, Arc::new(OracleKey(op_value(node.op()), arg, dtype, src)));
    }
    keys
}

/// Dense ranks the oracle assigns: position in a stable sort on whole keys.
fn oracle_ranks(nodes: &[Arc<UOp>]) -> Vec<u32> {
    let keys = oracle_keys(nodes);
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by(|&a, &b| keys[&nodes[a].id].cmp(&keys[&nodes[b].id]));
    let mut ranks = vec![0; nodes.len()];
    order.iter().enumerate().for_each(|(position, &node)| ranks[node] = position as u32);
    ranks
}

/// The previous `linearize`: a stable sort on whole keys followed by the same heap toposort.
fn oracle_linearize(sink: &Arc<UOp>) -> Vec<Arc<UOp>> {
    let lst = sink.toposort();
    let mut out_degree: HashMap<u64, usize> = HashMap::new();
    for source in lst.iter().flat_map(|u| u.op().sources()) {
        *out_degree.entry(source.id).or_default() += 1;
    }
    let priorities: HashMap<u64, (u64, i32, Option<i64>)> =
        lst.iter().map(|u| (u.id, (run_count(u), priority(u).0, priority(u).1))).collect();
    let keys = oracle_keys(&lst);
    let mut sorted: Vec<u64> = lst.iter().map(|u| u.id).collect();
    sorted.sort_by(|a, b| priorities[a].cmp(&priorities[b]).then_with(|| keys[a].cmp(&keys[b])));
    let nkey: HashMap<u64, usize> = sorted.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let by_id: HashMap<u64, Arc<UOp>> = lst.iter().map(|u| (u.id, u.clone())).collect();
    let (mut heap, mut visited, mut out) =
        (BinaryHeap::from([(nkey[&sink.id], sink.id)]), std::collections::HashSet::new(), Vec::new());
    while let Some((_, id)) = heap.pop() {
        if !visited.insert(id) {
            continue;
        }
        let u = &by_id[&id];
        out.push(u.clone());
        for v in u.op().sources() {
            let deg = out_degree.entry(v.id).or_default();
            *deg = deg.saturating_sub(1);
            if *deg == 0 && !visited.contains(&v.id) {
                heap.push((nkey[&v.id], v.id));
            }
        }
    }
    out.reverse();
    out
}

#[track_caller]
fn assert_matches_pairwise_oracle(sink: &Arc<UOp>) {
    let topo = sink.toposort();
    assert_eq!(tuplize_ranks(&topo, &node_index(&topo)), oracle_ranks(&topo));
    let (expected, actual) = (oracle_linearize(sink), linearize(sink.clone()));
    assert_eq!(actual.len(), expected.len(), "the linearizations have different lengths");
    assert!(actual.iter().zip(&expected).all(|(a, b)| Arc::ptr_eq(a, b)), "diverged from the oracle");
}

/// A shared-source diamond inside a loop, sunk beside a stored load.
fn kernel_like_graph() -> Arc<UOp> {
    let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let left = shared.try_add(&UOp::const_(DType::Float32, ConstValue::Float(2.0))).unwrap();
    let right = shared.try_add(&UOp::const_(DType::Float32, ConstValue::Float(3.0))).unwrap();
    let range = UOp::range_const(10, 0);
    let looped = left.try_add(&right).unwrap().end(smallvec![range.clone()]);
    let input = UOp::param(0, 16, DType::Float32, None);
    let output = UOp::param(1, 16, DType::Float32, None);
    let load = UOp::load().index(UOp::index().buffer(input).indices(vec![range.clone()]).call().unwrap()).call();
    let store_index = UOp::index().buffer(output).indices(vec![range]).call().unwrap();
    let store = store_index.store(load.try_mul(&UOp::const_(DType::Float32, ConstValue::Float(2.0))).unwrap());
    UOp::sink(vec![looped, store])
}

/// VCONST lanes compare as STACK(CONST...) sources, and two VCONSTs differ only in lane order.
fn vconst_graph() -> Arc<UOp> {
    let ascending = UOp::vconst(vec![ConstValue::Int(1), ConstValue::Int(2)], DType::WeakInt);
    let descending = UOp::vconst(vec![ConstValue::Int(2), ConstValue::Int(1)], DType::WeakInt);
    let stack = UOp::stack(smallvec![UOp::range_const(8, 0), UOp::special(UOp::index_const(8), "gidx0".to_string())]);
    UOp::sink(vec![descending, stack, ascending])
}

/// Distinct UOps with identical keys: `Index` and `Int64` share a dtype key, so their
/// constants and PRECAST chains tie and must keep toposort order.
fn tied_keys_graph() -> Arc<UOp> {
    let index = UOp::const_(DType::Index, ConstValue::Int(1));
    let long = UOp::const_(DType::Int64, ConstValue::Int(1));
    let index_chain = UOp::new(Op::Precast(ops::Precast { src: index.clone() }), DType::Index);
    let long_chain = UOp::new(Op::Precast(ops::Precast { src: long.clone() }), DType::Int64);
    UOp::sink(vec![long_chain, index_chain, UOp::const_(DType::Int64, ConstValue::Int(2)), index, long])
}

#[test_case(kernel_like_graph; "a kernel-like graph")]
#[test_case(vconst_graph; "vconst lanes")]
#[test_case(tied_keys_graph; "tied keys in toposort order")]
fn linearize_is_topological_and_matches_the_pairwise_oracle(graph: fn() -> Arc<UOp>) {
    let sink = graph();
    assert_linearize_is_topological(&sink);
    assert_matches_pairwise_oracle(&sink);
}

proptest! {
    #![proptest_config(equivalence())]
    #[test]
    fn ranked_linearize_matches_the_pairwise_oracle_on_arithmetic_forests(
        trees in proptest::collection::vec(arb_arithmetic_tree_up_to(DType::Int32, 4), 1..4)
    ) {
        let sink = UOp::sink(trees);
        prop_assert!(linearize(sink.clone()).last().is_some_and(|last| Arc::ptr_eq(last, &sink)), "the SINK is emitted last");
        assert_linearize_is_topological(&sink);
        assert_matches_pairwise_oracle(&sink);
    }
}
