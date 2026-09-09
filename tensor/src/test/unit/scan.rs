//! Schedule-level scan loops: cache identity, structural loop membership and
//! the kernel-graph grammar the loop rewrite installs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use smallvec::smallvec;
use svod_device::Buffer;
use svod_ir::{AxisId, AxisType, CallInfo, DType, DeviceSpec, Op, UOp, ops};
use svod_runtime::ExecutionPlan;
use test_case::test_case;

use crate::nn::RnnDirection;
use crate::scan::{BODY_SCAN_NAME, ScanVar, canonical_scan_names, wrap_scan_loops};
use crate::schedule::{InputBuffers, ScheduleItem, create_schedule};
use crate::{PrepareConfig, Tensor};

const T: usize = 8;
const B: usize = 2;
const I: usize = 16;
const H: usize = 16;

fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect()
}

fn cfg() -> PrepareConfig {
    PrepareConfig::from(svod_schedule::OptimizerConfig::default())
}

fn input(seed: f32) -> Tensor {
    Tensor::from_slice(seq(T * B * I, seed)).try_reshape([T as isize, B as isize, I as isize]).unwrap()
}

fn weight(rows: usize, cols: usize, seed: f32) -> Tensor {
    Tensor::from_slice(seq(rows * cols, seed)).try_reshape([rows as isize, cols as isize]).unwrap()
}

fn gru(x: &Tensor, direction: RnnDirection, seed: f32) -> Tensor {
    let d = direction.num_directions();
    x.gru()
        .weight_ih(&weight(d * 3 * H, I, seed))
        .weight_hh(&weight(d * 3 * H, H, seed + 0.06))
        .direction(direction)
        .call()
        .unwrap()
        .output
}

fn lstm(x: &Tensor, direction: RnnDirection, seed: f32) -> Tensor {
    let d = direction.num_directions();
    x.lstm()
        .weight_ih(&weight(d * 4 * H, I, seed))
        .weight_hh(&weight(d * 4 * H, H, seed + 0.06))
        .direction(direction)
        .call()
        .unwrap()
        .output
}

fn in_loop(kernel: &svod_runtime::PreparedKernel) -> bool {
    kernel.fixedvars.contains_key(BODY_SCAN_NAME)
}

/// Distinct compiled programs behind the loop kernels.
fn step_programs(plan: &ExecutionPlan) -> usize {
    plan.prepared_kernels()
        .into_iter()
        .filter(|k| in_loop(k))
        .map(|k| Arc::as_ptr(&k.kernel))
        .collect::<HashSet<_>>()
        .len()
}

/// Launches per kernel id, split into loop members and everything else.
fn launches(plan: &ExecutionPlan) -> (BTreeMap<u64, usize>, BTreeMap<u64, usize>) {
    let (mut members, mut others) = (BTreeMap::new(), BTreeMap::new());
    for kernel in plan.prepared_kernels() {
        let counts = if in_loop(kernel) { &mut members } else { &mut others };
        *counts.entry(kernel.id).or_insert(0) += 1;
    }
    (members, others)
}

fn kernel_graph(out: &Tensor) -> Arc<UOp> {
    let sink = UOp::sink(vec![out.uop().contiguous()]);
    let normalized = crate::realize::normalize_for_schedule_cache(&sink).unwrap().normalized;
    let rangeified = svod_schedule::rangeify_with_map(normalized).unwrap();
    svod_schedule::try_get_kernel_graph(rangeified.sink).unwrap().0
}

// =========================================================================
// Cache identity
// =========================================================================

#[test]
fn identical_scans_share_the_schedule_cache_entry() {
    let _structural = svod_ir::origin::capture_for_thread(false);
    let cfg = cfg();
    let first = gru(&input(0.31), RnnDirection::Forward, 0.17);
    let second = gru(&input(0.37), RnnDirection::Forward, 0.19);

    let key = crate::schedule_cache::cache_key_for(&first, &cfg).unwrap();
    assert_eq!(key, crate::schedule_cache::cache_key_for(&second, &cfg).unwrap());

    first.prepare_with(&cfg).unwrap();
    let cache = crate::schedule_cache::schedule_cache();
    let entry = {
        let guard = cache.guard();
        cache.get(&key, &guard).cloned().expect("entry after first prepare")
    };
    second.prepare_with(&cfg).unwrap();
    let again = {
        let guard = cache.guard();
        cache.get(&key, &guard).cloned().expect("entry after second prepare")
    };
    assert!(Arc::ptr_eq(&entry, &again), "the second build must hit the schedule cache");
}

/// Two scans in one graph have distinct counters, so their step bodies differ
/// only in the counter's name — which the loop rewrite canonicalizes away.
#[test]
fn scans_of_one_graph_share_the_step_program() {
    let cfg = cfg();
    let single = gru(&input(0.31), RnnDirection::Forward, 0.17).prepare_with(&cfg).unwrap();
    let pair = Tensor::stack(&[&gru(&input(0.31), RnnDirection::Forward, 0.17), &two_scan_sibling()], 0).unwrap();
    let pair = pair.prepare_with(&cfg).unwrap();

    let programs = step_programs(&single);
    assert!(programs >= 1);
    assert_eq!(step_programs(&pair), programs, "the second scan compiled its own copy of the step");
    let (single_members, _) = launches(&single);
    let (pair_members, _) = launches(&pair);
    assert_eq!(pair_members.values().sum::<usize>(), 2 * single_members.values().sum::<usize>());
}

fn two_scan_sibling() -> Tensor {
    gru(&input(0.37), RnnDirection::Forward, 0.19)
}

#[test]
fn stacked_layers_share_the_step_program() {
    let cfg = cfg();
    let single = gru(&input(0.31), RnnDirection::Forward, 0.17).prepare_with(&cfg).unwrap();
    let lower = gru(&input(0.31), RnnDirection::Forward, 0.17);
    let stacked = gru(&lower, RnnDirection::Forward, 0.23).prepare_with(&cfg).unwrap();

    assert_eq!(step_programs(&stacked), step_programs(&single), "the upper layer recompiled the step");
    let (members, _) = launches(&stacked);
    assert_eq!(members.len(), 2 * launches(&single).0.len());
}

// =========================================================================
// Loop membership
// =========================================================================

/// Every loop member launches once per slot; everything else, including the
/// history zero-fill an LSTM carries for its cell state, launches once.
#[test_case(gru as fn(&Tensor, RnnDirection, f32) -> Tensor; "gru step")]
#[test_case(lstm as fn(&Tensor, RnnDirection, f32) -> Tensor; "lstm step")]
fn loop_invariant_kernels_launch_once(build: fn(&Tensor, RnnDirection, f32) -> Tensor) {
    let plan = build(&input(0.31), RnnDirection::Bidirectional, 0.17).prepare_with(&cfg()).unwrap();
    let (members, others) = launches(&plan);
    assert!(!members.is_empty() && !others.is_empty());
    assert!(members.values().all(|&n| n == T), "loop members launch T times: {members:?}");
    assert!(others.values().all(|&n| n == 1), "loop-invariant kernels launch once: {others:?}");
}

fn cpu_buffer(numel: usize) -> Buffer {
    Buffer::new(svod_device::registry::cpu().expect("cpu allocator"), DType::Float32, vec![numel], Default::default())
}

/// A three-trip schedule loop over a hand-built kernel graph.
struct Loop {
    range: Arc<UOp>,
    bind: Arc<UOp>,
    inputs: InputBuffers,
}

impl Loop {
    fn new() -> Self {
        let range = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(0), AxisType::Loop);
        let bind = UOp::define_var("t".to_string(), 0, 2).bind(range.clone());
        Self { range, bind, inputs: InputBuffers::new() }
    }

    fn buffer(&mut self) -> Arc<UOp> {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
        self.inputs.insert(buffer.id, cpu_buffer(4));
        buffer
    }

    /// A CALL whose body is distinguished by `tag`, reading `args`.
    fn call(&self, tag: f32, args: Vec<Arc<UOp>>) -> Arc<UOp> {
        UOp::sink(vec![UOp::native_const(tag)]).call(args.into(), CallInfo::default())
    }

    fn member(&self, tag: f32, args: Vec<Arc<UOp>>) -> Arc<UOp> {
        self.call(tag, [vec![self.bind.clone()], args].concat())
    }

    fn end(&self, call: &Arc<UOp>) -> Arc<UOp> {
        call.end(smallvec![self.range.clone()])
    }

    fn schedule(&self, roots: Vec<Arc<UOp>>) -> crate::Result<Vec<ScheduleItem>> {
        create_schedule(UOp::sink(roots), &self.inputs, &HashMap::new()).map(|result| result.items)
    }

    fn rejection(&self, roots: Vec<Arc<UOp>>) -> String {
        let err = match self.schedule(roots) {
            Ok(_) => panic!("the schedule must be rejected"),
            Err(err) => err,
        };
        match err.kind() {
            crate::ErrorKind::IrConstruction { details } => details.clone(),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

fn trips(items: &[ScheduleItem]) -> Vec<Option<i64>> {
    items.iter().map(|item| item.fixedvars.get("t").copied()).collect()
}

#[test]
fn a_loop_invariant_kernel_inside_the_span_runs_once() {
    let mut l = Loop::new();
    let history = l.buffer();
    let fill = l.call(1.0, vec![history.clone()]);
    // The BIND comes first so the topological order places the fill after
    // the RANGE — inside the span, where a positional replay would repeat it.
    let step = l.member(2.0, vec![history.after(smallvec![fill.clone()])]);
    let items = l.schedule(vec![fill.clone(), l.end(&step)]).unwrap();

    let ids: Vec<u64> = items.iter().map(|item| item.kernel.id).collect();
    assert_eq!(ids, vec![fill.id, step.id, step.id, step.id]);
    assert_eq!(trips(&items), vec![None, Some(0), Some(1), Some(2)]);
    let chain: Vec<&[usize]> = items.iter().map(|item| item.instance_dependencies.as_slice()).collect();
    assert_eq!(chain, vec![&[][..], &[], &[1], &[2]], "each trip waits for the previous one");
}

#[test]
fn a_consumer_of_the_loop_output_runs_after_the_last_trip() {
    let mut l = Loop::new();
    let history = l.buffer();
    let output = l.buffer();
    let step = l.member(1.0, vec![history.clone()]);
    let consumer = l.call(2.0, vec![output, history.after(smallvec![step.clone()])]);
    // The consumer is a SINK output ahead of the END, so it precedes the END
    // in topological order and must wait for the loop regardless.
    let items = l.schedule(vec![consumer.clone(), l.end(&step)]).unwrap();

    let ids: Vec<u64> = items.iter().map(|item| item.kernel.id).collect();
    assert_eq!(ids, vec![step.id, step.id, step.id, consumer.id]);
    assert_eq!(items[3].dependencies, vec![step.id]);
}

#[test]
fn independent_members_chain_trip_to_trip() {
    let mut l = Loop::new();
    let (a_buf, b_buf) = (l.buffer(), l.buffer());
    let a = l.member(1.0, vec![a_buf]);
    let b = l.member(2.0, vec![b_buf]);
    let items = l.schedule(vec![a.clone(), l.end(&b)]).unwrap();

    let ids: Vec<u64> = items.iter().map(|item| item.kernel.id).collect();
    assert_eq!(ids, vec![a.id, b.id, a.id, b.id, a.id, b.id]);
    assert_eq!(trips(&items), vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]);
    let chain: Vec<&[usize]> = items.iter().map(|item| item.instance_dependencies.as_slice()).collect();
    assert_eq!(chain, vec![&[][..], &[], &[0, 1], &[0, 1], &[2, 3], &[2, 3]]);
}

#[test]
fn a_member_after_the_end_is_rejected() {
    let mut l = Loop::new();
    let history = l.buffer();
    let a = l.member(1.0, vec![history.clone()]);
    let end = l.end(&a);
    let b = l.member(2.0, vec![history.after(smallvec![end.clone()])]);
    let details = l.rejection(vec![end, b]);
    assert!(details.contains("outside the range's RANGE…END span"), "{details}");
}

#[test]
fn a_body_kernel_reading_the_loop_output_is_rejected() {
    let mut l = Loop::new();
    let (history, scratch) = (l.buffer(), l.buffer());
    let a = l.member(1.0, vec![history.clone()]);
    let outside = l.call(2.0, vec![scratch.clone(), history.after(smallvec![a.clone()])]);
    let b = l.member(3.0, vec![scratch.after(smallvec![outside.clone()])]);
    let details = l.rejection(vec![outside, l.end(&b)]);
    assert!(details.contains("reads the loop's output"), "{details}");
}

// =========================================================================
// The rewrite itself
// =========================================================================

#[test]
fn canonical_names_follow_graph_order_and_hash_alike() {
    let x = input(0.31);
    let build = || {
        let (first, second) = (ScanVar::new(T), ScanVar::new(T));
        let a = x.narrow(0, first.index(), 1usize).unwrap();
        let b = x.narrow(0, second.index(), 1usize).unwrap();
        UOp::sink(vec![(&a + &b).unwrap().uop().contiguous()])
    };
    let names = |sink: &Arc<UOp>| -> Vec<String> {
        sink.toposort()
            .iter()
            .filter_map(|node| match node.op() {
                Op::Param(ops::Param { arg, .. }) if arg.addrspace.is_none() => arg.name.clone(),
                _ => None,
            })
            .collect()
    };
    let (one, two) = (canonical_scan_names(&build()), canonical_scan_names(&build()));
    assert_eq!(names(&one), vec!["__scan#0", "__scan#1"]);
    assert_eq!(one.content_hash, two.content_hash, "fresh counters must not leak into the cache key");
}

#[test]
fn a_body_reading_two_scan_variables_is_rejected() {
    let x = input(0.31);
    let (first, second) = (ScanVar::new(T), ScanVar::new(T));
    let a = x.narrow(0, first.index(), 1usize).unwrap();
    let b = x.narrow(0, second.index(), 1usize).unwrap();
    let err = wrap_scan_loops(kernel_graph(&(&a + &b).unwrap())).expect_err("two counters in one body");
    let crate::ErrorKind::IrConstruction { details } = err.kind() else { panic!("unexpected error: {err:?}") };
    assert!(details.contains("reads 2 scan variables"), "{details}");
}

/// The loop grammar (RANGE, BIND and END over a CALL) is only present after
/// the rewrite, so this is where the kernel-graph spec sees it.
#[test]
fn wrapped_gru_kernel_graph_passes_the_spec() {
    let pair = Tensor::stack(&[&gru(&input(0.31), RnnDirection::Forward, 0.17), &two_scan_sibling()], 0).unwrap();
    let graph = kernel_graph(&pair);
    let wrapped = wrap_scan_loops(graph.clone()).unwrap();
    assert!(!Arc::ptr_eq(&wrapped, &graph), "the GRU step must be wrapped");

    svod_schedule::spec::verify_kernel_graph(&wrapped).unwrap();

    let mut bodies = Vec::new();
    let mut ends = 0;
    for node in wrapped.toposort_call_aware(false) {
        match node.op() {
            Op::End(ops::End { computation, .. }) if matches!(computation.op(), Op::Call(..)) => ends += 1,
            Op::Call(ops::Call { body, args, .. }) => {
                let Some(bound) = args.iter().find_map(|arg| match arg.op() {
                    Op::Bind(ops::Bind { var, value }) if matches!(value.op(), Op::Range(..)) => Some(var),
                    _ => None,
                }) else {
                    continue;
                };
                let Op::Param(ops::Param { arg, .. }) = bound.op() else { panic!("BIND over a scalar PARAM") };
                assert_eq!(arg.name.as_deref(), Some(BODY_SCAN_NAME));
                bodies.push(body.content_hash);
            }
            _ => {}
        }
    }
    let distinct: HashSet<u64> = bodies.iter().copied().collect();
    assert_eq!(ends, 2, "one END per scan");
    assert_eq!(bodies.len(), 2 * distinct.len(), "the two scans must share their step bodies");
}
