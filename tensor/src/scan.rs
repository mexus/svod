//! Schedule-level scan loops: one compiled step kernel, launched `T` times.
//!
//! A [`ScanVar`] is an ordinary symbolic [`Variable`](crate::Variable) that a
//! builder uses *only* as an additive index offset (`gx.narrow(0, t, 1)`), so
//! every axis extent it touches stays a constant and the optimizer keeps
//! upcasting and vectorizing the step. After rangeify, [`wrap_scan_loops`]
//! turns the kernels that read the variable into the body of the schedule-level
//! loop `RANGE → CALL … CALL → END(CALL, [RANGE])`, which `create_pre_schedule`
//! replays once per slot with the counter bound into the kernel's arguments.
//!
//! The rewrite is deliberately post-rangeify: rangeify is index-functional and
//! would never synthesize a recurrence, but it happily indexes a symbolic
//! offset, so the step is compiled exactly once and re-launched.
//!
//! Names are canonicalized twice so identical scans hash identically: the
//! schedule-cache normalization renumbers the per-instance names in graph
//! order ([`canonical_scan_names`]), and [`wrap_scan_loops`] gives every step
//! body the one name [`BODY_SCAN_NAME`], so two scans whose bodies differ only
//! in their counter — the layers of a stack, say — are one compiled program.
//! The three spellings (`__scan{N}` per instance, `__scan#{slot}` per graph,
//! `__scan` per body) never coincide, so each rename is a plain substitution
//! with no chance of one counter taking another's name.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::smallvec;
use svod_ir::{AxisId, AxisType, Op, ParamArg, SInt, UOp, UOpKey, ops};

use crate::Variable;
use crate::error::IrConstructionSnafu;

/// Name prefix that marks a variable as a loop counter rather than a runtime
/// input. Everything the rewrite needs is in the variable itself, so there is
/// no side table to keep in sync.
const SCAN_PREFIX: &str = "__scan";

/// The name every step body reads its counter under after [`wrap_scan_loops`].
pub(crate) const BODY_SCAN_NAME: &str = "__scan";

static SCAN_SEQ: AtomicUsize = AtomicUsize::new(0);

/// A loop counter for a schedule-level scan.
///
/// The name is unique per instance: two scans sharing one name would be merged
/// into a single loop, which is only correct when their bodies are mutually
/// independent.
#[derive(Clone, Debug)]
pub(crate) struct ScanVar {
    var: Variable,
}

impl ScanVar {
    pub(crate) fn new(trips: usize) -> Self {
        assert!(trips > 0, "scan trip count must be positive");
        let name = format!("{SCAN_PREFIX}{}", SCAN_SEQ.fetch_add(1, Ordering::Relaxed));
        Self { var: Variable::new(&name, 0, trips as i64 - 1) }
    }

    /// The counter as an index offset.
    pub(crate) fn index(&self) -> SInt {
        self.var.as_sint()
    }
}

fn scan_var_name(node: &Arc<UOp>) -> Option<&str> {
    let Op::Param(ops::Param { arg, .. }) = node.op() else { return None };
    arg.name.as_deref().filter(|name| arg.addrspace.is_none() && name.starts_with(SCAN_PREFIX))
}

fn renamed(node: &Arc<UOp>, name: String) -> Arc<UOp> {
    let Op::Param(ops::Param { shape, arg }) = node.op() else { unreachable!("scan variables are scalar PARAMs") };
    let arg = ParamArg { name: Some(name), ..(**arg).clone() };
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: Box::new(arg) }), node.dtype())
}

/// Renumber scan variables in graph order so two builds of one model hash to
/// the same schedule-cache key.
pub(crate) fn canonical_scan_names(root: &Arc<UOp>) -> Arc<UOp> {
    let mut substitutions: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for node in root.toposort_call_aware(false) {
        if scan_var_name(&node).is_some() {
            let canonical = format!("{SCAN_PREFIX}#{}", substitutions.len());
            substitutions.insert(UOpKey(node.clone()), renamed(&node, canonical));
        }
    }
    root.substitute(&substitutions)
}

/// The scan variable a kernel body reads, with its trip count.
///
/// The trip count is the variable's own upper bound — a counter over `[0, T)`
/// is declared as `vmin = 0, vmax = T - 1`. A body may read at most one: the
/// schedule loop binds one counter per CALL, and a second would silently keep
/// its default at prepare.
fn body_scan_var(body: &Arc<UOp>) -> crate::Result<Option<(Arc<UOp>, i64)>> {
    let scans: Vec<_> = svod_runtime::execution_plan::collect_runtime_vars(body)
        .into_iter()
        .filter(|var| var.name.starts_with(SCAN_PREFIX))
        .collect();
    let [scan] = scans.as_slice() else {
        snafu::ensure!(
            scans.is_empty(),
            IrConstructionSnafu {
                details: format!("kernel body reads {} scan variables; a step may read at most one", scans.len())
            }
        );
        return Ok(None);
    };
    let node = body
        .toposort()
        .into_iter()
        .find(|node| scan_var_name(node) == Some(scan.name.as_str()))
        .expect("collect_runtime_vars saw the PARAM");
    Ok(Some((node, scan.max_val + 1)))
}

struct ScanLoop {
    var: Arc<UOp>,
    trips: i64,
    calls: Vec<Arc<UOp>>,
}

/// Wrap every kernel that reads a scan variable in a schedule-level loop.
///
/// Returns the graph unchanged when no kernel reads one, which is every graph
/// that has no recurrence in it.
pub(crate) fn wrap_scan_loops(root: Arc<UOp>) -> crate::Result<Arc<UOp>> {
    // `toposort_call_aware` is topological, so the last member of a group is
    // the one every other member precedes — the only sound place for the END.
    let mut loops: Vec<ScanLoop> = Vec::new();
    for node in root.toposort_call_aware(false) {
        let Op::Call(ops::Call { body, .. }) = node.op() else { continue };
        let Some((var, trips)) = body_scan_var(body)? else { continue };
        match loops.iter_mut().find(|l| Arc::ptr_eq(&l.var, &var)) {
            Some(l) => l.calls.push(node),
            None => loops.push(ScanLoop { var, trips, calls: vec![node] }),
        }
    }

    let mut substitutions: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    for (axis, ScanLoop { var, trips, calls }) in loops.into_iter().enumerate() {
        let range = UOp::range_axis(UOp::index_const(trips), AxisId::Renumbered(axis), AxisType::Loop);
        let body_var = renamed(&var, BODY_SCAN_NAME.to_string());
        let bind = body_var.bind(range.clone());
        let rename = HashMap::from([(UOpKey(var), body_var)]);

        let last = calls.len() - 1;
        for (i, call) in calls.into_iter().enumerate() {
            let Op::Call(ops::Call { body, args, info }) = call.op() else { unreachable!("collected from CALLs") };
            let mut args = args.clone();
            args.push(bind.clone());
            let looped = body.substitute(&rename).call(args, (**info).clone());
            // The END closes the loop after the body's last kernel, and takes
            // the original CALL's place in the graph so it stays reachable
            // without becoming a SINK output.
            let replacement = if i == last { looped.end(smallvec![range.clone()]) } else { looped };
            substitutions.insert(UOpKey(call), replacement);
        }
    }
    Ok(root.substitute(&substitutions))
}
