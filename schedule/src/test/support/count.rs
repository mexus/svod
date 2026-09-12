//! Distinct-node counting. Every count here is over [`UOp::toposort`], so a shared
//! subexpression is counted once regardless of how many paths reach it.

use std::sync::Arc;

use svod_dtype::AddrSpace;
use svod_ir::{AxisType, Op, UOp, ops};

use crate::optimizer::Scheduler;

pub fn count(uop: &Arc<UOp>, pred: impl Fn(&Arc<UOp>) -> bool) -> usize {
    uop.toposort().iter().filter(|node| pred(node)).count()
}

/// Distinct-node tallies for the node kinds schedulers care about.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub loads: usize,
    pub stores: usize,
    pub calls: usize,
    pub ends: usize,
    pub stages: usize,
    pub ranges: usize,
    pub params: usize,
    pub locals: usize,
    pub regs: usize,
}

pub fn count_kinds(uop: &Arc<UOp>) -> OpCounts {
    let mut counts = OpCounts::default();
    for node in uop.toposort() {
        match node.op() {
            Op::Load(..) => counts.loads += 1,
            Op::Store(..) => counts.stores += 1,
            Op::Call(..) => counts.calls += 1,
            Op::End(..) => counts.ends += 1,
            Op::Stage(..) => counts.stages += 1,
            Op::Range(..) => counts.ranges += 1,
            Op::Param(..) => counts.params += 1,
            Op::Buffer(ops::Buffer { arg, .. }) => match arg.addrspace {
                Some(AddrSpace::Local) => counts.locals += 1,
                Some(AddrSpace::Reg) => counts.regs += 1,
                _ => {}
            },
            _ => {}
        }
    }
    counts
}

pub fn kernels(uop: &Arc<UOp>) -> usize {
    count(uop, |node| matches!(node.op(), Op::Call(..)))
}

/// The first `CALL` in topological order.
pub fn first_call(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    uop.toposort().into_iter().find(|node| matches!(node.op(), Op::Call(..)))
}

pub fn axis_count(s: &Scheduler, axis: AxisType) -> usize {
    s.axes_of(&[axis]).len()
}
