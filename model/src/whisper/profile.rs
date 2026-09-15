//! Internal Whisper profiling: graph executions and the copies around them.
//!
//! Both recorders carry an `enabled` flag so a call site is one line whether or
//! not a profile is being collected: disabled, they run the work and record
//! nothing.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use svod_device::Buffer;
use svod_runtime::{KernelProfile, StageProfile};

type DeviceError = svod_device::error::Error;

#[derive(Debug, Default)]
pub(crate) struct GraphProfile {
    enabled: bool,
    pub(crate) wall: Duration,
    pub(crate) executions: usize,
    pub(crate) kernels: Vec<KernelProfile>,
}

impl GraphProfile {
    pub(crate) fn new(enabled: bool) -> Self {
        Self { enabled, ..Self::default() }
    }

    /// Run a prepared graph. Profiling swaps the plain execution for the
    /// instrumented one and charges its synchronized wall to this stage; the
    /// instrumented closure must wait for the graph's output before returning.
    pub(crate) fn execute<J, E>(
        &mut self,
        jit: &mut J,
        run: impl FnOnce(&mut J) -> Result<(), E>,
        profiled: impl FnOnce(&mut J) -> Result<Vec<KernelProfile>, E>,
    ) -> Result<(), E> {
        if !self.enabled {
            return run(jit);
        }
        let started = Instant::now();
        let kernels = profiled(jit)?;
        self.record(started.elapsed(), kernels);
        Ok(())
    }

    pub(crate) fn record(&mut self, wall: Duration, kernels: Vec<KernelProfile>) {
        self.wall = self.wall.saturating_add(wall);
        self.executions = self.executions.saturating_add(1);
        self.kernels.extend(kernels);
    }

    /// Every numeric entry is a plain counter so per-window profiles sum when
    /// the pipeline merges them.
    pub(crate) fn stage(self, name: &str) -> StageProfile {
        let kernel_dispatches = self.kernels.len();
        let mut stage = StageProfile::gpu(name, self.wall, self.kernels);
        stage.meta.insert("executions".into(), self.executions.to_string());
        stage.meta.insert("kernel_dispatches".into(), kernel_dispatches.to_string());
        stage.meta.insert("accumulated_wall_ms".into(), format!("{:.3}", self.wall.as_secs_f64() * 1e3));
        stage.meta.insert(
            "timing_semantics".into(),
            "accumulated host wall per execution from profiled submission through explicit output synchronization"
                .into(),
        );
        stage
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CopyStats {
    ops: usize,
    bytes: usize,
    wall: Duration,
}

impl CopyStats {
    fn add(&mut self, ops: usize, bytes: usize, wall: Duration) {
        self.ops = self.ops.saturating_add(ops);
        self.bytes = self.bytes.saturating_add(bytes);
        self.wall = self.wall.saturating_add(wall);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CopyCategory {
    total: CopyStats,
    breakdown: BTreeMap<&'static str, CopyStats>,
}

impl CopyCategory {
    fn record(&mut self, name: &'static str, ops: usize, bytes: usize, wall: Duration) {
        self.total.add(ops, bytes, wall);
        self.breakdown.entry(name).or_default().add(ops, bytes, wall);
    }

    fn stage(&self, name: &str, semantics: &str) -> Option<StageProfile> {
        (self.total.bytes != 0).then(|| {
            let mut stage = StageProfile::host(name, self.total.wall);
            stage.meta.insert("ops".into(), self.total.ops.to_string());
            stage.meta.insert("bytes".into(), self.total.bytes.to_string());
            stage.meta.insert("timing_semantics".into(), semantics.into());
            for (breakdown, stats) in &self.breakdown {
                stage.meta.insert(format!("{breakdown}_ops"), stats.ops.to_string());
                stage.meta.insert(format!("{breakdown}_bytes"), stats.bytes.to_string());
                stage.meta.insert(format!("{breakdown}_wall_ms"), format!("{:.3}", stats.wall.as_secs_f64() * 1e3));
            }
            stage
        })
    }
}

/// Synchronized host wall around groups of transfers, by direction. These are
/// not DMA timestamps: a fence drains prior graph work before the clock starts
/// so the producing graph is not charged to the copy, and a second fence
/// after device-to-device groups waits for the whole asynchronous group.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CopyProfile {
    enabled: bool,
    h2d: CopyCategory,
    d2d: CopyCategory,
    d2h: CopyCategory,
}

impl CopyProfile {
    pub(crate) fn new(enabled: bool) -> Self {
        Self { enabled, ..Self::default() }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    fn begin<E: From<DeviceError>>(&self, fence: &Buffer) -> Result<Option<Instant>, E> {
        if !self.enabled {
            return Ok(None);
        }
        fence.synchronize()?;
        Ok(Some(Instant::now()))
    }

    pub(crate) fn h2d<T, E: From<DeviceError>>(
        &mut self,
        name: &'static str,
        ops: usize,
        bytes: usize,
        fence: &Buffer,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let Some(started) = self.begin(fence)? else { return work() };
        let value = work()?;
        self.h2d.record(name, ops, bytes, started.elapsed());
        Ok(value)
    }

    pub(crate) fn d2h<T, E: From<DeviceError>>(
        &mut self,
        name: &'static str,
        ops: usize,
        bytes: usize,
        fence: &Buffer,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let Some(started) = self.begin(fence)? else { return work() };
        let value = work()?;
        self.d2h.record(name, ops, bytes, started.elapsed());
        Ok(value)
    }

    pub(crate) fn d2d<T, E: From<DeviceError>>(
        &mut self,
        name: &'static str,
        ops: usize,
        bytes: usize,
        fence: &Buffer,
        work: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let Some(started) = self.begin(fence)? else { return work() };
        let value = work()?;
        fence.synchronize()?;
        self.d2d.record(name, ops, bytes, started.elapsed());
        Ok(value)
    }

    pub(crate) fn stages(&self) -> impl Iterator<Item = StageProfile> + '_ {
        [
            self.h2d.stage("copy_h2d", "prior device work fenced before host-visible writes; synchronized host wall"),
            self.d2d.stage("copy_d2d", "device synchronized before and after each transfer group; host wall"),
            self.d2h.stage("copy_d2h", "producer work fenced before host-visible reads; synchronized host wall"),
        ]
        .into_iter()
        .flatten()
    }
}
