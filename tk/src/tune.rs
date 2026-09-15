//! First-use tuning of the hand kernels' tile tables.
//!
//! A kernel's per-family table is a search space, not an answer: the first time
//! a device meets a shape, every candidate that fits it is compiled and timed
//! once, serially, on synthetic operands of that shape, and the winner is kept
//! — in this process, and on disk so the next process starts tuned. A device
//! whose family has no measured table thus gets a correct kernel at once and a
//! tuned one after one run. Measurement is skipped, and the table's static
//! choice used, when `SVOD_TK_TUNE=0`, when the device stamps no timings, or
//! when every candidate fails to build; nothing unmeasured is ever cached.
//!
//! The store is one line per entry (`key index ns`) in
//! `$SVOD_TK_TUNE_DIR` (else `$XDG_CACHE_HOME/svod/tk_tune`, else
//! `$HOME/.cache/svod/tk_tune`), one file per device; an unreadable or
//! unwritable store is a miss, never an error.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use svod_dtype::{DeviceSpec, GpuArch};

/// What a measurement is keyed by: the device (arch and compute units), the
/// kernel, its shape, and the candidate list (so a changed table re-measures).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TuneKey {
    pub kernel: &'static str,
    pub device: String,
    pub shape: Vec<usize>,
    pub candidates: u64,
}

impl TuneKey {
    /// The key for `kernel` at `shape` on the device behind `spec`, over the
    /// candidate list `candidates` (hashed through its `Debug` form).
    pub fn new<C: std::fmt::Debug>(
        kernel: &'static str,
        spec: &DeviceSpec,
        arch: GpuArch,
        shape: &[usize],
        candidates: &[C],
    ) -> Self {
        let mut hasher = std::hash::DefaultHasher::new();
        format!("{candidates:?}").hash(&mut hasher);
        let units = crate::target::compute_units(spec).unwrap_or(0);
        Self {
            kernel,
            device: format!("{}-{units}cu", arch.target_name()),
            shape: shape.to_vec(),
            candidates: hasher.finish(),
        }
    }

    fn line(&self) -> String {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        format!("{}|{}|{}|{:016x}", self.kernel, self.device, shape.join("x"), self.candidates)
    }
}

/// The on-disk store: one file per device under `root`.
#[derive(Clone, Debug)]
pub struct TuneStore {
    root: Option<PathBuf>,
}

impl TuneStore {
    /// The store rooted at `root` (`None`: memory only).
    pub fn at(root: Option<PathBuf>) -> Self {
        Self { root }
    }

    /// The process-wide store per the environment (see the module docs).
    pub fn global() -> &'static Self {
        static STORE: OnceLock<TuneStore> = OnceLock::new();
        STORE.get_or_init(|| {
            let root = if let Some(dir) = std::env::var_os("SVOD_TK_TUNE_DIR") {
                Some(PathBuf::from(dir))
            } else if let Some(cache) = std::env::var_os("XDG_CACHE_HOME") {
                Some(PathBuf::from(cache).join("svod/tk_tune"))
            } else {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/svod/tk_tune"))
            };
            Self::at(root.filter(|dir| std::fs::create_dir_all(dir).is_ok()))
        })
    }

    fn path(&self, key: &TuneKey) -> Option<PathBuf> {
        let name: String = key.device.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        self.root.as_ref().map(|root| root.join(format!("{name}.txt")))
    }

    /// Every `key line -> (index, ns)` the device's file holds; empty when unreadable.
    fn read(&self, key: &TuneKey) -> HashMap<String, (usize, u64)> {
        let Some(text) = self.path(key).and_then(|p| std::fs::read_to_string(p).ok()) else { return HashMap::new() };
        text.lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let (line, index) = (fields.next()?.to_string(), fields.next()?.parse().ok()?);
                Some((line, (index, fields.next()?.parse().ok()?)))
            })
            .collect()
    }

    fn get(&self, key: &TuneKey) -> Option<usize> {
        self.read(key).remove(&key.line()).map(|(index, _)| index)
    }

    /// Record `index` (measured at `ns`) for `key`: re-read, merge, and replace
    /// the file atomically, so concurrent writers lose at most each other's
    /// newest line, never the file.
    fn put(&self, key: &TuneKey, index: usize, ns: u64) {
        let Some(path) = self.path(key) else { return };
        let mut entries = self.read(key);
        entries.insert(key.line(), (index, ns));
        let mut lines: Vec<String> = entries.iter().map(|(line, (i, ns))| format!("{line} {i} {ns}")).collect();
        lines.sort();
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, lines.join("\n") + "\n").is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// The winning candidate index for `key` among `count` candidates: this
    /// process's memo, then the store, else each candidate is measured once
    /// through `measure` (its minimum device time in ns; `None` when it cannot
    /// run) and the fastest is kept. `None` when nothing measured — the caller
    /// keeps its static choice.
    pub fn select(&self, key: &TuneKey, count: usize, mut measure: impl FnMut(usize) -> Option<u64>) -> Option<usize> {
        static MEMO: OnceLock<Mutex<HashMap<TuneKey, usize>>> = OnceLock::new();
        let memo = MEMO.get_or_init(Default::default);
        if let Some(i) = memo.lock().expect("tune memo").get(key) {
            return Some(*i);
        }
        let cached = self.get(key).filter(|i| *i < count);
        let chosen = cached.or_else(|| {
            let best = (0..count).filter_map(|i| measure(i).map(|ns| (ns, i))).min()?;
            self.put(key, best.1, best.0);
            Some(best.1)
        })?;
        memo.lock().expect("tune memo").insert(key.clone(), chosen);
        Some(chosen)
    }
}

/// Whether first-use measurement is on (`SVOD_TK_TUNE` is not `0`).
pub fn enabled() -> bool {
    std::env::var("SVOD_TK_TUNE").map(|v| v != "0").unwrap_or(true)
}

/// The minimum device time (ns) of `launch` over `runs` dispatches after one
/// warm-up; `None` where the device stamps no timings or a dispatch fails.
pub fn min_dispatch_ns(launch: &crate::launch::CompiledLaunch, runs: usize) -> Option<u64> {
    launch.dispatch_gpu_ns().ok()??;
    (0..runs).map(|_| launch.dispatch_gpu_ns().ok().flatten()).min().flatten()
}
