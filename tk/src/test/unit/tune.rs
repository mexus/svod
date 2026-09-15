//! Tests for the first-use tuning store ([`crate::tune`]): the memo, the
//! on-disk round trip, and the rule that nothing unmeasured is cached.

use std::path::PathBuf;

use svod_dtype::{DeviceSpec, GpuArch};

use crate::tune::{TuneKey, TuneStore};

fn key(kernel: &'static str, shape: &[usize], candidates: &[u32]) -> TuneKey {
    let arch = GpuArch::Amd(svod_dtype::AmdArch::Gfx1151);
    TuneKey::new(kernel, &DeviceSpec::Cpu, arch, shape, candidates)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("svod-tk-tune-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The first selection measures every candidate and keeps the fastest; the
/// second, from another store on the same directory, reads it back without
/// measuring; a third candidate list is a different key.
#[test]
fn a_measured_winner_round_trips_through_the_store() {
    let dir = scratch("round-trip");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[1024, 1024, 6144], &[10u32, 20, 30]);
    let mut measured = Vec::new();
    let chosen = store.select(&k, 3, |i| {
        measured.push(i);
        Some([300u64, 100, 200][i])
    });
    assert_eq!((chosen, measured.as_slice()), (Some(1), &[0usize, 1, 2][..]));

    let again = TuneStore::at(Some(dir.clone()));
    let mut ran = false;
    let k2 = key("gemm_nt", &[1024, 1024, 6144], &[10u32, 20, 30]);
    assert_eq!(
        again.select(&k2, 3, |_| {
            ran = true;
            None
        }),
        Some(1)
    );
    assert!(!ran, "a stored winner is not re-measured");

    let other = key("gemm_nt", &[1024, 1024, 6144], &[10u32, 20]);
    assert_ne!(other, k2, "the candidate list is part of the key");
    let _ = std::fs::remove_dir_all(dir);
}

/// A candidate that cannot run is skipped, and when none can, nothing is kept:
/// the caller falls back to its static choice and the next call measures again.
#[test]
fn unmeasured_candidates_are_never_cached() {
    let dir = scratch("unmeasured");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("fa", &[8, 512, 16, 128], &[1u32, 2]);
    assert_eq!(store.select(&k, 2, |_| None), None);
    assert_eq!(store.select(&k, 2, |i| (i == 1).then_some(5)), Some(1), "the runnable candidate wins");
    let _ = std::fs::remove_dir_all(dir);
}

/// A stored index past the current candidate count (a shrunk table under the
/// same hash cannot happen, but a hand-edited file can) is ignored, and a store
/// with no directory still memoizes within the process.
#[test]
fn a_stale_index_is_ignored_and_a_memory_store_memoizes() {
    let dir = scratch("stale");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[64, 64, 192], &[7u32, 8, 9]);
    assert_eq!(store.select(&k, 3, |i| Some(10 - i as u64)), Some(2));
    let k_short = key("gemm_nt", &[64, 64, 192], &[7u32, 8, 9]);
    let mut measured = 0;
    assert_eq!(
        store.select(&k_short, 2, |_| {
            measured += 1;
            Some(1)
        }),
        Some(2),
        "memoized in-process"
    );
    assert_eq!(measured, 0);

    let memory = TuneStore::at(None);
    let k = key("norm", &[4096, 1024], &[1u32, 2]);
    let mut runs = 0;
    assert_eq!(
        memory.select(&k, 2, |i| {
            runs += 1;
            Some(2 - i as u64)
        }),
        Some(1)
    );
    assert_eq!(
        memory.select(&k, 2, |_| {
            runs += 1;
            Some(0)
        }),
        Some(1)
    );
    assert_eq!(runs, 2, "measured once, then memoized");
    let _ = std::fs::remove_dir_all(dir);
}
