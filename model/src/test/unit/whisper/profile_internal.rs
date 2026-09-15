//! Whisper copy/graph profiling internals: the `enabled` gate, saturating
//! accumulation, and the stage metadata the profile renders.

use std::sync::Arc;
use std::time::Duration;

use svod_device::{Buffer, BufferSpec, CpuAllocator};
use svod_dtype::DType;

use crate::whisper::Result;
use crate::whisper::profile::{CopyProfile, GraphProfile};

/// The buffer a copy group fences on. Any allocation will do: the recorders
/// only synchronize it, they never read it.
fn fence() -> Buffer {
    Buffer::allocate(Arc::new(CpuAllocator), DType::Float32, vec![1], BufferSpec::default()).unwrap()
}

/// The counters are `usize` sums over a whole run, so they must clamp rather
/// than wrap when a pathological run overflows them.
#[test]
fn copy_totals_saturate_instead_of_wrapping() {
    let (mut profile, fence) = (CopyProfile::new(true), fence());
    for _ in 0..2 {
        profile.d2d("cache_append", usize::MAX, usize::MAX, &fence, || -> Result<()> { Ok(()) }).unwrap();
    }

    let stage = profile.stages().next().unwrap();
    assert_eq!(stage.meta["ops"], usize::MAX.to_string());
    assert_eq!(stage.meta["bytes"], usize::MAX.to_string());
    assert_eq!(stage.meta["cache_append_ops"], usize::MAX.to_string());
    assert_eq!(stage.meta["cache_append_bytes"], usize::MAX.to_string());
}

#[test]
fn copy_stage_formats_totals_and_breakdown() {
    let (mut profile, fence) = (CopyProfile::new(true), fence());
    assert!(profile.enabled());
    let moved = profile.d2d("cache_append", 2, 1024, &fence, || -> Result<u32> { Ok(7) }).unwrap();
    assert_eq!(moved, 7, "the recorder returns the work's own value");

    let stages: Vec<_> = profile.stages().collect();
    assert_eq!(stages.len(), 1, "only the direction that moved bytes becomes a stage");
    let stage = &stages[0];
    assert_eq!(stage.name, "copy_d2d");
    assert_eq!(stage.meta["ops"], "2");
    assert_eq!(stage.meta["bytes"], "1024");
    assert_eq!(stage.meta["cache_append_ops"], "2");
    assert_eq!(stage.meta["cache_append_bytes"], "1024");
    assert!(stage.meta.contains_key("cache_append_wall_ms"));
    assert!(!stage.meta.contains_key("effective_gbps"), "rates are derived by the consumer, not stored");
    assert!(stage.meta["timing_semantics"].contains("synchronized before and after"));
}

/// Disabled, a recorder is a pass-through: the work still runs, nothing is
/// timed, and no stage is rendered.
#[test]
fn a_disabled_copy_profile_runs_the_work_and_records_nothing() {
    let (mut profile, fence) = (CopyProfile::new(false), fence());
    let mut ran = 0;
    assert!(!profile.enabled());
    let bump = |ran: &mut u32| -> Result<()> {
        *ran += 1;
        Ok(())
    };
    profile.h2d("tokens", 1, 64, &fence, || bump(&mut ran)).unwrap();
    profile.d2d("cache_append", 1, 64, &fence, || bump(&mut ran)).unwrap();
    profile.d2h("logits", 1, 64, &fence, || bump(&mut ran)).unwrap();

    assert_eq!(ran, 3);
    assert_eq!(profile.stages().count(), 0);
}

#[test]
fn graph_profile_accumulates_execution_wall_and_metadata() {
    let mut profile = GraphProfile::new(true);
    profile.record(Duration::from_millis(2), Vec::new());
    profile.record(Duration::from_millis(3), Vec::new());

    let stage = profile.stage("graph");
    assert_eq!(stage.wall, Duration::from_millis(5));
    assert_eq!(stage.meta["executions"], "2");
    assert_eq!(stage.meta["kernel_dispatches"], "0");
    assert_eq!(stage.meta["accumulated_wall_ms"], "5.000");
    assert!(
        !stage.meta.contains_key("average_execution_wall_ms"),
        "every entry must be a plain counter so window profiles sum on merge"
    );
    assert!(stage.meta["timing_semantics"].contains("output synchronization"));
}

/// `execute` chooses between the plain and the instrumented run, and only the
/// instrumented one is charged to the stage.
#[test]
fn graph_execute_takes_the_instrumented_path_only_when_enabled() {
    for enabled in [false, true] {
        let mut profile = GraphProfile::new(enabled);
        let mut calls = (0u32, 0u32);
        profile
            .execute::<_, std::convert::Infallible>(
                &mut calls,
                |calls: &mut (u32, u32)| {
                    calls.0 += 1;
                    Ok(())
                },
                |calls| {
                    calls.1 += 1;
                    Ok(Vec::new())
                },
            )
            .unwrap();

        assert_eq!(calls, if enabled { (0, 1) } else { (1, 0) });
        assert_eq!(profile.executions, usize::from(enabled));
    }
}
