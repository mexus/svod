use super::super::types::{OptArgExt, OptOps};
use super::*;

/// A SINK with one WEAK axis at `constant`, so `generate_actions` has something to split; the WEAK extent caps the renderer's global max at 32.
fn weak_axis_scheduler(constant: i32) -> Scheduler {
    use svod_ir::{AxisId, AxisType};
    let mut renderer = crate::optimizer::Renderer::cpu();
    renderer.global_max = Some(vec![32]);
    Scheduler::new(
        UOp::sink(vec![
            UOp::native_const(constant),
            UOp::range_axis(UOp::index_const(64), AxisId::Renumbered(0), AxisType::Weak),
        ]),
        renderer,
    )
}

/// Stable artifact id for a plan, so cold and warm runs score it identically.
fn plan_identity(opts: &[Opt]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    opts.hash(&mut hasher);
    hasher.finish()
}

/// Benchmark timing for an artifact identity, so cold and warm runs agree.
fn plan_timing(identity: u64) -> Option<Duration> {
    Some(Duration::from_nanos(1 + identity % 10_000))
}

/// A compiled candidate identified by `artifact`, its little-endian binary key.
fn compiled(artifact: u64, compute_ops: Option<u64>) -> CompiledCandidate<u64> {
    CompiledCandidate {
        artifact,
        binary_key: artifact.to_le_bytes().to_vec(),
        compute_ops,
        preparation: Duration::ZERO,
        compilation: Duration::ZERO,
    }
}

/// The cache tests would pass vacuously if the BEAM sled cache never opened.
#[track_caller]
fn cache_ready() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(format!("beam-cache-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("create the test cache root");
    // SAFETY: `dirs::cache_dir()` is not writable in every test environment.
    unsafe { std::env::set_var("XDG_CACHE_HOME", &root) };
    assert!(CACHE_DB.is_some(), "the BEAM sled cache must open under {}", root.display());
}

/// A cache test's run guard: the key a run under this scheduler, config and compiler identity uses, holding an invalidate that finally drops the entry.
struct CacheGuard {
    key: CacheKey,
}

impl CacheGuard {
    /// Invalidate the entry a leftover on-disk key would otherwise hit.
    fn new(scheduler: &Scheduler, config: &BeamConfig, identity: &str, fingerprint: u64) -> Self {
        let key = CacheKey::from_scheduler(scheduler, config, identity, fingerprint);
        cache_invalidate(&key);
        Self { key }
    }
    fn invalidate(&self) {
        cache_invalidate(&self.key);
    }
}

/// Run `compile`/`bench` through the cached remote search.
fn cached<T>(
    scheduler: &Scheduler,
    config: &BeamConfig,
    identity: &str,
    fingerprint: u64,
    compile: impl FnMut(&[Vec<Opt>], &mut dyn FnMut(usize, CompiledCandidate<T>)) -> Result<(), OptError>,
    bench: impl Fn(&T, Option<Duration>) -> Option<Duration>,
) -> BeamResult {
    beam_search_cached_remote(scheduler.clone(), config, identity, fingerprint, compile, bench).expect("cached")
}

/// The staged counterpart of [`cached`].
fn cached_staged<T>(
    scheduler: &Scheduler,
    config: &BeamConfig,
    identity: &str,
    fingerprint: u64,
    compile: impl FnMut(&[Scheduler], &mut dyn FnMut(usize, CompiledCandidate<T>)),
    bench: impl Fn(&T, Option<Duration>) -> Option<Duration>,
) -> BeamResult {
    beam_search_cached_staged(scheduler.clone(), config, identity, fingerprint, compile, bench).expect("cached")
}

/// A worker that emits every candidate replayable on the parent worker, keyed by its plan.
fn remote_wave(
    worker: &Scheduler,
    base: usize,
    config: &BeamConfig,
    candidates: &[Vec<Opt>],
    emit: &mut dyn FnMut(usize, CompiledCandidate<u64>),
) -> Result<(), OptError> {
    for (index, opts) in candidates.iter().enumerate() {
        if apply_remote_candidate(worker.clone(), base, opts, config).is_some() {
            emit(index, compiled(plan_identity(opts), Some(1)));
        }
    }
    Ok(())
}

/// `BEAM_ACTIONS` is tinygrad's `actions` grid: every opt kind is offered, the amount-major order is fixed, and PADTO/NOLOCALS stay env-gated.
#[test]
fn beam_actions_cover_every_opt_kind_in_amount_major_order() {
    for op in [
        OptOps::UPCAST,
        OptOps::UNROLL,
        OptOps::LOCAL,
        OptOps::GROUP,
        OptOps::GROUPTOP,
        OptOps::THREAD,
        OptOps::SWAP,
        OptOps::TC,
    ] {
        assert!(BEAM_ACTIONS.iter().any(|action| action.op == op), "{op:?} must be offered");
    }
    assert!(!BEAM_ACTIONS.iter().any(|action| action.op == OptOps::NOLOCALS), "NOLOCALS is env-gated");
    let padto = std::env::var("BEAM_PADTO").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
    let padding = usize::from(padto != 0) * 7;
    assert_eq!(
        BEAM_ACTIONS.iter().any(|action| action.op == OptOps::PADTO),
        padding != 0,
        "PADTO is offered iff BEAM_PADTO is set"
    );
    assert_eq!(BEAM_ACTIONS.len(), 48 + 15 + 42 + 24 + 12 + 2 + 10 + 10 + 30 + padding, "grid size");
    assert_eq!(BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::THREAD).count(), 30);
    let upcasts: Vec<_> = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::UPCAST).collect();
    assert_eq!(upcasts.len(), 48);
    assert_eq!(upcasts[..8].iter().map(|action| action.axis).collect::<Vec<_>>(), (0..8).map(Some).collect::<Vec<_>>());
    assert!(upcasts[..8].iter().all(|action| action.arg.int() == Ok(0)), "amount-major: all axes at amount zero");
    assert_eq!((upcasts[8].axis, upcasts[8].arg.int()), (Some(0), Ok(2)), "then all axes at amount two");
    // TC: one strict default-axis action plus one action per axis choice, all
    // carrying the `TC_OPT` level the grid was built with.
    let use_tc = std::env::var("TC").ok().and_then(|value| value.parse().ok()).unwrap_or(1usize);
    let tc_opt = std::env::var("TC_OPT").ok().and_then(|value| value.parse().ok()).unwrap_or(2usize);
    let tensor_cores: Vec<_> = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::TC).collect();
    assert_eq!(tensor_cores.len(), 10);
    assert_eq!(tensor_cores.iter().filter(|action| action.arg.tc().unwrap().1 == 0).count(), 1);
    assert!(tensor_cores[1..].iter().all(|action| action.arg.tc() == Ok((-1, tc_opt, use_tc))));
}

/// The persistent BEAM cache replays a winning plan, so its key must separate behavior from execution-only knobs and pin the action space.
#[test]
fn beam_cache_key_separates_behavior_and_ignores_execution_details() {
    use svod_dtype::AmdArch;
    let scheduler = weak_axis_scheduler(0x1111);
    let base = BeamConfig::default();
    let key = |config: &BeamConfig, compiler: &str, ast_hash| {
        CacheKey::from_scheduler(&scheduler, config, compiler, ast_hash).to_bytes()
    };
    let variant = |config: BeamConfig| key(&config, "compiler", 0);
    let base_key = key(&base, "compiler", 0);
    for (what, variant) in [
        ("ast hash", key(&base, "compiler", 1)),
        ("compiler identity", key(&base, "cpu-clang:18", 0)),
        ("beam width", variant(BeamConfig { beam_width: base.beam_width + 1, ..base.clone() })),
        ("upcast cap", variant(BeamConfig { max_upcast: base.max_upcast - 1, ..base.clone() })),
        ("local cap", variant(BeamConfig { max_local: base.max_local - 1, ..base.clone() })),
        ("uop cap", variant(BeamConfig { max_uops: base.max_uops - 1, ..base.clone() })),
        ("min progress", variant(BeamConfig { min_progress_ns: base.min_progress_ns + 1, ..base.clone() })),
        ("nolocals", variant(BeamConfig { enable_nolocals: !base.enable_nolocals, ..base.clone() })),
        (
            "compile timeout",
            variant(BeamConfig { compile_timeout_secs: base.compile_timeout_secs + 1, ..base.clone() }),
        ),
        ("num runs", variant(BeamConfig { num_runs: base.num_runs + 1, ..base.clone() })),
    ] {
        assert_ne!(base_key, variant, "{what} must change the key");
    }
    for (what, variant) in [
        ("compile workers", variant(BeamConfig { compile_workers: base.compile_workers + 1, ..base.clone() })),
        ("child recycling", variant(BeamConfig { max_tasks_per_child: base.max_tasks_per_child + 1, ..base.clone() })),
    ] {
        assert_eq!(base_key, variant, "{what} must not change the key");
    }
    let ast = UOp::sink(vec![UOp::native_const(1i32)]);
    let amd = |arch| Scheduler::new(ast.clone(), crate::optimizer::Renderer::for_amd_arch(arch));
    assert_ne!(
        CacheKey::from_scheduler(&amd(AmdArch::Gfx1100), &base, "amd", 0).to_bytes(),
        CacheKey::from_scheduler(&amd(AmdArch::Gfx1151), &base, "amd", 0).to_bytes(),
        "the exact AMD target must change the key"
    );
    // A replayed plan is only valid under the action space that produced it, and
    // `BEAM_ACTIONS` is built from `BEAM_PADTO` / `TC` / `TC_OPT`.
    let full = CacheKey::from_scheduler(&scheduler, &base, "compiler", 0);
    assert_eq!(full.action_space, action_space_hash(&BEAM_ACTIONS));
    assert_ne!(full.action_space, action_space_hash(&BEAM_ACTIONS[1..]));
    assert_ne!(base_key, CacheKey { action_space: full.action_space ^ 1, ..full }.to_bytes());
}

/// Every opt kind must survive the persistent-cache encoding unchanged.
#[test]
fn opts_survive_the_cache_encoding_roundtrip() {
    let every_kind = vec![
        Opt::upcast(0, 4),
        Opt::local(1, 16),
        Opt::unroll(0, 8),
        Opt::group(0, 4),
        Opt::grouptop(1, 8),
        Opt::thread(0, 4),
        Opt::padto(1, 32),
        Opt::swap(0, 2),
        Opt::tc(None, -1, 2, 1),
        Opt::nolocals(),
    ];
    for opts in [vec![], every_kind] {
        assert_eq!(deserialize_opts(&serialize_opts(&opts)), Some(opts));
    }
    assert_eq!(deserialize_opts(&[0xff; 3]), None, "a corrupt entry is not a plan");
}

/// `validate_limits` rejects a candidate whose upcast product outgrows the target's cap.
#[test]
fn validate_limits_rejects_oversized_upcasts() {
    let mut scheduler = weak_axis_scheduler(0x1102);
    assert!(validate_limits(&scheduler, &BeamConfig::default()));
    apply_opt(&mut scheduler, &Opt::upcast(0, 4), true).expect("UPCAST(0, 4) on a 64-wide Weak axis");
    assert!(validate_limits(&scheduler, &BeamConfig::default()));
    assert!(!validate_limits(&scheduler, &BeamConfig { max_upcast: 2, ..Default::default() }));
}

/// `generate_actions` extends the parent by exactly one recorded opt and yields only limit-respecting candidates.
#[test]
fn generate_actions_extend_the_parent_by_one_opt() {
    let scheduler = weak_axis_scheduler(0x1104);
    let candidates = generate_actions(&scheduler, &BeamConfig::default());
    assert!(!candidates.is_empty());
    for candidate in &candidates {
        assert_eq!(candidate.applied_opts.len(), scheduler.applied_opts.len() + 1);
        assert!(validate_limits(candidate, &BeamConfig::default()));
    }
    assert!(candidates.iter().any(|candidate| !candidate.axes_of(&[svod_ir::AxisType::Thread]).is_empty()));
}

/// The prefilter drops an action whose logical axis cannot resolve and one whose arg duplicates the full-axis variant.
#[test]
fn prefilter_drops_unresolvable_and_duplicate_full_axis_actions() {
    let scheduler = weak_axis_scheduler(0);
    assert!(!passes_prefilter(&scheduler, &Opt::upcast(0, 64)), "arg == full_shape duplicates the arg=0 variant");
    assert!(passes_prefilter(&scheduler, &Opt::upcast(0, 2)));
    assert!(!passes_prefilter(&scheduler, &Opt::upcast(9, 2)), "an axis past the shape cannot resolve");
    assert!(passes_prefilter(&scheduler, &Opt::tc(None, -1, 2, 1)), "TC has no logical axis");
    assert!(passes_prefilter(&scheduler, &Opt::nolocals()));
}

/// The bloat filter folds `compute_ops` into the running minimum and only fires past a thousand-fold.
#[test]
fn bloated_is_a_thousand_fold_check() {
    let mut least = u64::MAX;
    assert!(!bloated(&mut least, Some(1000)));
    assert_eq!(least, 1000);
    assert!(!bloated(&mut least, Some(1_000_000)), "exactly a thousand-fold is still allowed");
    assert!(bloated(&mut least, Some(1_000_001)));
    let mut untouched = 5000;
    assert!(!bloated(&mut untouched, None));
    assert_eq!(untouched, 5000, "an unknown count must not lower the bar");
}

/// A remote candidate is only accepted when the worker's plan extends the recorded prefix.
#[test]
fn apply_remote_candidate_requires_the_recorded_prefix() {
    let mut scheduler = weak_axis_scheduler(0x4b31);
    apply_opt(&mut scheduler, &Opt::upcast(0, 2), true).expect("UPCAST");
    let config = BeamConfig::default();
    let base = scheduler.applied_opts.len();
    let mut extended = scheduler.applied_opts.clone();
    extended.push(Opt::thread(0, 4));
    assert!(apply_remote_candidate(scheduler.clone(), base, &extended, &config).is_some());
    assert!(
        apply_remote_candidate(scheduler.clone(), base, &[Opt::local(0, 2), Opt::thread(0, 4)], &config).is_none(),
        "a differently-prefixed plan is rejected"
    );
    assert!(apply_remote_candidate(scheduler, base, &[], &config).is_none(), "a shorter plan is rejected");
}

/// The plain `beam_search` entry point drives the same loop as the staged ones: it threads the incumbent's early stop into the scorer and keeps the winner.
#[test]
fn beam_search_threads_early_stop_and_picks_the_winner() {
    use std::sync::{Arc, Mutex};
    let early_stops = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&early_stops);
    let score = move |scheduler: &Scheduler, early_stop: Option<Duration>| {
        record.lock().unwrap().push(early_stop);
        Some(CandidateMetrics {
            timing: Duration::from_millis(1),
            ir_hash: plan_identity(&scheduler.applied_opts),
            compute_ops: Some(1),
        })
    };
    let config = BeamConfig { beam_width: 2, disable_cache: true, ..Default::default() };
    let result = beam_search(weak_axis_scheduler(0x1103), &config, score).expect("beam search");
    assert!(result.iterations >= 2, "min_progress is the only stop, so the loop runs again");
    assert_eq!(result.candidates_evaluated, result.benchmarked);
    assert!(result.benchmarked > 0);
    assert!(!result.scheduler.applied_opts.is_empty(), "the winner carries at least one action");
    let stops = early_stops.lock().unwrap();
    assert_eq!(stops[0], None, "the first wave has no incumbent");
    assert!(
        stops.iter().flatten().any(|stop| *stop == Duration::from_millis(3)),
        "the incumbent's best time must reach the scorer as 3x"
    );
}

#[test]
fn test_remote_beam_parent_tracks_only_opt_sequences() {
    let scheduler = weak_axis_scheduler(0x4b31);
    let config =
        BeamConfig { beam_width: 2, min_progress_ns: 1_000_000_000, disable_cache: true, ..Default::default() };
    let base = scheduler.applied_opts.len();
    let worker = scheduler.clone();
    let result = beam_search_remote_staged(
        scheduler,
        &config,
        |candidates, emit| {
            assert!(candidates.iter().all(|opts| opts.len() == base + 1));
            for (index, opts) in candidates.iter().enumerate() {
                if apply_remote_candidate(worker.clone(), base, opts, &config).is_some() {
                    emit(index, compiled(index as u64, Some(1)));
                }
            }
            Ok(())
        },
        |index, _| Some(Duration::from_nanos(10_000 - *index)),
    )
    .unwrap();
    assert_eq!(result.iterations, 1);
    assert_eq!(result.scheduler.applied_opts.len(), base + 1);
    assert!(result.compiled > 0);
}

/// The staged loop streams out-of-order compiles, drops duplicates by binary key and bloated candidates, and serializes backend timing.
#[test]
fn test_staged_beam_streams_unordered_compiles_dedups_and_serializes_timing() {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    struct FakeArtifact {
        index: usize,
    }
    let scheduler = weak_axis_scheduler(0x51a9);
    let config = BeamConfig {
        beam_width: 2,
        min_progress_ns: 1_000_000_000,
        compile_workers: 3,
        disable_cache: true,
        ..Default::default()
    };
    let (opts_by_index, compile_calls) = (Arc::new(Mutex::new(HashMap::new())), Arc::new(AtomicUsize::new(0)));
    let (benchmark_calls, active, maximum) =
        (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let result = beam_search_staged(
        scheduler,
        &config,
        {
            let (opts_by_index, calls) = (Arc::clone(&opts_by_index), Arc::clone(&compile_calls));
            move |candidates: &[Scheduler], emit: &mut dyn FnMut(usize, CompiledCandidate<FakeArtifact>)| {
                for index in (0..candidates.len()).rev() {
                    calls.fetch_add(1, Ordering::SeqCst);
                    opts_by_index.lock().unwrap().insert(index, candidates[index].applied_opts.clone());
                    let binary_key = if matches!(index, 3 | 4) { vec![0xdd] } else { index.to_le_bytes().to_vec() };
                    let compute_ops = Some(if index == 2 { 1001 } else { 1 });
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: FakeArtifact { index },
                            binary_key,
                            compute_ops,
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            }
        },
        {
            let (calls, active, maximum) = (Arc::clone(&benchmark_calls), Arc::clone(&active), Arc::clone(&maximum));
            move |artifact: &FakeArtifact, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                maximum.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(1));
                active.fetch_sub(1, Ordering::SeqCst);
                Some(Duration::from_nanos(10_000 - artifact.index as u64))
            }
        },
    )
    .unwrap();
    assert!(compile_calls.load(Ordering::SeqCst) > 6, "the fixture scheduler must expose enough candidates");
    assert_eq!(result.generated, compile_calls.load(Ordering::SeqCst));
    assert_eq!(result.unique_ir, 0);
    assert_eq!(result.compiled, compile_calls.load(Ordering::SeqCst));
    assert_eq!(result.unique_binary, result.compiled - 2, "one bloated and one duplicate binary are removed");
    assert_eq!(benchmark_calls.load(Ordering::SeqCst), result.unique_binary);
    assert_eq!(result.benchmarked, result.unique_binary);
    assert_eq!(maximum.load(Ordering::SeqCst), 1, "backend timing must be serialized");
    let opts = opts_by_index.lock().unwrap();
    let winning = opts.keys().copied().filter(|index| !matches!(index, 1 | 2 | 4)).max().unwrap();
    assert_eq!(result.scheduler.applied_opts, opts[&winning]);
}

#[test]
fn test_staged_beam_cache_cold_and_warm_choose_same_winner() {
    cache_ready();
    let scheduler = weak_axis_scheduler(0x6b17);
    let config =
        BeamConfig { min_progress_ns: 1_000_000_000, compile_workers: 2, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:beam-cold-warm-v1";
    let guard = CacheGuard::new(&scheduler, &config, identity, 0x1234);
    let run = |scheduler: &Scheduler| {
        cached_staged(
            scheduler,
            &config,
            identity,
            0x1234,
            |candidates, emit| {
                for (index, candidate) in candidates.iter().enumerate() {
                    emit(index, compiled(plan_identity(&candidate.applied_opts), Some(1)));
                }
            },
            |identity, _| plan_timing(*identity),
        )
    };
    let cold = run(&scheduler);
    let warm = run(&scheduler);
    guard.invalidate();
    assert!(cold.iterations > 0);
    assert_eq!(warm.iterations, 0, "the second search must replay the persistent BEAM entry");
    assert_eq!(cold.scheduler.applied_opts, warm.scheduler.applied_opts);
    assert_eq!(cold.timing, warm.timing);
}

#[test]
fn test_remote_beam_cache_reuses_winner_across_parallel_and_recycling_changes() {
    cache_ready();
    let scheduler = weak_axis_scheduler(0x7193);
    let cold = BeamConfig {
        min_progress_ns: 1_000_000_000,
        compile_workers: 1,
        max_tasks_per_child: 1,
        disable_cache: false,
        ..Default::default()
    };
    let warm = BeamConfig { compile_workers: 8, max_tasks_per_child: 99, ..cold.clone() };
    let identity = "fake-compiler:remote-cache-v1";
    let guard = CacheGuard::new(&scheduler, &cold, identity, 0x22);
    let base = scheduler.applied_opts.len();
    let worker = scheduler.clone();
    let run = |config: &BeamConfig| {
        cached(
            &scheduler,
            config,
            identity,
            0x22,
            |candidates, emit| remote_wave(&worker, base, config, candidates, emit),
            |artifact, _| plan_timing(*artifact),
        )
    };
    let cold_run = run(&cold);
    let warm_run = run(&warm);
    guard.invalidate();
    assert!(cold_run.iterations > 0);
    assert_eq!(warm_run.iterations, 0, "the execution-only knobs must not change the key");
    assert_eq!(cold_run.scheduler.applied_opts, warm_run.scheduler.applied_opts);
    assert_eq!(cold_run.timing, warm_run.timing);
}

#[test]
fn test_remote_beam_does_not_cache_unbenchmarked_search() {
    cache_ready();
    let scheduler = weak_axis_scheduler(0x7a21);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-no-empty-cache-v1";
    let guard = CacheGuard::new(&scheduler, &config, identity, 0x31);
    let failed = cached(
        &scheduler,
        &config,
        identity,
        0x31,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| Ok(()),
        |_artifact, _| Some(Duration::from_nanos(1)),
    );
    assert_eq!(failed.benchmarked, 0);
    assert_eq!(failed.timing, Duration::MAX);
    assert!(cache_get(&guard.key).is_none(), "a search that benchmarks nothing must not be cached");
    let cold = cached(
        &scheduler,
        &config,
        identity,
        0x31,
        |candidates, emit| {
            for index in 0..candidates.len() {
                emit(index, compiled(index as u64, Some(1)));
            }
            Ok(())
        },
        |artifact, _| Some(Duration::from_nanos(10_000 - *artifact)),
    );
    assert!(cold.iterations > 0, "the failed search must not create a cache hit");
    assert!(cache_get(&guard.key).is_some());
    guard.invalidate();
}

#[test]
fn test_remote_beam_worker_error_invalidates_cache() {
    cache_ready();
    let scheduler = weak_axis_scheduler(0x7a22);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-worker-error-v1";
    let guard = CacheGuard::new(&scheduler, &config, identity, 0x32);
    cache_put(&guard.key, &[Opt::upcast(0, 2)]);
    let result = beam_search_cached_remote(
        scheduler,
        &config,
        identity,
        0x32,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| {
            Err(OptError::BeamWorker { message: "disconnected".into() })
        },
        |_artifact, _| Some(Duration::from_nanos(1)),
    );
    assert!(matches!(result, Err(OptError::BeamWorker { .. })));
    assert!(cache_get(&guard.key).is_none(), "a stale entry must be dropped when the worker fails");
}
