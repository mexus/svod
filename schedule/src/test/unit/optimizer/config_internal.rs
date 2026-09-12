use super::*;

use test_case::test_case;

#[test_case(OptStrategy::None, true, false; "None disables optimization")]
#[test_case(OptStrategy::Heuristic, false, false; "Heuristic is neither")]
#[test_case(OptStrategy::Beam { width: 4 }, false, true; "Beam keeps a width")]
fn test_opt_strategy_predicates(strategy: OptStrategy, is_none: bool, is_beam: bool) {
    assert_eq!((strategy.is_none(), strategy.is_beam()), (is_none, is_beam), "{strategy:?}");
    assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
}

/// The defaults are tinygrad's, and a builder call reaches every field.
#[test]
fn test_beam_config_default_and_builder() {
    let config = BeamConfig::default();
    assert_eq!((config.beam_width, config.max_upcast, config.max_local), (4, 256, 1024));
    assert_eq!((config.max_uops, config.num_runs, config.min_progress_ns), (3000, 3, 10));
    assert!(!config.enable_nolocals);
    assert!(!config.disable_cache);
    assert_eq!((config.compile_workers, config.compile_timeout_secs, config.max_tasks_per_child), (0, 10, 16));
    let built = BeamConfig::builder()
        .beam_width(8)
        .max_upcast(512)
        .max_local(2048)
        .max_uops(4096)
        .num_runs(5)
        .min_progress_ns(25)
        .enable_nolocals(true)
        .compile_workers(3)
        .compile_timeout_secs(7)
        .max_tasks_per_child(5)
        .disable_cache(true)
        .build();
    assert_eq!(
        (built.beam_width, built.max_upcast, built.max_local, built.max_uops, built.num_runs, built.min_progress_ns),
        (8, 512, 2048, 4096, 5, 25)
    );
    assert!(built.enable_nolocals);
    assert!(built.disable_cache);
    assert_eq!((built.compile_workers, built.compile_timeout_secs, built.max_tasks_per_child), (3, 7, 5));
}

/// tinygrad `helpers.py`: `BEAM_MIN_PROGRESS` is in microseconds.
#[test_case(None, 10; "unset falls back to the default")]
#[test_case(Some("0.01"), 10; "sub-nanosecond rounds to the default")]
#[test_case(Some("1"), 1_000; "one microsecond")]
#[test_case(Some("1000000"), 1_000_000_000; "one second")]
#[test_case(Some("invalid"), 10; "unparseable falls back to the default")]
#[test_case(Some("inf"), u64::MAX; "infinity clamps to the maximum")]
#[test_case(Some("nan"), 0; "NaN clamps to zero")]
#[test_case(Some("-1"), 0; "a negative value clamps to zero")]
fn test_beam_min_progress_matches_tinygrad_microseconds_env(raw: Option<&str>, expected: u64) {
    assert_eq!(parse_beam_min_progress(raw), expected);
}

/// `SVOD_THREADS` is the one thread budget; only a positive integer overrides
/// the host's parallelism.
#[test_case(Some("4"), Some(4); "positive integer")]
#[test_case(Some("0"), None; "zero falls back")]
#[test_case(Some("many"), None; "unparseable falls back")]
#[test_case(Some(" 4 "), None; "whitespace is not a number")]
#[test_case(Some("18446744073709551616"), None; "an overflowing value falls back")]
#[test_case(None, None; "unset falls back")]
fn test_thread_budget_parsing(raw: Option<&str>, expected: Option<usize>) {
    let fallback = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    assert_eq!(parse_thread_budget(raw), expected.unwrap_or(fallback));
}

#[test]
fn test_heuristics_config_default_and_builder() {
    let config = HeuristicsConfig::default();
    assert_eq!(config.tc_enabled, TcUsage::Enabled);
    // One above tinygrad's `helpers.py:238` TC_OPT=0: multi-reduce kernels
    // (convolutions) get the tensor core by default, padding stays opt-in.
    assert_eq!(config.tc_opt, TcOpt::Relaxed);
    assert_eq!(config.tc_select, TcSelect::Auto);
    assert!(config.matvec_enabled);
    assert_eq!(config.matvec_blocksize, 4);
    assert_eq!((config.threads_per_row, config.rows_per_thread), (8, 4));
    assert_eq!((config.grouped_threshold, config.unroll_threshold), (256, 32));
    assert!(!config.disable_locals);
    assert_eq!(config.thread_count, thread_budget());
    assert!(!config.k_vectorize);
    assert!(config.output_upcast);
    assert_eq!(config.debug_level, 0);
    let built = HeuristicsConfig::builder()
        .tc_enabled(TcUsage::Disabled)
        .tc_opt(TcOpt::Padded)
        .tc_select(TcSelect::Index(2))
        .matvec_enabled(false)
        .matvec_blocksize(16)
        .threads_per_row(16)
        .rows_per_thread(2)
        .grouped_threshold(128)
        .unroll_threshold(8)
        .disable_locals(true)
        .thread_count(3)
        .k_vectorize(true)
        .output_upcast(false)
        .debug_level(2)
        .build();
    assert_eq!(
        (built.tc_enabled, built.tc_opt, built.tc_select),
        (TcUsage::Disabled, TcOpt::Padded, TcSelect::Index(2))
    );
    assert_eq!((built.matvec_enabled, built.matvec_blocksize), (false, 16));
    assert_eq!((built.threads_per_row, built.rows_per_thread), (16, 2));
    assert_eq!((built.grouped_threshold, built.unroll_threshold), (128, 8));
    assert_eq!((built.disable_locals, built.thread_count), (true, 3));
    assert_eq!((built.k_vectorize, built.output_upcast, built.debug_level), (true, false, 2));
}

#[test]
fn test_optimizer_config_default_and_builder() {
    let config = OptimizerConfig::default();
    assert_eq!(config.strategy, OptStrategy::Heuristic);
    assert_eq!(config.beam, BeamConfig::default());
    assert_eq!(config.heuristics, HeuristicsConfig::default());
    // tinygrad `helpers.py:245`: DISABLE_FAST_IDIV defaults to 1.
    assert!(config.disable_fast_idiv);
    assert_eq!(config.transcendental, 1);
    assert!(config.opts_to_apply.is_none());
    let heuristics = HeuristicsConfig::builder().debug_level(3).build();
    let built = OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .beam(BeamConfig::builder().max_upcast(512).build())
        .heuristics(heuristics.clone())
        .transcendental(2)
        .disable_fast_idiv(false)
        .opts_to_apply(vec![Opt::upcast(0, 4)])
        .build();
    assert_eq!(built.strategy, OptStrategy::Beam { width: 8 });
    assert_eq!((built.beam.beam_width, built.beam.max_upcast), (8, 512), "the strategy width overrides BEAM");
    assert_eq!(built.heuristics, heuristics);
    assert_eq!((built.transcendental, built.disable_fast_idiv), (2, false));
    assert_eq!(built.opts_to_apply, Some(vec![Opt::upcast(0, 4)]));
}

/// `disable_fast_idiv` gates the magic-multiply rewrite in the late pattern set.
#[test_case(true, true; "disabled keeps cdiv")]
#[test_case(false, false; "enabled rewrites cdiv")]
fn test_disable_fast_idiv_gates_late_rewrites(disable_fast_idiv: bool, expect_cdiv: bool) {
    use svod_ir::{BinaryOp, DType, Op, UOp};
    let x = UOp::var("x", DType::Int32, 0, 255);
    let cdiv = UOp::new(Op::Binary(BinaryOp::CDiv, x, UOp::native_const(3i32)), DType::Int32);
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let patterns = crate::optimizer::get_late_rewrite_patterns(&renderer, disable_fast_idiv);
    let rewritten = crate::rewrite::graph_rewrite(&patterns, cdiv, &mut ());
    let has_cdiv = rewritten.toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::CDiv, ..)));
    assert_eq!(has_cdiv, expect_cdiv, "{}", rewritten.tree());
}

/// The numeric encodings tinygrad's `TC` / `TC_OPT` / `TC_SELECT` env vars use; they
/// reach the BEAM cache key and the remote worker protocol.
#[test]
fn test_tc_env_encodings_match_tinygrad() {
    assert_eq!([TcUsage::Disabled.as_usize(), TcUsage::Enabled.as_usize(), TcUsage::ShapeOnly.as_usize()], [0, 1, 2]);
    assert_eq!([TcOpt::Strict.as_usize(), TcOpt::Relaxed.as_usize(), TcOpt::Padded.as_usize()], [0, 1, 2]);
    assert_eq!([TcSelect::Auto.as_i32(), TcSelect::Index(5).as_i32()], [-1, 5]);
}

const PROBE: &str = "SVOD_FROM_ENV_PROBE";
const PROBE_NAME: &str = "optimizer::config::tests::test_from_env_maps_the_documented_variables";

/// The `(case, environment)` pairs the probe re-runs itself with; the child's
/// environment is otherwise empty, so no ambient variable can leak in.
const PROBE_CASES: &[(&str, &[(&str, &str)])] = &[
    ("defaults", &[]),
    (
        "overrides",
        &[
            ("SVOD_TC", "0"),
            ("SVOD_TC_OPT", "2"),
            ("SVOD_TC_SELECT", "3"),
            ("SVOD_MV", "0"),
            ("SVOD_MV_BLOCKSIZE", "5"),
            ("SVOD_K_VECTORIZE", "1"),
            ("SVOD_NO_OUTPUT_UPCAST", "1"),
            ("SVOD_NOLOCALS", "1"),
        ],
    ),
    ("tc_edges", &[("SVOD_TC", "2"), ("TC_OPT", "0"), ("TC_SELECT", "-1")]),
    ("beam", &[("BEAM", "8")]),
    ("noopt", &[("SVOD_NOOPT", "1")]),
    ("beam_zero", &[("BEAM", "0")]),
    ("numerics", &[("TRANSCENDENTAL", "2"), ("DISABLE_FAST_IDIV", "0")]),
];

/// Environment variables are process-global, so the `from_env` mapping is
/// asserted in child processes: each case re-executes this test binary with an
/// empty environment plus its own variables, and the in-process environment is
/// never mutated.
#[test]
fn test_from_env_maps_the_documented_variables() {
    let Ok(case) = std::env::var(PROBE) else {
        let exe = std::env::current_exe().expect("current test binary");
        for (case, env) in PROBE_CASES {
            let mut command = std::process::Command::new(&exe);
            command.args(["--exact", "--nocapture", PROBE_NAME]).env_clear().env(PROBE, case).envs(env.iter().copied());
            let output = command.output().expect("probe child");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success() && stdout.contains("probe ok"), "probe {case}:\n{stdout}");
        }
        return;
    };
    assert_from_env(&case);
    println!("probe ok");
}

fn assert_from_env(case: &str) {
    match case {
        "defaults" => assert_eq!(HeuristicsConfig::from_env(), HeuristicsConfig::default()),
        "overrides" => {
            let config = HeuristicsConfig::from_env();
            assert_eq!(
                (config.tc_enabled, config.tc_opt, config.tc_select),
                (TcUsage::Disabled, TcOpt::Padded, TcSelect::Index(3))
            );
            assert_eq!((config.matvec_enabled, config.matvec_blocksize), (false, 5));
            assert_eq!((config.k_vectorize, config.output_upcast, config.disable_locals), (true, false, true));
        }
        "tc_edges" => {
            let config = HeuristicsConfig::from_env();
            assert_eq!(
                (config.tc_enabled, config.tc_opt, config.tc_select),
                (TcUsage::ShapeOnly, TcOpt::Strict, TcSelect::Auto)
            );
        }
        "beam" => {
            let config = OptimizerConfig::from_env();
            assert_eq!((config.strategy, config.beam.beam_width), (OptStrategy::Beam { width: 8 }, 8));
        }
        "noopt" => assert_eq!(OptimizerConfig::from_env().strategy, OptStrategy::None),
        "beam_zero" => assert_eq!(OptStrategy::from_env(), OptStrategy::Heuristic, "BEAM=0 is not a beam search"),
        "numerics" => {
            let from_env = OptimizerConfig::from_env();
            let built = OptimizerConfig::builder().build();
            assert_eq!((from_env.transcendental, from_env.disable_fast_idiv), (2, false));
            assert_eq!((built.transcendental, built.disable_fast_idiv), (2, false), "builder defaults read the env");
        }
        other => panic!("unknown probe case {other}"),
    }
}
