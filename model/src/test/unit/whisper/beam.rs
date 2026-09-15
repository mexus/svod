use std::collections::HashSet;
use std::time::Duration;

use svod_runtime::{RunProfile, StageProfile};

use crate::whisper::decode::{
    BeamCandidate, BeamHypothesis, BeamSurvivor, DecodeScheduleStats, SlotAllocator, attempt_strategies,
    collect_ordered, derived_sampling_seed, finalize_beam_hypotheses, plan_beam_rows, select_beam_candidates,
    strategy_width,
};
use crate::whisper::{DecodeOptions, DecodeStrategy};

fn parent(logical_id: usize, sum_logprob: f32) -> BeamHypothesis {
    BeamHypothesis { logical_id, tokens: vec![logical_id as u32], token_probs: vec![], sum_logprob }
}

fn candidate(parent_index: usize, parent_logical_id: usize, parent_row: usize, token_id: u32) -> BeamCandidate {
    BeamCandidate { parent_index, parent_logical_id, parent_row, token_id, token_logprob: -0.25, sum_logprob: -1.0 }
}

#[test]
fn candidate_ties_use_logical_parent_then_token_not_physical_row() {
    let parents = [parent(20, -0.75), parent(10, -0.75)];
    let candidates = vec![candidate(0, 20, 0, 1), candidate(1, 10, 3, 9), candidate(1, 10, 3, 2)];
    let mut next_id = 100;

    let (active, finished, survivors) = select_beam_candidates(&parents, candidates, 3, 99, 3, &mut next_id);

    assert!(finished.is_empty());
    assert_eq!(active.iter().map(|beam| *beam.tokens.last().unwrap()).collect::<Vec<_>>(), [2, 9, 1]);
    assert_eq!(survivors.iter().map(|beam| beam.parent_row).collect::<Vec<_>>(), [3, 3, 0]);
    assert_eq!(active.iter().map(|beam| beam.logical_id).collect::<Vec<_>>(), [100, 101, 102]);
}

#[test]
fn candidate_selection_preserves_eot_probability_for_final_accounting() {
    let parents = [parent(0, -0.75)];
    let mut eot = candidate(0, 0, 2, 7);
    eot.token_logprob = -0.5;
    eot.sum_logprob = -1.25;
    let mut next_id = 1;

    let (active, finished, survivors) = select_beam_candidates(&parents, vec![eot], 1, 7, 1, &mut next_id);

    assert!(active.is_empty());
    assert!(survivors.is_empty());
    assert_eq!(finished[0].tokens, [0, 7]);
    assert_eq!(finished[0].token_probs, [-0.5f32].map(f32::exp));
    assert_eq!(finished[0].sum_logprob, -1.25);
}

#[test]
fn finalization_adds_eot_without_changing_generated_tokens_or_probabilities() {
    let active = vec![BeamHypothesis {
        logical_id: 4,
        tokens: vec![50, 10, 11],
        token_probs: vec![0.8, 0.7],
        sum_logprob: -0.4,
    }];

    let best = finalize_beam_hypotheses(active, vec![], 1, 99, 1).unwrap();

    assert_eq!(best.tokens, [50, 10, 11, 99]);
    assert_eq!(best.tokens.len() - 1 - 1, 2);
    assert_eq!(best.token_probs, [0.8, 0.7]);
}

#[test]
fn row_assignment_retains_one_child_per_parent_and_uses_inactive_rows() {
    let survivors = [
        BeamSurvivor { logical_id: 8, parent_row: 3 },
        BeamSurvivor { logical_id: 2, parent_row: 1 },
        BeamSurvivor { logical_id: 9, parent_row: 3 },
        BeamSurvivor { logical_id: 4, parent_row: 1 },
    ];

    let plan = plan_beam_rows(&[0, 1, 2, 3], &survivors).unwrap();

    assert_eq!(plan.rows, [3, 1, 0, 2]);
    assert_eq!(plan.copies.iter().map(|copy| (copy.src_row, copy.dst_row)).collect::<Vec<_>>(), [(3, 0), (1, 2)]);
}

#[test]
fn row_assignment_rejects_invalid_geometry() {
    let survivor = BeamSurvivor { logical_id: 0, parent_row: 4 };
    assert!(plan_beam_rows(&[0, 1], &[survivor]).is_err());
    assert!(plan_beam_rows(&[0, 0], &[]).is_err());
}

#[test]
fn row_assignment_invariants_hold_for_all_small_parent_sequences() {
    // Exhaust all parent selections up to four survivors. This gives the same
    // invariant coverage as a generated property while remaining deterministic.
    for len in 0usize..=4 {
        for encoded in 0usize..4usize.pow(len as u32) {
            let mut value = encoded;
            let survivors: Vec<_> = (0..len)
                .map(|logical_id| {
                    let parent_row = value % 4;
                    value /= 4;
                    BeamSurvivor { logical_id: len - logical_id, parent_row }
                })
                .collect();
            let plan = plan_beam_rows(&[0, 1, 2, 3], &survivors).unwrap();
            let parents: HashSet<_> = survivors.iter().map(|survivor| survivor.parent_row).collect();
            let destinations: HashSet<_> = plan.rows.iter().copied().collect();

            assert_eq!(destinations.len(), survivors.len());
            assert!(plan.copies.iter().all(|copy| !parents.contains(&copy.dst_row)));
            assert!(plan.copies.iter().all(|copy| parents.contains(&copy.src_row)));
            assert_eq!(plan.copies.len(), survivors.len() - parents.len());
        }
    }
}

#[test]
fn admission_is_atomic_for_mixed_widths_and_refills_after_release() {
    let mut slots = SlotAllocator::new(5);
    let beam = slots.reserve(10, 3).unwrap().unwrap();
    let sample = slots.reserve(11, 1).unwrap().unwrap();
    assert_eq!(beam, [0, 1, 2]);
    assert_eq!(sample, [3]);

    let before = slots.owners().to_vec();
    assert_eq!(slots.reserve(12, 2).unwrap(), None);
    assert_eq!(slots.owners(), before); // no partial reservation

    slots.release(10);
    let refill = slots.reserve(12, 2).unwrap().unwrap();
    assert_eq!(refill, [0, 1]);
    let rows: HashSet<_> = beam.into_iter().chain(sample).chain(refill.iter().copied()).collect();
    assert_eq!(rows.len(), 4);
    assert_eq!(slots.owners().iter().filter(|owner| **owner == Some(12)).count(), 2);
}

#[test]
fn admission_rejects_width_over_capacity_without_ownership() {
    let mut slots = SlotAllocator::new(2);
    assert_eq!(slots.reserve(0, 3), Err("decode attempt width exceeds decoder slots"));
    assert!(slots.owners().iter().all(Option::is_none));
}

#[test]
fn fallback_sequence_changes_beam_attempt_to_width_one_sampling() {
    let options = DecodeOptions {
        strategy: DecodeStrategy::Beam { size: 5 },
        fallback_temperatures: vec![0.4, 0.8],
        ..DecodeOptions::default()
    };
    let strategies = attempt_strategies(&options);
    assert_eq!(
        strategies,
        [
            DecodeStrategy::Beam { size: 5 },
            DecodeStrategy::Sample { temperature: 0.4 },
            DecodeStrategy::Sample { temperature: 0.8 },
        ]
    );
    assert_eq!(strategies.into_iter().map(strategy_width).collect::<Vec<_>>(), [5, 1, 1]);

    // An empty temperature list is how fallback is switched off.
    let without = DecodeOptions { fallback_temperatures: Vec::new(), ..options };
    assert_eq!(attempt_strategies(&without), [DecodeStrategy::Beam { size: 5 }]);
}

#[test]
fn sampling_seed_is_stable_and_request_specific() {
    let base = 0x1234_5678_9abc_def0;
    assert_eq!(derived_sampling_seed(base, 0), base);
    assert_eq!(derived_sampling_seed(base, 1), derived_sampling_seed(base, 1));
    assert_ne!(derived_sampling_seed(base, 1), derived_sampling_seed(base, 2));
    assert_ne!(derived_sampling_seed(base, 0), derived_sampling_seed(base, 1));
}

#[test]
fn cache_copy_plan_moves_only_the_valid_prefix_in_payload_simulation() {
    let survivors = [BeamSurvivor { logical_id: 0, parent_row: 2 }, BeamSurvivor { logical_id: 1, parent_row: 2 }];
    let plan = plan_beam_rows(&[0, 1, 2], &survivors).unwrap();
    let row_stride = 6;
    let valid_prefix = 3;
    let mut payload: Vec<_> = (0..3).flat_map(|row| (0..row_stride).map(move |column| row * 10 + column)).collect();
    for copy in plan.copies {
        let src = copy.src_row * row_stride;
        let dst = copy.dst_row * row_stride;
        let prefix = payload[src..src + valid_prefix].to_vec();
        payload[dst..dst + valid_prefix].copy_from_slice(&prefix);
    }
    let cloned_row = plan.rows[1];
    assert_eq!(&payload[cloned_row * row_stride..cloned_row * row_stride + valid_prefix], &[20, 21, 22]);
    assert_eq!(&payload[cloned_row * row_stride + valid_prefix..(cloned_row + 1) * row_stride], &[3, 4, 5]);
}

#[test]
fn scheduler_collection_preserves_input_order_after_out_of_order_completion() {
    let mut completed = vec![None; 3];
    for (request, value) in [(2, "third"), (0, "first"), (1, "second")] {
        completed[request] = Some(value);
    }
    assert_eq!(collect_ordered(completed).unwrap(), ["first", "second", "third"]);
}

/// The schedule counters no longer merge themselves: each window annotates its
/// own stage and the run profile sums the numeric metadata. The accumulation
/// across encoder batches must still be a plain per-counter sum.
#[test]
fn scheduler_stats_accumulate_across_encoder_batches_through_the_run_profile() {
    let annotated = |stats: DecodeScheduleStats, wall| {
        let mut stage = StageProfile::host("decode", wall);
        stats.annotate(&mut stage);
        let mut profile = RunProfile::default();
        profile.push(stage);
        profile
    };

    let mut total = annotated(
        DecodeScheduleStats {
            dispatches: 2,
            active_row_steps: 6,
            reserved_row_steps: 8,
            capacity_row_steps: 10,
            cache_clone_ops: 1,
            cache_clone_bytes: 128,
            attempts: 2,
            fallback_attempts: 0,
        },
        Duration::from_millis(4),
    );
    total.merge(annotated(
        DecodeScheduleStats {
            dispatches: 1,
            active_row_steps: 2,
            reserved_row_steps: 3,
            capacity_row_steps: 5,
            cache_clone_ops: 2,
            cache_clone_bytes: 256,
            attempts: 2,
            fallback_attempts: 1,
        },
        Duration::from_millis(1),
    ));

    let stage = total.stage("decode").unwrap();
    assert_eq!(stage.wall, Duration::from_millis(5));
    for (key, expected) in [
        ("dispatches", "3"),
        ("active_row_steps", "8"),
        ("reserved_row_steps", "11"),
        ("capacity_row_steps", "15"),
        ("cache_clone_ops", "3"),
        ("cache_clone_bytes", "384"),
        ("attempts", "4"),
        ("fallback_attempts", "1"),
    ] {
        assert_eq!(stage.meta[key], expected, "counter {key}");
    }
}

/// A single window's counters are the values the scheduler held, rendered as
/// the plain integers the merge above can sum.
#[test]
fn scheduler_stats_annotate_every_counter() {
    let stats = DecodeScheduleStats {
        dispatches: 7,
        active_row_steps: 11,
        reserved_row_steps: 13,
        capacity_row_steps: 17,
        cache_clone_ops: 19,
        cache_clone_bytes: 23,
        attempts: 29,
        fallback_attempts: 31,
    };
    let mut stage = StageProfile::host("decode", Duration::ZERO);
    stats.annotate(&mut stage);

    let mut rendered: Vec<(&str, &str)> = stage.meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    rendered.sort_unstable();
    assert_eq!(
        rendered,
        [
            ("active_row_steps", "11"),
            ("attempts", "29"),
            ("cache_clone_bytes", "23"),
            ("cache_clone_ops", "19"),
            ("capacity_row_steps", "17"),
            ("dispatches", "7"),
            ("fallback_attempts", "31"),
            ("reserved_row_steps", "13"),
        ]
    );
}
