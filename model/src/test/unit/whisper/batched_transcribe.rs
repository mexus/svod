//! Correctness tests for fixed-slot decode scheduling.
//!
//! The heavy tests (marked `#[ignore]`) load real `whisper-tiny` weights. Run
//! them with `cargo test -- --ignored`.

use svod_arch::pipelines::audio::Transcriber;

use crate::whisper::{
    DecodeOptions, DecodeStrategy, ModelDimensions, N_TEXT_CTX, WhisperAlignedTranscriber, WhisperPlan, WhisperSize,
    WhisperTask, WhisperTokenizer,
};

/// Loads whisper-tiny + tokenizer from HuggingFace Hub and builds a
/// transcriber with the same options soroka uses.
fn tiny_transcriber(encoder_batch: usize, decoder_slots: usize) -> WhisperAlignedTranscriber {
    let repo = "openai/whisper-tiny";
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = crate::whisper::Whisper::from_hub(repo, "main", dims).unwrap();
    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let tokenizer = WhisperTokenizer::from_hub(multilingual, num_languages).unwrap();
    // Greedy + no fallback is deterministic across slot geometries.
    let options = DecodeOptions {
        task: WhisperTask::Transcribe,
        language: None,
        strategy: DecodeStrategy::Greedy,
        fallback_temperatures: Vec::new(),
        ..DecodeOptions::default()
    };
    let mut plan = WhisperPlan::for_model(&model.dims, WhisperSize::Tiny);
    plan.encoder_batch = encoder_batch;
    plan.decoder_slots = decoder_slots;
    plan.alignment_batch = 1;
    WhisperAlignedTranscriber::new_with_plan(model, tokenizer, options, WhisperSize::Tiny, 480_000, plan).unwrap()
}

/// Two sine-wave "audio" windows (distinct frequencies) — enough to drive the
/// encoder/decoder without needing a real speech fixture. The point is
/// *deterministic equality* between paths, not transcript quality.
fn fake_windows() -> Vec<Vec<f32>> {
    let sr = 16_000usize;
    (0..2)
        .map(|wi| {
            let freq = 220.0 * 2_f32.powi(wi);
            (0..sr).map(|t| (freq * t as f32 * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3).collect()
        })
        .collect()
}

/// Slot refill must not change deterministic greedy output.
#[test]
#[ignore = "heavy: real whisper-tiny weights + JIT compile"]
fn generalized_scheduler_runs_greedy_with_slot_refill() {
    // `encoder_batch = 1` also walks the bounded-retention path: the two
    // windows are consumed one encoder batch at a time, so a batch's cross-K/V
    // snapshots are released before the next one is projected.
    let mut refill = tiny_transcriber(1, 1);
    let mut concurrent = tiny_transcriber(2, 2);
    refill.set_language(Some("en".to_string()));
    concurrent.set_language(Some("en".to_string()));

    let windows = fake_windows();
    let refs: Vec<&[f32]> = windows.iter().map(|w| w.as_slice()).collect();

    let (refilled, _) = refill.transcribe_windows(&refs, false).expect("refilled greedy transcribe");
    let (concurrent, profile) = concurrent.transcribe_windows(&refs, true).expect("concurrent greedy transcribe");
    assert_eq!(refilled.len(), concurrent.len());
    for (index, (refilled, concurrent)) in refilled.iter().zip(concurrent).enumerate() {
        assert_eq!(refilled.text, concurrent.text, "window {index}: slot geometry changed greedy output");
    }
    let profile = profile.unwrap();
    let decode = profile.stage("decoder_step").unwrap();
    assert!(!decode.kernels.is_empty());
    assert!(decode.meta["dispatches"].parse::<usize>().unwrap() > 0);
    assert_eq!(decode.meta["executions"], decode.meta["dispatches"]);
    assert!(decode.meta["active_row_steps"].parse::<usize>().unwrap() > 0);
    let active_steps = decode.meta["active_row_steps"].parse::<usize>().unwrap();
    let capped_steps = N_TEXT_CTX / 2 - 1;
    assert!(
        active_steps < capped_steps,
        "fake-window greedy decode approached the per-window token cap instead of emitting EOT: {active_steps}/{capped_steps} aggregate steps"
    );
    // Utilization is no longer pre-divided: the stage reports the two sums.
    let capacity_steps = decode.meta["capacity_row_steps"].parse::<usize>().unwrap();
    assert!(capacity_steps >= active_steps, "more row-steps ran than the compiled slots could hold");
    assert!(active_steps as f64 / capacity_steps as f64 > 0.0, "the scheduler reported no row utilization");
    assert!(profile.stage("mel").is_some());
    assert!(profile.stage("encoder").is_some());
    // The cross-attention caches now come out of the prefill graph, so there is
    // no separate cross-K/V projection stage to report.
    assert!(profile.stage("prefill").is_some());
    assert!(profile.stage("cross_kv_projection").is_none());
    assert!(profile.stage("decoder_scheduler_total").is_some());
    assert!(profile.stage("alignment_graph").is_some());
    assert!(profile.stage("alignment_cpu_dtw").is_some());
}

#[test]
#[ignore = "heavy: real whisper-tiny weights + JIT compile"]
fn generalized_scheduler_runs_beam_sizes_two_and_five() {
    for size in [2, 5] {
        let repo = "openai/whisper-tiny";
        let dims = ModelDimensions::for_size(WhisperSize::Tiny);
        let model = crate::whisper::Whisper::from_hub(repo, "main", dims).unwrap();
        let multilingual = model.is_multilingual();
        let num_languages = model.dims.num_languages();
        let options = DecodeOptions {
            language: Some("en".to_string()),
            strategy: DecodeStrategy::Beam { size },
            fallback_temperatures: Vec::new(),
            sample_len: Some(4),
            ..DecodeOptions::default()
        };
        let mut refill_plan = WhisperPlan::for_model(&model.dims, WhisperSize::Tiny);
        refill_plan.decoder_slots = size;
        let mut concurrent_plan = refill_plan.clone();
        concurrent_plan.decoder_slots = size * 2;
        let mut refill = WhisperAlignedTranscriber::new_with_plan(
            model.clone(),
            WhisperTokenizer::from_hub(multilingual, num_languages).unwrap(),
            options.clone(),
            WhisperSize::Tiny,
            480_000,
            refill_plan,
        )
        .unwrap();
        let mut concurrent = WhisperAlignedTranscriber::new_with_plan(
            model,
            WhisperTokenizer::from_hub(multilingual, num_languages).unwrap(),
            options,
            WhisperSize::Tiny,
            480_000,
            concurrent_plan,
        )
        .unwrap();
        let windows = fake_windows();
        let refs: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        let (refilled, _) = refill.transcribe_windows(&refs, false).unwrap();
        let (concurrent, _) = concurrent.transcribe_windows(&refs, false).unwrap();
        assert_eq!(refilled, concurrent, "beam-{size} output changed with physical slot geometry");
    }
}

#[test]
#[ignore = "heavy: real whisper-tiny weights + JIT compile"]
fn seeded_sampling_is_independent_of_slot_geometry() {
    let repo = "openai/whisper-tiny";
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = crate::whisper::Whisper::from_hub(repo, "main", dims).unwrap();
    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let options = DecodeOptions {
        language: Some("en".to_string()),
        strategy: DecodeStrategy::Sample { temperature: 0.8 },
        fallback_temperatures: Vec::new(),
        sampling_seed: Some(42),
        sample_len: Some(4),
        ..DecodeOptions::default()
    };
    let mut serial_plan = WhisperPlan::for_model(&model.dims, WhisperSize::Tiny);
    serial_plan.decoder_slots = 1;
    let mut concurrent_plan = serial_plan.clone();
    concurrent_plan.decoder_slots = 2;
    let mut serial = WhisperAlignedTranscriber::new_with_plan(
        model.clone(),
        WhisperTokenizer::from_hub(multilingual, num_languages).unwrap(),
        options.clone(),
        WhisperSize::Tiny,
        480_000,
        serial_plan,
    )
    .unwrap();
    let mut concurrent = WhisperAlignedTranscriber::new_with_plan(
        model,
        WhisperTokenizer::from_hub(multilingual, num_languages).unwrap(),
        options,
        WhisperSize::Tiny,
        480_000,
        concurrent_plan,
    )
    .unwrap();
    let windows = fake_windows();
    let refs: Vec<_> = windows.iter().map(Vec::as_slice).collect();
    let (serial, _) = serial.transcribe_windows(&refs, false).unwrap();
    let (concurrent, _) = concurrent.transcribe_windows(&refs, false).unwrap();
    assert_eq!(serial, concurrent);
}
