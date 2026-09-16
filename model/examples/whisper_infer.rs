//! Whisper inference demo: transcribe a WAV file using Whisper ASR.
//!
//! Loads a WAV and runs it through an [`Asr`]: a [`FixedLengthSplitter`] feeds
//! a [`WhisperAlignedTranscriber`]. Each 30-second window is mel → encoder JIT →
//! cached beam decoding with temperature fallback, followed by DTW alignment.
//!
//! Usage:
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav --profile
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav --size base
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav --warmup 1 --runs 5 --profile
//!
//! Options:
//!   --size       Model size: tiny, base, small, medium, large-v3, turbo (default: tiny)
//!   --language   Language code: en, fr, de, ..., or auto (default: auto)
//!   --task       transcribe or translate (default: transcribe)
//!   --profile    Run and print a separate per-stage GPU profile
//!   --timestamps Print word timestamps
//!   --repo       HF Hub repo (default: vpermilp/whisper)
//!   --revision   HF Hub revision (default: main)

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::audio::{Asr, FixedLengthSplitter, RunOptions};
use svod_model::whisper::{
    CHUNK_LENGTH, DecodeOptions, DecodeStrategy, SAMPLE_RATE, Whisper, WhisperAlignedTranscriber, WhisperPlan,
    WhisperSize, WhisperTask, WhisperTokenizer,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Strategy {
    Greedy,
    Beam,
    Sample,
}

#[derive(Parser, Debug)]
#[command(about = "Whisper ASR transcription demo", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// Model size name.
    #[arg(long, default_value = "tiny")]
    size: String,

    /// HF Hub repo override (default: vpermilp/whisper).
    #[arg(long)]
    repo: Option<String>,

    /// HF Hub branch, tag, or commit (default: model size, or main with --repo).
    #[arg(long)]
    revision: Option<String>,

    /// Local checkpoint directory, overriding --repo and --revision.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Safetensors filename within the repository or local directory.
    #[arg(long, default_value = "model.safetensors")]
    weights: String,

    /// Spoken language code, or "auto" to detect.
    #[arg(long, default_value = "auto")]
    language: String,

    /// Task: transcribe or translate.
    #[arg(long, default_value = "transcribe")]
    task: WhisperTask,

    /// Primary decoding strategy.
    #[arg(long, value_enum, default_value_t = Strategy::Beam)]
    strategy: Strategy,

    /// Beam width used by `--strategy beam`.
    #[arg(long, default_value_t = 5)]
    beam_size: usize,

    /// Temperature used by `--strategy sample`.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Disable quality-gated sampling fallback.
    #[arg(long)]
    no_fallback: bool,

    /// Reproducible base seed for sampling and sampling fallback.
    #[arg(long)]
    sampling_seed: Option<u64>,

    /// Override the concrete decoder row capacity.
    #[arg(long)]
    decoder_slots: Option<usize>,

    /// Untimed warmup iterations.
    #[arg(long, default_value_t = 0)]
    warmup: usize,

    /// Timed steady-state iterations.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// Print word timestamps.
    #[arg(long)]
    timestamps: bool,

    /// Collect and print per-stage profile.
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let t_total = Instant::now();
    let args = Args::parse();

    let size = WhisperSize::from_name(&args.size)
        .ok_or_else(|| format!("unknown size {:?}; try: tiny, base, small, medium, large-v3, turbo", args.size))?;

    println!("Loading audio: {}", args.wav.display());
    let (waveform, sample_rate) = load_wav(&args.wav)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    if sample_rate != SAMPLE_RATE as u32 {
        return Err(format!("WAV is {sample_rate} Hz; Whisper expects {} Hz", SAMPLE_RATE).into());
    }

    let dims = svod_model::whisper::ModelDimensions::for_size(size);
    let model = if let Some(model_dir) = &args.model_dir {
        println!("\nLoading Whisper from {} ...", model_dir.display());
        Whisper::from_dir_with_weights(model_dir, &args.weights, dims)?
    } else {
        let repo = args.repo.clone().unwrap_or_else(|| "vpermilp/whisper".to_string());
        let revision =
            args.revision.as_deref().unwrap_or_else(|| if args.repo.is_some() { "main" } else { size.name() });
        println!("\nLoading Whisper from {repo}@{revision} ...");
        Whisper::from_hub_with_weights(&repo, revision, &args.weights, dims)?
    };

    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let tokenizer = WhisperTokenizer::from_hub(multilingual, num_languages)?;

    let options = DecodeOptions {
        task: args.task,
        language: if args.language == "auto" { None } else { Some(args.language.clone()) },
        strategy: match args.strategy {
            Strategy::Greedy => DecodeStrategy::Greedy,
            Strategy::Beam => DecodeStrategy::Beam { size: args.beam_size },
            Strategy::Sample => DecodeStrategy::Sample { temperature: args.temperature },
        },
        fallback_temperatures: if args.no_fallback {
            Vec::new()
        } else {
            DecodeOptions::default().fallback_temperatures
        },
        sampling_seed: args.sampling_seed,
        ..Default::default()
    };
    if args.runs == 0 {
        return Err("--runs must be non-zero".into());
    }

    // Fixed-length 30s windows (no VAD dependency)
    let window_samples = CHUNK_LENGTH * SAMPLE_RATE;
    let splitter = FixedLengthSplitter::new(window_samples, SAMPLE_RATE);

    // The fixed-length splitter under `Asr` hands over one window at a time.
    let mut plan = WhisperPlan::sequential(&model.dims, size);
    if let Some(decoder_slots) = args.decoder_slots {
        plan.decoder_slots = decoder_slots;
    }
    println!(
        "Plan: encoder batch {}, decoder slots {}, alignment batch {}",
        plan.encoder_batch, plan.decoder_slots, plan.alignment_batch,
    );
    let transcriber = WhisperAlignedTranscriber::new_with_plan(model, tokenizer, options, size, window_samples, plan)?;

    let mut asr = Asr::new(splitter, transcriber);

    for iteration in 0..args.warmup {
        let started = Instant::now();
        let _ = asr.transcribe(&waveform, RunOptions::default())?;
        println!("Warmup {}/{}: {:.3}s", iteration + 1, args.warmup, started.elapsed().as_secs_f64());
    }

    println!("Running {} measured iteration(s)...", args.runs);
    let mut durations = Vec::with_capacity(args.runs);
    let mut result = None;
    for iteration in 0..args.runs {
        let started = Instant::now();
        let current =
            asr.transcribe(&waveform, RunOptions { words: args.timestamps, profile: false, ..Default::default() })?;
        let elapsed = started.elapsed();
        let rtf = rtf(elapsed, duration_s);
        println!("Run {}/{}: {:.3}s, RTF {:.5}x", iteration + 1, args.runs, elapsed.as_secs_f64(), rtf);
        durations.push(elapsed);
        result = Some(current);
    }
    let result = result.expect("at least one measured run");

    if args.profile {
        println!("\nRunning separate profiling pass (excluded from RTF)...");
        let profiled =
            asr.transcribe(&waveform, RunOptions { words: args.timestamps, profile: true, ..Default::default() })?;
        if let Some(profile) = &profiled.profile {
            println!("\n--- Profile ---\n{}", profile.render_table());
            for stage in &profile.stages {
                if !stage.meta.is_empty() {
                    println!("{} metadata:", stage.name);
                    for (key, value) in &stage.meta {
                        println!("  {key}: {value}");
                    }
                }
            }
        }
    }

    if args.timestamps {
        for chunk in &result.chunks {
            for word in chunk.words.as_deref().unwrap_or_default() {
                println!(
                    "  [{:>6.2}s - {:>6.2}s] {}",
                    chunk.start_sec + word.start,
                    chunk.start_sec + word.end,
                    word.text.trim(),
                );
            }
        }
    } else {
        for chunk in &result.chunks {
            if !chunk.text.is_empty() {
                println!("  [{:>6.1}s] {}", chunk.start_sec, chunk.text);
            }
        }
    }

    println!("\n--- Transcript ---\n{}", result.text);
    let min = durations.iter().copied().min().unwrap_or(Duration::ZERO);
    let max = durations.iter().copied().max().unwrap_or(Duration::ZERO);
    let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
    println!(
        "\nMeasured: min {:.3}s / avg {:.3}s / max {:.3}s over {} run(s)",
        min.as_secs_f64(),
        avg.as_secs_f64(),
        max.as_secs_f64(),
        durations.len(),
    );
    println!(
        "RTF: min {:.5}x / avg {:.5}x; throughput {:.2}x realtime; total process {:.2}s",
        rtf(min, duration_s),
        rtf(avg, duration_s),
        if avg.is_zero() { 0.0 } else { duration_s / avg.as_secs_f64() as f32 },
        t_total.elapsed().as_secs_f32(),
    );

    Ok(())
}

fn rtf(elapsed: Duration, audio_seconds: f32) -> f64 {
    if audio_seconds > 0.0 { elapsed.as_secs_f64() / audio_seconds as f64 } else { 0.0 }
}

fn load_wav(path: &PathBuf) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    Ok((samples, spec.sample_rate))
}
