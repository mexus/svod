//! Transcribe a labelled manifest with one prepared Whisper pipeline.
//!
//! The demo (`whisper_infer`) pays a JIT prepare per process, which makes it
//! useless for a corpus. This keeps one transcriber and walks the manifest, so
//! scoring a set costs one prepare. It prints `index<TAB>hypothesis` per line;
//! the reference text stays in the manifest and WER is scored downstream, where
//! the text normalizer lives.
//!
//! Usage:
//!   cargo run -p svod-model --release --example whisper_wer -- manifest.json --size tiny
//!
//! The manifest is `[{"wav": "<path>", "text": "<reference>"}, ...]`.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use svod_arch::pipelines::audio::{Asr, FixedLengthSplitter, RunOptions};
use svod_model::whisper::{
    CHUNK_LENGTH, DecodeOptions, DecodeStrategy, SAMPLE_RATE, Whisper, WhisperAlignedTranscriber, WhisperPlan,
    WhisperSize, WhisperTask, WhisperTokenizer,
};

#[derive(Parser, Debug)]
#[command(about = "Transcribe a labelled manifest for WER scoring", long_about = None)]
struct Args {
    /// Manifest JSON: `[{"wav": "...", "text": "..."}, ...]`.
    manifest: PathBuf,

    /// Model size name.
    #[arg(long, default_value = "tiny")]
    size: String,

    /// HF Hub repo override.
    #[arg(long)]
    repo: Option<String>,

    /// Spoken language code, or "auto" to detect.
    #[arg(long, default_value = "en")]
    language: String,

    /// Beam width.
    #[arg(long, default_value_t = 5)]
    beam_size: usize,

    /// Disable quality-gated sampling fallback.
    #[arg(long)]
    no_fallback: bool,

    /// Stop after this many entries.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(serde::Deserialize)]
struct Entry {
    wav: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let size = WhisperSize::from_name(&args.size).ok_or_else(|| format!("unknown size {:?}", args.size))?;
    let entries: Vec<Entry> = serde_json::from_reader(std::fs::File::open(&args.manifest)?)?;
    let entries = &entries[..args.limit.unwrap_or(entries.len()).min(entries.len())];

    let dims = svod_model::whisper::ModelDimensions::for_size(size);
    let repo = args.repo.clone().unwrap_or_else(|| "vpermilp/whisper".to_string());
    let model = Whisper::from_hub_with_weights(&repo, size.name(), "model.safetensors", dims)?;
    let tokenizer = WhisperTokenizer::from_hub(model.is_multilingual(), model.dims.num_languages())?;

    let options = DecodeOptions {
        task: WhisperTask::Transcribe,
        language: (args.language != "auto").then(|| args.language.clone()),
        strategy: DecodeStrategy::Beam { size: args.beam_size },
        fallback_temperatures: if args.no_fallback {
            Vec::new()
        } else {
            DecodeOptions::default().fallback_temperatures
        },
        ..Default::default()
    };
    let window = CHUNK_LENGTH * SAMPLE_RATE;
    // The fixed-length splitter under `Asr` hands over one window at a time.
    let plan = WhisperPlan::sequential(&model.dims, size);
    let transcriber = WhisperAlignedTranscriber::new_with_plan(model, tokenizer, options, size, window, plan)?;
    let mut asr = Asr::new(FixedLengthSplitter::new(window, SAMPLE_RATE), transcriber);

    let started = Instant::now();
    let mut audio_seconds = 0.0f64;
    for (index, entry) in entries.iter().enumerate() {
        let waveform = load_wav(&entry.wav)?;
        audio_seconds += waveform.len() as f64 / SAMPLE_RATE as f64;
        let result = asr.transcribe(&waveform, RunOptions::default())?;
        println!("{index}\t{}", result.text.trim());
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "transcribed {} entries, {audio_seconds:.1}s audio in {elapsed:.2}s (RTF {:.5}, {:.1}x realtime)",
        entries.len(),
        elapsed / audio_seconds,
        audio_seconds / elapsed,
    );
    Ok(())
}

fn load_wav(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE as u32 {
        return Err(format!("{} is {} Hz; Whisper expects {SAMPLE_RATE}", path.display(), spec.sample_rate).into());
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    })
}
