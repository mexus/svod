//! GTCRN speech enhancement: noisy WAV in → enhanced WAV out.
//!
//! Loads the converted checkpoint and runs the JIT (STFT → network → ISTFT,
//! one graph), writing the enhanced waveform.
//!
//! ## Chunking
//!
//! The GRU recurrence unrolls one IR node per time step, so a single JIT plan
//! over the whole waveform would explode the symbolic graph. This example
//! processes the audio in fixed 32-frame chunks (one `prepare`, reused per
//! chunk). The GRU hidden state resets at each chunk boundary, and each chunk
//! is reflect-padded against its own ends, so the output diverges slightly
//! from a full-sequence run near boundaries — the
//! [`gtcrn::parity`](../../src/test/unit/gtcrn/parity.rs) test verifies exact
//! PyTorch parity on a single chunk.
//!
//! Usage:
//!   cargo run -p svod-model --release --example gtcrn_enhance -- \
//!       --in noisy.wav --out enhanced.wav
//!   cargo run -p svod-model --release --example gtcrn_enhance -- \
//!       --in noisy.wav --hub        # pull weights from vpermilp/gtcrn
//!
//! Env:
//!   SVOD_GTCRN=/path/to/data/gtcrn   Local weights dir (gtcrn.safetensors).

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use svod_model::gtcrn::{Gtcrn, GtcrnJit, HOP, num_frames};
use svod_model::jit::InputSpec;

#[derive(Parser, Debug)]
#[command(about = "GTCRN speech enhancement", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono).
    #[arg(long)]
    r#in: PathBuf,

    /// Output enhanced WAV.
    #[arg(long)]
    out: PathBuf,

    /// Pull weights from HuggingFace Hub (vpermilp/gtcrn) instead of a local file.
    #[arg(long)]
    hub: bool,

    /// Local safetensors path (used when --hub is not set). Defaults to
    /// data/gtcrn/gtcrn.safetensors or $SVOD_GTCRN/gtcrn.safetensors.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Re-run one steady-state chunk under the profiler and print the per-kernel
    /// report. The enhancement pass itself is timed unprofiled, so the RTF above
    /// stays comparable.
    #[arg(long)]
    profile: bool,

    /// Origin rollup depth for `--profile`; defaults to `SVOD_ORIGIN_DEPTH`.
    #[arg(long)]
    origin_depth: Option<usize>,
}

fn resolve_weights(args: &Args) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if args.hub {
        return Ok(PathBuf::from(".")); // Gtcrn::from_hub resolves it.
    }
    if let Some(w) = &args.weights {
        return Ok(w.clone());
    }
    if let Ok(dir) = std::env::var("SVOD_GTCRN") {
        let p = PathBuf::from(dir).join("gtcrn.safetensors");
        if p.exists() {
            return Ok(p);
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/gtcrn/gtcrn.safetensors");
    if p.exists() {
        return Ok(p);
    }
    Err("no --weights given and no data/gtcrn/gtcrn.safetensors found (pass --hub to download)".into())
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
    // Mono: if stereo, take the first channel.
    let mono = if spec.channels > 1 { samples.chunks(spec.channels as usize).map(|c| c[0]).collect() } else { samples };
    Ok((mono, spec.sample_rate))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let (waveform, sr) = load_wav(&args.r#in)?;
    if sr != 16000 {
        return Err(format!("expected 16 kHz input, got {sr} Hz").into());
    }
    println!("Input: {} ({} samples, {:.1}s)", args.r#in.display(), waveform.len(), waveform.len() as f32 / sr as f32);

    // Load model.
    let t = Instant::now();
    let model = if args.hub {
        println!("Pulling weights from vpermilp/gtcrn...");
        Gtcrn::from_hub()?
    } else {
        let w = resolve_weights(&args)?;
        println!("Loading weights from {}...", w.display());
        Gtcrn::from_safetensors(&w)?
    };
    println!("  loaded in {:.2}s", t.elapsed().as_secs_f32());

    // The GRU recurrence unrolls one IR node per time step, so a single JIT
    // plan over the full audio would explode the symbolic graph. Process it in
    // fixed-size sample chunks (one prepare, reused per chunk): a chunk of
    // CHUNK_FRAMES · HOP samples is CHUNK_FRAMES + 1 STFT frames in and the
    // same sample count back out, so the enhanced chunks concatenate directly.
    //
    // 128 is the measured knee. The DPGRNN's intra RNN runs along frequency,
    // not time, so its 33 steps cost the same whatever the chunk length and
    // amortise as the chunk grows: 36 dispatches per frame at 32 frames, 24 at
    // 128. Past that the whole-chunk convolutions start losing more than the
    // recurrence saves (256 is slower than 128 on both devices).
    const CHUNK_FRAMES: usize = 128;
    const CHUNK_SAMPLES: usize = CHUNK_FRAMES * HOP;
    let mut enh = vec![0.0f32; waveform.len()];

    let t = Instant::now();
    let mut jit = GtcrnJit::new(model);
    jit.prepare(InputSpec::f32(&[1, CHUNK_SAMPLES]))?;
    println!("JIT prepare ({CHUNK_FRAMES} frames/chunk): {:.2}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let mut done = 0usize;
    // Reusable zero-padded chunk buffers (only the final chunk is partial).
    let mut buf = vec![0.0f32; CHUNK_SAMPLES];
    let mut out_buf = vec![0.0f32; CHUNK_SAMPLES];
    while done < waveform.len() {
        let take = (waveform.len() - done).min(CHUNK_SAMPLES);
        buf[..take].copy_from_slice(&waveform[done..done + take]);
        buf[take..].fill(0.0);
        jit.waveform_mut()?.copyin(bytemuck::cast_slice(&buf))?;
        jit.execute()?;
        jit.output()?.copyout(bytemuck::cast_slice_mut(&mut out_buf))?;
        enh[done..done + take].copy_from_slice(&out_buf[..take]);
        done += take;
    }
    println!(
        "JIT execute ({} chunks, {} frames): {:.2}s",
        waveform.len().div_ceil(CHUNK_SAMPLES),
        num_frames(waveform.len()),
        t.elapsed().as_secs_f32()
    );

    if args.profile {
        // One chunk, after the loop above has warmed every cache, so the report
        // is steady state rather than first-dispatch.
        buf[..CHUNK_SAMPLES].copy_from_slice(&waveform[..CHUNK_SAMPLES.min(waveform.len())]);
        jit.waveform_mut()?.copyin(bytemuck::cast_slice(&buf))?;
        let t = Instant::now();
        let kernels = jit.execute_profiled()?;
        let stage = svod_runtime::StageProfile::gpu("enhance_chunk", t.elapsed(), kernels);
        let mut run = svod_runtime::RunProfile::default();
        run.push(stage);
        let depth = args.origin_depth.or_else(|| svod_runtime::ProfileOptions::from_env().origin_depth);
        println!("\n--- Profile (one {CHUNK_FRAMES}-frame chunk) ---\n{}", run.render_report_at(depth));

        // Split by dispatch count: a kernel launched once per STFT frame is the
        // recurrence re-launched per time slot; a kernel launched a handful of
        // times is a whole-chunk op (conv, STFT, mask).
        let frames = CHUNK_FRAMES + 1;
        let aggs = svod_runtime::aggregate_profiles(&run.stages[0].kernels);
        let (mut rec_ms, mut rec_n, mut one_ms, mut one_n) = (0.0f64, 0usize, 0.0f64, 0usize);
        for a in &aggs {
            let ms = a.total.as_secs_f64() * 1e3;
            if a.count >= frames {
                rec_ms += ms;
                rec_n += a.count;
            } else {
                one_ms += ms;
                one_n += a.count;
            }
        }
        let tot = rec_ms + one_ms;
        println!(
            "per-frame (recurrence): {rec_n:5} dispatches  {rec_ms:6.3} ms  {:5.1}%\n\
             whole-chunk (conv/stft): {one_n:5} dispatches  {one_ms:6.3} ms  {:5.1}%",
            100.0 * rec_ms / tot,
            100.0 * one_ms / tot
        );
    }

    // Write enhanced WAV.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.out, spec)?;
    for &s in &enh {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    println!("Wrote {} ({} samples)", args.out.display(), enh.len());
    Ok(())
}
