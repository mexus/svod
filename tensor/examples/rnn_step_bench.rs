//! Looped vs. host-unrolled GRU: kernel inventory and timings.
//!
//! `Tensor::gru` compiles one step kernel and re-launches it per time slot
//! through the schedule-level scan loop. The unrolled baseline here is the
//! shape the builder had before that: the same cell, driven by a host `for`
//! loop over constant time offsets, so every step carries its own AST.
//!
//! Run with `cargo run --release -p svod-tensor --example rnn_step_bench`;
//! `SVOD_DEVICE` selects the backend.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;
use svod_tensor::nn::{GruCell, RecurrentCell};

const I: usize = 256;
const H: usize = 256;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect()
}

struct Weights {
    x: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
}

fn weights(t: usize, b: usize) -> Weights {
    let x = Tensor::from_slice(values(t * b * I, 0.31)).try_reshape([t as isize, b as isize, I as isize]).unwrap();
    let w_ih = Tensor::from_slice(values(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(values(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    Weights { x, w_ih, w_hh }
}

/// The looped builder: one step graph indexed by a scan variable.
fn looped(w: &Weights) -> Tensor {
    w.x.gru().weight_ih(&w.w_ih).weight_hh(&w.w_hh).call().unwrap().output
}

/// The host-unrolled baseline: `T` structurally identical step graphs whose
/// input slices differ only by a constant offset.
fn unrolled(w: &Weights, t_len: usize, batch: usize) -> Tensor {
    let cell = GruCell::new(w.w_ih.clone(), w.w_hh.clone(), None, None).unwrap();
    let gx = cell.project_input(&w.x).unwrap().contiguous();
    let mut state = Tensor::zeros(&[batch, H], DType::Float32);
    let mut outs = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let gx_t = gx.narrow(0, t, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        state = cell.step_projected(&gx_t, &state).unwrap();
        outs.push(state.clone());
    }
    Tensor::stack(&outs.iter().collect::<Vec<_>>(), 0).unwrap()
}

/// Launch count per compiled program, keyed by entry point.
fn inventory(plan: &ExecutionPlan) -> BTreeMap<String, usize> {
    let mut launches = BTreeMap::new();
    for kernel in plan.prepared_kernels() {
        *launches.entry(kernel.kernel.entry_point.clone()).or_insert(0) += 1;
    }
    launches
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed())
}

fn main() {
    println!("I = {I}, H = {H}, f32\n");
    println!(
        "{:<9} {:>4} {:>7} {:>9} {:>9} {:>11} {:>11}",
        "variant", "T", "B", "programs", "launches", "prepare ms", "exec ms"
    );

    for batch in [1usize, 8] {
        for t_len in [8usize, 64, 256] {
            for (name, build) in [
                ("looped", &looped as &dyn Fn(&Weights) -> Tensor),
                ("unrolled", &|w: &Weights| unrolled(w, t_len, batch)),
            ] {
                let w = weights(t_len, batch);
                let out = build(&w);
                let (plan, prepare) = timed(|| out.prepare().unwrap());
                plan.execute().unwrap();
                // The readback is inside the timer on purpose: a device submit
                // returns before the work does, so without it the GPU numbers
                // would measure dispatch, not execution. It costs both variants
                // the same T*B*H copyout.
                let (_, exec) = timed(|| {
                    for _ in 0..3 {
                        plan.execute().unwrap();
                    }
                    out.to_vec::<f32>().unwrap()
                });
                let inv = inventory(&plan);
                let launches: usize = inv.values().sum();
                println!(
                    "{name:<9} {t_len:>4} {batch:>7} {:>9} {launches:>9} {:>11.1} {:>11.2}",
                    inv.len(),
                    prepare.as_secs_f64() * 1e3,
                    exec.as_secs_f64() * 1e3 / 3.0
                );
            }
        }
    }
}
