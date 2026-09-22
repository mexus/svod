//! Criterion GPU-device-time bench for `svod_tk::conv2d_nhwc` — the implicit-GEMM
//! convolution (`act(x ⊛ w + bias)`, channels-last, bf16/f16 in and out) — against
//! svod's generic `Tensor::conv2d` on the YOLO26-x layer shapes. See [`common`]
//! for device-time stamping and self-skip.
//!
//! What the kernel can run is bounded by whichever tile set it draws on —
//! the family's hand table, or the device's own lattice
//! (`svod_tk::kernels::tiling`) where the gathered form stands alone. The
//! 96-channel rows are here precisely because no hand-table tile serves them:
//! they are 16% of a YOLO26-x frame's MACs and never reached this kernel.
//!
//! The `k1` rows are the other 27%. `YoloConv::tk_eligible` turns every 1x1
//! away, but `ConvGeom` has no objection to one — `k = cin`, the tap walk
//! degenerates — so what the model is really declining is the layout, not the
//! kernel. These rows say what it would be declining if the input were already
//! channels-last, which inside a C3k chain it is.
//!
//! Read the `generic` rows under `BEAM=4`: unset, they are the heuristics
//! path, which is not what the model gets and is 3-7x slower than what it does.
//!
//! Run: `SVOD_DEVICE={CUDA,AMD}:0 cargo bench -p svod-tk --bench conv`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

mod common;
use common::{bench_kernel, bench_plan, requirements_met};

/// `(cin, cout, side, stride, k, label)` — one `pad = k / 2` convolution each,
/// at batch 1; `side` is the *input* side.
///
/// The first four are the shapes commit 7f6e8134 measured on gfx1201, so the
/// numbers here sit beside that message's. The rest are the YOLO26-x shapes the
/// model does not route here, largest MAC share first.
const SHAPES: &[(usize, usize, usize, usize, usize, &str)] = &[
    (768, 768, 80, 2, 3, "768-768-s2-80"),
    (768, 768, 20, 1, 3, "768-768-s1-20"),
    (384, 384, 160, 2, 3, "384-384-s2-160"),
    (192, 192, 40, 1, 3, "192-192-s1-40"),
    // Turned away by `cout % 64`: the C3k bottleneck body (9.3% of the x MACs)
    // and the neck's 384-to-96 reduction (2.2%).
    (96, 96, 80, 1, 3, "96-96-s1-80"),
    (384, 96, 80, 1, 3, "384-96-s1-80"),
    // Turned away by `kh * kw > 1`: the C3k2 splits and joins, 1x1 and already
    // channels-last where they sit.
    (1536, 768, 40, 1, 1, "1536-768-k1-40"),
    (768, 768, 80, 1, 1, "768-768-k1-80"),
    (1536, 384, 80, 1, 1, "1536-384-k1-80"),
    (384, 384, 160, 1, 1, "384-384-k1-160"),
    // The head's box branch, which asks for no tk kernel at all though its
    // channels already pass the gate: `hb` is 64 at m/l and 96 at x.
    (512, 64, 20, 1, 3, "512-64-s1-20"),
    (768, 96, 20, 1, 3, "768-96-s1-20"),
    // The same branch at n, where the feature stacks are a quarter as wide: the
    // shallowest K in the model, and the case that says whether asking for tk in
    // the head is safe at every scale or only where the stack is deep.
    (64, 64, 80, 1, 3, "64-64-s1-80"),
    (128, 64, 40, 1, 3, "128-64-s1-40"),
    (256, 64, 20, 1, 3, "256-64-s1-20"),
];

/// A realized random tensor on the env-selected device, at the bench dtype.
fn randn(shape: &[usize], dtype: DType) -> Tensor {
    let t = Tensor::randn(shape).expect("randn").cast(dtype);
    t.realize().expect("realize");
    t
}

fn bench_conv2d_nhwc(c: &mut Criterion) {
    // `RUST_LOG=svod_tk::kernels::tiling=debug` then says which tiles the walk
    // timed, and what each cost in registers, LDS and spill.
    let _ = tracing_subscriber::fmt::try_init();
    if !requirements_met(svod_tk::CONV_SUPPORTED_ARCHS) {
        eprintln!("svod-tk conv bench: skipped (no supported GPU / toolchain)");
        return;
    }
    // f16 is what the YOLO path computes in; the matrix core needs it either way.
    let dtype = DType::Float16;
    let mut group = c.benchmark_group("conv2d_nhwc");
    for &(cin, cout, side, stride, k, label) in SHAPES {
        let (kh, kw, pad) = (k, k, k / 2);
        let out = (side + 2 * pad - kh) / stride + 1;
        group.throughput(Throughput::Elements(2 * (out * out * cout * kh * kw * cin) as u64));

        // Channels-last activation and taps-major weight: what the kernel binds.
        let x = randn(&[1, side, side, cin], dtype.clone());
        let w = randn(&[cout, kh, kw, cin], dtype.clone());
        let bias = randn(&[cout], dtype.clone());

        match svod_tk::conv2d_nhwc(&x, &w, &bias, None, stride, pad, true) {
            Ok(Some(y)) => {
                let plan = y.prepare().expect("prepare conv2d_nhwc");
                group.bench_with_input(BenchmarkId::new("tk", label), &label, |b, _| {
                    bench_kernel(b, &plan, "conv2d_nhwc")
                });
            }
            // No tile of the device's table serves this shape — the model keeps
            // its graph conv. Recording nothing says so louder than a zero would.
            Ok(None) => eprintln!("conv2d_nhwc: no tile serves {label}; only the generic row is recorded"),
            Err(err) => panic!("conv2d_nhwc build {label}: {err}"),
        }

        // Reference: the optimizer's own conv, NCHW as a model would hold it.
        let xn = randn(&[1, cin, side, side], dtype.clone());
        let wn = randn(&[cout, cin, kh, kw], dtype.clone());
        let reference = xn
            .conv2d()
            .weight(&wn)
            .stride(&[stride, stride])
            .padding(&[(pad as isize, pad as isize), (pad as isize, pad as isize)])
            .call()
            .expect("reference conv2d")
            .contiguous();
        let ref_plan = reference.prepare().expect("prepare reference");
        group.bench_with_input(BenchmarkId::new("generic", label), &label, |b, _| bench_plan(b, &ref_plan));
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_conv2d_nhwc
}
criterion_main!(benches);
