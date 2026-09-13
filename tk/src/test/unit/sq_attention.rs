use std::sync::Arc;

use svod_dtype::{AmdArch, CudaArch, DType, DeviceSpec, GpuArch};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;
use test_case::test_case;

use crate::kernels::sq_attention::{
    HeadSelection, SQ_ATTENTION_SUPPORTED_ARCHS, SqAttentionOpts, SqGeom, build_single_query_attention,
    build_single_query_attention_merge, build_single_query_attention_partial,
};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

const SM_86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));

/// Every arch the kernel is built for: the AMD pair plus CUDA warp32.
fn all_caps() -> [ArchCaps; 3] {
    [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151), ArchCaps::for_arch(SM_86)]
}

fn buffers(b: usize, n: usize, h: usize, d: usize, masked: bool) -> Vec<Arc<UOp>> {
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
    ];
    if masked {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32));
    }
    bufs
}

fn sink(caps: ArchCaps, masked: bool) -> Arc<UOp> {
    let (b, n, h, h_total, d, head_offset) = (2, 5, 3, 7, 64, 2);
    let ker = Kernel::new(
        "sq_attention",
        [h as i64, b as i64, 1],
        caps.wave_size as i64,
        buffers(b, n, h_total, d, masked),
        caps,
    );
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };
    build_single_query_attention(&ker, SqGeom { b, kv_batch: b, n, heads, d }, masked, masked, false);
    ker.finish(1)
}

fn split_sinks(caps: ArchCaps, splits: usize, d: usize) -> (Arc<UOp>, Arc<UOp>) {
    let (b, n, h, h_total, head_offset) = (2, 20, 3, 7, 2);
    let partial_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_total * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_total * d, DType::Float32),
    ];
    let partial = Kernel::new(
        "sq_attention_partial",
        [h as i64, b as i64, splits as i64],
        caps.wave_size as i64,
        partial_buffers,
        caps,
    );
    let heads = HeadSelection { count: h, total: h_total, offset: head_offset };
    build_single_query_attention_partial(&partial, SqGeom { b, kv_batch: b, n, heads, d }, splits, false);
    let partial = partial.finish(2);

    let merge_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
    ];
    let merge = Kernel::new("sq_attention_merge", [h as i64, b as i64, 1], caps.wave_size as i64, merge_buffers, caps);
    build_single_query_attention_merge(&merge, b, h, d, splits);
    (partial, merge.finish(1))
}

#[test]
fn sq_attention_graph_shape_all_arches() {
    for caps in all_caps() {
        for masked in [false, true] {
            let topo = sink(caps, masked).toposort();
            let shuffles = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
            assert_eq!(shuffles, caps.wave_size.ilog2() as usize, "{:?}: one XOR reduction", caps.arch);
            assert!(
                topo.iter().any(|u| matches!(u.op(), Op::Unary(svod_ir::UnaryOp::Exp2, ..))),
                "{:?}: exp2",
                caps.arch
            );
            assert!(topo.iter().any(|u| matches!(u.op(), Op::Range(..))), "{:?}: streamed N loop", caps.arch);
            assert!(
                !topo.iter().any(
                    |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))
                ),
                "{:?}: no LDS",
                caps.arch
            );
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{:?}: no MFMA/WMMA", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "{:?}: no barrier", caps.arch);
        }
        for d in [64, 128] {
            let (partial, merge) = split_sinks(caps, 4, d);
            let partial_topo = partial.toposort();
            let partial_shuffles = partial_topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
            let groups = caps.wave_size / 8;
            let expected = 3 + 2 * caps.wave_size.ilog2() as usize + groups;
            assert_eq!(
                partial_shuffles, expected,
                "{:?} D={d}: width-8 dot, tile reductions, and beta broadcasts",
                caps.arch
            );
            for (name, graph) in [("partial", partial), ("merge", merge)] {
                let topo = graph.toposort();
                assert!(topo.iter().any(|u| matches!(u.op(), Op::Range(..))), "{:?}: split {name} loop", caps.arch);
                assert!(
                    !topo.iter().any(
                        |u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))
                    ),
                    "{:?}: split {name} no LDS",
                    caps.arch
                );
                assert!(
                    !topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))),
                    "{:?}: split {name} no MFMA/WMMA",
                    caps.arch
                );
                assert!(
                    !topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))),
                    "{:?}: split {name} no barrier",
                    caps.arch
                );
            }
        }
    }
}

/// The three kernels render on every supported arch with the arch's own
/// cross-lane primitive — `ds_bpermute` on AMD, `shfl.sync.bfly` (the XOR
/// reductions) plus `shfl.sync.idx` (the per-subgroup beta broadcast, partial
/// kernel only) on NVPTX — and never allocate LDS.
#[test_case(GpuArch::Amd(AmdArch::Gfx942), &["llvm.amdgcn.ds.bpermute"], &[]; "gfx942")]
#[test_case(GpuArch::Amd(AmdArch::Gfx1151), &["llvm.amdgcn.ds.bpermute"], &[]; "gfx1151")]
#[test_case(SM_86, &["llvm.nvvm.shfl.sync.bfly.i32"], &["llvm.nvvm.shfl.sync.idx.i32"]; "sm_86")]
fn sq_attention_renders_per_arch(arch: GpuArch, reduce_intrinsics: &[&str], broadcast_intrinsics: &[&str]) {
    let caps = ArchCaps::for_arch(arch);
    let (renderer, opt_renderer) = match arch {
        GpuArch::Amd(amd) => {
            (svod_codegen::llvm::LlvmTextRenderer::amd(amd), svod_schedule::OptimizerRenderer::for_amd_arch(amd))
        }
        GpuArch::Cuda(cuda) => {
            (svod_codegen::llvm::LlvmTextRenderer::nvptx(cuda), svod_schedule::OptimizerRenderer::for_cuda_arch(cuda))
        }
        GpuArch::Metal(_) => unreachable!("no Metal case"),
    };
    let opt_renderer = opt_renderer.with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        svod_codegen::traits::Renderer::decompositor(&renderer),
        None,
    );
    let (partial, merge) = split_sinks(caps, 4, 64);
    for (name, graph, reduces, broadcasts) in [
        ("sq_attention", sink(caps, true), true, false),
        ("sq_attention_partial", partial, true, true),
        ("sq_attention_merge", merge, false, false),
    ] {
        let optimized =
            svod_schedule::apply_post_optimization_with_renderer(graph, &opt_renderer).expect("post optimization");
        let program =
            svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Cpu).expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize");
        let linear = linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear(..))).expect("LINEAR");
        let code = svod_codegen::traits::Renderer::render(&renderer, &linear, Some(name)).expect("render").code;
        for intrinsic in reduce_intrinsics {
            assert_eq!(code.contains(intrinsic), reduces, "{arch:?}: {name} reduce shuffle {intrinsic}");
        }
        for intrinsic in broadcast_intrinsics {
            assert_eq!(code.contains(intrinsic), broadcasts, "{arch:?}: {name} broadcast shuffle {intrinsic}");
        }
        assert!(!code.contains("@local"), "{arch:?}: {name} no LDS allocation");
    }
}

fn supported_device() -> bool {
    crate::target::check_target(&Tensor::empty(&[1], DType::Float32).device(), SQ_ATTENTION_SUPPORTED_ARCHS).is_ok()
}

#[test]
fn sq_attention_packed_validates_head_geometry() {
    let q = Tensor::empty(&[1, 1, 3, 64], DType::Float32);
    let k = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let v = Tensor::empty(&[1, 5, 4, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &v, 2, SqAttentionOpts::default())
        .expect_err("out-of-range heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));

    let bad_v = Tensor::empty(&[1, 5, 5, 64], DType::Float32);
    let err = crate::single_query_attention_packed(&q, &k, &bad_v, 1, SqAttentionOpts::default())
        .expect_err("mismatched total heads");
    assert!(matches!(err, crate::LaunchError::OperandDimMismatch { .. }));
}

fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize, usize, usize, usize),
    head_offset: usize,
    lens: Option<&[i32]>,
) -> Vec<f32> {
    let (b, n, h, h_total, d) = dims;
    let mut out = vec![0.0; b * h * d];
    for bi in 0..b {
        for hi in 0..h {
            let valid = |ni: usize| lens.is_none_or(|ls| ni < ls[bi] as usize || ni + 1 == n);
            let mut scores = vec![f32::NEG_INFINITY; n];
            for (ni, score) in scores.iter_mut().enumerate().filter(|(ni, _)| valid(*ni)) {
                let mut dot = 0.0;
                for di in 0..d {
                    dot += q[(bi * h + hi) * d + di] * k[((bi * n + ni) * h_total + hi + head_offset) * d + di];
                }
                *score = dot / (d as f32).sqrt();
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
            let norm: f32 = weights.iter().sum();
            for di in 0..d {
                for ni in 0..n {
                    out[(bi * h + hi) * d + di] +=
                        weights[ni] * v[((bi * n + ni) * h_total + hi + head_offset) * d + di] / norm;
                }
            }
        }
    }
    out
}

/// One K/V cache serving every batch row must decode exactly as the same cache
/// replicated per row. Beam search reads an identical cross-attention cache from
/// every hypothesis, so binding it once is the difference between reading the
/// largest tensor in the step `B` times and reading it once.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_broadcast -- --ignored`.
#[test]
#[ignore]
fn sq_attention_broadcast_cache_matches_a_replicated_one() {
    if !supported_device() {
        eprintln!("skip sq_attention_broadcast_cache_matches_a_replicated_one: unsupported device/toolchain");
        return;
    }
    // Whisper's cross-attention geometry: five hypotheses over one 1500-frame window.
    let (b, n, h, h_total, d, head_offset) = (5usize, 1500usize, 20usize, 24usize, 64usize, 2usize);
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let shared_k = Tensor::randn(&[1, n, h_total, d]).expect("shared k");
    let shared_v = Tensor::randn(&[1, n, h_total, d]).expect("shared v");
    for t in [&q, &shared_k, &shared_v] {
        t.realize().expect("realize");
    }
    // The same bytes, laid out once per row, is what the kernel used to require.
    let tile = |t: &Tensor| {
        let wide = t.try_expand([b, n, h_total, d]).expect("expand").contiguous();
        wide.realize().expect("realize tiled");
        wide
    };
    let (wide_k, wide_v) = (tile(&shared_k), tile(&shared_v));

    for split in [1usize, 4] {
        let run = |k: &Tensor, v: &Tensor| {
            let opts = SqAttentionOpts { key_lens: None, include_last: false, split, cache_map: None };
            let out = crate::single_query_attention_packed(&q, k, v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            out.realize().expect("realize");
            out.as_vec::<f32>().expect("vec")
        };
        let shared = run(&shared_k, &shared_v);
        let replicated = run(&wide_k, &wide_v);
        assert_eq!(shared.len(), replicated.len());
        let max_abs = shared.iter().zip(&replicated).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert_eq!(max_abs, 0.0, "split {split}: broadcast diverged from the replicated cache by {max_abs}");
    }
}

/// With rows pointing at different caches, each must read the one it names.
///
/// This is the shape a batched decoder actually has: several hypothesis sets in
/// flight, each against its own audio window, so neither "one cache for all rows"
/// nor "a cache per row" describes it. The map lets one copy serve a whole
/// hypothesis set without serializing the windows.
///
/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_cache_map -- --ignored`.
#[test]
#[ignore]
fn sq_attention_cache_map_reads_the_row_it_names() {
    if !supported_device() {
        eprintln!("skip sq_attention_cache_map_reads_the_row_it_names: unsupported device/toolchain");
        return;
    }
    // Four query rows over two windows, interleaved so a row never reads its own index.
    let (b, caches, n, h, h_total, d, head_offset) = (4usize, 2usize, 256usize, 3usize, 7usize, 64usize, 2usize);
    let owners = [0i32, 1, 0, 1];
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let k = Tensor::randn(&[caches, n, h_total, d]).expect("k");
    let v = Tensor::randn(&[caches, n, h_total, d]).expect("v");
    for t in [&q, &k, &v] {
        t.realize().expect("realize");
    }
    let map = Tensor::from_slice(owners.as_slice());
    map.realize().expect("realize map");

    // The same caches laid out one per row — what the kernel required before.
    let widen = |t: &Tensor| {
        let rows: Vec<Tensor> = owners.iter().map(|&o| t.narrow(0, o as usize, 1usize).expect("row")).collect();
        let wide = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).expect("cat").contiguous();
        wide.realize().expect("realize wide");
        wide
    };
    let (wide_k, wide_v) = (widen(&k), widen(&v));

    for split in [1usize, 4] {
        let mapped = crate::single_query_attention_packed(
            &q,
            &k,
            &v,
            head_offset,
            SqAttentionOpts { cache_map: Some(&map), split, ..Default::default() },
        )
        .expect("mapped")
        .expect("supported");
        mapped.realize().expect("realize mapped");
        let replicated = crate::single_query_attention_packed(
            &q,
            &wide_k,
            &wide_v,
            head_offset,
            SqAttentionOpts { split, ..Default::default() },
        )
        .expect("replicated")
        .expect("supported");
        replicated.realize().expect("realize replicated");

        let a = mapped.as_vec::<f32>().expect("mapped vec");
        let e = replicated.as_vec::<f32>().expect("replicated vec");
        let max_abs = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert_eq!(max_abs, 0.0, "split {split}: mapped cache diverged from the replicated one by {max_abs}");
    }
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib sq_attention_numerical_gpu -- --ignored`.
#[test]
#[ignore]
fn sq_attention_numerical_gpu() {
    if !supported_device() {
        eprintln!("skip sq_attention_numerical_gpu: unsupported device/toolchain");
        return;
    }
    let (b, n, h, h_total, d, head_offset) = (2, 20, 3, 7, 64, 2);
    let q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let k = Tensor::randn(&[b, n, h_total, d]).expect("k");
    let v = Tensor::randn(&[b, n, h_total, d]).expect("v");
    q.realize().expect("realize q");
    k.realize().expect("realize k");
    v.realize().expect("realize v");
    let qv = q.as_vec::<f32>().expect("q vec");
    let kv = k.as_vec::<f32>().expect("k vec");
    let vv = v.as_vec::<f32>().expect("v vec");

    for (lens, splits) in [(None, vec![1, 2, 4]), (Some(vec![7i32, 13]), vec![1])] {
        for split in splits {
            let mut lens_t = lens.as_ref().map(|x| Tensor::from_slice(x.as_slice()));
            if let Some(t) = &mut lens_t {
                t.realize().expect("realize lens");
            }
            let opts =
                SqAttentionOpts { key_lens: lens_t.as_ref(), include_last: lens.is_some(), split, cache_map: None };
            let got = crate::single_query_attention_packed(&q, &k, &v, head_offset, opts)
                .expect("sq attention")
                .expect("supported");
            got.realize().expect("realize output");
            let got = got.as_vec::<f32>().expect("output vec");
            let expected = cpu_reference(&qv, &kv, &vv, (b, n, h, h_total, d), head_offset, lens.as_deref());
            let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 2e-4, "split {split} max abs error {max_abs}");
        }
    }

    // Production cross-cache geometry. A 150-key chunk leaves a ragged width-8
    // tile on both wave64 (8 groups) and wave32 (4 groups).
    let (b, n, h, h_total, d, head_offset) = (5, 1500, 20, 24, 64, 2);
    let q = Tensor::randn(&[b, 1, h, d]).expect("production q");
    let k = Tensor::randn(&[b, n, h_total, d]).expect("production k");
    let v = Tensor::randn(&[b, n, h_total, d]).expect("production v");
    q.realize().expect("realize production q");
    k.realize().expect("realize production k");
    v.realize().expect("realize production v");
    let expected = cpu_reference(
        &q.as_vec::<f32>().expect("production q vec"),
        &k.as_vec::<f32>().expect("production k vec"),
        &v.as_vec::<f32>().expect("production v vec"),
        (b, n, h, h_total, d),
        head_offset,
        None,
    );
    let got = crate::single_query_attention_packed(
        &q,
        &k,
        &v,
        head_offset,
        SqAttentionOpts { split: 10, ..Default::default() },
    )
    .expect("production sq attention")
    .expect("production supported");
    got.realize().expect("realize production output");
    let got = got.as_vec::<f32>().expect("production output vec");
    let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 2e-4, "production split 10 max abs error {max_abs}");
}

/// A head dim the arch's wave size does not divide is a fit failure, not a
/// caller bug: the launch declines with `Ok(None)` so a dispatch layer (the
/// whisper decoder) falls back to its generic attention path. Host-portable:
/// off-GPU the arch resolution declines the same way, so the assertion holds
/// on every runner, and on real hardware it exercises the `applies` arm (no
/// supported arch has a wave size dividing 4).
#[test]
fn undivisible_head_dim_declines_instead_of_erroring() {
    let (b, n, h, d) = (1usize, 3usize, 2usize, 4usize);
    let q = Tensor::zeros(&[b, 1, h, d], DType::Float32);
    let k = Tensor::zeros(&[b, n, h, d], DType::Float32);
    let v = Tensor::zeros(&[b, n, h, d], DType::Float32);
    let out = crate::single_query_attention(&q, &k, &v, SqAttentionOpts::default()).expect("declines, not errors");
    assert!(out.is_none(), "head dim 4 divides no supported wave size (32/64); the launch must fall back");
}
