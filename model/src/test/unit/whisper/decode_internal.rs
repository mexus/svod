//! Whisper decode-seed internals: the prefill seed owns its own device caches
//! (never aliasing the prefill buffers) and seeds every scheduler row.

use crate::whisper::decode::{PrefillMetadata, build_decode_seed, clone_device_cache, copy_device_cache_row};
use crate::whisper::vocab::top_k_logprobs;
use std::sync::Arc;
use svod_device::{Buffer, BufferSpec, CpuAllocator};
use svod_dtype::DType;
use test_case::test_case;

fn cache(allocator: Arc<CpuAllocator>, values: &[f32]) -> Buffer {
    let mut buffer = Buffer::allocate(allocator, DType::Float32, vec![values.len()], BufferSpec::default()).unwrap();
    buffer.copyin(bytemuck::cast_slice(values)).unwrap();
    buffer
}

fn metadata(sample_begin: usize) -> PrefillMetadata {
    PrefillMetadata {
        initial_tokens: (0..sample_begin as u32).collect(),
        sample_begin,
        suppress_tokens: Vec::new(),
        logits: vec![0.0; 4],
        no_speech_prob: f32::NAN,
    }
}

#[test]
fn scheduler_seed_owns_device_buffers_and_seeds_multiple_rows() {
    let allocator = Arc::new(CpuAllocator);
    let self_values = [1.0f32, 2.0, 3.0, 4.0];
    let cross_values = [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
    let self_k_source = cache(allocator.clone(), &self_values);
    let self_v_source = cache(allocator.clone(), &self_values);
    let cross_k_source = cache(allocator.clone(), &cross_values);
    let cross_v_source = cache(allocator.clone(), &cross_values);

    // The seed takes snapshots, never the prefill graph's own buffers: the next
    // window overwrites those while fallback attempts are still reading them.
    let self_k = clone_device_cache(&self_k_source).unwrap();
    assert_ne!(self_k.storage_id(), self_k_source.storage_id());
    assert_eq!(self_k.size(), std::mem::size_of_val(&self_values));

    let seed = build_decode_seed(
        metadata(2),
        self_k,
        clone_device_cache(&self_v_source).unwrap(),
        clone_device_cache(&cross_k_source).unwrap(),
        clone_device_cache(&cross_v_source).unwrap(),
    )
    .unwrap();

    assert_eq!(seed.metadata.initial_tokens, [0, 1]);
    assert_eq!(seed.metadata.sample_begin, 2);
    assert_ne!(seed.cross_k.storage_id(), cross_k_source.storage_id());
    assert_ne!(seed.cross_v.storage_id(), cross_v_source.storage_id());

    // Every scheduler row is seeded from the one snapshot.
    let cross_bytes = cross_k_source.size();
    let mut cross_rows = Buffer::allocate(
        allocator,
        DType::Float32,
        vec![2 * cross_bytes / std::mem::size_of::<f32>()],
        BufferSpec::default(),
    )
    .unwrap();
    copy_device_cache_row(&mut cross_rows, 0, cross_bytes, &seed.cross_k).unwrap();
    copy_device_cache_row(&mut cross_rows, 1, cross_bytes, &seed.cross_v).unwrap();
    let rows = cross_rows.as_host_bytes().unwrap();
    let expected_cross: &[u8] = bytemuck::cast_slice(&cross_values);
    assert_eq!(&rows[..expected_cross.len()], expected_cross);
    assert_eq!(&rows[cross_bytes..], expected_cross);

    // The cross caches outlive the attempt that consumed the self cache.
    let (kept_k, kept_v) = (seed.cross_k.storage_id(), seed.cross_v.storage_id());
    let (cross_k, cross_v) = seed.into_cross_kv();
    assert_eq!((cross_k.storage_id(), cross_v.storage_id()), (kept_k, kept_v));
    assert_eq!((cross_k.size(), cross_v.size()), (cross_bytes, cross_bytes));
}

/// The seed derives its per-position stride from the prompt length, so every
/// geometry that would make that stride meaningless has to be rejected here
/// rather than mis-slicing a cache later.
#[test]
fn seed_construction_rejects_inconsistent_cache_geometry() {
    let allocator = Arc::new(CpuAllocator);
    let other = Arc::new(CpuAllocator);
    let caches = |values: &[f32]| (cache(allocator.clone(), values), cache(allocator.clone(), values));
    let (self_k, self_v) = caches(&[1.0, 2.0, 3.0, 4.0]);
    let (cross_k, cross_v) = caches(&[5.0, 6.0]);
    let seed = |metadata, self_k: &Buffer, self_v: &Buffer, cross_k: &Buffer, cross_v: &Buffer| {
        build_decode_seed(
            metadata,
            clone_device_cache(self_k).unwrap(),
            clone_device_cache(self_v).unwrap(),
            clone_device_cache(cross_k).unwrap(),
            clone_device_cache(cross_v).unwrap(),
        )
    };

    assert!(seed(metadata(2), &self_k, &self_v, &cross_k, &cross_v).is_ok());
    assert!(seed(metadata(0), &self_k, &self_v, &cross_k, &cross_v).is_err(), "an empty prompt has no stride");
    assert!(seed(metadata(3), &self_k, &self_v, &cross_k, &cross_v).is_err(), "positions must divide the self cache");

    let short = cache(allocator.clone(), &[1.0]);
    assert!(seed(metadata(2), &self_k, &short, &cross_k, &cross_v).is_err(), "k and v must hold the same positions");
    assert!(seed(metadata(2), &self_k, &self_v, &cross_k, &short).is_err(), "the cross caches must match too");

    let foreign = cache(other, &[5.0, 6.0]);
    assert!(seed(metadata(2), &self_k, &self_v, &cross_k, &foreign).is_err(), "one attempt cannot span allocators");
}

// ─── Bounded top-k token selection ──────────────────────────────────────────

/// What `top_k_logprobs` replaced: a full log-softmax over the vocabulary,
/// sorted in its entirety. Kept here as the oracle the fast path must match.
fn reference_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logsum = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln() + max;
    let mut ranked: Vec<(usize, f32)> = logits.iter().map(|&l| l - logsum).enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().take(k).collect()
}

/// Deterministic spread of logits with deliberate exact ties and a run of
/// suppressed (`-inf`) entries, as the logit filters produce.
fn synthetic_logits(len: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match i % 23 {
                0 => f32::NEG_INFINITY,
                1 => 4.25,
                _ => (state >> 40) as f32 / 1024.0 - 12.0,
            }
        })
        .collect()
}

#[test]
fn top_k_matches_a_full_sort_of_the_vocabulary() {
    let logits = synthetic_logits(51865);
    for k in [1usize, 2, 6, 33] {
        let expected = reference_top_k(&logits, k);
        let actual = top_k_logprobs(&logits, k);
        assert_eq!(actual.len(), k);
        for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(got.0, want.0, "token at rank {index} for k={k}");
            assert_eq!(got.1.to_bits(), want.1.to_bits(), "logprob at rank {index} for k={k}");
        }
    }
}

#[test]
fn top_k_breaks_ties_toward_the_lower_token_id() {
    // Ranks 0..3 are all the same logit, so only the id ordering decides.
    let logits = vec![1.0f32, 0.0, 1.0, 1.0, 0.5, 1.0];
    let picked: Vec<usize> = top_k_logprobs(&logits, 4).into_iter().map(|(token, _)| token).collect();
    assert_eq!(picked, vec![0, 2, 3, 5]);
}

#[test]
fn top_k_is_clamped_to_the_vocabulary_and_empty_at_zero() {
    let logits = vec![0.5f32, -1.0, 2.0];
    assert_eq!(top_k_logprobs(&logits, 0), Vec::new());
    let all = top_k_logprobs(&logits, 99);
    assert_eq!(all.len(), 3, "k above the vocabulary size returns every token, once");
    assert_eq!(all.iter().map(|(token, _)| *token).collect::<Vec<_>>(), vec![2, 0, 1]);
}

#[test]
fn top_k_returns_normalized_logprobs_that_suppressed_tokens_cannot_reach() {
    let logits = vec![f32::NEG_INFINITY, 1.0, 2.0, f32::NEG_INFINITY];
    let picked = top_k_logprobs(&logits, 4);
    assert_eq!(picked.iter().map(|(token, _)| *token).collect::<Vec<_>>(), vec![2, 1, 0, 3]);
    // Two finite candidates, so the surviving mass must sum back to 1.
    let mass: f32 = picked.iter().take(2).map(|(_, logprob)| logprob.exp()).sum();
    assert!((mass - 1.0).abs() < 1e-6, "probability mass was {mass}");
    assert!(picked[2].1 == f32::NEG_INFINITY && picked[3].1 == f32::NEG_INFINITY);
}

/// The vector scans must agree with the scalar ones they replaced, at the row
/// width Whisper actually uses. `n_vocab` is 51,866 at large-v3; the widths
/// below straddle the dispatch floor, the lane multiples and the remainder tail.
#[test_case(0; "empty")]
#[test_case(1; "single")]
#[test_case(63; "below the simd floor")]
#[test_case(64; "at the simd floor")]
#[test_case(255; "ragged tail")]
#[test_case(1501; "timestamp range")]
#[test_case(51866; "large-v3 vocabulary")]
fn vector_logsumexp_matches_the_scalar_scan(n: usize) {
    use crate::whisper::vocab::{logsumexp, logsumexp_scalar};
    let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7919).sin() * 6.0).collect();
    let (vector, scalar) = (logsumexp(&logits), logsumexp_scalar(&logits));
    if n == 0 {
        assert_eq!(vector, f32::NEG_INFINITY);
        return;
    }
    // The two summation orders differ, so they cannot be bit-equal. A shift this
    // small is common to every candidate in the row and cannot reorder them.
    assert!((vector - scalar).abs() < 1e-3, "n={n}: vector {vector} vs scalar {scalar}");
}

/// An all-suppressed row is the degenerate case the vector cut cannot express:
/// every entry ties at `-inf`, so the `>= kth` filter would admit the whole
/// vocabulary. It must fall back to the scalar scan and reproduce it exactly --
/// including the `-inf - -inf` NaN logprobs, which is what the scan has always
/// returned here and what the caller's quality gate already rejects.
#[test]
fn vector_top_k_falls_back_when_every_token_is_suppressed() {
    use crate::whisper::vocab::top_k_scalar;
    let logits = vec![f32::NEG_INFINITY; 512];
    let picked = top_k_logprobs(&logits, 5);
    let expected = top_k_scalar(&logits, 5, f32::NEG_INFINITY);
    assert_eq!(picked.len(), 5);
    assert_eq!(picked.iter().map(|(token, _)| *token).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    for (got, want) in picked.iter().zip(&expected) {
        assert_eq!(got.0, want.0);
        assert_eq!(got.1.is_nan(), want.1.is_nan(), "the fallback must not invent a finite logprob");
    }
}

/// Tokens the filters suppressed sit at `-inf` among finite ones, which is the
/// shape every real decode step has once `apply_logit_filters` has run.
#[test]
fn vector_top_k_matches_the_scalar_scan_with_suppressed_tokens() {
    use crate::whisper::vocab::{logsumexp_scalar, top_k_scalar};
    let n = 51866usize;
    let logits: Vec<f32> =
        (0..n).map(|i| if i % 7 == 0 { f32::NEG_INFINITY } else { ((i as f32) * 1.37).cos() * 9.0 }).collect();
    let logsum = logsumexp_scalar(&logits);
    for k in [1usize, 5, 6, 8] {
        let expected = top_k_scalar(&logits, k, logsum);
        let actual = top_k_logprobs(&logits, k);
        assert_eq!(actual.len(), k);
        for (rank, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(got.0, want.0, "token at rank {rank} for k={k}");
            assert!((got.1 - want.1).abs() < 1e-3, "logprob at rank {rank} for k={k}");
        }
    }
}

/// The sampling weights the vector path writes must match the scalar reference
/// entry for entry, at the widths the dispatch floor, the lane multiples and the
/// remainder tail all straddle. The one intended difference is the `-inf` floor:
/// the vector path clamps a suppressed token to `exp(-87)` where the scalar path
/// reaches exactly zero, which is far below any weight the sampler can draw.
#[test_case(0; "empty")]
#[test_case(1; "single")]
#[test_case(63; "below the simd floor")]
#[test_case(64; "at the simd floor")]
#[test_case(255; "ragged tail")]
#[test_case(1501; "timestamp range")]
#[test_case(51866; "large-v3 vocabulary")]
fn vector_scaled_exp_matches_the_scalar_scan(n: usize) {
    use crate::whisper::vocab::{scaled_exp, scaled_exp_scalar};
    let temperature = 0.7f32;
    // Suppressed tokens sit among finite ones, as `apply_logit_filters` leaves them.
    let logits: Vec<f32> =
        (0..n).map(|i| if i % 11 == 5 { f32::NEG_INFINITY } else { ((i as f32) * 0.7919).sin() * 6.0 }).collect();

    let (weights, sum) = scaled_exp(&logits, temperature);
    let mut expected = vec![0f32; n];
    let expected_sum = scaled_exp_scalar(&logits, temperature.recip(), &mut expected);

    assert_eq!(weights.len(), n);
    for (index, (&got, &want)) in weights.iter().zip(&expected).enumerate() {
        // Two ulp of the larger magnitude, plus the `-inf` floor's own offset.
        let tolerance = 2.0 * f32::EPSILON * got.abs().max(want.abs()) + 1e-37;
        assert!((got - want).abs() <= tolerance, "entry {index} of {n}: vector {got} vs scalar {want}");
        if logits[index] == f32::NEG_INFINITY {
            assert!(got < 1e-37, "a suppressed token kept a drawable weight: {got}");
        }
    }

    // The returned sum is the sum of the weights that were written.
    let total: f64 = weights.iter().map(|&weight| weight as f64).sum();
    assert!((sum as f64 - total).abs() <= 1e-4 * total.max(1.0), "n={n}: sum {sum} vs weights {total}");
    assert!((sum - expected_sum).abs() <= 1e-4 * expected_sum.max(1.0), "n={n}: {sum} vs scalar {expected_sum}");
}
