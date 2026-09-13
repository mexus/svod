//! Whisper decode-seed internals: the prefill seed owns its own device caches
//! (never aliasing the prefill buffers) and seeds every scheduler row.

use crate::whisper::decode::{
    PrefillMetadata, build_decode_seed, clone_device_cache, copy_device_cache_row, top_k_logprobs,
};
use std::sync::Arc;
use svod_device::{Buffer, BufferSpec, CpuAllocator};
use svod_dtype::DType;

fn cache(allocator: Arc<CpuAllocator>, values: &[f32]) -> Buffer {
    let mut buffer = Buffer::allocate(allocator, DType::Float32, vec![values.len()], BufferSpec::default()).unwrap();
    buffer.copyin(bytemuck::cast_slice(values)).unwrap();
    buffer
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
    let metadata = PrefillMetadata {
        initial_tokens: vec![1, 2],
        sample_begin: 2,
        init_len: 2,
        suppress_tokens: Vec::new(),
        prefill_logits: vec![0.0; 4],
        no_speech_prob: f32::NAN,
        pos_embedding: Vec::new(),
        n_state: 0,
    };

    let seed = build_decode_seed(
        metadata,
        clone_device_cache(&self_k_source).unwrap(),
        clone_device_cache(&self_v_source).unwrap(),
        clone_device_cache(&cross_k_source).unwrap(),
        clone_device_cache(&cross_v_source).unwrap(),
    )
    .unwrap();

    assert_eq!(seed.metadata.initial_tokens, [1, 2]);
    assert_eq!(seed.per_pos_bytes, 2 * std::mem::size_of::<f32>());
    assert_eq!(seed.self_cache_bytes, std::mem::size_of_val(&self_values));
    assert_eq!(seed.cross_cache_bytes, std::mem::size_of_val(&cross_values));
    assert_eq!((seed.self_positions, seed.cross_positions), (2, 3));
    assert_ne!(seed.self_k_cache.storage_id(), self_k_source.storage_id());
    assert_ne!(seed.self_v_cache.storage_id(), self_v_source.storage_id());
    assert_ne!(seed.cross_k.storage_id(), cross_k_source.storage_id());
    assert_ne!(seed.cross_v.storage_id(), cross_v_source.storage_id());

    let self_stride = 4 * seed.per_pos_bytes;
    let mut self_rows = Buffer::allocate(
        allocator.clone(),
        DType::Float32,
        vec![2 * self_stride / std::mem::size_of::<f32>()],
        BufferSpec::default(),
    )
    .unwrap();
    copy_device_cache_row(&mut self_rows, 0, self_stride, &seed.self_k_cache).unwrap();
    copy_device_cache_row(&mut self_rows, 1, self_stride, &seed.self_k_cache).unwrap();
    let self_bytes = self_rows.as_host_bytes().unwrap();
    let expected_self: &[u8] = bytemuck::cast_slice(&self_values);
    assert_eq!(&self_bytes[..expected_self.len()], expected_self);
    assert_eq!(&self_bytes[self_stride..self_stride + expected_self.len()], expected_self);

    let mut cross_rows = Buffer::allocate(
        allocator,
        DType::Float32,
        vec![2 * seed.cross_cache_bytes / std::mem::size_of::<f32>()],
        BufferSpec::default(),
    )
    .unwrap();
    copy_device_cache_row(&mut cross_rows, 0, seed.cross_cache_bytes, &seed.cross_k).unwrap();
    copy_device_cache_row(&mut cross_rows, 1, seed.cross_cache_bytes, &seed.cross_k).unwrap();
    let cross_bytes = cross_rows.as_host_bytes().unwrap();
    let expected_cross: &[u8] = bytemuck::cast_slice(&cross_values);
    assert_eq!(&cross_bytes[..expected_cross.len()], expected_cross);
    assert_eq!(&cross_bytes[seed.cross_cache_bytes..], expected_cross);
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
