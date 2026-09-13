//! Whisper encoder internals: the encoder sequence-padding policy and the
//! per-block flash-attention dispatch count it produces.

use crate::whisper::config::ModelDimensions;
use crate::whisper::encoder::{AudioEncoder, encoder_padded_sequence_len};
use svod_dtype::{DType, DeviceSpec};
use svod_tensor::Tensor;

fn encoder_dims(layers: usize) -> ModelDimensions {
    ModelDimensions {
        n_mels: 4,
        n_audio_ctx: 1500,
        n_audio_state: 128,
        n_audio_head: 2,
        n_audio_layer: layers,
        n_vocab: 16,
        n_text_ctx: 8,
        n_text_state: 8,
        n_text_head: 2,
        n_text_layer: 1,
        dtype: DType::Float16,
    }
}

#[test]
fn unsupported_device_keeps_original_encoder_sequence() {
    assert_eq!(encoder_padded_sequence_len(&DeviceSpec::Cpu, &DType::Float16, 1500), None);

    let encoder = AudioEncoder::empty(&encoder_dims(1));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32);
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dims().unwrap(), [1, 1500, 128]);
}

#[test]
#[ignore = "GPU: inspect full padded Whisper encoder execution plan"]
fn padded_encoder_plan_has_one_flash_attention_per_block() {
    // The encoder gates on its activations' device, which follows the weights
    // onto the process default device.
    let device = svod_dtype::default_device::default_device();
    if !svod_tk::flash_attention_supported(&device) {
        eprintln!("skipping: flash-attention is not supported on {device:?}");
        return;
    }
    let encoder = AudioEncoder::empty(&encoder_dims(32));
    let mel = Tensor::zeros(&[1, 4, 3000], DType::Float32);
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dims().unwrap(), [1, 1500, 128]);

    let plan = out.prepare().unwrap();
    // `unique_kernel_name` suffixes the n-th kernel sharing a name with `n{n-1}`
    // from a PROCESS-wide counter, so whether these dispatches are named
    // `flash_attention` or `flash_attentionn7` depends on what else the test
    // binary compiled first. Match the base name, not the whole entry point.
    let is_flash_attention = |entry: &str| {
        entry.strip_prefix("flash_attention").is_some_and(|suffix| {
            suffix.is_empty()
                || suffix.strip_prefix('n').is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
    };
    let flash_attention = plan.kernels().filter(|kernel| is_flash_attention(&kernel.entry_point)).count();
    assert_eq!(flash_attention, 32, "expected one handwritten flash-attention dispatch per encoder block");
}

/// fp32 activations must not pad, on any device. The padding buys nothing on its
/// own — it exists so flash attention can tile the sequence — and that kernel's
/// mma operands are 16-bit, so reaching it from fp32 means a silent downcast. That
/// downcast moved the encoder from 1.0e-3 to 1.8 against the PyTorch golden, so
/// the dtype gate is what keeps an fp32 model on SDPA.
#[test]
fn fp32_activations_are_never_padded_for_flash_attention() {
    for device in [DeviceSpec::Cpu, DeviceSpec::Cuda { device_id: 0 }] {
        assert_eq!(
            encoder_padded_sequence_len(&device, &DType::Float32, 1500),
            None,
            "fp32 must not pad on {device:?}"
        );
    }
    // The 16-bit dtypes stay eligible; whether they pad is then the device's call.
    for dtype in [DType::Float16, DType::BFloat16] {
        assert_eq!(encoder_padded_sequence_len(&DeviceSpec::Cpu, &dtype, 1500), None, "CPU has no FA kernel");
    }
}
