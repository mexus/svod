//! The bottleneck's residual on and off the kernel.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::yolo::YoloBottleneck;

/// The residual reaches the block's output whichever path the block takes: at
/// f32 there is no kernel, so this is the elementwise fallback, and it has to
/// equal the two convs plus the input exactly.
#[test]
fn the_residual_follows_the_block_off_the_kernel() {
    let block = YoloBottleneck::empty(8, 8, true);
    let x = Tensor::randn(&[1, 8, 6, 6]).unwrap();
    let want = block.cv2.forward(&block.cv1.forward(&x).unwrap()).unwrap().try_add(&x).unwrap();
    let got = block.forward(&x).unwrap();
    assert_eq!(got.dtype(), DType::Float32);
    assert_eq!(got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
}
