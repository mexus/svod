//! Device semantics at the kernel boundary: only a COPY may straddle devices;
//! a compute kernel must read and write one.
//!
//! `extract_device_from_graph` is covered by the rows in `buffer_limits.rs`.

use svod_dtype::DeviceSpec;
use svod_ir::UOp;

use crate::test::support::prelude::*;

#[test]
fn mixed_device_kernel_is_rejected() {
    let cpu = buffer_on(8, svod_dtype::ScalarDType::Float32, DeviceSpec::Cpu);
    let amd = buffer_on(8, svod_dtype::ScalarDType::Float32, DeviceSpec::Amd { device_id: 0 });
    let sink = UOp::sink(vec![cpu.try_mul(&amd).expect("mul").contiguous()]);

    let (rangeified, _ctx) = crate::rangeify::rangeify(sink).expect("rangeify accepts the mixed graph");
    assert!(crate::rangeify::try_get_kernel_graph(rangeified).is_err());
}
