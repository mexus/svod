//! Devectorizer tests: tinygrad's `devectorizer2` plus Svod's shaped STACK mapping.
pub mod alu_devectorization;
pub mod bool_storage;
pub mod edge_cases;
pub mod fp8_decomp;
pub mod gep_movement;
pub mod helpers;
pub mod late_gater;
pub mod pipeline;
pub mod reduce_to_acc;
