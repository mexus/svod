//! Property-based tests for the schedule passes: algebraic laws, pass-level
//! fixpoints, and the metamorphic relations every lowering stage must preserve.
//!
//! `oracles` is z3-gated and only runs with `--features z3`.

mod algebra;
pub(crate) mod checks;
pub(crate) mod generators;
pub mod long_shift;
#[cfg(feature = "z3")]
mod oracles;
mod passes;
mod ranges;
mod symbolic_meta;
mod symbolic_props;
