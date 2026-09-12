//! Proptest case budgets shared by the suite.

use proptest::prelude::ProptestConfig;

pub const CHEAP: u32 = 256;
/// Semantic-equivalence checks, which are more expensive per case.
pub const EQUIVALENCE: u32 = 512;

pub fn proptest_config(cases: u32) -> ProptestConfig {
    ProptestConfig::with_cases(cases)
}

pub fn cheap() -> ProptestConfig {
    proptest_config(CHEAP)
}

pub fn equivalence() -> ProptestConfig {
    proptest_config(EQUIVALENCE)
}
