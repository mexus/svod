//! `RangeifyContext` carries no test: every field is dead in production.
//!
//! `rangeify()` builds it with `range_map: HashMap::new()` and no caller ever
//! reads the result, `record_transform`/`get_rangeified` have no caller outside
//! this file, and `range_counter` only mirrors `IndexingContext::range_counter()`.
//! Tests that exercised the map pinned a `HashMap` round-trip, not a rangeify
//! behaviour, so they were deleted rather than kept as coverage theatre.
