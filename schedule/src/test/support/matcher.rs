//! Frozen pattern-matcher entry points and the thin rewrite wrappers that hide
//! `&mut ()` from the rest of the test suite.

use std::sync::Arc;
use std::sync::LazyLock;

use svod_ir::TypedPatternMatcher;

use crate::rewrite::graph_rewrite;

pub struct Matchers;

impl Matchers {
    /// Rules that never need value analysis.
    pub fn simple() -> &'static TypedPatternMatcher {
        crate::symbolic::symbolic_simple()
    }

    /// `symbolic_simple() + pm_fold_cast_const()`: the table the DCE tests use.
    pub fn full() -> &'static TypedPatternMatcher {
        static FULL: LazyLock<TypedPatternMatcher> =
            LazyLock::new(|| crate::symbolic::symbolic_simple() + crate::symbolic::pm_fold_cast_const());
        &FULL
    }

    /// Dead-code elimination folds exactly like [`Self::full`].
    pub fn dce() -> &'static TypedPatternMatcher {
        Self::full()
    }
}

pub fn rewrite(matcher: &TypedPatternMatcher, expr: Arc<svod_ir::UOp>) -> Arc<svod_ir::UOp> {
    graph_rewrite(matcher, expr, &mut ())
}

pub fn rewrite_with<C>(matcher: &TypedPatternMatcher<C>, ctx: &mut C, expr: Arc<svod_ir::UOp>) -> Arc<svod_ir::UOp> {
    graph_rewrite(matcher, expr, ctx)
}
