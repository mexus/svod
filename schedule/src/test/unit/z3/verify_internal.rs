use super::*;
use crate::test::support::prelude::*;
use crate::test::unit::z3::helpers::assert_not_equivalent;
use test_case::test_case;

/// Both the integer and the boolean arms of `verify_equivalence` must accept what really is
/// equivalent. The `x + 0 = x` case is proven over generated inputs by
/// `test::property::oracles`, so it is not repeated here.
#[test_case(|v| (v.x.add(&v.y), v.y.add(&v.x)) ; "addition commutes")]
#[test_case(|v| (v.x.sub(&v.x), v.c(0)) ; "a value minus itself is zero")]
#[test_case(|v| (v.x.try_cmpeq(&v.c(50)).unwrap(), v.c(50).try_cmpeq(&v.x).unwrap()) ; "equality commutes")]
#[test_case(|v| (v.x.try_cmpne(&v.c(50)).unwrap(), v.c(50).try_cmpne(&v.x).unwrap()) ; "inequality commutes")]
fn verify_equivalence_accepts_equivalent_expressions(pair: fn(&TestVars) -> (Arc<UOp>, Arc<UOp>)) {
    let (lhs, rhs) = pair(&TestVars::new());
    verify_equivalence(&lhs, &rhs).expect("the expressions must be equivalent");
}

/// The model is rendered one `name -> value` line per constant, so a counterexample that
/// distinguishes `x` from `y` must bind *both*: a bare `contains('x')` also matches the
/// `x` inside any other identifier, and an `||` passes on a model that names only one.
#[test]
fn a_disproved_equivalence_yields_a_counterexample_model() {
    let vars = TestVars::new();
    let (lhs, rhs) = (vars.x.lt(&vars.c(50)), vars.x.try_cmpeq(&vars.c(50)).unwrap());
    assert!(matches!(assert_not_equivalent(&lhs, &rhs), CounterExample::Found { .. }));

    match assert_not_equivalent(&vars.x, &vars.y) {
        CounterExample::Found { model, .. } => {
            let binds =
                |name: &str| model.lines().any(|line| line.split("->").next().is_some_and(|lhs| lhs.trim() == name));
            assert!(binds("x") && binds("y"), "the model must bind both inputs: {model}");
        }
        other => panic!("expected a counterexample model, got {other:?}"),
    }
}

/// An expression with no Z3 encoding is a conversion failure, not a counterexample. The
/// mismatched-shape arm (`verify.rs`'s `TypeMismatch -> ConversionFailed`) is reached when
/// both sides convert but one lands on `Int` and the other on `Bool`; `property::oracles`
/// tolerates `ConversionFailed`, so without this row a rewrite that turned an integer
/// expression into a boolean one would read there as "nothing to see".
#[test_case(|_| (UOp::native_const(1.0f32), UOp::native_const(1.0f32)),
    |error| matches!(error, ConversionError::UnsupportedType { .. }) ; "a float has no encoding at all")]
#[test_case(|v: &TestVars| (v.x.clone(), v.x.lt(&v.c(0))),
    |error| matches!(error, ConversionError::TypeMismatch { .. }) ; "an integer and a boolean cannot be compared")]
fn an_unencodable_expression_is_reported_as_a_conversion_failure(
    pair: fn(&TestVars) -> (Arc<UOp>, Arc<UOp>),
    expected: fn(&ConversionError) -> bool,
) {
    let (lhs, rhs) = pair(&TestVars::new());
    match verify_equivalence(&lhs, &rhs) {
        Err(CounterExample::ConversionFailed { source }) => {
            assert!(expected(&source), "unexpected conversion error: {source:?}")
        }
        other => panic!("expected a ConversionFailed, got {other:?}"),
    }
}

/// A `RANGE` and a narrowing `CAST` each mint an unconstrained `Int::fresh_const`.
/// The conversion memo lives on the `Z3Context` (`z3/convert.rs`), not on the single
/// `convert_uop` call, so both sides of an equivalence see the SAME Z3 variable for
/// such a node and `e == e` is provable. With a per-call memo the two sides got two
/// unrelated variables and the verifier refuted even an expression against itself.
///
/// The second half is what stops this passing on a verifier that proves everything:
/// perturbing one side must still be refuted.
#[test_case(|_| UOp::range(UOp::index_const(8), 0).try_add(&UOp::index_const(1)).unwrap() ; "a RANGE mints a fresh loop variable")]
#[test_case(|_| UOp::var("narrow", svod_dtype::DType::Int32, 0, 1000).cast(svod_dtype::DType::Int8) ; "a narrowing CAST mints a fresh bounded variable")]
fn a_freshly_minted_variable_is_shared_across_both_sides(build: Term) {
    let expr = build(&TestVars::new());
    verify_equivalence(&expr, &expr)
        .unwrap_or_else(|error| panic!("an expression must verify against itself: {error:?}\n{}", expr.tree()));

    let perturbed = expr.try_add(&expr.const_like(1)).expect("a shifted operand");
    assert!(
        matches!(verify_equivalence(&expr, &perturbed), Err(CounterExample::Found { .. })),
        "shifting one side must be refuted, or the verifier proves anything\n{}",
        expr.tree()
    );
}
