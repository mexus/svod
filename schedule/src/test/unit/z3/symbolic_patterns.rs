//! Z3/SMT equivalence pins for `symbolic_simple`.
//!
//! Structural results are covered by `test::property::symbolic_props`; pinned here is that
//! each rewrite preserves semantics, and that untouched expressions still mean what we think.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;
use test_case::test_case;

use crate::test::support::prelude::*;
use crate::test::unit::z3::helpers::verify_roundtrip;
use crate::z3::verify::verify_equivalence;

type Case = fn(&TestVars) -> (Arc<UOp>, Option<Arc<UOp>>);

/// A second and third divisor whose range excludes zero, so a chain of divisions can be
/// written with *every* divisor symbolic; [`TestVars`] carries only one such operand.
///
/// The ranges are small on purpose: nested division by two symbolic divisors is nonlinear,
/// and the solver's time on it is a cliff — [1, 4] over a 16-element numerator proves the
/// theorem in 30 ms; over 40 it took 2 s here, 10 s on the 4-core CI runner, and once the
/// solver gave up.
fn divisor(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::Int32, 1, 4)
}

/// The numerator of the chained-division row, kept inside the divisors' reach.
fn chain_numerator() -> Arc<UOp> {
    UOp::var("chain_x", DType::Int32, 0, 16)
}

/// Rows are closures rather than keys into a string dispatch, so the compiler type-checks
/// every expression. Divisors stay away from zero so the solver never sees undefined division.
#[test_case(|v| (v.x.add(&v.c(0)), None) ; "adding zero")]
#[test_case(|v| (v.x.sub(&v.c(0)), None) ; "subtracting zero")]
#[test_case(|v| (v.x.mul(&v.c(1)), None) ; "multiplying by one")]
#[test_case(|v| (v.x.try_div(&v.c(1)).unwrap(), None) ; "dividing by one")]
#[test_case(|v| (v.x.mul(&v.c(0)), None) ; "multiplying by zero")]
#[test_case(|v| (v.x.mod_(&v.c(1)), Some(v.c(0))) ; "modulo one")]
#[test_case(|v| (v.c(0).try_div(&v.n).unwrap(), Some(v.c(0))) ; "zero numerator")]
#[test_case(|v| (v.x.sub(&v.x), Some(v.c(0))) ; "subtracting itself")]
#[test_case(|v| (v.n.try_div(&v.n).unwrap(), Some(v.c(1))) ; "dividing by itself")]
#[test_case(|v| (v.n.mod_(&v.n), Some(v.c(0))) ; "modulo itself")]
#[test_case(|v| (v.x.add(&v.x), Some(v.c(2).mul(&v.x))) ; "adding itself")]
#[test_case(|v| (v.x.mul(&v.n).try_div(&v.n).unwrap(), Some(v.x.clone())) ; "division cancelling a multiplication")]
// Two *symbolic* divisors. `symbolic_simple` leaves the chain alone, so the reference is
// what carries the claim: Z3 proves `(x / b) / c == x / (b * c)` over the declared ranges,
// a theorem a literal outer divisor reduces to arithmetic the solver decides without it.
#[test_case(|_| (
    chain_numerator().try_div(&divisor("chain_b")).unwrap().try_div(&divisor("chain_c")).unwrap(),
    Some(chain_numerator().try_div(&divisor("chain_b").mul(&divisor("chain_c"))).unwrap()),
) ; "chained division by two symbolic divisors")]
#[test_case(|v| (v.x.mul(&v.c(6)).try_div(&v.n.mul(&v.c(6))).unwrap(), None) ; "common factor in a division")]
#[test_case(|v| (v.c(2).mul(&v.x).add(&v.c(3).mul(&v.x)), Some(v.c(5).mul(&v.x))) ; "combining coefficients")]
#[test_case(|v| (v.x.add(&v.c(3)).add(&v.c(5)), Some(v.x.add(&v.c(8)))) ; "folding added constants")]
#[test_case(|v| (v.x.mul(&v.c(2)).mul(&v.c(3)), Some(v.x.mul(&v.c(6)))) ; "folding multiplied constants")]
// Boolean and unsigned expressions reach the non-integer and unsigned arms of the
// converter, which no integer row touches.
#[test_case(|v| (v.x.lt(&v.c(50)).and_(&v.b(false)), Some(v.b(false))) ; "false absorbs a conjunction")]
#[test_case(|v| (v.x.lt(&v.c(50)).and_(&v.b(true)), Some(v.x.lt(&v.c(50)))) ; "true is the conjunction identity")]
#[test_case(|v| (v.x.lt(&v.c(50)).or_(&v.b(false)), Some(v.x.lt(&v.c(50)))) ; "false is the disjunction identity")]
#[test_case(|_| (UOp::native_const(3u32).try_add(&UOp::native_const(4u32)).unwrap(), Some(UOp::native_const(7u32))) ; "unsigned constants fold")]
fn symbolic_simple_rewrites_preserve_semantics(case: Case) {
    let vars = TestVars::new();
    let (expr, reference) = case(&vars);
    let simplified = verify_roundtrip(Matchers::simple(), expr);
    if let Some(reference) = reference {
        verify_equivalence(&simplified, &reference).expect("the result must equal its reference value");
    }
}
