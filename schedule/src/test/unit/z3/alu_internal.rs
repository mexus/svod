use super::*;
use test_case::test_case;

use crate::test::unit::z3::helpers::{assert_valid, assert_valid_in};

/// `z3_cdiv`/`z3_cmod` must match Rust's truncated division and remainder in all four sign
/// quadrants. Each row refutes the negation of the asserted value, so it proves the concrete
/// result: a hypothetical `z3_cdiv ≡ 0` fails every row but `(0, 5, 0, 0)`.
#[test_case(7, 3, 2, 1 ; "both operands positive")]
#[test_case(-7, 3, -2, -1 ; "a negative dividend")]
#[test_case(7, -3, -2, 1 ; "a negative divisor")]
#[test_case(-7, -3, 2, -1 ; "both operands negative")]
#[test_case(6, 3, 2, 0 ; "an exact multiple")]
#[test_case(-6, 3, -2, 0 ; "an exact negative multiple")]
#[test_case(7, 1, 7, 0 ; "division by one")]
#[test_case(-7, -1, 7, 0 ; "division by minus one")]
#[test_case(0, 5, 0, 0 ; "a zero dividend")]
#[test_case(1, 2, 0, 1 ; "a truncated positive quotient")]
#[test_case(-1, 2, 0, -1 ; "a truncated negative quotient")]
fn c_division_and_modulo_match_rust(a: i64, b: i64, cdiv: i64, cmod: i64) {
    let (a, b) = (Int::from_i64(a), Int::from_i64(b));
    assert_valid(&z3_cdiv(&a, &b), &Int::from_i64(cdiv));
    assert_valid(&z3_cmod(&a, &b), &Int::from_i64(cmod));
}

/// The positive-quadrant shortcut is a proof under its assumptions: refuting
/// `cdiv(a, b) != a / b` with `a >= 0, b > 0`.
#[test]
fn c_division_reduces_to_euclidean_division_on_positive_operands() {
    let (a, b) = (Int::new_const("a"), Int::new_const("b"));
    let solver = z3::Solver::new();
    solver.assert(a.ge(Int::from_i64(0)));
    solver.assert(b.gt(Int::from_i64(0)));
    assert_valid_in(&solver, &z3_cdiv(&a, &b), &(&a / &b));
}
