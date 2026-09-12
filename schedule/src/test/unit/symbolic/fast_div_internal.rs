use super::*;

use std::collections::HashMap;

use svod_ir::{Op, UOpKey};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::rewrite::graph_rewrite;
use crate::symbolic::{pm_fold_cast_const, symbolic};
use crate::test::support::prelude::*;

fn var(name: &str, min: i64, max: i64) -> Arc<UOp> {
    UOp::var(name, DType::Int32, min, max)
}

fn c(value: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(value))
}

/// `magic_unsigned(max, d)` returns `(m, s)` with `(x * m) >> s == x / d` for every `x` in
/// `0..=max`, checked exhaustively. The larger maxima are the ones a power-of-two
/// factorization leaves behind (`x / 6` becomes `(x >> 1) / 3`, `x / 12` becomes `(x >> 2) / 3`).
#[test_case(100, 3; "small max")]
#[test_case(500, 3; "max left by factoring out 2")]
#[test_case(1000, 7; "odd divisor")]
#[test_case(10000, 10; "even divisor with a wide range")]
fn magic_unsigned_reproduces_integer_division(max: i64, divisor: i64) {
    let (m, s) = magic_unsigned(max, divisor).expect("a magic number exists");
    for x in 0..=max {
        assert_eq!(x / divisor, ((x as i128 * m as i128) >> s) as i64, "{x} / {divisor}");
    }
}

#[test_case(0, "zero")]
#[test_case(-5, "negative")]
fn magic_unsigned_rejects_non_positive_divisors(divisor: i64, _label: &str) {
    assert!(magic_unsigned(100, divisor).is_none());
}

/// The `fast_idiv` edges: a signed range that straddles zero is not rewritten, a
/// range inside one divisor short-circuits to zero, and a widening needs the cast.
#[test]
fn fast_idiv_declines_and_short_circuits_at_the_range_edges() {
    let signed = UOp::var("x", DType::Int32, -100, 100);
    let div = signed.cdiv(&signed.const_like(7));
    assert!(matches!(fast_division_patterns(HashSet::new()).rewrite(&div, &mut ()), RewriteResult::NoMatch));

    let small = UOp::var("x", DType::Int32, -5, 5);
    assert_const!(fast_idiv(&small, 7, false, &HashSet::new()).expect("|x| < 7 divides to zero"), 0);

    let narrow = UOp::var("x", DType::Int16, 0, i16::MAX as i64);
    let supported = HashSet::from([ScalarDType::Int16, ScalarDType::Int32]);
    // 7 is odd, so the factorisation cannot reduce the range either.
    assert!(fast_idiv(&narrow, 7, true, &supported).is_none());
    assert!(fast_idiv(&narrow, 7, false, &supported).is_some(), "the same row succeeds when the cast is allowed");
}

/// Every non-power-of-two divisor must fold to `value / divisor` over the whole 8-bit range,
/// exhaustively in both the divisor and the dividend: the magic-number replacement is only
/// correct up to the `vmax` it was derived for, and an off-by-one there shows up as a single
/// wrong quotient at one end of the range.
#[test_case(DType::UInt8, ScalarDType::UInt16 ; "unsigned byte")]
#[test_case(DType::Int8, ScalarDType::Int16 ; "signed byte")]
fn fast_division_replacements_are_exhaustive_for_eight_bit_ranges(dtype: DType, wider: ScalarDType) {
    let vmax = if dtype.is_unsigned() { u8::MAX as i64 } else { i8::MAX as i64 };
    let variable = UOp::var("x", dtype.clone(), 0, vmax);
    let supported = HashSet::from([dtype.base(), wider]);
    for divisor in (2..=vmax).filter(|divisor| !(*divisor as u64).is_power_of_two()) {
        let Some(replacement) = fast_idiv(&variable, divisor, false, &supported) else { continue };
        for value in 0..=vmax {
            let substituted = replacement.substitute(&HashMap::from([(
                UOpKey(variable.clone()),
                UOp::const_(dtype.clone(), ConstValue::Int(value)),
            )]));
            let folded = graph_rewrite(&(symbolic() + pm_fold_cast_const()), substituted, &mut ());
            let Op::Const(actual) = folded.op() else {
                panic!("replacement did not fold for {dtype:?} {value}/{divisor}: {}", folded.tree())
            };
            assert_eq!(actual.0.try_int(), Some(value / divisor), "{dtype:?} {value}/{divisor}");
        }
    }
}

/// A width whose multiplier does not fit is computed in the next width and cast back, so the
/// replacement still has the operand's own dtype while the product is evaluated without
/// overflow. The check samples the whole range rather than the low end, where a too-narrow
/// multiplication has not yet wrapped.
#[test_case(DType::Int16, i16::MAX as i64, ScalarDType::Int32 ; "int16 widens to int32")]
#[test_case(DType::Int32, i32::MAX as i64, ScalarDType::Int64 ; "int32 widens to int64")]
fn fast_idiv_widens_to_a_supported_next_width(dtype: DType, vmax: i64, wider: ScalarDType) {
    let x = UOp::var("x", dtype.clone(), 0, vmax);
    let supported = HashSet::from([dtype.base(), wider]);
    for divisor in [7i64, 10, 1000] {
        let replacement = fast_idiv(&x, divisor, false, &supported).expect("a supported widening must exist");
        assert_eq!(replacement.dtype(), dtype);
        for value in (0..=vmax).step_by((vmax / 64) as usize + 1) {
            let computed = fold_at(&replacement, &Bindings::at("x", value)).expect("replacement must fold");
            assert_eq!(computed, ConstValue::Int(value / divisor), "{dtype:?} {value}/{divisor}");
        }
    }
}

/// `fast_division_patterns` must take the caller's supported-dtype set into account rather
/// than assume the widening width exists: an i16 division strength-reduces only when the
/// backend also has i32, and declines when i16 is all it has.
#[test]
fn fast_division_patterns_honour_the_supported_dtype_set() {
    let x = UOp::var("x", DType::Int16, 0, i16::MAX as i64);
    let div = x.cdiv(&x.const_like(7));

    let supported = HashSet::from([ScalarDType::Int16, ScalarDType::Int32]);
    let RewriteResult::Rewritten(replacement) = fast_division_patterns(supported).rewrite(&div, &mut ()) else {
        panic!("i16 division should be strength-reduced through i32");
    };
    assert_eq!(replacement.dtype(), DType::Int16);
    for value in [0, 1, 6, 7, 8, 100, 32767] {
        let computed = fold_at(&replacement, &Bindings::at("x", value)).expect("replacement must fold");
        assert_eq!(computed, ConstValue::Int(value / 7), "{value}/7");
    }

    let narrow = HashSet::from([ScalarDType::Int16]);
    assert!(matches!(fast_division_patterns(narrow).rewrite(&div, &mut ()), RewriteResult::NoMatch));
}

#[test_case(8, 64; "byte periods")]
#[test_case(16, 16; "fixed period")]
#[test_case(64, 4096; "wide periods")]
fn symbolic_divisor_factors_out_of_an_affine_numerator(period_min: i64, period_max: i64) {
    // (N*i + j) // N -> i and (N*i + j) % N -> j for a symbolic N, with j in one period.
    let n = UOp::var("n", DType::Index, period_min, period_max);
    let i = UOp::var("i", DType::Index, 0, 7);
    let j = UOp::var("j", DType::Index, 0, period_min - 1);
    let numerator = n.try_mul(&i).unwrap().try_add(&j).unwrap();

    let quotient = graph_rewrite(symbolic(), numerator.floor_div(&n), &mut ());
    assert_same!(quotient, i);
    let remainder = graph_rewrite(symbolic(), numerator.mod_(&n), &mut ());
    assert_same!(remainder, j);
}

/// One expression per rule ported from tinygrad's `uop/divandmod.py`, with the numerator,
/// the divisor and the folded form taken verbatim from tinygrad's
/// `test/null/test_uop_symbolic.py`.
#[test_case(
    || var("a", 0, 31).mod_(&c(12)).mod_(&c(4)),
    || var("a", 0, 31).mod_(&c(4)) ;
    "remove_nested_mod: (a%12)%4 -> a%4")]
#[test_case(
    || var("a", 0, 31).mul(&c(4)).mod_(&c(12)).mod_(&c(4)),
    || c(0) ;
    "remove_nested_mod: (a*4%12)%4 -> 0")]
#[test_case(
    || var("x", 0, 23).mod_(&c(6)).floor_div(&c(3)),
    || var("x", 0, 23).floor_div(&c(3)).mod_(&c(2)) ;
    "nested_div: x%6//3 -> x//3%2")]
#[test_case(
    || var("x", 0, 23).mod_(&c(12)).floor_div(&c(4)),
    || var("x", 0, 23).floor_div(&c(4)).mod_(&c(3)) ;
    "nested_div: x%12//4 -> x//4%3")]
#[test_case(
    || var("idx", 0, 16).mul(&c(4)).mod_(&c(8)).floor_div(&c(4)),
    || var("idx", 0, 16).mod_(&c(2)) ;
    "nested_div: idx*4%8//4 -> idx%2")]
#[test_case(
    || var("a", 0, 2).mul(&c(4)).floor_div(&c(6)),
    || var("a", 0, 2).mul(&c(2)).floor_div(&c(3)) ;
    "gcd_with_remainder: a*4//6 -> a*2//3")]
#[test_case(
    || var("a", 0, 2).mul(&c(4)).add(&c(2)).floor_div(&c(6)),
    || var("a", 0, 2).mul(&c(2)).add(&c(1)).floor_div(&c(3)) ;
    "gcd_with_remainder: (a*4+2)//6 -> (a*2+1)//3")]
#[test_case(
    || var("a", 0, 2).mul(&c(4)).add(&c(3)).mod_(&c(6)),
    || var("a", 0, 2).mul(&c(2)).add(&c(1)).mod_(&c(3)).mul(&c(2)).add(&c(1)) ;
    "gcd_with_remainder: (a*4+3)%6 -> (a*2+1)%3*2+1")]
#[test_case(
    || var("gidx0", 0, 15).mul(&c(4)).add(&var("lidx0", 0, 3)).mod_(&c(8)),
    || var("gidx0", 0, 15).mod_(&c(2)).mul(&c(4)).add(&var("lidx0", 0, 3)) ;
    "nest_by_factor: (gidx0*4+lidx0)%8 -> lidx0+gidx0%2*4")]
#[test_case(
    || var("a", 0, 10).mul(&c(3)).add(&var("b", 0, 2)).mod_(&c(9)),
    || var("a", 0, 10).mod_(&c(3)).mul(&c(3)).add(&var("b", 0, 2)) ;
    "nest_by_factor: (a*3+b)%9 -> b+a%3*3")]
#[test_case(
    || var("a", 0, 7).mul(&c(4)).add(&var("b", 0, 1)).add(&c(2)).mod_(&c(8)),
    || var("b", 0, 1).add(&var("a", 0, 7).mod_(&c(2)).mul(&c(4))).add(&c(2)) ;
    "nest_by_factor: (a*4+b+2)%8 -> b+a%2*4+2")]
#[test_case(
    || var("a", 0, 10).mul(&c(-4)).add(&c(4)).mod_(&c(8)),
    || var("a", 0, 10).mul(&c(-1)).add(&c(1)).mod_(&c(2)).mul(&c(4)) ;
    "divide_by_gcd: (a*-4+4)%8 -> (a*-1+1)%2*4")]
#[test_case(
    || var("a", 0, 10).mul(&c(-7)).add(&var("b", 70, 100)).floor_div(&c(5)),
    || var("a", 0, 10).mul(&c(3)).add(&var("b", 70, 100)).floor_div(&c(5)).add(&var("a", 0, 10).mul(&c(-2))) ;
    "factor_remainder: (a*-7+b)//5 floors the carry")]
#[test_case(
    || var("b", 0, 100).mul(&c(31)).add(&c(1)).floor_div(&c(18)),
    || var("b", 0, 100).mul(&c(13)).add(&c(1)).floor_div(&c(18)).add(&var("b", 0, 100)) ;
    "factor_remainder: (b*31+1)//18 -> (b*13+1)//18+b")]
fn tinygrad_divmod_examples_fold_to_the_upstream_form(input: fn() -> Arc<UOp>, expected: fn() -> Arc<UOp>) {
    let original = input();
    let folded = rewrite(symbolic(), original.clone());
    let want = rewrite(symbolic(), expected());
    assert_same!(folded, want);

    // Sweep the whole box the row declares: every rule here has at most two small ranges.
    let mut vars: Vec<Arc<UOp>> = Vec::new();
    for node in original.toposort() {
        if var_range(&node).is_some() && !vars.iter().any(|seen| Arc::ptr_eq(seen, &node)) {
            vars.push(node);
        }
    }
    assert!(!vars.is_empty(), "a divmod row must declare at least one ranged operand");
    for bindings in range_points(&vars, 4096) {
        let (lhs, rhs) = (fold_at(&original, &bindings), fold_at(&folded, &bindings));
        assert!(lhs.is_some(), "input did not evaluate at {bindings:?}");
        assert_eq!(lhs, rhs, "identity broken at {bindings:?}");
    }
}
