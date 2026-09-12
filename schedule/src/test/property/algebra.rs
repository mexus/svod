//! Algebraic laws of the commutative, associative and idempotent operators, on the op
//! surface the `ir` generators expose but the rest of the suite never used.

use proptest::prelude::*;

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;
use svod_ir::types::BinaryOp;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic;
use crate::test::property::checks::{same_value, swapped};
use crate::test::property::generators::{arb_op_tree_up_to, build_binary};
use crate::test::support::prelude::*;

use svod_ir::test::property::generators::*;

/// The depth-`depth` Int32 tree the commutative and idempotent laws quantify over.
fn arb_int_tree(depth: usize) -> impl Strategy<Value = Arc<UOp>> {
    arb_op_tree_up_to(DType::Int32, depth)
}

proptest! {
    #![proptest_config(cheap())]

    /// Swapping the operands of a commutative op preserves the value at every point.
    #[test]
    fn commutative_ops_agree_with_swapped_operands(
        op in arb_commutative_binary_op(),
        lhs in arb_int_tree(1),
        rhs in arb_int_tree(1),
    ) {
        let Some((forward, backward)) = swapped(op, lhs, rhs) else { return Ok(()) };
        same_value(&graph_rewrite(symbolic(), forward, &mut ()), &graph_rewrite(symbolic(), backward, &mut ()))?;
    }

    /// The fold cannot depend on which constant operand the rule saw first.
    #[test]
    fn commutative_constants_canonicalize_identically(
        op in arb_commutative_binary_op(),
        lhs in arb_const_uop(DType::Int32),
        rhs in arb_const_uop(DType::Int32),
    ) {
        let Some((forward, backward)) = swapped(op, lhs, rhs) else { return Ok(()) };
        let forward = graph_rewrite(symbolic(), forward, &mut ());
        let backward = graph_rewrite(symbolic(), backward, &mut ());
        assert_same!(forward, backward);
    }

    /// Associativity: bracketing the constants the other way reaches the same form.
    #[test]
    fn associativity_is_bracketing_independent(
        op in arb_associative_binary_op(),
        x in arb_var_uop(DType::Int32),
        a in -30i32..30,
        b in -30i32..30,
    ) {
        let constant = |value: i32| UOp::native_const(value);
        let left = build_binary(op, build_binary(op, x.clone(), constant(a)).unwrap(), constant(b)).unwrap();
        // `a op b` is the same constant the left fold collapses to, computed natively.
        let folded = match op {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::And => a & b,
            BinaryOp::Or => a | b,
            BinaryOp::Max => a.max(b),
            _ => unreachable!("the table is restricted to folding ops"),
        };
        let right = graph_rewrite(symbolic(), build_binary(op, x, constant(folded)).unwrap(), &mut ());
        let left = graph_rewrite(symbolic(), left, &mut ());
        assert_same!(left, right);
    }

    /// The idempotent ops collapse `x op x` to `x` for an arbitrary operand tree.
    #[test]
    fn idempotent_ops_collapse_self_application(
        op in prop_oneof![Just(BinaryOp::And), Just(BinaryOp::Or), Just(BinaryOp::Max)],
        x in arb_int_tree(2),
    ) {
        let folded = graph_rewrite(symbolic(), build_binary(op, x.clone(), x.clone()).unwrap(), &mut ());
        let canonical = graph_rewrite(symbolic(), x, &mut ());
        assert_same!(folded, canonical);
    }
}
