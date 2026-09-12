//! Assertion macros: they capture the call-site expression for their panic
//! message, and the pattern form stays a real `match` arm for exhaustiveness.

/// `assert_op!(x, Op::Reduce(..))`, or `assert_op!(x, Op::Const(v) => v)` to bind.
#[macro_export]
macro_rules! assert_op {
    ($uop:expr, $pat:pat => $binding:ident) => {
        match $uop.op() {
            $pat => $binding,
            other => ::std::panic!("assert_op! expected {}, got {:?}\n{}", ::std::stringify!($pat), other, $uop.tree()),
        }
    };
    ($uop:expr, $pat:pat) => {
        match $uop.op() {
            $pat => {}
            other => ::std::panic!("assert_op! expected {}, got {:?}\n{}", ::std::stringify!($pat), other, $uop.tree()),
        }
    };
}

/// `unwrap_op!(x, Op::Call(c) => c)` — [`assert_op!`]'s binding form, returning the binding.
#[macro_export]
macro_rules! unwrap_op {
    ($uop:expr, $pat:pat => $binding:ident) => {
        match $uop.op() {
            $pat => $binding,
            other => ::std::panic!("unwrap_op! expected {}, got {:?}\n{}", ::std::stringify!($pat), other, $uop.tree()),
        }
    };
}

/// `assert_same!(a, b)` — the two operands must be the same interned node.
#[macro_export]
macro_rules! assert_same {
    ($a:expr, $b:expr) => {
        if !::std::sync::Arc::ptr_eq(&$a, &$b) {
            ::std::panic!("assert_same! failed\nleft:  {}\nright: {}", $a.tree(), $b.tree());
        }
    };
}

/// `assert_const!(x, 7)`, `assert_const!(x, 1.5)`, `assert_const!(x, true)` or a `ConstValue`.
#[macro_export]
macro_rules! assert_const {
    ($uop:expr, $expected:expr) => {{
        let expected: ::svod_ir::types::ConstValue = ::core::convert::Into::into($expected);
        match $uop.op() {
            ::svod_ir::Op::Const(value) => {
                ::core::assert_eq!(value.0, expected, "assert_const! failed: got {:?}\n{}", value.0, $uop.tree())
            }
            other => ::std::panic!("assert_const! expected Const({:?}), got {:?}\n{}", expected, other, $uop.tree()),
        }
    }};
}

/// `assert_axis!(sched, Global: 2)` — the scheduler has exactly that many axes of the named type.
#[macro_export]
macro_rules! assert_axis {
    ($sched:expr, $axis:ident : $count:expr) => {{
        let actual = $crate::test::support::count::axis_count(&$sched, ::svod_ir::AxisType::$axis);
        ::core::assert_eq!(
            actual,
            $count,
            "assert_axis! expected {} {:?} axes, got {}",
            $count,
            ::svod_ir::AxisType::$axis,
            actual
        );
    }};
}
