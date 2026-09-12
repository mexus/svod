//! Symbolic test operands: the named variable set every rewrite table is written
//! against, plus the constant constructors that go with them.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::{ConstValue, Op, UOp, ops};

/// An inclusive integer interval `[lo, hi]`, the bound vocabulary `UOp::var` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeSpec {
    pub lo: i64,
    pub hi: i64,
}

impl RangeSpec {
    /// `x`, `y`: non-negative.
    pub const NON_NEG: Self = Self { lo: 0, hi: 100 };
    /// `a`, `b`: straddling zero.
    pub const SIGNED: Self = Self { lo: -100, hi: 100 };
    /// `n`: excludes zero, so `n % n` is defined.
    pub const NONZERO: Self = Self { lo: 1, hi: 100 };
    /// `i`: an index-typed variable.
    pub const INDEX: Self = Self { lo: 0, hi: 1024 };
}

/// Integer, unsigned, float, bool and index constants in one vocabulary, committed
/// either to [`TestVars`]' dtype family or, for `Arc<UOp>`, to `self.dtype()`.
pub trait Consts {
    fn c(&self, v: i64) -> Arc<UOp>;
    fn u(&self, v: u64) -> Arc<UOp>;
    fn f(&self, v: f64) -> Arc<UOp>;
    fn b(&self, v: bool) -> Arc<UOp>;
    fn ic(&self, v: i64) -> Arc<UOp>;
}

impl Consts for Arc<UOp> {
    fn c(&self, v: i64) -> Arc<UOp> {
        self.const_like(v)
    }

    fn u(&self, v: u64) -> Arc<UOp> {
        self.const_like(v)
    }

    fn f(&self, v: f64) -> Arc<UOp> {
        self.const_like(v)
    }

    fn b(&self, v: bool) -> Arc<UOp> {
        self.const_like(v)
    }

    fn ic(&self, v: i64) -> Arc<UOp> {
        index_const(v)
    }
}

/// An Index-typed constant. `UOp::index_const` builds a WeakInt one; prefer that
/// where the pipeline's own arithmetic is WeakInt, or the fixture stops matching
/// the shape production actually emits.
pub fn index_const(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// The operands the rewrite tables are written against.
#[derive(Debug, Clone)]
pub struct TestVars {
    /// Non-negative integers.
    pub x: Arc<UOp>,
    pub y: Arc<UOp>,
    /// Integers that straddle zero, for the rules that must survive a sign change.
    pub a: Arc<UOp>,
    pub b: Arc<UOp>,
    /// An integer whose range excludes zero, so `n % n` is defined.
    pub n: Arc<UOp>,
    /// Booleans.
    pub p: Arc<UOp>,
    pub q: Arc<UOp>,
    /// An index-typed variable.
    pub i: Arc<UOp>,
    /// A float the analyses can bound.
    pub bounded: Arc<UOp>,
    /// A float the analyses cannot bound (NaN, infinity or signed zero), so no
    /// value-sensitive rule may fire on it.
    pub unknown: Arc<UOp>,
}

impl Default for TestVars {
    fn default() -> Self {
        Self::new()
    }
}

impl TestVars {
    /// Int32 integers, Bool flags, an Index variable, and a Float32 bounded/unboundable pair.
    pub fn new() -> Self {
        Self::typed(DType::Int32)
    }

    /// The same names in another dtype family: `bounded`/`unknown` take `dtype`
    /// when it is a float and Float32 otherwise, `p`/`q` stay Bool, `i` stays Index.
    pub fn typed(dtype: DType) -> Self {
        let float = if dtype.is_float() { dtype.clone() } else { DType::Float32 };
        let var = |name: &str, range: RangeSpec| UOp::var(name, dtype.clone(), range.lo, range.hi);
        Self {
            x: var("x", RangeSpec::NON_NEG),
            y: var("y", RangeSpec::NON_NEG),
            a: var("a", RangeSpec::SIGNED),
            b: var("b", RangeSpec::SIGNED),
            n: var("n", RangeSpec::NONZERO),
            p: UOp::var("p", DType::Bool, 0, 1),
            q: UOp::var("q", DType::Bool, 0, 1),
            i: UOp::var("i", DType::Index, RangeSpec::INDEX.lo, RangeSpec::INDEX.hi),
            bounded: UOp::var("bounded", float.clone(), -1, 1),
            unknown: unboundable(float),
        }
    }

    /// The weak-integer family, where literals have no storage width yet.
    pub fn weak() -> Self {
        Self::typed(DType::WeakInt)
    }

    /// Pin the named variables to constants; panics on a name outside the fixed set.
    #[track_caller]
    pub fn at(&self, point: &[(&str, i64)]) -> Self {
        let mut vars = self.clone();
        for (name, value) in point {
            let slot = match *name {
                "x" => &mut vars.x,
                "y" => &mut vars.y,
                "a" => &mut vars.a,
                "b" => &mut vars.b,
                "n" => &mut vars.n,
                "p" => &mut vars.p,
                "q" => &mut vars.q,
                "i" => &mut vars.i,
                "bounded" => &mut vars.bounded,
                "unknown" => &mut vars.unknown,
                other => panic!("unknown test variable {other:?}; expected x, y, a, b, n, p, q, i, bounded or unknown"),
            };
            let dtype = slot.dtype();
            let constant = match dtype.base() {
                ScalarDType::Bool => ConstValue::Bool(*value != 0),
                _ => ConstValue::Int(*value),
            };
            *slot = UOp::const_(dtype, constant);
        }
        vars
    }
}

impl Consts for TestVars {
    fn c(&self, v: i64) -> Arc<UOp> {
        self.x.const_like(v)
    }

    fn u(&self, v: u64) -> Arc<UOp> {
        self.x.const_like(v)
    }

    fn f(&self, v: f64) -> Arc<UOp> {
        self.bounded.const_like(v)
    }

    fn b(&self, v: bool) -> Arc<UOp> {
        self.p.const_like(v)
    }

    fn ic(&self, v: i64) -> Arc<UOp> {
        index_const(v)
    }
}

/// A `LOAD` from a one-element buffer of `dtype`: a value no analysis can bound.
pub fn unboundable(dtype: DType) -> Arc<UOp> {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 1, dtype);
    let index = UOp::index().buffer(buffer).indices(vec![index_const(0)]).call().expect("scalar INDEX must build");
    UOp::load().index(index).call()
}

pub fn var_name(uop: &Arc<UOp>) -> Option<String> {
    match uop.op() {
        Op::DefineVar(ops::DefineVar { name, .. }) => Some(name.clone()),
        Op::Param(ops::Param { arg, .. }) => arg.name.clone(),
        _ => None,
    }
}

pub fn var_range(uop: &Arc<UOp>) -> Option<(i64, i64)> {
    match uop.op() {
        Op::DefineVar(ops::DefineVar { min_val, max_val, .. }) => Some((*min_val, *max_val)),
        Op::Param(ops::Param { arg, .. }) => {
            let (lo, hi) = arg.vmin_vmax?;
            Some((lo.0.try_int()?, hi.0.try_int()?))
        }
        Op::Range(ops::Range { end, .. }) | Op::Special(ops::Special { end, .. }) => match end.op() {
            Op::Const(value) => Some((0, value.0.try_int()? - 1)),
            _ => None,
        },
        _ => None,
    }
}

/// A rewrite table row: a symbolic expression as a function of [`TestVars`].
pub type Term = fn(&TestVars) -> Arc<UOp>;
