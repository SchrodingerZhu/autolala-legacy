//! Concrete evaluation of MLIR affine maps and integer sets.
//!
//! The symbolic counterpart lives in `analyzer/src/isl.rs::ExprConverter`, which
//! lowers the same expressions into isl affines. The operand indexing rule is
//! copied from there rather than from `cachegrind-runner`'s C emitter: an
//! affine map's operand list is `[dims.., symbols..]`, so a `Symbol` at position
//! `s` resolves to `operands[num_dims + s]`, while a `Dim` at position `p`
//! resolves to `operands[p]`.
//!
//! Unlike the isl path, evaluation is total: `mod` and `ceildiv` are rejected
//! there but are ordinary arithmetic here. That is deliberate — the sampler is
//! the ground truth, so it should be able to run kernels the symbolic analysis
//! cannot yet model.

use anyhow::{Result, bail};
use raffine::affine::{AffineExpr, AffineExprKind, AffineMap, IntegerSet};
use raffine::tree::ValID;

/// Concrete values for the induction variables and symbols in scope.
///
/// `ivars` is indexed by the `ValID::IVar` number, which `raffine` assigns by
/// nesting depth: `TranslationContext::loop_scope` restores the counter when a
/// loop closes, so sibling loops at the same depth share a slot and shadowing
/// falls out for free. `analyzer/src/isl.rs` relies on the same property when it
/// indexes `ivar_map` by depth.
#[derive(Debug, Default, Clone)]
pub struct Env {
    ivars: Vec<i64>,
    symbols: Vec<i64>,
}

impl Env {
    pub fn new(symbols: Vec<i64>) -> Self {
        Self {
            ivars: Vec::new(),
            symbols,
        }
    }

    pub fn set_ivar(&mut self, index: usize, value: i64) {
        if self.ivars.len() <= index {
            self.ivars.resize(index + 1, 0);
        }
        self.ivars[index] = value;
    }

    pub fn ivar(&self, index: usize) -> Result<i64> {
        match self.ivars.get(index) {
            Some(value) => Ok(*value),
            None => bail!("induction variable i{index} is not in scope"),
        }
    }

    pub fn symbol(&self, index: usize) -> Result<i64> {
        match self.symbols.get(index) {
            Some(value) => Ok(*value),
            None => bail!(
                "symbol s{index} has no binding; pass one with `--symbol s{index}=<value>` \
                 (the program declares {} symbol(s))",
                self.symbols.len()
            ),
        }
    }

    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }
}

/// Resolves one operand slot to its concrete value.
fn operand_value(operands: &[ValID], position: usize, env: &Env) -> Result<i64> {
    let Some(val_id) = operands.get(position) else {
        bail!("affine operand {position} is out of range ({} present)", operands.len());
    };
    match *val_id {
        ValID::IVar(index) => env.ivar(index),
        ValID::Symbol(index) => env.symbol(index),
        ValID::Memref(_) | ValID::Global(_) => {
            bail!("memref {val_id} cannot appear as an affine operand")
        }
    }
}

/// Evaluates one affine expression.
///
/// `num_dims` is the dimension count of the enclosing map or set, used to shift
/// symbol positions past the dims in `operands`.
pub fn eval_expr(
    expr: AffineExpr<'_>,
    operands: &[ValID],
    num_dims: usize,
    env: &Env,
) -> Result<i64> {
    let kind = expr.get_kind();
    match kind {
        AffineExprKind::Constant => match expr.get_value() {
            Some(value) => Ok(value),
            None => bail!("constant affine expression carries no value"),
        },
        AffineExprKind::Dim => {
            let Some(position) = expr.get_position() else {
                bail!("dim affine expression carries no position");
            };
            operand_value(operands, position as usize, env)
        }
        AffineExprKind::Symbol => {
            let Some(position) = expr.get_position() else {
                bail!("symbol affine expression carries no position");
            };
            operand_value(operands, num_dims + position as usize, env)
        }
        AffineExprKind::Add
        | AffineExprKind::Mul
        | AffineExprKind::Mod
        | AffineExprKind::FloorDiv
        | AffineExprKind::CeilDiv => {
            let (Some(lhs), Some(rhs)) = (expr.get_lhs(), expr.get_rhs()) else {
                bail!("binary affine expression is missing an operand");
            };
            let lhs = eval_expr(lhs, operands, num_dims, env)?;
            let rhs = eval_expr(rhs, operands, num_dims, env)?;
            // MLIR restricts the divisor and modulus of affine expressions to
            // positive constants and defines them with floored (not truncated)
            // semantics, which is what the `_euclid` family gives for rhs > 0.
            match kind {
                AffineExprKind::Add => Ok(lhs + rhs),
                AffineExprKind::Mul => Ok(lhs * rhs),
                AffineExprKind::Mod if rhs > 0 => Ok(lhs.rem_euclid(rhs)),
                AffineExprKind::FloorDiv if rhs > 0 => Ok(lhs.div_euclid(rhs)),
                AffineExprKind::CeilDiv if rhs > 0 => Ok(lhs.div_euclid(rhs) + i64::from(lhs.rem_euclid(rhs) != 0)),
                _ => bail!("affine mod/div requires a positive divisor, found {rhs}"),
            }
        }
    }
}

/// Evaluates every result of an affine map into `out`.
pub fn eval_map(map: &AffineMap<'_>, operands: &[ValID], env: &Env, out: &mut Vec<i64>) -> Result<()> {
    out.clear();
    let num_dims = map.num_dims();
    for index in 0..map.num_results() {
        let Some(expr) = map.get_result_expr(index as isize) else {
            bail!("affine map is missing result {index}");
        };
        out.push(eval_expr(expr, operands, num_dims, env)?);
    }
    Ok(())
}

/// Evaluates an `affine.for` lower bound: the maximum over the map's results.
pub fn eval_lower_bound(map: &AffineMap<'_>, operands: &[ValID], env: &Env) -> Result<i64> {
    let mut results = Vec::new();
    eval_map(map, operands, env, &mut results)?;
    match results.into_iter().max() {
        Some(value) => Ok(value),
        None => bail!("affine.for lower bound has no results"),
    }
}

/// Evaluates an `affine.for` upper bound: the minimum over the map's results.
pub fn eval_upper_bound(map: &AffineMap<'_>, operands: &[ValID], env: &Env) -> Result<i64> {
    let mut results = Vec::new();
    eval_map(map, operands, env, &mut results)?;
    match results.into_iter().min() {
        Some(value) => Ok(value),
        None => bail!("affine.for upper bound has no results"),
    }
}

/// Tests membership in an `affine.if` condition.
///
/// An integer set is a conjunction whose constraints are either `expr == 0` or
/// `expr >= 0`, matching the equality/inequality split that
/// `get_access_map_impl` feeds to isl.
pub fn eval_integer_set(set: &IntegerSet<'_>, operands: &[ValID], env: &Env) -> Result<bool> {
    let num_dims = set.num_dims();
    for index in 0..set.num_constraints() {
        let expr = set.get_constraint(index as isize);
        let value = eval_expr(expr, operands, num_dims, env)?;
        let satisfied = if set.is_constraint_equal(index as isize) {
            value == 0
        } else {
            value >= 0
        };
        if !satisfied {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_symbol_binding_names_the_flag() {
        let env = Env::new(vec![10]);
        let message = env.symbol(3).expect_err("unbound").to_string();
        assert!(message.contains("--symbol s3="), "{message}");
        assert!(message.contains("declares 1 symbol"), "{message}");
    }

    #[test]
    fn ivars_are_addressed_by_depth_and_shadow_in_place() {
        let mut env = Env::new(Vec::new());
        env.set_ivar(2, 7);
        assert_eq!(env.ivar(2).expect("set"), 7);
        assert_eq!(env.ivar(0).expect("resized"), 0);
        env.set_ivar(2, 9);
        assert_eq!(env.ivar(2).expect("set"), 9);
    }

    #[test]
    fn memref_operands_are_rejected() {
        let env = Env::new(Vec::new());
        let operands = [ValID::Memref(0)];
        assert!(operand_value(&operands, 0, &env).is_err());
    }
}
