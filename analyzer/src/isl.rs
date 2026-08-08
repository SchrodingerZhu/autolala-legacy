use crate::utils::{Poly, get_max_array_dim};
use ahash::AHashMap;
use anyhow::Result;
use barvinok::{
    DimType,
    aff::Affine,
    constraint::Constraint,
    list::List,
    local_space::LocalSpace,
    map::{BasicMap, Map},
    point::Point,
    polynomial::{PiecewiseQuasiPolynomial, QuasiPolynomial, Term},
    set::Set,
    space::Space,
    value::Value,
};
use comfy_table::Table;
use denning::MissRatioCurve;
use raffine::{
    affine::{AffineExpr, AffineExprKind, AffineMap},
    tree::{Tree, ValID},
};
use std::path::Path;
use std::{
    collections::HashSet,
    collections::hash_map::Entry,
    num::NonZero,
    time::{Duration, Instant},
};

use serde::Serialize;
use symbolica::{atom::Atom, domains::Field, domains::integer::IntegerRing};
use symbolica::{atom::AtomCore, symbol};
use symbolica::{
    domains::{Ring, RingOps, rational_polynomial::RationalPolynomialField},
    printer::PrintOptions,
};

use crate::AnalysisContext;

struct ConvertedIVar<'a> {
    lower_bound: AffineMap<'a>,
    step_size: i64,
    index: usize,
    operands: &'a [ValID],
}

type IVarMap<'a> = Vec<ConvertedIVar<'a>>;

pub fn get_timestamp_space<'a, 'b: 'a>(
    num_params: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
) -> Result<Set<'b>> {
    let mut ivar_map = Vec::new();
    let res = get_timestamp_space_impl(num_params, 0, context, tree, &mut ivar_map);
    tracing::trace!("timestamp space: {res:?}");
    res
}

fn align_sets<'i, 'a: 'i>(
    longest: Set<'a>,
    depth: usize,
    iter: impl Iterator<Item = &'i mut Set<'a>>,
    add_dim_constraint: bool,
) -> Result<()> {
    let space = longest.get_space()?;
    let longest_dim = longest.n_dim()?;
    let local_space = LocalSpace::try_from(space.clone())?;
    for (idx, i) in iter.enumerate() {
        let length = i.n_dim()?;
        let mut s = i
            .clone()
            .insert_dims(DimType::Out, length, longest_dim - length)?;
        for j in length..longest_dim {
            // add constraint eq 0
            let constraint = Constraint::new_equality(local_space.clone())?.set_coefficient_si(
                DimType::Out,
                j as i32,
                1,
            )?;
            s = s.add_constraint(constraint)?;
            // Padding dims must carry the longest statement's dim names, or
            // the union erases the names and downstream output prints
            // auto-generated ones that nothing else can refer to.
            if let Some(name) = space.get_dim_name(DimType::Out, j as u32)? {
                let name = name.to_string();
                s = s.set_dim_name(DimType::Out, j as u32, &name)?;
            }
        }
        if add_dim_constraint {
            let current_dim_eq_i = Constraint::new_equality(local_space.clone())?
                .set_coefficient_si(DimType::Out, depth as i32, 1)?
                .set_constant_si(-(idx as i32))?;
            s = s.add_constraint(current_dim_eq_i)?;
        }
        // The aligned copy must always be written back; writing it back only
        // when the dim constraint was requested silently discarded the
        // alignment for the `add_dim_constraint == false` (if/else) callers.
        *i = s;
    }
    Ok(())
}

fn align_maps<'i, 'a: 'i>(
    longest: Map<'a>,
    depth: usize,
    iter: impl Iterator<Item = &'i mut Map<'a>>,
    add_dim_constraint: bool,
) -> Result<()> {
    let space = longest.get_space()?;
    let local_space = LocalSpace::try_from(space.clone())?;
    let longest_length = longest.get_space()?.dim(DimType::In)?;
    for (idx, i) in iter.enumerate() {
        let length = i.get_space()?.dim(DimType::In)?;
        let mut s = i.clone().add_dims(DimType::In, longest_length - length)?;
        for j in length..longest_length {
            // add constraint eq 0
            let constraint = Constraint::new_equality(local_space.clone())?.set_coefficient_si(
                DimType::In,
                j as i32,
                1,
            )?;
            s = s.add_constraint(constraint)?;
            // See align_sets: padding dims keep the longest statement's names.
            if let Some(name) = space.get_dim_name(DimType::In, j as u32)? {
                let name = name.to_string();
                s = s.set_dim_name(DimType::In, j as u32, &name)?;
            }
        }
        if add_dim_constraint {
            let current_dim_eq_i = Constraint::new_equality(local_space.clone())?
                .set_coefficient_si(DimType::In, depth as i32, 1)?
                .set_constant_si(-(idx as i32))?;
            s = s.add_constraint(current_dim_eq_i)?;
        }
        // Always write the aligned copy back (see align_sets).
        *i = s;
    }
    Ok(())
}

fn get_timestamp_space_impl<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    ivar_map: &mut IVarMap<'a>,
) -> Result<Set<'b>> {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            step,
            body,
            ..
        } => {
            {
                let step_size = *step as i64;
                let index = depth;
                let ivar = ConvertedIVar {
                    lower_bound: *lower_bound,
                    step_size,
                    index,
                    operands: lower_bound_operands,
                };
                ivar_map.push(ivar);
            }
            let set = get_timestamp_space_impl(num_params, depth + 1, context, body, ivar_map)?;
            let space = set.get_space()?;
            let lower_converter =
                ExprConverter::new(space.clone(), *lower_bound, lower_bound_operands, ivar_map)?;
            let lower_bound =
                lower_converter.convert_polynomial(lower_bound.get_result_expr(0).ok_or_else(
                    || anyhow::anyhow!("invalid affine expression: at least one result expression"),
                )?)?;
            let upper_converter =
                ExprConverter::new(space.clone(), *upper_bound, upper_bound_operands, ivar_map)?;
            let upper_bound =
                upper_converter.convert_polynomial(upper_bound.get_result_expr(0).ok_or_else(
                    || anyhow::anyhow!("invalid affine expression: at least one result expression"),
                )?)?;
            let local_space = LocalSpace::try_from(space.clone())?;
            let step = Value::int_from_si(context.bcontext(), *step as i64)?;
            let step = Affine::val_on_domain(local_space.clone(), step)?;
            let trip_size = upper_bound
                .checked_sub(lower_bound)?
                .checked_div(step)?
                .ceil()?;
            let ge_0 = Constraint::new_inequality(local_space.clone())?.set_coefficient_si(
                DimType::Out,
                depth as i32,
                1,
            )?;
            let affine_minus_ivar = trip_size
                .checked_sub(Affine::var_on_domain(
                    local_space.clone(),
                    DimType::Out,
                    depth as u32,
                )?)?
                .checked_sub(Affine::val_on_domain(
                    local_space.clone(),
                    Value::int_from_si(context.bcontext(), 1)?,
                )?)?;
            let affine_minus_ivar_gt_0 = Constraint::new_inequality_from_affine(affine_minus_ivar);
            ivar_map.pop();
            Ok(set
                .add_constraint(ge_0)?
                .add_constraint(affine_minus_ivar_gt_0)?
                .set_dim_name(
                    DimType::Out,
                    depth as u32,
                    &format!("i{}", ivar_map.len() + 1),
                )?)
        }
        Tree::Block(stmts) => {
            if stmts.is_empty() {
                // An empty block contributes no timestamps; erroring here
                // (the old "no sets found") rejected programs with empty
                // branches instead of treating them as empty schedules.
                let space = Space::set(context.bcontext(), num_params as u32, depth as u32)?;
                return Ok(Set::empty(space)?);
            }
            let mut sub_sets = stmts
                .iter()
                .map(|stmt| {
                    get_timestamp_space_impl(num_params, depth + 1, context, stmt, ivar_map)
                })
                .collect::<Result<Vec<_>>>()?;
            let longest = sub_sets
                .iter()
                .max_by_key(|set| set.n_dim().unwrap_or_default())
                .ok_or_else(|| anyhow::anyhow!("no sets found"))?
                .clone();
            let space = longest.get_space()?;
            align_sets(longest, depth, sub_sets.iter_mut(), true)?;
            let total_set = sub_sets
                .into_iter()
                .try_fold(Set::empty(space.clone())?, |acc, set| acc.union(set))?
                .set_dim_name(
                    DimType::Out,
                    depth as u32,
                    &format!("t{}", depth - ivar_map.len()),
                )?;
            Ok(total_set)
        }
        Tree::Access { .. } => {
            let space = Space::set(context.bcontext(), num_params as u32, depth as u32)?;
            Ok(Set::universe(space)?)
        }
        Tree::If {
            condition,
            operands,
            r#then,
            r#else,
        } => {
            let then_set = get_timestamp_space_impl(num_params, depth, context, r#then, ivar_map)?;
            let else_set = if let Some(r#else) =
                r#else.filter(|x| !matches!(**x, Tree::Block(ref v) if v.is_empty()))
            {
                get_timestamp_space_impl(num_params, depth, context, r#else, ivar_map)?
            } else {
                Set::empty(then_set.get_space()?)?
            };
            // similar to block, align with longest set
            let longest = if then_set.n_dim()? > else_set.n_dim()? {
                then_set.clone()
            } else {
                else_set.clone()
            };

            let mut subsets = [then_set, else_set];
            let space = longest.get_space()?;
            align_sets(longest, depth, subsets.iter_mut(), false)?;
            let conv = ExprConverter::new_with_dims(
                space.clone(),
                condition.num_dims(),
                operands,
                ivar_map,
            )?;
            let mut then_cond = Set::universe(space.clone())?;
            for i in 0..condition.num_constraints() {
                let expr = condition.get_constraint(i as isize);
                let converted = conv.convert_polynomial(expr)?;
                let constraint = if condition.is_constraint_equal(i as isize) {
                    Constraint::new_equality_from_affine(converted)
                } else {
                    Constraint::new_inequality_from_affine(converted)
                };
                then_cond = then_cond.add_constraint(constraint)?;
            }
            let complement = then_cond.clone().complement()?;
            let [x, y] = subsets;
            x.intersect(then_cond)?
                .union(y.intersect(complement)?)
                .map_err(Into::into)
        }
    }
}

pub fn get_access_map<'a, 'b: 'a>(
    num_params: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    block_size: usize,
    num_sets: NonZero<usize>,
) -> Result<Map<'b>> {
    let mut ivar_map = Vec::new();
    let max_array_dim = get_max_array_dim(tree)?;
    get_access_map_impl(
        num_params,
        0,
        context,
        tree,
        &mut ivar_map,
        block_size,
        max_array_dim,
        num_sets,
    )
}

fn get_access_map_impl<'a, 'b: 'a>(
    num_params: usize,
    depth: usize,
    context: &AnalysisContext<'b>,
    tree: &Tree<'a>,
    ivar_map: &mut IVarMap<'a>,
    block_size: usize,
    max_array_dim: usize,
    num_sets: NonZero<usize>,
) -> Result<Map<'b>> {
    match tree {
        Tree::For {
            body,
            lower_bound,
            lower_bound_operands,
            step,
            ..
        } => {
            {
                let step_size = *step as i64;
                let index = depth;
                let ivar = ConvertedIVar {
                    lower_bound: *lower_bound,
                    step_size,
                    index,
                    operands: lower_bound_operands,
                };
                ivar_map.push(ivar);
            }
            let res = get_access_map_impl(
                num_params,
                depth + 1,
                context,
                body,
                ivar_map,
                block_size,
                max_array_dim,
                num_sets,
            )?;
            ivar_map.pop();
            Ok(res.set_dim_name(
                DimType::In,
                depth as u32,
                &format!("i{}", ivar_map.len() + 1),
            )?)
        }
        Tree::Block(stmts) => {
            if stmts.is_empty() {
                // Empty block: no accesses. Mirror the range layout used by
                // Tree::Access (memref id + padded array dims + optional set
                // tag) so the empty map unions cleanly with siblings.
                let range_dims =
                    (max_array_dim + 1 + usize::from(num_sets.get() > 1)) as u32;
                let space = Space::new(
                    context.bcontext(),
                    num_params as u32,
                    depth as u32,
                    range_dims,
                )?;
                return Ok(Map::empty(space)?);
            }
            let mut sub_maps = stmts
                .iter()
                .map(|stmt| {
                    get_access_map_impl(
                        num_params,
                        depth + 1,
                        context,
                        stmt,
                        ivar_map,
                        block_size,
                        max_array_dim,
                        num_sets,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let longest = sub_maps
                .iter()
                .max_by_key(|map| {
                    map.get_space()
                        .and_then(|s| s.dim(DimType::In))
                        .unwrap_or_default()
                })
                .ok_or_else(|| anyhow::anyhow!("no maps found"))?
                .clone();
            let space = longest.get_space()?;
            align_maps(longest.clone(), depth, sub_maps.iter_mut(), true)?;
            let total_map = sub_maps
                .into_iter()
                .try_fold(Map::empty(space)?, |acc, set| acc.union(set))?
                .set_dim_name(
                    DimType::In,
                    depth as u32,
                    &format!("t{}", depth - ivar_map.len()),
                )?;
            Ok(total_map)
        }
        Tree::Access {
            map,
            operands,
            memref,
            ..
        } => {
            let domain_space = Space::set(context.bcontext(), num_params as u32, depth as u32)?;
            let converter = ExprConverter::new(domain_space.clone(), *map, operands, ivar_map)?;
            let mut aff_list = List::new(domain_space.context_ref(), map.num_results());
            let ValID::Memref(memref) = *memref else {
                return Err(anyhow::anyhow!("invalid access map: invalid memref"));
            };
            let val = Value::int_from_ui(domain_space.context_ref(), memref as u64)?;
            let aff = Affine::val_on_domain_space(domain_space.clone(), val)?;
            aff_list.push(aff);
            // align array dimension
            for _ in 0..max_array_dim - map.num_results() {
                let val = Value::int_from_ui(domain_space.context_ref(), 0)?;
                let aff = Affine::val_on_domain_space(domain_space.clone(), val)?;
                aff_list.push(aff);
            }
            for i in 0..map.num_results() {
                let expr = map
                    .get_result_expr(i as isize)
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid result"))?;
                tracing::debug!("expr: {}", expr);
                let mut aff = converter.convert_polynomial(expr)?;
                if block_size > 1 && i == map.num_results() - 1 {
                    let block_size =
                        Value::int_from_ui(domain_space.context_ref(), block_size as u64)?;
                    let block_size = Affine::val_on_domain_space(domain_space.clone(), block_size)?;
                    aff = aff.checked_div(block_size)?;
                    aff = aff.floor()?;
                }
                if num_sets.get() > 1 && i == map.num_results() - 1 {
                    // add set dimension
                    let num_sets_value =
                        Value::int_from_ui(domain_space.context_ref(), num_sets.get() as u64)?;
                    let set_tag = aff.clone().modulo(num_sets_value)?;
                    aff_list.push(set_tag);
                }
                aff_list.push(aff);
            }
            let basic_map = BasicMap::from_affine_list(domain_space, aff_list)?;
            Ok(basic_map.try_into()?)
        }
        Tree::If {
            condition,
            operands,
            r#then,
            r#else,
        } => {
            let then_map = get_access_map_impl(
                num_params,
                depth,
                context,
                r#then,
                ivar_map,
                block_size,
                max_array_dim,
                num_sets,
            )?;
            let else_map = if let Some(r#else) =
                r#else.filter(|x| !matches!(**x, Tree::Block(ref v) if v.is_empty()))
            {
                get_access_map_impl(
                    num_params,
                    depth,
                    context,
                    r#else,
                    ivar_map,
                    block_size,
                    max_array_dim,
                    num_sets,
                )?
            } else {
                Map::empty(then_map.get_space()?)?
            };
            // similar to block, align with longest set
            let longest = if then_map.dim(DimType::In)? > else_map.dim(DimType::In)? {
                then_map.clone()
            } else {
                else_map.clone()
            };

            let mut submaps = [then_map, else_map];
            let dom_space = longest.clone().domain()?.get_space()?;
            align_maps(longest, depth, submaps.iter_mut(), false)?;
            let conv = ExprConverter::new_with_dims(
                dom_space.clone(),
                condition.num_dims(),
                operands,
                ivar_map,
            )?;
            let mut then_cond = Set::universe(dom_space)?;
            for i in 0..condition.num_constraints() {
                let expr = condition.get_constraint(i as isize);
                let converted = conv.convert_polynomial(expr)?;
                let constraint = if condition.is_constraint_equal(i as isize) {
                    Constraint::new_equality_from_affine(converted)
                } else {
                    Constraint::new_inequality_from_affine(converted)
                };
                then_cond = then_cond.add_constraint(constraint)?;
            }
            let complement = then_cond.clone().complement()?;
            let [x, y] = submaps;
            x.intersect_domain(then_cond)?
                .union(y.intersect_domain(complement)?)
                .map_err(Into::into)
        }
    }
}

struct ExprConverter<'isl, 'mlir, 'map> {
    local_space: LocalSpace<'isl>,
    ivar_map: &'map IVarMap<'mlir>,
    symbol_shift: usize,
    operands: &'mlir [ValID],
}

impl<'isl, 'mlir, 'map> ExprConverter<'isl, 'mlir, 'map> {
    pub fn new(
        space: Space<'isl>,
        map: AffineMap<'mlir>,
        operands: &'mlir [ValID],
        ivar_map: &'map IVarMap<'mlir>,
    ) -> Result<Self> {
        let local_space = LocalSpace::try_from(space)?;
        let symbol_shift = map.num_dims();
        Ok(Self {
            local_space,
            symbol_shift,
            operands,
            ivar_map,
        })
    }

    pub fn new_with_dims(
        space: Space<'isl>,
        symbol_shift: usize,
        operands: &'mlir [ValID],
        ivar_map: &'map IVarMap<'mlir>,
    ) -> Result<Self> {
        let local_space = LocalSpace::try_from(space)?;
        Ok(Self {
            local_space,
            symbol_shift,
            operands,
            ivar_map,
        })
    }

    pub fn convert_polynomial<'a>(&self, expr: AffineExpr<'a>) -> Result<Affine<'isl>> {
        let kind = expr.get_kind();
        match kind {
            AffineExprKind::Add => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_add(rhs)?)
            }
            AffineExprKind::Mod => Err(anyhow::anyhow!(
                "invalid affine expression: mod is not supported"
            )),
            AffineExprKind::Mul => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_mul(rhs)?)
            }
            AffineExprKind::Symbol | AffineExprKind::Dim => {
                let position = expr.get_position().ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid position")
                })?;
                self.position_to_var(position as usize, kind)
            }
            AffineExprKind::CeilDiv => todo!(),
            AffineExprKind::Constant => {
                let constant = expr.get_value().ok_or_else(|| {
                    anyhow::anyhow!("invalid affine expression: invalid constant")
                })?;
                let value = Value::int_from_si(self.local_space.context_ref(), constant)?;
                Ok(Affine::val_on_domain(self.local_space.clone(), value)?)
            }
            AffineExprKind::FloorDiv => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid rhs"))?;
                let lhs = self.convert_polynomial(lhs)?;
                let rhs = self.convert_polynomial(rhs)?;
                Ok(lhs.checked_div(rhs)?.floor()?)
            }
        }
    }

    pub fn position_to_var(
        &self,
        mut position: usize,
        kind: AffineExprKind,
    ) -> Result<Affine<'isl>> {
        let dim_type = if matches!(kind, raffine::affine::AffineExprKind::Symbol) {
            position += self.symbol_shift;
            DimType::Param
        } else {
            DimType::Out
        };

        let val_id = *self
            .operands
            .get(position)
            .ok_or_else(|| anyhow::anyhow!("invalid affine expression: invalid position"))?;
        match val_id {
            ValID::Symbol(n) => Ok(Affine::var_on_domain(
                self.local_space.clone(),
                dim_type,
                n as u32,
            )?),
            ValID::IVar(n) => {
                let ivar = Affine::var_on_domain(
                    self.local_space.clone(),
                    dim_type,
                    self.ivar_map[n].index as u32,
                )?;
                let step_size =
                    Value::int_from_si(self.local_space.context_ref(), self.ivar_map[n].step_size)?;
                let step_size = Affine::val_on_domain(self.local_space.clone(), step_size)?;
                let converter = Self {
                    local_space: self.local_space.clone(),
                    ivar_map: self.ivar_map,
                    symbol_shift: self.ivar_map[n].lower_bound.num_dims(),
                    operands: self.ivar_map[n].operands,
                };
                let lower_bound = converter.convert_polynomial(
                    self.ivar_map[n]
                        .lower_bound
                        .get_result_expr(0)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "invalid affine expression: at least one result expression"
                            )
                        })?,
                )?;
                Ok(ivar.checked_mul(step_size)?.checked_add(lower_bound)?)
            }
            _ => Err(anyhow::anyhow!("invalid affine expression")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RIProcessor<'a> {
    pw_qpoly: PiecewiseQuasiPolynomial<'a>,
    /// True support of the counted relation (its domain set). The cardinality
    /// pw_qpolynomial may legally represent "count = 0" points *inside* a
    /// coalesced piece cell (isl is free to choose that representation), so a
    /// piece domain can strictly contain the accesses that actually reuse.
    /// Every piece is restricted to this support before it is counted;
    /// otherwise zero-count points are tallied as warm accesses.
    support: Set<'a>,
}

#[derive(Clone, Debug)]
pub struct Piece<'a> {
    domain: Set<'a>,
    qpoly: QuasiPolynomial<'a>,
}

#[derive(Clone, Debug)]
pub struct DistItem<'a> {
    qpoly: QuasiPolynomial<'a>,
    cardinality: PiecewiseQuasiPolynomial<'a>,
}

impl<'a> RIProcessor<'a> {
    pub fn new(pw_qpoly: PiecewiseQuasiPolynomial<'a>, support: Set<'a>) -> Self {
        RIProcessor { pw_qpoly, support }
    }
    pub fn get_all_pieces(&self) -> Result<Box<[Piece<'a>]>, barvinok::Error> {
        let mut pieces: Vec<Piece<'_>> = Vec::new();
        self.pw_qpoly.foreach_piece(|qpoly, domain| {
            let mut to_merge = None;
            for existing in pieces.iter().enumerate() {
                // TODO: this can be sped up. currently O(n^2)
                if qpoly.plain_is_equal(&existing.1.qpoly)? {
                    to_merge = Some(existing.0);
                    break;
                }
            }
            if let Some(index) = to_merge {
                pieces[index].domain = pieces[index].domain.clone().union(domain)?;
            } else {
                pieces.push(Piece { domain, qpoly });
            }
            Ok(())
        })?;
        Ok(pieces.into_boxed_slice())
    }
    fn get_processed_pieces(&self) -> Result<Box<[Piece<'a>]>, barvinok::Error> {
        let mut pieces = Vec::new();
        for mut piece in self.get_all_pieces()?.into_vec() {
            // Restrict each piece to the true support of the counted
            // relation and drop pieces that become empty (see the `support`
            // field doc for why the raw piece domains over-approximate).
            piece.domain = piece.domain.intersect(self.support.clone())?;
            if piece.domain.clone().is_empty()? {
                continue;
            }
            let involved_dims = piece.involved_input_dims()?;
            // move involved_dims into params space (currently for domain only)
            let mut domain = piece.domain.clone();
            for (shift, dim) in involved_dims.iter().enumerate() {
                // TODO: this should not unwrap
                let num_params = domain.dim(DimType::Param).unwrap();
                domain = domain.move_dims(
                    DimType::Param,
                    num_params,
                    DimType::Out,
                    *dim - shift as u32,
                    1,
                )?;
            }
            piece.domain = domain;
            pieces.push(piece);
        }
        Ok(pieces.into_boxed_slice())
    }
    pub fn get_distribution(&self) -> Result<Box<[DistItem<'a>]>, barvinok::Error> {
        let mut pieces = self.get_processed_pieces()?;
        let mut dist_items: Vec<DistItem<'_>> = Vec::new();
        for piece in pieces.iter_mut() {
            let cardinality = piece.cardinality()?;
            dist_items.push(DistItem {
                qpoly: piece.qpoly.clone(),
                cardinality,
            });
        }
        Ok(dist_items.into_boxed_slice())
    }
}

impl<'a> Piece<'a> {
    pub fn domain(&self) -> Set<'a> {
        self.domain.clone()
    }
    pub fn cardinality(&self) -> Result<PiecewiseQuasiPolynomial<'a>, barvinok::Error> {
        self.domain().cardinality()
    }
    /// Input dimensions the value quasi-polynomial *actually* uses, judged
    /// from its rendered expression. `isl_qpolynomial_involves_dims` also
    /// reports dims that only occur in unused div definitions left over from
    /// earlier computations, which would needlessly expose extra iterators as
    /// distribution parameters. Unnamed dims (which the rendering cannot
    /// identify) fall back to the conservative isl answer.
    fn involved_input_dims(&self) -> Result<Box<[u32]>, barvinok::Error> {
        let space = self.qpoly.get_space()?;
        let rendered = convert_quasi_poly(self.qpoly.clone())?;
        // `false`: skip function-name symbols (e.g. `floor`) but still
        // collect the symbols inside their arguments.
        let symbols = rendered
            .to_expression()
            .get_all_symbols(false)
            .iter()
            .map(|s| s.get_stripped_name().to_string())
            .collect::<HashSet<String>>();
        let dims = self.qpoly.get_dim(DimType::In)?;
        let mut res = Vec::with_capacity(dims as usize);
        for i in 0..dims {
            match space.get_dim_name(DimType::In, i)? {
                Some(name) => {
                    if symbols.contains(name) {
                        res.push(i);
                    }
                }
                None => {
                    if self.qpoly.involves_dims(DimType::In, i, 1)? {
                        res.push(i);
                    }
                }
            }
        }
        Ok(res.into_boxed_slice())
    }
}

/// Extract the unique piece of the total-count pw_qpolynomial together with
/// its domain guard. A single piece keeps its guard (the value is only valid
/// on that domain), and a genuinely piecewise total cannot be collapsed into
/// one rational polynomial, so that case is an explicit error instead of the
/// old behavior of silently keeping whichever piece was visited last.
fn total_count_piece<'a>(
    total: &PiecewiseQuasiPolynomial<'a>,
) -> Result<(QuasiPolynomial<'a>, Set<'a>)> {
    let mut piece = None;
    let mut num_pieces = 0usize;
    total.foreach_piece(|qpoly, domain| {
        num_pieces += 1;
        if piece.is_none() {
            piece = Some((qpoly, domain));
        }
        Ok(())
    })?;
    match num_pieces {
        0 => Err(anyhow::anyhow!("no total count found")),
        1 => Ok(piece.expect("one piece")),
        n => Err(anyhow::anyhow!(
            "total count is piecewise ({n} pieces); cannot collapse it into a single polynomial"
        )),
    }
}

/// The constraint part of a set's textual form (empty for the universe).
fn domain_guard(domain: &Set<'_>) -> String {
    let raw = format!("{domain:?}");
    raw.split("{  : ")
        .nth(1)
        .unwrap_or_default()
        .split(" }")
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub fn create_table(dist: &[DistItem], total: PiecewiseQuasiPolynomial<'_>) -> Result<Table> {
    use comfy_table::ContentArrangement;
    use comfy_table::modifiers::UTF8_ROUND_CORNERS;
    use comfy_table::presets::UTF8_FULL;
    let (total_count, _total_domain) = total_count_piece(&total)?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["RI Value", "Count", "Symbol Range", "Portion"]);
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let value_str = format!("{value}");
        item.cardinality.foreach_piece(|qpoly, domain| {
            let poly = convert_quasi_poly(qpoly.clone())?;
            let count = format!("{poly}");
            let range = domain_guard(&domain);
            let portion = field.div(&poly, &total_count_poly);
            let portion_str = format!("{portion}");
            table.add_row([&value_str, &count, &range, &portion_str]);
            Ok(())
        })?;
    }
    Ok(table)
}

struct QpolyConverter<'a> {
    ring: IntegerRing,
    field: RationalPolynomialField<IntegerRing, u32>,
    space: Space<'a>,
}

/// The `floor` function symbol used to render isl div dimensions. Created
/// once, with a numeric evaluation hook, so that expressions containing
/// floors remain evaluable by `evaluate_poly` (symbolica has no builtin
/// floor). The `OnceLock` avoids re-registering the hook, which symbolica
/// would treat as redefining the symbol's attributes.
fn floor_symbol() -> symbolica::atom::Symbol {
    use symbolica::atom::EvaluationInfo;
    static FLOOR: std::sync::OnceLock<symbolica::atom::Symbol> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        symbol!(
            "floor",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0].floor())
        )
    })
}

fn convert_quasi_poly<'a>(
    qpoly: QuasiPolynomial<'a>,
) -> std::result::Result<Poly, barvinok::Error> {
    let space = qpoly.get_space()?;
    let converter = QpolyConverter::new(space)?;
    converter.quasi_poly_to_rational_poly(qpoly)
}

impl<'a> QpolyConverter<'a> {
    pub fn new(space: Space<'a>) -> std::result::Result<Self, barvinok::Error> {
        let ring = IntegerRing::new();
        let field = RationalPolynomialField::new(ring);
        Ok(QpolyConverter { ring, field, space })
    }
    fn value_to_rational_poly(&self, value: Value<'a>) -> Poly {
        let num = value.numerator();
        let denom = value.denominator();
        let num: Atom = Atom::num(num);
        let denom: Atom = Atom::num(denom);
        let num = num.to_rational_polynomial(&self.ring, &self.ring, None);
        let denom = denom.to_rational_polynomial(&self.ring, &self.ring, None);
        self.field.div(&num, &denom)
    }

    /// The rational value of an isl affine expression:
    /// `constant + Σ coeff·dim + Σ coeff·floor(nested div)`.
    /// The `*_val` getters already return fully divided rational values, so
    /// no further division by `get_denominator_val` may be applied (doing so
    /// was the double-division bug that turned `floor((i+1)/8)` into
    /// `(i+1)/64`).
    fn aff_to_rational_poly(&self, aff: Affine<'a>) -> Result<Poly, barvinok::Error> {
        let mut num = self.field.zero();
        let cst = self.value_to_rational_poly(aff.get_constant_val()?);
        if !cst.is_zero() {
            num = self.field.add(&num, &cst);
        }
        for ty in [DimType::Param, DimType::In] {
            let n = aff.dim(ty)?;
            for i in 0..n {
                let coeff = aff.get_coefficient_val(ty, i as i32)?;
                let coeff = self.value_to_rational_poly(coeff);
                if coeff.is_zero() {
                    continue;
                }
                let name = self.space.get_dim_name(ty, i)?.unwrap_or("unnamed");
                let v = Atom::var(symbol!(name));
                let term = self.field.mul(
                    &coeff,
                    &v.to_rational_polynomial(&self.ring, &self.ring, None),
                );
                num = self.field.add(&num, &term);
            }
        }
        // An affine may reference (earlier) div dimensions; each use is a
        // coefficient times the floor of that div's own affine.
        let div_dims = aff.dim(DimType::Div)?;
        for i in 0..div_dims {
            let coeff = self.value_to_rational_poly(aff.get_coefficient_val(DimType::Div, i as i32)?);
            if coeff.is_zero() {
                continue;
            }
            let div_aff = aff
                .get_div(i as i32)
                .ok_or(barvinok::Error::VariablePositionOutOfBounds)?;
            let div = self.floor_div(div_aff)?;
            num = self.field.add(&num, &self.field.mul(&coeff, &div));
        }
        Ok(num)
    }

    /// An isl div dimension is `floor(<affine>)`; dropping the floor turns an
    /// integer-valued step term into a wrong rational value. Symbolica 2.2
    /// has no builtin floor symbol, so the floor is rendered as an opaque
    /// `floor(...)` function atom; `to_rational_polynomial` maps such atoms
    /// to fresh polynomial variables (PolyVariable::Function), which is the
    /// minimal faithful counterpart of the reference's `FormulaExpr::floor`.
    /// The symbol carries a numeric evaluation hook (see [`floor_symbol`]) so
    /// the miss-ratio distribution path can still evaluate the expression at
    /// integer points. When the affine is already integral (denominator 1)
    /// the floor is the identity, and the plain affine is kept so the result
    /// stays a pure polynomial in that case.
    fn floor_div(&self, aff: Affine<'a>) -> Result<Poly, barvinok::Error> {
        let denominator = aff.get_denominator_val()?;
        let inner = self.aff_to_rational_poly(aff)?;
        if denominator.is_one()? {
            return Ok(inner);
        }
        let atom = symbolica::function!(Atom::var(floor_symbol()), inner.to_expression());
        Ok(atom.to_rational_polynomial(&self.ring, &self.ring, None))
    }

    fn term_to_rational_poly(&self, term: Term<'a>) -> std::result::Result<Poly, barvinok::Error> {
        let mut poly = self.value_to_rational_poly(term.coefficient()?);
        // isl terms expose their monomial exponents on the *Param* and *Out*
        // ("set") slots only, while the enclosing quasi-polynomial space
        // names those same set dims as In dims. Counting via the space's In
        // slot silently dropped set-dim monomials whenever the term carried
        // more set dims than the space had In dims (a value `-7 + 2*i1`
        // rendered as `-5`), so both the counts and the exponent reads come
        // from the term itself; only the *names* come from the space (In
        // first, then Out for genuine set spaces), bounds-checked because the
        // term may have more set dims than the space names.
        let space_in_dims = self.space.dim(DimType::In)?;
        let space_out_dims = self.space.dim(DimType::Out)?;
        for ty in [DimType::Param, DimType::Out] {
            let dims = term.dim(ty)?;
            for i in 0..dims {
                let exp = term.exponent(ty, i)?;
                if exp == 0 {
                    continue;
                }
                let mut name = None;
                if matches!(ty, DimType::Param) {
                    name = self.space.get_dim_name(DimType::Param, i)?;
                } else {
                    if i < space_in_dims {
                        name = self.space.get_dim_name(DimType::In, i)?;
                    }
                    if name.is_none() && i < space_out_dims {
                        name = self.space.get_dim_name(DimType::Out, i)?;
                    }
                }
                let symbol = symbol!(name.unwrap_or("unnamed"));
                let exp = Atom::num(exp as i64);
                let atom = Atom::var(symbol).pow(exp);
                let atom = atom.to_rational_polynomial(&self.ring, &self.ring, None);
                poly = self.field.mul(&poly, &atom);
            }
        }
        let div_dims = term.dim(DimType::Div)?;
        for i in 0..div_dims {
            let exp = term.exponent(DimType::Div, i)?;
            if exp > 0 {
                let div_aff = term.get_div(i)?;
                let div_poly = self.floor_div(div_aff)?;
                let p = self.field.pow(&div_poly, exp as u64);
                poly = self.field.mul(&poly, &p);
            }
        }
        Ok(poly)
    }

    fn quasi_poly_to_rational_poly(
        &self,
        qpoly: QuasiPolynomial<'a>,
    ) -> std::result::Result<Poly, barvinok::Error> {
        let mut poly = Atom::num(0).to_rational_polynomial(&self.ring, &self.ring, None);
        qpoly.foreach_term(|term| {
            let term_poly = self.term_to_rational_poly(term)?;
            poly = self.field.add(&poly, &term_poly);
            Ok(())
        })?;
        Ok(poly)
    }
}

pub(crate) fn ensure_set_name<'a>(mut set: Set<'a>) -> Result<Set<'a>> {
    let params = set.n_param()?;
    let dims = set.n_dim()?;
    for i in 0..params {
        if !set.has_dim_name(DimType::Param, i)? {
            set = set.set_dim_name(DimType::Param, i, &format!("p{i}"))?;
        }
    }
    for i in 0..dims {
        if !set.has_dim_name(DimType::Out, i)? {
            set = set.set_dim_name(DimType::Out, i, &format!("i{i}"))?;
        }
    }
    Ok(set)
}

pub(crate) fn ensure_map_domain_name<'a>(mut map: Map<'a>) -> Result<Map<'a>> {
    let params = map.dim(DimType::Param)?;
    let in_dims = map.dim(DimType::In)?;
    for i in 0..params {
        if !map.has_dim_name(DimType::Param, i)? {
            map = map.set_dim_name(DimType::Param, i, &format!("p{i}"))?;
        }
    }
    for i in 0..in_dims {
        if !map.has_dim_name(DimType::In, i)? {
            map = map.set_dim_name(DimType::In, i, &format!("i{i}"))?;
        }
    }
    Ok(map)
}

#[derive(Clone, Debug)]
struct ConvertedDistItem<'a> {
    value: Poly,
    portion: Poly,
    domain: Set<'a>,
}

#[derive(serde::Serialize)]
struct SerializableDistItem {
    qpoly: String,
    cardinality: String,
}

#[derive(serde::Serialize)]
struct SerializableDistro {
    total: String,
    items: Vec<SerializableDistItem>,
}

pub fn save_all_dist_items<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
    path: &Path,
) -> Result<()> {
    let mut items = Vec::new();
    for item in dist.iter() {
        let cardinality_str = format!("{:?}", item.cardinality);
        let qpoly_str = format!("{:?}", item.qpoly);
        items.push(SerializableDistItem {
            qpoly: qpoly_str,
            cardinality: cardinality_str,
        });
    }
    let total_str = format!("{:?}", total);
    let distro = SerializableDistro {
        total: total_str,
        items,
    };
    let mut file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &distro)?;
    Ok(())
}

impl<'a> ConvertedDistItem<'a> {
    fn evaluate(&self, point: Point<'a>) -> Result<(isize, f64)> {
        let value = evaluate_poly(&self.value, &point)? as isize;
        let portion = evaluate_poly(&self.portion, &point)?;
        Ok((value, portion))
    }
    fn add_to_dist(&self, dist: &mut AHashMap<isize, f64>) -> Result<()> {
        let mut points = Vec::new();
        self.domain.foreach_point(|point| {
            points.push(point);
            Ok(())
        })?;
        points.into_iter().try_for_each(|point| {
            let (value, portion) = self.evaluate(point)?;
            match dist.entry(value) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += portion;
                }
                Entry::Vacant(entry) => {
                    entry.insert(portion);
                }
            }
            Ok(())
        })
    }
}

fn evaluate_poly<'a>(poly: &Poly, point: &Point<'a>) -> Result<f64> {
    let expr = poly.to_expression();
    // Collect symbols from the full expression rather than from the
    // polynomial's variable list: dims that only occur inside rendered
    // `floor(...)` atoms are part of an opaque function variable and would
    // otherwise be missing from the evaluation map.
    let name_map = expr
        .get_all_symbols(false)
        .iter()
        .map(|x| (x.get_stripped_name().to_string(), Atom::var(*x)))
        .collect::<AHashMap<String, Atom>>();
    let space = point.get_space()?;
    let dims = space.dim(DimType::Out)?;
    let mut const_map = AHashMap::new();
    for i in 0..dims {
        let Some(name) = space.get_dim_name(DimType::Out, i)? else {
            continue;
        };
        let Some(atom) = name_map.get(name).cloned() else {
            continue;
        };
        let val = point.get_coordinate_val(DimType::Out, i as i32)?.to_f64();
        const_map.insert(atom, val);
    }
    expr.evaluate(&const_map)
        .map_err(|msg| anyhow::anyhow!("failed to evaluate polynomial: {msg}"))
}

fn convert_bounded_set<'a>(mut set: Set<'a>) -> Result<Set<'a>> {
    let num_params = set.n_param()?;
    for i in 0..num_params {
        let name = set.get_dim_name(DimType::Param, i)?.unwrap_or_default();
        if name.starts_with("p") {
            return Err(anyhow::anyhow!(
                "all parameters must be instantiated for numerical analysis"
            ));
        }
    }
    // move all parameters to set space
    set = set.move_dims(DimType::Out, 0, DimType::Param, 0, num_params)?;
    if !set.is_bounded()? {
        return Err(anyhow::anyhow!("set {set:?} is not bounded"));
    }
    Ok(set)
}

pub fn get_distro<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
) -> Result<Box<[(isize, f64)]>> {
    let dist = convert_dist(dist, total)?;
    let mut result = AHashMap::new();
    for item in dist.iter() {
        item.add_to_dist(&mut result)?;
    }
    let mut vector = vec![(0, 0.0)];
    vector.extend(result.iter().map(|(k, v)| (*k, *v)));
    vector.sort_unstable_by_key(|a| a.0);
    Ok(vector.into_boxed_slice())
}

fn convert_dist<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
) -> Result<Box<[ConvertedDistItem<'a>]>> {
    let (total_count, _total_domain) = total_count_piece(&total)?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut output = Vec::new();
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let mut res = Ok(());
        item.cardinality.foreach_piece(|qpoly, domain| {
            if res.is_err() {
                return Ok(());
            }
            let poly = convert_quasi_poly(qpoly.clone())?;
            let portion = field.div(&poly, &total_count_poly);
            match convert_bounded_set(domain) {
                Ok(domain) => {
                    let item = ConvertedDistItem {
                        value: value.clone(),
                        portion,
                        domain,
                    };
                    output.push(item);
                }
                Err(e) => res = Err(e),
            }
            Ok(())
        })?;
        res?;
    }
    Ok(output.into_boxed_slice())
}

#[derive(Serialize)]
struct BarvinokResult {
    ri_values: Box<[String]>,
    symbol_ranges: Box<[String]>,
    counts: Box<[String]>,
    portions: Box<[String]>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

pub fn create_json_output<'a>(
    dist: &[DistItem<'a>],
    total: PiecewiseQuasiPolynomial<'a>,
    start_time: Instant,
) -> Result<String> {
    let distribution = get_distro(dist, total.clone()).unwrap_or_default();
    let (total_count, total_domain) = total_count_piece(&total)?;
    let total_count_poly = convert_quasi_poly(total_count)?;
    let mut ri_values = Vec::new();
    let mut symbol_ranges = Vec::new();
    let mut counts = Vec::new();
    let mut portions = Vec::new();
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    for item in dist.iter() {
        let value = convert_quasi_poly(item.qpoly.clone())?;
        let value_str = format!("{}", value.to_expression().printer(PrintOptions::latex()));
        item.cardinality.foreach_piece(|qpoly, domain| {
            let poly = convert_quasi_poly(qpoly.clone())?;
            let count = format!("{}", poly.to_expression().printer(PrintOptions::latex()));
            let range = domain_guard(&domain);
            let portion = field.div(&poly, &total_count_poly);
            let portion_str = format!("{}", portion.to_expression().printer(PrintOptions::latex()));
            ri_values.push(value_str.clone());
            symbol_ranges.push(range);
            counts.push(count);
            portions.push(portion_str);
            Ok(())
        })?;
    }
    let ri_values = ri_values.into_boxed_slice();
    let symbol_ranges = symbol_ranges.into_boxed_slice();
    let counts = counts.into_boxed_slice();
    let portions = portions.into_boxed_slice();
    let total_count_expr = format!(
        "{}",
        total_count_poly
            .to_expression()
            .printer(PrintOptions::latex())
    );
    // A single-piece total keeps its domain guard: the expression is only
    // valid on the piece domain, and rendering it bare would silently claim
    // validity everywhere (e.g. outside the parameter range).
    let guard = domain_guard(&total_domain);
    let total_count = if guard.is_empty() {
        total_count_expr
    } else {
        format!("\\left[{guard}\\right] \\Rightarrow {total_count_expr}")
    };
    let miss_ratio_curve = MissRatioCurve::new(&distribution);
    let analysis_time = start_time.elapsed();
    let result = BarvinokResult {
        ri_values,
        symbol_ranges,
        counts,
        portions,
        miss_ratio_curve,
        total_count,
        analysis_time,
    };
    let json = serde_json::to_string(&result)
        .map_err(|e| anyhow::anyhow!("failed to serialize to JSON: {e}"))?;
    Ok(json)
}
