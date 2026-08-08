// TODO: fix this
#![allow(unused)]
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use ahash::AHashMap;
use denning::MissRatioCurve;
use raffine::{
    affine::{AffineExpr, AffineMap},
    tree::{Tree, ValID},
};
use serde::Serialize;
use symbolica::{atom::Atom, domains::rational_polynomial::FromNumeratorAndDenominator};
use symbolica::{atom::AtomCore, domains::integer::Integer};
use symbolica::{
    domains::{Ring, RingOps},
    symbol,
};
use symbolica::{
    domains::{Field, integer::IntegerRing, rational_polynomial::RationalPolynomialField},
    printer::PrintOptions,
};

use crate::{
    AnalysisContext,
    utils::{Poly, convert_affine_map},
};

fn isize_to_poly<'a>(value: isize, context: &AnalysisContext<'a>) -> Poly {
    let expr = AffineExpr::new_constant(context.rcontext().mlir_context(), value as i64);
    let map = AffineMap::new(
        context.rcontext().mlir_context(),
        0, // num_dims
        0, // num_symbols
        &[expr],
    );
    let step_converted = convert_affine_map(map, &[]).unwrap();
    step_converted[0].clone()
}

pub fn no_coefficient_for_block(tree: &Tree) -> bool {
    match tree {
        Tree::For { body, .. } => no_coefficient_for_block(body),
        Tree::Access { map, operands, .. } => {
            let converted_map = convert_affine_map(*map, operands);

            let mut has_coefficient = false;
            if let Result::Ok(polys) = converted_map {
                for poly in polys {
                    has_coefficient = false;
                    let mut block_position = 0;

                    for (i, var) in poly.numerator.variables.iter().enumerate() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            if index >= block_position {
                                if poly.numerator.coefficients[i] != 1 {
                                    has_coefficient = true;
                                }
                                block_position = index;
                            }
                        }
                    }
                }
            }
            !has_coefficient
        }
        Tree::Block(trees) => {
            for t in trees.iter() {
                if !no_coefficient_for_block(t) {
                    return false;
                }
            }
            true
        }
        Tree::If { .. } => false,
    }
}

fn has_reuses_helper(tree: &Tree, ivar_set: &mut HashSet<usize>) -> bool {
    match tree {
        Tree::For { body, ivar, .. } => {
            let ValID::IVar(id) = ivar else {
                unreachable!("not possible ")
            };
            ivar_set.insert(*id);
            has_reuses_helper(body, ivar_set)
        }
        Tree::Access { map, operands, .. } => {
            let mut reference_vector = vec![0; ivar_set.len()];
            let converted_map = convert_affine_map(*map, operands);
            if let Result::Ok(polys) = converted_map {
                for poly in polys {
                    for var in poly.numerator.variables.iter() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            reference_vector[index] = 1;
                        }
                    }
                }
            }
            let mut has_reuse = false;
            for i in reference_vector {
                if i == 0 {
                    has_reuse = true;
                    break;
                }
            }
            has_reuse
        }
        Tree::Block(trees) => {
            for t in trees.iter() {
                if !has_reuses_helper(t, ivar_set) {
                    return false;
                }
            }
            true
        }
        Tree::If { .. } => false,
    }
}

pub fn has_reuses(tree: &Tree) -> bool {
    let mut ivar_set = HashSet::new();
    has_reuses_helper(tree, &mut ivar_set)
}

pub fn is_perfectly_nested(tree: &Tree) -> bool {
    match tree {
        Tree::For { body, .. } => is_perfectly_nested(body),
        Tree::Block(trees) => {
            if trees.len() == 1 {
                return is_perfectly_nested(trees.first().unwrap());
            }
            trees
                .iter()
                .all(|subtree| matches!(subtree, Tree::Access { .. }))
        }
        Tree::Access { .. } => true,
        Tree::If { .. } => false,
    }
}

pub fn number_of_accesses(tree: &Tree) -> usize {
    match tree {
        Tree::For { body, .. } => number_of_accesses(body),
        Tree::Block(trees) => trees
            .iter()
            .filter(|subtree| matches!(subtree, Tree::Access { .. }))
            .count(),
        Tree::Access { .. } => 1,
        Tree::If { .. } => 0,
    }
}

// reuse interval distribution without block
pub fn get_reuse_interval_distribution<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> HashMap<Poly, Poly> {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            body,
            ivar,
            step,
        } => {
            let lower_bound_converted =
                convert_affine_map(*lower_bound, lower_bound_operands).unwrap();
            let upper_bound_converted =
                convert_affine_map(*upper_bound, upper_bound_operands).unwrap();
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);
            let tmp = upper_bound_converted[0].clone() - lower_bound_converted[0].clone();
            let trip_count = field.div(&tmp, &isize_to_poly(*step, context));
            if reuse_factors.is_empty() {
                reuse_factors.insert(usize::MAX, isize_to_poly(1, context));
            }

            for (_key, value) in reuse_factors.iter_mut() {
                *value = &*value * &trip_count.clone();
            }

            let ValID::IVar(id) = ivar else {
                unreachable!("not possible ")
            };
            reuse_factors.insert(*id, isize_to_poly(1, context));
            trip_counts.insert(*id, trip_count.clone());

            get_reuse_interval_distribution(body, reuse_factors, trip_counts, ref_count, context)
        }
        Tree::Block(trees) => {
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            for subtree in trees.iter() {
                let tmp = get_reuse_interval_distribution(
                    subtree,
                    reuse_factors,
                    trip_counts,
                    trees.len(),
                    context,
                );
                for (i, j) in tmp.iter() {
                    if ri_dist.contains_key(i) {
                        *ri_dist.get_mut(i).unwrap() = ri_dist.get(i).unwrap() + j;
                    } else {
                        ri_dist.insert(i.clone(), j.clone());
                    }
                }
            }
            ri_dist
        }
        Tree::Access { map, operands, .. } => {
            let field = RationalPolynomialField::new(symbolica::domains::integer::IntegerRing);
            let mut reference_vector = vec![0; reuse_factors.len()];
            let converted_map = convert_affine_map(*map, operands);
            let mut block_position = 0;
            let mut access_poly = isize_to_poly(1, context);
            if let Result::Ok(polys) = converted_map {
                for poly in polys {
                    block_position = 0;
                    for var in poly.numerator.variables.iter() {
                        if var.to_string().starts_with('i') {
                            let index_str = &var.to_string()[1..];
                            let index = index_str.parse::<usize>().unwrap();
                            block_position = block_position.max(index + 1);
                            reference_vector[index + 1] = 1;
                            access_poly = &access_poly * trip_counts.get(&index).unwrap();
                        }
                    }
                }
            }
            reference_vector[block_position] = 1;

            let mut portion_factors = vec![];
            let mut p_factor = isize_to_poly(1, context);
            let mut total_access = isize_to_poly(1, context);
            for i in (0..reference_vector.len()).rev() {
                portion_factors.push(field.div(&isize_to_poly(1, context), &p_factor.clone()));
                if reference_vector[i] == 0 && i != 0 {
                    p_factor = &p_factor * trip_counts.get(&(i - 1)).unwrap()
                }
                if i != 0 {
                    total_access = &total_access * trip_counts.get(&(i - 1)).unwrap()
                }
            }
            portion_factors.reverse();

            reference_vector[block_position] = 2;

            let mut shrinked_ref_vec = vec![];
            let mut coefficients = vec![];
            let mut b_found = false;
            let mut last_position_of_zero_group = reference_vector.len() - 1;

            for i in (0..reference_vector.len() - 1).rev() {
                if reference_vector[i] != reference_vector[i + 1] {
                    if i == block_position {
                        if reference_vector[i + 1] == 1 {
                            coefficients.push(0);
                            last_position_of_zero_group = block_position;
                        } else {
                            reference_vector[i] = 1;
                            b_found = true;
                            continue;
                        }
                    } else if reference_vector[i] == 1 {
                        coefficients.push(-1);
                    } else if reference_vector[i] == 0 {
                        coefficients.push(1);
                        if !b_found {
                            last_position_of_zero_group = i;
                        }
                    }
                    shrinked_ref_vec.push(i);
                }
                if i == block_position {
                    b_found = true;
                }
            }

            shrinked_ref_vec.reverse();
            coefficients.reverse();
            if reference_vector[reference_vector.len() - 1] == 0 {
                shrinked_ref_vec.push(reference_vector.len() - 1);
                coefficients.push(1);
            } else if reference_vector[reference_vector.len() - 1] == 2 {
                shrinked_ref_vec.push(reference_vector.len() - 1);
                coefficients.push(0);
                last_position_of_zero_group = reference_vector.len() - 1;
            }

            let n_ref = isize_to_poly(ref_count as isize, context);
            let mut ri_value = isize_to_poly(0, context);

            let block_atom = Atom::var(symbol!("b"));
            let block_poly =
                block_atom.to_rational_polynomial(&IntegerRing::new(), &IntegerRing::new(), None);
            let mut ri_values: Vec<(Poly, usize)> = vec![];

            let mut ri_block_poly = isize_to_poly(-1, context);

            for (place, i) in (shrinked_ref_vec.iter().rev()).enumerate() {
                let factor = if *i != 0 {
                    reuse_factors.get(&(*i - 1)).unwrap()
                } else {
                    reuse_factors.get(&usize::MAX).unwrap()
                };

                let coefficient = coefficients[coefficients.len() - 1 - place];

                if last_position_of_zero_group == *i {
                    ri_value = &ri_value + factor;
                    ri_values.push(((&ri_value.clone() * &n_ref), (*i)));

                    ri_block_poly = &ri_value * &n_ref;
                    let block_factor = if block_position != 0 {
                        reuse_factors.get(&(block_position - 1)).unwrap()
                    } else {
                        reuse_factors.get(&usize::MAX).unwrap()
                    };
                    ri_value = &ri_value - &(&block_poly * block_factor);
                } else if coefficient == -1 {
                    if ri_values.len() > 1 {
                        ri_value = &ri_value - factor;
                    }
                } else {
                    ri_value = &ri_value + factor;
                    ri_values.push(((&ri_value.clone() * &n_ref), (*i)));
                }
            }

            ri_values.reverse();

            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();

            let mut prev_portion = isize_to_poly(0, context);

            let mut addition = isize_to_poly(0, context);
            let mut real = false;
            for (i, j) in ri_values.iter() {
                let p_factor = portion_factors[*j].clone();
                if real {
                    if *j < block_position {
                        let without_block = &p_factor - &prev_portion;
                        let with_block = field.div(&without_block, &block_poly);
                        addition = &addition + &(&without_block - &with_block);
                        ri_dist.insert(i.clone(), field.div(&with_block, &n_ref));
                    } else {
                        let portion = &p_factor - &prev_portion;
                        ri_dist.insert(i.clone(), field.div(&portion, &n_ref));
                    }
                    prev_portion = p_factor;
                } else {
                    real = true;
                    let imaginary_portion = field.div(
                        &field.div(&field.div(&access_poly, &total_access), &n_ref),
                        &block_poly,
                    );
                    if *j < block_position {
                        let without_block = &p_factor - &prev_portion;
                        let with_block = field.div(&without_block, &block_poly);
                        addition = &addition + &(&without_block - &with_block);
                        let tmp = &field.div(&with_block, &n_ref) - &imaginary_portion;
                        if tmp != isize_to_poly(0, context) {
                            ri_dist.insert(i.clone(), tmp);
                        }
                    } else {
                        let portion = &p_factor - &prev_portion;
                        let tmp = &field.div(&portion, &n_ref) - &imaginary_portion;
                        if tmp != isize_to_poly(0, context) {
                            ri_dist.insert(i.clone(), tmp);
                        }
                    }
                    prev_portion = p_factor;
                }
            }

            *ri_dist.get_mut(&ri_block_poly).unwrap() =
                ri_dist.get(&ri_block_poly).unwrap() + &field.div(&addition, &n_ref);
            ri_dist
        }

        Tree::If { .. } => HashMap::new(),
    }
}

#[derive(Serialize)]
struct SaltResult {
    ri_values: Vec<String>,
    portions: Vec<String>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

pub fn get_total_count<'a, I>(accesses: usize, trip_counts: I) -> anyhow::Result<Poly>
where
    I: Iterator<Item = &'a Poly>,
{
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    let accesses = Atom::num(accesses as i64).to_rational_polynomial(&ring, &ring, None);
    let total_count = trip_counts.fold(accesses, |acc, poly| field.mul(&acc, poly));
    Ok(total_count)
}

pub fn get_ri_distro(dist: &[(Poly, Poly)]) -> anyhow::Result<Vec<(isize, f64)>> {
    let mut distro_map = AHashMap::new();
    let empty_const_map = AHashMap::<Atom, f64>::new();
    for (value, portion) in dist.iter() {
        let value = value
            .to_expression()
            .evaluate(&empty_const_map)
            .map_err(|e| anyhow::anyhow!("Failed to evaluate expression: {e}"))?
            as isize;
        let portion = portion
            .to_expression()
            .evaluate(&empty_const_map)
            .map_err(|e| anyhow::anyhow!("Failed to evaluate expression: {e}"))?;
        match distro_map.entry(value) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                *entry.get_mut() += portion;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(portion);
            }
        }
    }
    let mut distro = distro_map.into_iter().chain([(0, 0.0)]).collect::<Vec<_>>();
    distro.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(distro)
}

pub fn subsitute_block_size(poly: &Poly, block_size: usize) -> Poly {
    let vars = poly.get_variables();
    let var_idx = vars.iter().position(|v| {
        v.get_id()
            .map(|id| id.get_stripped_name() == "b")
            .unwrap_or(false)
    });
    match var_idx {
        Some(idx) => {
            let ring = IntegerRing::new();
            let integer = Integer::Single(block_size as i64);
            let numerator = poly.numerator.replace(idx, &integer);
            let denominator = poly.denominator.replace(idx, &integer);
            Poly::from_num_den(numerator, denominator, &ring, true)
        }
        None => poly.clone(),
    }
}

pub fn create_json_output<'a, I>(
    dist: &[(Poly, Poly)],
    accesses: usize,
    trip_counts: I,
    start_time: Instant,
) -> anyhow::Result<String>
where
    I: Iterator<Item = &'a Poly>,
{
    let total_count = get_total_count(accesses, trip_counts)?;
    let ri_values: Vec<String> = dist
        .iter()
        .map(|(poly, _)| {
            poly.to_expression()
                .printer(PrintOptions::latex())
                .to_string()
        })
        .collect();
    let portions: Vec<String> = dist
        .iter()
        .map(|(_, poly)| {
            poly.to_expression()
                .printer(PrintOptions::latex())
                .to_string()
        })
        .collect();
    let total_count = total_count
        .to_expression()
        .printer(PrintOptions::latex())
        .to_string();
    let distribution = get_ri_distro(dist).unwrap_or_default().into_boxed_slice();
    let miss_ratio_curve = MissRatioCurve::new(&distribution);
    let analysis_time = start_time.elapsed();
    let result = SaltResult {
        ri_values,
        portions,
        total_count,
        miss_ratio_curve,
        analysis_time,
    };
    serde_json::to_string(&result).map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {e}"))
}
