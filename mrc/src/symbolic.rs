//! Instantiation of a fully symbolic reuse-interval distribution.
//!
//! The analyzer's `--json` report carries a `symbolic` section: every piece
//! of the distribution as a reuse-interval expression, a count expression and
//! an isl domain over the program parameters and the piece's free iterators.
//! Given concrete parameter values this fixes them in each domain, enumerates
//! the remaining integer points with isl, evaluates the two expressions at
//! each point and accumulates the histogram -- the same procedure the
//! analyzer runs internally for constant-size programs, now on the saved
//! symbolic result instead of a re-derivation.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use barvinok::{DimType, aff::Affine, set::{BasicSet, Set}};
use denning::MissRatioCurve;
use serde::Deserialize;
use symbolica::atom::{Atom, AtomView, Symbol};
use symbolica::parser::ParseSettings;

#[derive(Deserialize)]
pub struct SymbolicDistribution {
    pub params: Vec<String>,
    pub total: SymbolicTotal,
    pub items: Vec<SymbolicItem>,
}

/// Piecewise access total. Reports written before totals were piecewise
/// carry a single `{expr, domain}`; both shapes are read.
#[derive(Deserialize)]
#[serde(from = "TotalShape")]
pub struct SymbolicTotal {
    pub pieces: Vec<SymbolicTotalPiece>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TotalShape {
    Pieces { pieces: Vec<SymbolicTotalPiece> },
    Single(SymbolicTotalPiece),
}

impl From<TotalShape> for SymbolicTotal {
    fn from(shape: TotalShape) -> Self {
        match shape {
            TotalShape::Pieces { pieces } => SymbolicTotal { pieces },
            TotalShape::Single(piece) => SymbolicTotal { pieces: vec![piece] },
        }
    }
}

#[derive(Deserialize)]
pub struct SymbolicTotalPiece {
    pub expr: String,
    pub domain: String,
}

#[derive(Deserialize)]
pub struct SymbolicItem {
    pub value: String,
    pub pieces: Vec<SymbolicPiece>,
}

#[derive(Deserialize)]
pub struct SymbolicPiece {
    pub count: String,
    pub domain: String,
}

/// The distribution at one parameter point.
pub struct Instantiated {
    pub curve: MissRatioCurve,
    /// Access total at the parameter point.
    pub total: f64,
    /// Distinct reuse-interval values in the histogram.
    pub support: usize,
    /// Integer points enumerated over all pieces.
    pub points: usize,
    /// Seconds spent enumerating and evaluating the pieces.
    pub histogram_seconds: f64,
    /// Seconds spent in the Denning recursion and associativity conversion.
    pub curve_seconds: f64,
}

/// Variables of an expression bound to slots of a value vector, so a point
/// is evaluated without any name lookup.
struct Slots(HashMap<Symbol, usize>);

impl Slots {
    fn new() -> Self {
        Slots(HashMap::new())
    }
    fn slot(&mut self, symbol: Symbol) -> usize {
        let next = self.0.len();
        *self.0.entry(symbol).or_insert(next)
    }
    /// The slot of the variable called `name`, if the expressions use it.
    fn by_name(&self, name: &str) -> Option<usize> {
        self.0
            .iter()
            .find(|(symbol, _)| symbol.get_stripped_name() == name)
            .map(|(_, slot)| *slot)
    }
}

pub fn parse_symbol_binding(text: &str) -> Result<(String, i64)> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| anyhow!("expected NAME=VALUE, got `{text}`"))?;
    let value = value
        .trim()
        .parse::<i64>()
        .with_context(|| format!("value of `{name}` is not an integer: `{value}`"))?;
    Ok((name.trim().to_string(), value))
}

pub fn instantiate(
    dist: &SymbolicDistribution,
    values: &HashMap<String, i64>,
    associativity: usize,
) -> Result<Instantiated> {
    for param in &dist.params {
        if !values.contains_key(param) {
            bail!("parameter `{param}` has no value; pass `--symbol {param}=<int>`");
        }
    }
    let total = barvinok::Context::new().scope(|ctx| -> Result<f64> {
        // The piece whose domain contains the parameter point; outside every
        // piece the derivation made no claim, so refuse rather than guess.
        for piece in &dist.total.pieces {
            let domain = Set::from_str(ctx, &piece.domain)
                .map_err(|e| anyhow!("total domain `{}`: {e:?}", piece.domain))?;
            if fix_params(domain, values)?.is_empty()? {
                continue;
            }
            let atom = parse_expr(&piece.expr)?;
            let mut slots = Slots::new();
            let expr = compile(atom.as_view(), &mut slots);
            let mut env = vec![0.0; slots.0.len()];
            bind_params(&slots, values, &mut env);
            return evaluate(&expr, &env);
        }
        bail!(
            "the parameter point lies outside the derivation's domain(s): {}",
            dist.total
                .pieces
                .iter()
                .map(|p| p.domain.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        )
    })?;
    if total <= 0.0 {
        bail!("access total evaluates to {total} at this parameter point");
    }

    // `MRC_ISL_POINTS=1` forces isl's own point enumeration; the fast path
    // is checked against it in the tests.
    let use_isl_points = std::env::var_os("MRC_ISL_POINTS").is_some();
    let started = Instant::now();
    let mut histogram: HashMap<isize, f64> = HashMap::new();
    let mut points = 0usize;
    barvinok::Context::new().scope(|ctx| -> Result<()> {
        for item in &dist.items {
            let value_atom = parse_expr(&item.value)?;
            for piece in &item.pieces {
                let count_atom = parse_expr(&piece.count)?;
                let mut slots = Slots::new();
                let value_expr = compile(value_atom.as_view(), &mut slots);
                let count_expr = compile(count_atom.as_view(), &mut slots);
                let domain = Set::from_str(ctx, &piece.domain)
                    .map_err(|e| anyhow!("piece domain `{}`: {e:?}", piece.domain))?;
                let domain = fix_params(domain, values)?;
                if domain.is_empty()? {
                    continue;
                }
                // The free iterators are isl parameters of the piece; make
                // them set dimensions so isl can enumerate them.
                let n_param = domain.n_param()?;
                let domain = domain.move_dims(DimType::Out, 0, DimType::Param, 0, n_param)?;
                if !domain.is_bounded()? {
                    bail!("piece domain is unbounded after fixing the parameters: {domain:?}");
                }
                let space = domain.get_space()?;
                let dims = space.dim(DimType::Out)?;
                // Which set dimension feeds which slot; dims the expressions
                // never mention are enumerated but not read.
                let mut bound_dims = Vec::new();
                for i in 0..dims {
                    if let Some(name) = space.get_dim_name(DimType::Out, i)?
                        && let Some(slot) = slots.by_name(name)
                    {
                        bound_dims.push((i as i32, slot));
                    }
                }
                let mut env = vec![0.0; slots.0.len()];
                bind_params(&slots, values, &mut env);
                let mut visit = |coordinates: &[i64]| -> Result<()> {
                    for (dim, slot) in &bound_dims {
                        env[*slot] = coordinates[*dim as usize] as f64;
                    }
                    let value = evaluate(&value_expr, &env)?;
                    let count = evaluate(&count_expr, &env)?;
                    if count == 0.0 {
                        return Ok(());
                    }
                    *histogram.entry(value.round() as isize).or_insert(0.0) += count;
                    points += 1;
                    Ok(())
                };
                match (!use_isl_points).then(|| Polytope::extract(&domain)).transpose()?.flatten() {
                    Some(polytope) => polytope.for_each_point(&mut visit)?,
                    None => {
                        // Fallback: let isl enumerate (slower, always correct).
                        let mut enumerated = Vec::new();
                        domain.foreach_point(|point| {
                            enumerated.push(point);
                            Ok(())
                        })?;
                        let mut coordinates = vec![0i64; dims as usize];
                        for point in enumerated {
                            for (i, c) in coordinates.iter_mut().enumerate() {
                                let v = point.get_coordinate_val(DimType::Out, i as i32)?;
                                *c = v.numerator() / v.denominator();
                            }
                            visit(&coordinates)?;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    let histogram_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();

    let mut distribution: Vec<(isize, f64)> = vec![(0, 0.0)];
    distribution.extend(histogram.iter().map(|(ri, count)| (*ri, *count / total)));
    distribution.sort_unstable_by_key(|(ri, _)| *ri);
    let mut curve = MissRatioCurve::new(&distribution);
    if associativity > 1 {
        curve = curve.compute_assoc(associativity);
    }
    Ok(Instantiated {
        curve,
        total,
        support: histogram.len(),
        points,
        histogram_seconds,
        curve_seconds: started.elapsed().as_secs_f64(),
    })
}

fn bind_params(slots: &Slots, values: &HashMap<String, i64>, env: &mut [f64]) {
    for (name, value) in values {
        if let Some(slot) = slots.by_name(name) {
            env[slot] = *value as f64;
        }
    }
}

/// Fixes every parameter of `set` that has a binding in `values`.
fn fix_params<'a>(mut set: Set<'a>, values: &HashMap<String, i64>) -> Result<Set<'a>> {
    for i in (0..set.n_param()?).rev() {
        let Some(name) = set.get_dim_name(DimType::Param, i)? else {
            continue;
        };
        if let Some(value) = values.get(name) {
            let value = i32::try_from(*value)
                .with_context(|| format!("parameter `{name}` does not fit isl's i32"))?;
            set = set.fix_si(DimType::Param, i, value)?;
        }
    }
    Ok(set)
}

fn parse_expr(text: &str) -> Result<Atom> {
    Atom::parse(text, "mrc", ParseSettings::default())
        .map_err(|e| anyhow!("cannot parse expression `{text}`: {e}"))
}

/// An expression compiled to a tree over slots, evaluated with no lookups.
enum Expr {
    Num(f64),
    Var(usize),
    Floor(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    /// A small integer power, the common case (`p0^2`), without `powf`.
    Powi(Box<Expr>, i32),
    Mul(Vec<Expr>),
    Add(Vec<Expr>),
    /// A function symbolica knows but this evaluator does not.
    Unsupported(String),
}

/// Symbolica's own `evaluate` needs the `floor` symbol registered with an
/// evaluation hook in *this* process under the namespace the analyzer used;
/// matching on the stripped name instead keeps the JSON independent of
/// which crate wrote it.
fn compile(view: AtomView<'_>, slots: &mut Slots) -> Expr {
    match view {
        AtomView::Num(_) => match number(view) {
            Ok(value) => Expr::Num(value),
            Err(e) => Expr::Unsupported(e.to_string()),
        },
        AtomView::Var(var) => Expr::Var(slots.slot(var.get_symbol())),
        AtomView::Fun(fun) => {
            let symbol = fun.get_symbol();
            if symbol.get_stripped_name() == "floor" && fun.get_nargs() == 1 {
                let arg = fun.iter().next().expect("one argument");
                Expr::Floor(Box::new(compile(arg, slots)))
            } else {
                Expr::Unsupported(format!(
                    "function `{}` with {} argument(s)",
                    symbol.get_stripped_name(),
                    fun.get_nargs()
                ))
            }
        }
        AtomView::Pow(pow) => {
            let (base, exp) = pow.get_base_exp();
            let base = Box::new(compile(base, slots));
            match number(exp) {
                Ok(e) if e.fract() == 0.0 && e.abs() <= 64.0 => Expr::Powi(base, e as i32),
                _ => Expr::Pow(base, Box::new(compile(exp, slots))),
            }
        }
        AtomView::Mul(mul) => Expr::Mul(mul.iter().map(|f| compile(f, slots)).collect()),
        AtomView::Add(add) => Expr::Add(add.iter().map(|t| compile(t, slots)).collect()),
    }
}

fn evaluate(expr: &Expr, env: &[f64]) -> Result<f64> {
    Ok(match expr {
        Expr::Num(value) => *value,
        Expr::Var(slot) => env[*slot],
        Expr::Floor(arg) => evaluate(arg, env)?.floor(),
        Expr::Pow(base, exp) => evaluate(base, env)?.powf(evaluate(exp, env)?),
        Expr::Powi(base, exp) => evaluate(base, env)?.powi(*exp),
        Expr::Mul(factors) => {
            let mut product = 1.0;
            for factor in factors {
                product *= evaluate(factor, env)?;
            }
            product
        }
        Expr::Add(terms) => {
            let mut sum = 0.0;
            for term in terms {
                sum += evaluate(term, env)?;
            }
            sum
        }
        Expr::Unsupported(what) => bail!("unsupported expression: {what}"),
    })
}

/// A rational number atom as f64. Rendered and re-read rather than matched on
/// symbolica's coefficient variants, so big integers and fractions take the
/// same path.
fn number(view: AtomView<'_>) -> Result<f64> {
    let text = format!("{view}");
    let parse = |s: &str| {
        s.trim()
            .parse::<f64>()
            .map_err(|_| anyhow!("cannot read number `{text}`"))
    };
    match text.split_once('/') {
        Some((num, den)) => Ok(parse(num)? / parse(den)?),
        None => parse(&text),
    }
}

/// One basic set as integer constraints, scanned over its bounding box
/// with integer arithmetic. isl's `foreach_point` costs microseconds per
/// point (a callback, a point object, a value per coordinate); pieces with
/// two free iterators reach millions of points, so this matters.
struct BasicPolytope {
    /// Existentially quantified dims, each `floor((c0 + Σ c·x + Σ c·div) / den)`
    /// in terms of the set dims and the earlier divs.
    divs: Vec<Div>,
    constraints: Vec<LinearConstraint>,
}

struct Div {
    dim_coefficients: Vec<i64>,
    div_coefficients: Vec<i64>,
    constant: i64,
    denominator: i64,
}

struct LinearConstraint {
    dim_coefficients: Vec<i64>,
    div_coefficients: Vec<i64>,
    constant: i64,
    equality: bool,
}

struct Polytope {
    dims: usize,
    lower: Vec<i64>,
    upper: Vec<i64>,
    pieces: Vec<BasicPolytope>,
}

impl Polytope {
    /// `None` when the set has a shape this extraction does not handle;
    /// the caller then falls back to isl.
    fn extract(set: &Set<'_>) -> Result<Option<Self>> {
        if set.n_param()? != 0 {
            return Ok(None);
        }
        let dims = set.dim(DimType::Out)? as usize;
        let mut lower = Vec::with_capacity(dims);
        let mut upper = Vec::with_capacity(dims);
        for i in 0..dims {
            let Some(low) = constant_bound(set.clone().dim_min(i as i32)?)? else {
                return Ok(None);
            };
            let Some(high) = constant_bound(set.clone().dim_max(i as i32)?)? else {
                return Ok(None);
            };
            lower.push(low.ceil() as i64);
            upper.push(high.floor() as i64);
        }
        let mut pieces = Vec::new();
        for basic in set.clone().get_basic_set_list()?.iter() {
            let Some(piece) = BasicPolytope::extract(&basic, dims)? else {
                return Ok(None);
            };
            pieces.push(piece);
        }
        Ok(Some(Polytope {
            dims,
            lower,
            upper,
            pieces,
        }))
    }

    fn for_each_point(&self, visit: &mut impl FnMut(&[i64]) -> Result<()>) -> Result<()> {
        let mut coordinates = self.lower.clone();
        if self.dims == 0 {
            if self.pieces.iter().any(|p| p.contains(&coordinates)) {
                visit(&coordinates)?;
            }
            return Ok(());
        }
        if self.lower.iter().zip(&self.upper).any(|(l, u)| l > u) {
            return Ok(());
        }
        loop {
            // A point in several basic sets is one point of the set.
            if self.pieces.iter().any(|p| p.contains(&coordinates)) {
                visit(&coordinates)?;
            }
            // Odometer increment, last dim fastest.
            let mut k = self.dims;
            loop {
                if k == 0 {
                    return Ok(());
                }
                k -= 1;
                if coordinates[k] < self.upper[k] {
                    coordinates[k] += 1;
                    break;
                }
                coordinates[k] = self.lower[k];
            }
        }
    }
}

impl BasicPolytope {
    fn extract(basic: &BasicSet<'_>, dims: usize) -> Result<Option<Self>> {
        let n_div = basic.dim(DimType::Div)? as usize;
        let mut divs = Vec::with_capacity(n_div);
        for i in 0..n_div {
            let Some(aff) = basic.clone().get_div(i as i32) else {
                return Ok(None);
            };
            let Some(div) = Div::extract(&aff, dims, n_div)? else {
                return Ok(None);
            };
            divs.push(div);
        }
        let mut constraints = Vec::new();
        for constraint in basic.clone().get_constraint_list()?.iter() {
            let mut dim_coefficients = Vec::with_capacity(dims);
            for i in 0..dims {
                dim_coefficients.push(integer(constraint.clone().get_coefficient(DimType::Out, i as i32)?)?);
            }
            let mut div_coefficients = Vec::with_capacity(n_div);
            for i in 0..n_div {
                div_coefficients.push(integer(constraint.clone().get_coefficient(DimType::Div, i as i32)?)?);
            }
            constraints.push(LinearConstraint {
                dim_coefficients,
                div_coefficients,
                constant: integer(constraint.clone().get_constant()?)?,
                equality: constraint.is_equality()?,
            });
        }
        Ok(Some(BasicPolytope { divs, constraints }))
    }

    fn contains(&self, x: &[i64]) -> bool {
        let mut div_values = [0i64; 8];
        let mut heap;
        let div_values: &mut [i64] = if self.divs.len() <= 8 {
            &mut div_values[..self.divs.len()]
        } else {
            heap = vec![0i64; self.divs.len()];
            &mut heap
        };
        for (i, div) in self.divs.iter().enumerate() {
            let mut numerator = div.constant;
            for (c, v) in div.dim_coefficients.iter().zip(x) {
                numerator += c * v;
            }
            for (c, v) in div.div_coefficients.iter().zip(div_values.iter()).take(i) {
                numerator += c * v;
            }
            div_values[i] = numerator.div_euclid(div.denominator);
        }
        self.constraints.iter().all(|constraint| {
            let mut value = constraint.constant;
            for (c, v) in constraint.dim_coefficients.iter().zip(x) {
                value += c * v;
            }
            for (c, v) in constraint.div_coefficients.iter().zip(div_values.iter()) {
                value += c * v;
            }
            if constraint.equality { value == 0 } else { value >= 0 }
        })
    }
}

impl Div {
    /// isl's affine getters return coefficients already divided by the
    /// affine's denominator; put them back over one common denominator so
    /// the floor is exact integer arithmetic.
    fn extract(aff: &Affine<'_>, dims: usize, n_div: usize) -> Result<Option<Self>> {
        let mut rationals = Vec::with_capacity(dims + n_div + 1);
        for i in 0..dims {
            let v = aff.clone().get_coefficient_val(DimType::In, i as i32)?;
            rationals.push((v.numerator(), v.denominator()));
        }
        for i in 0..n_div {
            let v = aff.clone().get_coefficient_val(DimType::Div, i as i32)?;
            rationals.push((v.numerator(), v.denominator()));
        }
        let constant = aff.clone().get_constant_val()?;
        rationals.push((constant.numerator(), constant.denominator()));
        let denominator = rationals.iter().fold(1i64, |acc, (_, d)| lcm(acc, *d));
        let scaled: Vec<i64> = rationals
            .iter()
            .map(|(n, d)| n * (denominator / d))
            .collect();
        Ok(Some(Div {
            dim_coefficients: scaled[..dims].to_vec(),
            div_coefficients: scaled[dims..dims + n_div].to_vec(),
            constant: scaled[dims + n_div],
            denominator,
        }))
    }
}

fn constant_bound(bound: barvinok::pw_aff::PiecewiseAffine<'_>) -> Result<Option<f64>> {
    if !bound.is_cst()? {
        return Ok(None);
    }
    let aff = bound.as_aff()?;
    Ok(Some(aff.get_constant_val()?.to_f64()))
}

fn integer(value: barvinok::value::Value<'_>) -> Result<i64> {
    if value.denominator() != 1 {
        bail!("constraint coefficient {}/{} is not an integer", value.numerator(), value.denominator());
    }
    Ok(value.numerator())
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

fn lcm(a: i64, b: i64) -> i64 {
    (a / gcd(a, b) * b).abs()
}
