//! Parallel-loop locality: private reuse intervals and the CRI model.
//!
//! Implements the model of *Parallel Loop Locality Analysis for Symbolic Thread
//! Counts* (PACT'24, DOI 10.1145/3656019.3676948) on top of this crate's
//! polyhedral machinery. A loop nest parallelized by `schedule(static, chunk)`
//! is analyzed in two steps:
//!
//! 1. compute the **private reuse interval** (PRI) distribution — reuse as one
//!    thread sees its own accesses, ignoring the other threads;
//! 2. map each PRI to a **concurrent reuse interval** (CRI) distribution — reuse
//!    as the shared cache sees it, after all threads interleave.
//!
//! The second step is Table 1 of the paper, a two-by-two over data sharing and
//! reuse length:
//!
//! | | Short RI | Long RI |
//! |---|---|---|
//! | Data sharing | `NBD(r, 1/T)` | racetrack, `CRI = T·r·X` |
//! | No sharing | `NBD(r, 1/T)` | `CRI = T·r` |
//!
//! Every cell here is checked against measurement by the `parallel-sampler`
//! crate (see its `tests/table1_cells.rs`). Two of them are easy to get wrong in
//! ways that cost orders of magnitude, and both were wrong in an earlier port of
//! this model:
//!
//!  * a long **private** reuse dilates to `T·r`. Applying the racetrack law
//!    there instead yields a mean of `r/T`, short by a factor of `T²`;
//!  * the racetrack law itself carries the interleaving dilation, `CRI = T·r·X`
//!    rather than `r·X`. `X` has mean `1/T`, so the two effects cancel and a
//!    shared reuse keeps `E[CRI] = r` — dropping the `T` leaves it short by a
//!    further factor of `T`.
//!
//! Both errors shorten the predicted reuse intervals, which shifts the whole
//! miss-ratio curve toward smaller caches and makes the kernel look far more
//! cache-friendly than it is.

use anyhow::{Result, anyhow, bail};
use barvinok::{
    DimType, constraint::Constraint, local_space::LocalSpace, map::Map, set::Set,
};
use raffine::tree::Tree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Gamma};
use statrs::function::beta::beta_reg;

use crate::isl;

/// How the parallel loop's iterations are distributed.
///
/// Only OpenMP's `schedule(static, chunk)` is modeled: normalized iteration `k`
/// belongs to thread `(k / chunk) mod T`. `chunk = ceil(trip / T)` is the
/// default `schedule(static)`, and `chunk = 1` is round-robin, so the single
/// parameter spans the range. Dynamic and guided schedules are out of scope;
/// keeping ownership a pure function of `k` is what makes the thread identity
/// an affine construction below.
/// Name given to the synthetic thread dimension.
///
/// Deliberately unlike the `i<n>` / `t<n>` names the tree lowering hands out, so
/// it cannot collide with a real loop or statement dimension.
pub const THREAD_DIM_NAME: &str = "__tid";

#[derive(Debug, Clone, Copy)]
pub struct ParallelSpec {
    /// Loop nesting level to parallelize, outermost being 0.
    pub loop_level: usize,
    pub threads: u32,
    pub chunk: i64,
}

/// Short/long boundary from Theorem 3.1 of the paper.
///
/// Above this bound the CRI has concentrated on its mean `T·r`, so the cheap
/// deterministic law is as good as expanding the full distribution. Below it,
/// the spread still matters and the negative binomial is used. The constants
/// are the theorem's: `c1 ∈ (0.5, 1)` and `c2 > 1` bracket the concentration
/// interval, `epsilon` is the probability of falling outside it.
#[derive(Debug, Clone, Copy)]
pub struct ShortBound {
    pub c1: f64,
    pub c2: f64,
    pub epsilon: f64,
}

impl Default for ShortBound {
    fn default() -> Self {
        Self {
            c1: 0.6,
            c2: 2.0,
            epsilon: 1e-3,
        }
    }
}

impl ShortBound {
    /// Reuse intervals at or below this are expanded as negative binomials.
    pub fn value(&self) -> Result<f64> {
        if !(self.c1 > 0.5 && self.c1 < 1.0) {
            bail!("short-RI bound needs 0.5 < c1 < 1, got {}", self.c1);
        }
        if self.c2 <= 1.0 {
            bail!("short-RI bound needs c2 > 1, got {}", self.c2);
        }
        if !(self.epsilon > 0.0 && self.epsilon < 1.0) {
            bail!("short-RI bound needs 0 < epsilon < 1, got {}", self.epsilon);
        }
        let log = (1.0 / self.epsilon).ln();
        let first = 2.0 * log / (self.c2 * (1.0 / self.c2 - 1.0).powi(2));
        let second = 3.0 * log / (self.c1 * (1.0 / self.c1 - 1.0).powi(2));
        Ok(first.max(second))
    }
}

/// Locates the timestamp-space dimension holding a loop's normalized counter.
///
/// The timestamp space interleaves loop counters with statement selectors — a
/// `Tree::Block` consumes a dimension just as a `Tree::For` does — so a loop's
/// *nesting level* is not its dimension index. This walk mirrors the depth
/// accounting of `isl::get_timestamp_space_impl` exactly; any divergence would
/// silently pick the wrong dimension and model the wrong loop as parallel.
pub fn timestamp_dim_of_loop(tree: &Tree<'_>, loop_level: usize) -> Result<usize> {
    fn walk(
        tree: &Tree<'_>,
        depth: usize,
        level: usize,
        target: usize,
        found: &mut Option<usize>,
    ) -> Result<()> {
        match tree {
            Tree::For { body, .. } => {
                if level == target {
                    match found {
                        Some(previous) if *previous != depth => bail!(
                            "loop level {target} appears at timestamp dimensions {previous} and \
                             {depth}; the nest is not uniform enough to parallelize by level"
                        ),
                        _ => *found = Some(depth),
                    }
                }
                walk(body, depth + 1, level + 1, target, found)
            }
            Tree::Block(stmts) => {
                for stmt in stmts.iter() {
                    walk(stmt, depth + 1, level, target, found)?;
                }
                Ok(())
            }
            Tree::If { then, r#else, .. } => {
                walk(then, depth, level, target, found)?;
                if let Some(otherwise) = r#else {
                    walk(otherwise, depth, level, target, found)?;
                }
                Ok(())
            }
            Tree::Access { .. } => Ok(()),
        }
    }

    let mut found = None;
    walk(tree, 0, 0, loop_level, &mut found)?;
    found.ok_or_else(|| anyhow!("the loop nest has no loop at nesting level {loop_level}"))
}

/// Trip count of a loop whose bounds are compile-time constants.
///
/// Used only to resolve a default chunk size of `ceil(trip / T)`. Returns `None`
/// for symbolic bounds, where the caller must be given an explicit chunk: a
/// symbolic chunk would make thread ownership non-affine and defeat the whole
/// construction.
pub fn constant_trip_count(tree: &Tree<'_>, loop_level: usize) -> Option<i64> {
    fn walk(tree: &Tree<'_>, level: usize, target: usize) -> Option<i64> {
        match tree {
            Tree::For {
                lower_bound,
                upper_bound,
                step,
                body,
                ..
            } => {
                if level == target {
                    let lower = lower_bound.get_result_expr(0)?.get_value()?;
                    let upper = upper_bound.get_result_expr(0)?.get_value()?;
                    let step = *step as i64;
                    if step <= 0 {
                        return None;
                    }
                    return Some(((upper - lower) + step - 1).div_euclid(step).max(0));
                }
                walk(body, level + 1, target)
            }
            Tree::Block(stmts) => stmts.iter().find_map(|stmt| walk(stmt, level, target)),
            Tree::If { then, r#else, .. } => walk(then, level, target)
                .or_else(|| r#else.and_then(|otherwise| walk(otherwise, level, target))),
            Tree::Access { .. } => None,
        }
    }
    walk(tree, 0, loop_level)
}

/// Appends a thread-identity dimension to the timestamp space.
///
/// The thread that owns normalized iteration `k` is `(k / chunk) mod T`. Written
/// with an existential quotient `q` that is projected out afterwards, that is
/// purely affine:
///
/// ```text
/// 0 <= tid < T,   0 <= k - chunk*T*q - chunk*tid < chunk
/// ```
///
/// Because `tid` is a function of `k`, adding it changes neither the number of
/// timestamps nor their order — so cardinalities are preserved, and "same
/// thread" reduces to an equality on one dimension.
///
/// It is appended *last* rather than prepended, which matters: a leading thread
/// dimension would make the lexicographic order run all of thread 0, then all
/// of thread 1, and the unrestricted reuse relation over that order would be
/// meaningless. Appended, the order stays the original program order, so the
/// same space yields both the sequential reuse interval (unrestricted) and the
/// private one (restricted to equal `tid`).
///
/// Returns the space and the index of the thread dimension.
///
/// The dimension is named explicitly rather than left for `ensure_set_name` to
/// fill in: that names by index, while the tree builder names loop dimensions by
/// depth, so an auto-named thread dimension can collide with a real one.
pub fn add_thread_dim<'ctx>(
    space: Set<'ctx>,
    ts_dim: usize,
    spec: &ParallelSpec,
) -> Result<(Set<'ctx>, usize)> {
    if spec.chunk <= 0 {
        bail!("chunk size must be positive, got {}", spec.chunk);
    }
    if spec.threads < 1 {
        bail!("thread count must be at least one");
    }
    let threads = i64::from(spec.threads);
    let chunk = spec.chunk;
    let stride = chunk
        .checked_mul(threads)
        .ok_or_else(|| anyhow!("chunk {chunk} times {threads} threads overflows"))?;

    // `tid` and the existential quotient `q` are appended, leaving every
    // existing dimension where it was.
    let thread_dim = space.n_dim()? as usize;
    let mut space = space.insert_dims(DimType::Out, thread_dim as u32, 2)?;
    let iteration = ts_dim as i32;
    let tid = thread_dim as i32;
    let quotient = thread_dim as i32 + 1;
    let local_space: LocalSpace = space.get_space()?.try_into()?;

    // 0 <= tid
    space = space.add_constraint(
        Constraint::new_inequality(local_space.clone())?.set_coefficient_si(DimType::Out, tid, 1)?,
    )?;
    // tid <= T - 1
    space = space.add_constraint(
        Constraint::new_inequality(local_space.clone())?
            .set_coefficient_si(DimType::Out, tid, -1)?
            .set_constant_si((threads - 1) as i32)?,
    )?;
    // k - stride*q - chunk*tid >= 0
    space = space.add_constraint(
        Constraint::new_inequality(local_space.clone())?
            .set_coefficient_si(DimType::Out, iteration, 1)?
            .set_coefficient_si(DimType::Out, quotient, -(stride as i32))?
            .set_coefficient_si(DimType::Out, tid, -(chunk as i32))?,
    )?;
    // chunk - 1 - (k - stride*q - chunk*tid) >= 0
    space = space.add_constraint(
        Constraint::new_inequality(local_space)?
            .set_coefficient_si(DimType::Out, iteration, -1)?
            .set_coefficient_si(DimType::Out, quotient, stride as i32)?
            .set_coefficient_si(DimType::Out, tid, chunk as i32)?
            .set_constant_si((chunk - 1) as i32)?,
    )?;

    // Drop `q`; isl keeps it as an existential so `tid` stays exact.
    let space = space
        .project_out(DimType::Out, quotient as u32, 1)?
        .set_dim_name(DimType::Out, thread_dim as u32, THREAD_DIM_NAME)?;
    Ok((space, thread_dim))
}

/// Relation holding pairs of timestamps executed by the same thread.
///
/// With the thread identity materialized at dimension 0 by [`add_thread_dim`],
/// this is one equality rather than a modular condition.
pub fn same_thread<'ctx>(map: Map<'ctx>, thread_dim: usize) -> Result<Map<'ctx>> {
    let local_space: LocalSpace = map.get_space()?.try_into()?;
    Ok(map.add_constraint(
        Constraint::new_equality(local_space)?
            .set_coefficient_si(DimType::In, thread_dim as i32, 1)?
            .set_coefficient_si(DimType::Out, thread_dim as i32, -1)?,
    )?)
}

/// Tunables for turning a PRI distribution into a CRI distribution.
/// How the dilation of a non-intercepted reuse window is distributed.
///
/// Both laws describe the same thing -- a window holding `r` of this thread's
/// accesses, stretched by the other threads' -- and both have mean `T*r`. They
/// differ in cost and in what happens to long windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DilationLaw {
    /// `CRI = r + X` with `X ~ NB(r, 1/T)`, expanded term by term. Exact, but
    /// the support grows with `r`, so beyond the Theorem 3.1 bound it is
    /// abandoned for a point mass at `T*r` and the spread is lost.
    NegativeBinomial,
    /// `CRI = r + Gamma(shape = r*(1 - 1/T), scale = T)`, the continuous limit
    /// of the same law: a sum of geometric waits becomes a sum of exponential
    /// ones.
    ///
    /// The shift is not cosmetic. `CRI = r + X` cannot fall below `r` -- the
    /// window contains this thread's own `r` accesses whatever the others do --
    /// and an unshifted Gamma puts mass where the CRI cannot go, which costs a
    /// factor of thirty on short reuse. Shifted, and with the shape chosen so
    /// both moments match `NB(r, 1/T)` exactly, it agrees with the negative
    /// binomial and needs a fixed number of buckets whatever `r` is -- so no
    /// short/long cutoff, and no collapse to a point mass on long windows.
    Gamma,
    /// The negative binomial below the Theorem 3.1 bound, where it is exact and
    /// affordable, and the shifted Gamma above it, where the negative binomial
    /// would otherwise be abandoned for a point mass.
    ///
    /// Each law is used where it is better, and measurably so: the negative
    /// binomial alone loses an order of magnitude on long private reuse by
    /// discarding its spread, and the Gamma alone loses a factor of five on very
    /// short reuse, where the window holds few enough accesses that the
    /// distribution's discreteness still matters.
    Hybrid,
    /// `CRI = r + X` with `X` the continuous negative binomial, evaluated
    /// through its beta form: `P[X <= k] = I_{1/T}(r, k+1)`, the regularized
    /// incomplete beta, continuous in both `r` and `k` and defined for real
    /// `r`. One law at every window length -- exact NB support and skew where
    /// the window is short, converging to the shifted Gamma where it is long
    /// -- with no Theorem 3.1 cutoff, no rounding of `r`, and no point-mass
    /// collapse. Quantized by equal-probability buckets like the Gamma.
    Beta,
}

#[derive(Debug, Clone, Copy)]
pub struct CriKnobs {
    pub short_bound: f64,
    pub racetrack_bins: usize,
    pub nbd_epsilon: f64,
    pub dilation: DilationLaw,
}

/// Outcome of a parallel analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelReport {
    pub threads: u32,
    pub chunk: i64,
    pub short_ri_bound: f64,
    /// Portion of all accesses that reuse a datum only this thread touches.
    pub private_portion: f64,
    /// Portion of all accesses that reuse a datum some other thread also
    /// touches. These are the racetrack ones.
    pub shared_portion: f64,
    /// Portion of all accesses with no prior access by the same thread. These
    /// are the compulsory misses the Denning recursion infers from the gap.
    pub cold_portion: f64,
    /// Average number of distinct threads touching a shared datum. Equal to the
    /// thread count when a shared array is swept by everyone, and far below it
    /// for neighbour-only sharing such as a stencil halo.
    pub mean_sharing_degree: f64,
    /// The reuse-interval distribution *before* any CRI law is applied, as
    /// `(reuse interval, portion of all accesses, shared)`.
    ///
    /// This is the quantity a symbolic-in-`T` pipeline would cache: the laws
    /// downstream are closed forms in `T`, so if this is invariant across thread
    /// counts it need only be derived once. Exposed so that invariance can be
    /// checked rather than assumed.
    pub reuse_interval_distribution: Vec<(f64, f64, bool)>,
    /// `(CRI, portion)` support, ready for `denning::MissRatioCurve::new`.
    pub support: Vec<(isize, f64)>,
    /// Wall time of the polyhedral part inside this function: the reuse
    /// relations and their barvinok counts. The lowering that precedes it is
    /// timed by the caller.
    /// Access total of the analysed loop nest, as the sequential report's
    /// `total_count`; lets `mrc` report miss counts. Absent in older files.
    #[serde(default)]
    pub total_count: String,
    pub derivation_seconds: f64,
    /// Wall time of applying the CRI laws to the derived distribution.
    pub scaling_seconds: f64,
}

/// Runs the parallel pipeline over a thread-extended timestamp space.
///
/// `space` and `access_map` must already carry the thread dimension added by
/// [`add_thread_dim`]. The reuse construction mirrors the sequential one in
/// `main_entry`, with every order relation restricted to a single thread so
/// that what it counts is the *private* reuse interval.
pub fn analyze<'ctx>(
    space: Set<'ctx>,
    access_map: Map<'ctx>,
    thread_dim: usize,
    spec: &ParallelSpec,
    knobs: &CriKnobs,
) -> Result<ParallelReport> {
    let derivation_started = std::time::Instant::now();
    let total = space.clone().cardinality()?;

    let same_element = access_map
        .clone()
        .apply_range(access_map.clone().reverse()?)?;

    // A datum is shared when two *different* threads touch it. Subtracting the
    // same-thread pairs from the same-datum pairs leaves exactly the
    // cross-thread ones, and their domain is the set of accesses to shared
    // data. This is exact, unlike the paper's syntactic rule of "does the
    // subscript mention the parallel induction variable" -- in a stencil every
    // subscript mentions it, yet the halo rows are shared.
    let cross_thread = same_element
        .clone()
        .subtract(same_thread(same_element.clone(), thread_dim)?)?;
    let shared_times = cross_thread.domain()?;
    let private_times = space.clone().subtract(shared_times.clone())?;

    // How many distinct threads touch each shared datum. Projecting the
    // same-datum relation's range down to the thread dimension turns
    // `cardinality` into a count of *threads*, since isl counts distinct range
    // points per domain point.
    //
    // The paper's racetrack always uses T runners, which is right for a matrix
    // every thread sweeps but wrong for a stencil halo, where a row is shared
    // with two neighbours no matter how many threads run.
    let sharing_degrees = if shared_times.clone().is_empty()? {
        Vec::new()
    } else {
        let sharers = same_element
            .clone()
            .project_out(DimType::Out, 0, thread_dim as u32)?
            .intersect_domain(shared_times.clone())?;
        let shared_total = shared_times.clone().cardinality()?;
        let distribution =
            isl::RIProcessor::new(sharers.clone().cardinality()?, sharers.domain()?)
                .get_distribution()?;
        isl::get_distro(&distribution, shared_total)?
            .iter()
            .copied()
            .filter(|(degree, weight)| *degree >= 2 && *weight > 0.0)
            .map(|(degree, weight)| (degree as u32, weight))
            .collect::<Vec<_>>()
    };
    let degree_mass: f64 = sharing_degrees.iter().map(|(_, weight)| *weight).sum();
    let mean_sharing_degree = if degree_mass > 0.0 {
        sharing_degrees
            .iter()
            .map(|(degree, weight)| f64::from(*degree) * weight)
            .sum::<f64>()
            / degree_mass
    } else {
        0.0
    };

    let greater = space.clone().lex_gt_set(space.clone())?;
    let at_least = space.clone().lex_ge_set(space.clone())?;

    // Two reuse relations over the same space. Because the thread dimension is
    // last, the unrestricted order is the original program order, so:
    //
    //  * `sequential` counts every access in the reuse window, which is the
    //    interval the racetrack consumes ("for T threads and a sequential reuse
    //    interval r", Section 3.3);
    //  * `private` counts only this thread's, which is the interval the
    //    negative binomial and the scale-by-T law consume.
    //
    // Feeding a private interval to the racetrack would be wrong twice over: it
    // is the wrong quantity, and for data a thread touches only once -- a
    // stencil halo, say -- it does not exist at all, so those accesses would be
    // silently reclassified as compulsory misses.
    let sequential_reuse = reuse_relation(&same_element, greater.clone(), at_least.clone())?;
    let private_reuse = reuse_relation(
        &same_element,
        same_thread(greater, thread_dim)?,
        same_thread(at_least, thread_dim)?,
    )?;

    let shared_reuse = sequential_reuse.intersect_domain(shared_times)?;
    let unshared_reuse = private_reuse.intersect_domain(private_times)?;

    // Both counts are the last of the polyhedral work; everything after this
    // point is closed-form arithmetic over the resulting distributions.
    let total_count = isl::total_count_string(&total)?;
    let private_distribution = distribution_of(unshared_reuse, total.clone())?;
    let shared_distribution = distribution_of(shared_reuse, total)?;
    let derivation_seconds = derivation_started.elapsed().as_secs_f64();
    let scaling_started = std::time::Instant::now();

    let mut jobs = Vec::new();
    let mut reuse_interval_distribution = Vec::new();
    let mut shared_portion = 0.0;
    let mut private_portion = 0.0;

    for (reuse_interval, portion) in private_distribution {
        private_portion += portion;
        reuse_interval_distribution.push((reuse_interval, portion, false));
        jobs.push(ExpansionJob {
            reuse_interval,
            weight: portion,
            sharing: Sharing::Private,
        });
    }

    // One traversal of the shared data, in accesses: the longest reuse interval
    // any shared datum shows. A datum touched once per sweep has exactly this
    // interval, so it is the natural yardstick for whether a shorter window can
    // expect a sharer to arrive.
    let lap = shared_distribution
        .iter()
        .map(|(reuse_interval, _)| *reuse_interval)
        .fold(0.0f64, f64::max);
    for (reuse_interval, portion) in shared_distribution {
        shared_portion += portion;
        reuse_interval_distribution.push((reuse_interval, portion, true));
        if degree_mass <= 0.0 {
            // No degree information: fall back to the paper's assumption that
            // every thread shares.
            jobs.push(ExpansionJob {
                reuse_interval,
                weight: portion,
                sharing: Sharing::Shared {
                    sharers: spec.threads,
                    lap,
                },
            });
            continue;
        }
        // The degree distribution is measured over all shared accesses rather
        // than per reuse interval, so this treats the two as independent. The
        // joint law would cost one barvinok run per distinct degree.
        for (sharers, degree_weight) in sharing_degrees.iter().copied() {
            jobs.push(ExpansionJob {
                reuse_interval,
                weight: portion * degree_weight / degree_mass,
                sharing: Sharing::Shared { sharers, lap },
            });
        }
    }
    let expansion = expand_all(&jobs, spec.threads, knobs)?;

    Ok(ParallelReport {
        threads: spec.threads,
        chunk: spec.chunk,
        short_ri_bound: knobs.short_bound,
        private_portion,
        shared_portion,
        cold_portion: (1.0 - private_portion - shared_portion).max(0.0),
        mean_sharing_degree,
        reuse_interval_distribution,
        support: to_denning_support(expansion),
        total_count,
        derivation_seconds,
        scaling_seconds: scaling_started.elapsed().as_secs_f64(),
    })
}

/// Re-applies the CRI laws at a new thread count, reusing a distribution that
/// was derived once.
///
/// At a fixed chunk the extracted reuse-interval distribution does not depend on
/// the thread count -- ownership is cyclic with period `T`, so which accesses a
/// thread sees, and how far apart, is the same pattern at every `T`. Only the
/// laws downstream consume `T`, and they are closed forms. One derivation
/// therefore serves every thread count, which is the sense in which the analysis
/// is parametric in parallelism.
///
/// What does *not* carry over is the sharing degree: how many threads touch a
/// datum is genuinely `T`-dependent, and not always equal to `T`. This function
/// assumes it is `T` -- the assumption PLUSS makes -- so comparing its output
/// against a full re-derivation isolates exactly what that assumption costs. On
/// a stencil, where the degree saturates at the halo width, it costs a lot.
pub fn rescale(path: &std::path::Path, threads: u32, knobs: &CriKnobs) -> Result<ParallelReport> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| anyhow!("reading {}: {error}", path.display()))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| anyhow!("{} is not a JSON report: {error}", path.display()))?;
    let Some(source) = document.get("parallel") else {
        bail!(
            "{} has no `parallel` section; it must come from `analyzer --json ... \
             --parallel-loop-depth`",
            path.display()
        );
    };
    let source: ParallelReport = serde_json::from_value(source.clone())
        .map_err(|error| anyhow!("{} has an unreadable parallel report: {error}", path.display()))?;
    if source.reuse_interval_distribution.is_empty() {
        bail!(
            "{} carries no reuse-interval distribution; it predates the parametric path",
            path.display()
        );
    }

    let lap = source
        .reuse_interval_distribution
        .iter()
        .filter(|(_, _, is_shared)| *is_shared)
        .map(|(reuse_interval, _, _)| *reuse_interval)
        .fold(0.0f64, f64::max);

    let scaling_started = std::time::Instant::now();
    let mut jobs = Vec::new();
    let mut private_portion = 0.0;
    let mut shared_portion = 0.0;
    for (reuse_interval, portion, is_shared) in source.reuse_interval_distribution.iter().copied() {
        let sharing = if is_shared {
            shared_portion += portion;
            Sharing::Shared {
                sharers: threads,
                lap,
            }
        } else {
            private_portion += portion;
            Sharing::Private
        };
        jobs.push(ExpansionJob {
            reuse_interval,
            weight: portion,
            sharing,
        });
    }
    let expansion = expand_all(&jobs, threads, knobs)?;

    Ok(ParallelReport {
        threads,
        chunk: source.chunk,
        short_ri_bound: knobs.short_bound,
        private_portion,
        shared_portion,
        cold_portion: (1.0 - private_portion - shared_portion).max(0.0),
        // Assumed, not measured -- that is the point of this path.
        mean_sharing_degree: if shared_portion > 0.0 {
            f64::from(threads)
        } else {
            0.0
        },
        reuse_interval_distribution: source.reuse_interval_distribution,
        support: to_denning_support(expansion),
        total_count: source.total_count,
        derivation_seconds: 0.0,
        scaling_seconds: scaling_started.elapsed().as_secs_f64(),
    })
}

/// Builds a reuse relation: each access paired with every access from just
/// after its most recent predecessor on the same datum up to itself, so that
/// counting the pairs yields the reuse interval.
///
/// This is the sequential construction from `main_entry`, lifted so the same
/// code can be run with unrestricted or same-thread order relations.
fn reuse_relation<'ctx>(
    same_element: &Map<'ctx>,
    greater: Map<'ctx>,
    at_least: Map<'ctx>,
) -> Result<Map<'ctx>> {
    let less = greater.clone().reverse()?;
    let immediate_pred = same_element.clone().intersect(greater)?.lexmax()?;
    Ok(immediate_pred.apply_range(less)?.intersect(at_least)?)
}

/// Counts a reuse relation into `(reuse interval, portion of all accesses)`.
fn distribution_of<'ctx>(
    relation: Map<'ctx>,
    total: barvinok::polynomial::PiecewiseQuasiPolynomial<'ctx>,
) -> Result<Vec<(f64, f64)>> {
    let support = relation.clone().domain()?;
    if support.clone().is_empty()? {
        return Ok(Vec::new());
    }
    let distribution = isl::RIProcessor::new(relation.cardinality()?, support).get_distribution()?;
    Ok(isl::get_distro(&distribution, total)?
        .iter()
        .copied()
        .filter(|(_, portion)| *portion > 0.0)
        .map(|(reuse_interval, portion)| (reuse_interval as f64, portion))
        .collect())
}

/// Which Table-1 row a reuse belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sharing {
    /// The datum is touched by exactly one thread.
    Private,
    /// The datum is touched by `sharers` distinct threads, which between them
    /// traverse the shared data once every `lap` accesses.
    ///
    /// The paper writes the racetrack with `T` runners, because in the kernels
    /// it targets a shared array is swept by *every* thread. That is not
    /// general: in a stencil, a row is touched only by the threads owning the
    /// adjacent rows, so the number of runners stays around three however many
    /// threads run. Carrying the measured degree separately from the thread
    /// count keeps the two effects apart -- the window still dilates by `T`,
    /// but it is split by `sharers - 1` competitors, not `T - 1`.
    Shared { sharers: u32, lap: f64 },
}

/// Expands one reuse interval into the concurrent intervals it becomes,
/// appending `(cri, weight)` pairs.
///
/// `weight` is the portion of all accesses this reuse interval accounts for; it
/// is split across the expansion so total mass is preserved.
#[allow(clippy::too_many_arguments)]
pub fn expand(
    out: &mut Vec<(f64, f64)>,
    reuse_interval: f64,
    weight: f64,
    sharing: Sharing,
    threads: u32,
    short_bound: f64,
    racetrack_bins: usize,
    nbd_epsilon: f64,
    dilation: DilationLaw,
) -> Result<()> {
    if weight <= 0.0 {
        return Ok(());
    }
    if reuse_interval <= 0.0 {
        // A zero interval is a repeat of the immediately preceding access; it
        // is unaffected by interleaving.
        out.push((reuse_interval, weight));
        return Ok(());
    }
    if threads < 2 {
        out.push((reuse_interval, weight));
        return Ok(());
    }

    // Two effects act on every reuse window, and which dominates decides the
    // law. *Dilation*: the other threads' accesses to anything at all stretch
    // the window by about `T`. *Interception*: another thread touching this
    // same datum cuts the window short. Dilation always applies; interception
    // only when a sharer actually reaches the datum before the window closes.
    //
    // The racetrack is the interception law and already carries the dilation
    // inside it. So it is right exactly when interception is likely, and that
    // is not the same question as the paper's short/long split -- which is a
    // statement about when a *private* CRI has concentrated on its mean
    // (Theorem 3.1), and says nothing about whether a sharer arrives.
    //
    // Interception is likely when the window, spread across the `s` sharers,
    // covers a full traversal of the shared data: `r * s >= lap`. Both ways of
    // getting this wrong were measured. Racetracking a sub-traversal reuse --
    // two back-to-back accesses to one cache block -- charges it an
    // interception that never happens. Dilating a full-traversal reuse because
    // it happens to fall under the Chernoff bound costs 0.44 mean miss-ratio
    // error on a kernel whose shared array is small enough to sweep in fewer
    // accesses than the bound.
    let intercepted = match sharing {
        Sharing::Shared { sharers, lap } => {
            sharers >= 2 && reuse_interval * f64::from(sharers) >= lap
        }
        Sharing::Private => false,
    };

    if intercepted {
        let Sharing::Shared { sharers, .. } = sharing else {
            unreachable!("interception is only decided for shared data");
        };
        return quantize_racetrack(out, reuse_interval, sharers, weight, racetrack_bins);
    }

    // Not intercepted: the window only dilates.
    let short = reuse_interval <= short_bound;
    match dilation {
        // The continuous forms carry the spread at every `r` for a fixed cost,
        // so there is no cutoff to apply.
        DilationLaw::Gamma => quantize_gamma(out, reuse_interval, threads, weight, racetrack_bins),
        DilationLaw::Beta => {
            quantize_continuous_nb(out, reuse_interval, threads, weight, racetrack_bins)
        }
        DilationLaw::Hybrid if short => expand_negative_binomial(
            out,
            reuse_interval.round() as u64,
            threads,
            weight,
            nbd_epsilon,
        ),
        DilationLaw::Hybrid => {
            quantize_gamma(out, reuse_interval, threads, weight, racetrack_bins)
        }
        // Theorem 3.1 says the dilated CRI concentrates on `T*r` once the window
        // is long enough; below the bound it has not, and the negative binomial
        // carries the spread. Above it, the expansion would be unaffordable and
        // the point mass is the stated approximation.
        DilationLaw::NegativeBinomial if short => expand_negative_binomial(
            out,
            reuse_interval.round() as u64,
            threads,
            weight,
            nbd_epsilon,
        ),
        DilationLaw::NegativeBinomial => {
            out.push((f64::from(threads) * reuse_interval, weight));
            Ok(())
        }
    }
}

/// `CRI = r + Gamma(shape = r*(1 - 1/T), scale = T)`, quantized into
/// equal-probability buckets at their midpoints -- the same scheme
/// [`quantize_racetrack`] uses.
///
/// `X ~ NB(r, 1/T)` has mean `r*(T-1)` and variance `r*(T-1)*T`. Matching both
/// with a Gamma gives scale `T` and shape `r*(1 - 1/T)`, and adding the shift
/// back puts the mean at `r*T` with the support starting where it must.
fn quantize_gamma(
    out: &mut Vec<(f64, f64)>,
    reuse_interval: f64,
    threads: u32,
    weight: f64,
    bins: usize,
) -> Result<()> {
    if bins == 0 {
        bail!("gamma quantization needs at least one bin");
    }
    let threads = f64::from(threads);
    let shape = reuse_interval * (1.0 - 1.0 / threads);
    if shape <= 0.0 || shape.is_nan() {
        // One thread, or a zero-length window: nothing dilates it.
        out.push((reuse_interval, weight));
        return Ok(());
    }
    // statrs parameterizes by rate, so a scale of `T` is a rate of `1/T`.
    let distribution = Gamma::new(shape, 1.0 / threads).map_err(|error| {
        anyhow!("gamma({shape}, 1/{threads}) is not a distribution: {error}")
    })?;
    let bucket = weight / bins as f64;
    for index in 0..bins {
        let u = (index as f64 + 0.5) / bins as f64;
        out.push((reuse_interval + distribution.inverse_cdf(u), bucket));
    }
    Ok(())
}

/// `CRI = r + X` with `X` the continuous negative binomial `NB(r, 1/T)`,
/// evaluated through its beta form `P[X <= k] = I_{1/T}(r, k+1)` -- continuous
/// in both `r` and `k`, defined for real `r`, exact NB support and skew at
/// every window length, converging to the shifted Gamma for long windows.
/// Quantized into equal-probability buckets at their midpoints, the same
/// scheme [`quantize_gamma`] uses; the quantile has no closed-form inverse, so
/// each bucket bisects the beta CDF in `k`.
fn quantize_continuous_nb(
    out: &mut Vec<(f64, f64)>,
    reuse_interval: f64,
    threads: u32,
    weight: f64,
    bins: usize,
) -> Result<()> {
    if bins == 0 {
        bail!("continuous-NB quantization needs at least one bin");
    }
    let t = f64::from(threads);
    let p = 1.0 / t;
    let r = reuse_interval;
    let cdf = |k: f64| beta_reg(r, k + 1.0, p);
    // NB(r, 1/T) has mean r*(T-1) and variance r*(T-1)*T. Cantelli's
    // one-sided inequality bounds the u-quantile of ANY such distribution by
    // mean + sd*sqrt(u/(1-u)); the largest u a bucket midpoint reaches is
    // (bins-0.5)/bins, so this bracket provably contains every quantile.
    let (mean, sd) = (r * (t - 1.0), (r * (t - 1.0) * t).sqrt());
    let u_max = (bins as f64 - 0.5) / bins as f64;
    let upper = mean + sd * (u_max / (1.0 - u_max)).sqrt();
    let bucket = weight / bins as f64;
    // `k` counts other threads' accesses, so the emitted quantile is the
    // smallest INTEGER with `F(k) >= u`. The interpolant agrees with the
    // discrete CDF at integers and rises strictly between them, so that is
    // the ceiling of the continuous solution -- except at the first atom,
    // where bisection converges to 0+eps and the ceiling would shift the
    // whole `P[X = 0] = p^r` mass one slot up; guard it explicitly.
    let mass_at_zero = cdf(0.0);
    for index in 0..bins {
        let u = (index as f64 + 0.5) / bins as f64;
        if mass_at_zero >= u {
            out.push((reuse_interval, bucket));
            continue;
        }
        let (mut lo, mut hi) = (0.0_f64, upper);
        while hi - lo > 1e-9 * hi.max(1.0) {
            let mid = 0.5 * (lo + hi);
            if cdf(mid) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        out.push((reuse_interval + (0.5 * (lo + hi)).ceil(), bucket));
    }
    Ok(())
}

/// `CRI = r + X` with `X ~ NB(r, 1/T)`: the number of other threads' accesses
/// that land inside a window containing `r` of this thread's own.
fn expand_negative_binomial(
    out: &mut Vec<(f64, f64)>,
    reuse_interval: u64,
    threads: u32,
    weight: f64,
    epsilon: f64,
) -> Result<()> {
    if reuse_interval == 0 {
        bail!("negative-binomial expansion needs a positive reuse interval");
    }
    let p = 1.0 / f64::from(threads);
    let q = f64::from(threads - 1) / f64::from(threads);
    let mut pmf = p.powi(reuse_interval as i32);
    let mut cumulative = 0.0;
    let mut extra = 0u64;
    loop {
        out.push(((reuse_interval + extra) as f64, weight * pmf));
        cumulative += pmf;
        if (1.0 - cumulative).max(0.0) <= epsilon {
            return Ok(());
        }
        pmf *= (extra + reuse_interval) as f64 / (extra + 1) as f64 * q;
        extra += 1;
        if extra > 20_000_000 {
            bail!(
                "negative-binomial expansion of reuse interval {reuse_interval} at {threads} \
                 threads did not converge. Its first term is (1/T)^r, which underflows to zero \
                 for a window this long, so the expansion never accumulates any mass. Use \
                 `--dilation hybrid` (the default) or `gamma`, which have no such limit, or \
                 lower --short-ri-bound"
            );
        }
    }
}

/// `CRI = s·r·X` with `X ~ (s-1)(1-x)^(s-2)` on `[0, 1]`, for `s` sharing
/// threads and a *sequential* reuse interval `r`.
///
/// Section 3.3: "for T threads and a sequential reuse interval r, the racetrack
/// model computes the distribution of RIs resulting from randomly sampling
/// T - 1 points in the interval [0, r] and partitioning the interval using
/// those points."
///
/// Two departures from that wording, both forced:
///
/// The window is dilated by `s`. Around a cycle, the gaps between successive
/// accesses to one datum sum to the trace length, and parallelizing a loop
/// reorders the trace without changing its length -- so for a datum touched
/// often enough for that identity to bite, the *mean* reuse interval is the
/// same sequentially and concurrently. `E[X] = 1/s`, so the dilation must be
/// `s` for `E[CRI] = r` to come out. Sampling `[0, r]` with no dilation would
/// shrink every shared reuse by a factor of `s`.
///
/// The runner count is the measured sharing degree, not `T`. Only threads that
/// actually touch the datum can split the window, and when every thread does --
/// the case the paper targets -- `s = T` and this is exactly its formula.
///
/// Quantized into equal-probability buckets through the inverse CDF
/// `F⁻¹(u) = 1 - (1-u)^(1/(s-1))`, evaluated at bucket midpoints.
fn quantize_racetrack(
    out: &mut Vec<(f64, f64)>,
    reuse_interval: f64,
    sharers: u32,
    weight: f64,
    bins: usize,
) -> Result<()> {
    if bins == 0 {
        bail!("racetrack quantization needs at least one bin");
    }
    let span = f64::from(sharers) * reuse_interval;
    let exponent = 1.0 / f64::from(sharers - 1);
    let bucket = weight / bins as f64;
    for index in 0..bins {
        let u = (index as f64 + 0.5) / bins as f64;
        let x = 1.0 - (1.0 - u).powf(exponent);
        out.push((span * x, bucket));
    }
    Ok(())
}

/// One reuse-interval entry awaiting expansion into its CRI law.
#[derive(Clone, Copy, Debug)]
struct ExpansionJob {
    reuse_interval: f64,
    weight: f64,
    sharing: Sharing,
}

/// Expands every job with [`expand`], in parallel.
///
/// Entries are independent, so this is a plain data-parallel map. The results
/// are collected in job order and concatenated, so the output is exactly what
/// the sequential loop produced -- the parallelism changes the time, not the
/// answer. On kernels with many distinct intervals this loop, not barvinok, is
/// most of the analysis: covariance spends 17s deriving and 831s expanding when
/// run on one core.
fn expand_all(jobs: &[ExpansionJob], threads: u32, knobs: &CriKnobs) -> Result<Vec<(f64, f64)>> {
    let parts: Vec<Result<Vec<(f64, f64)>>> = jobs
        .par_iter()
        .map(|job| {
            let mut out = Vec::new();
            expand(
                &mut out,
                job.reuse_interval,
                job.weight,
                job.sharing,
                threads,
                knobs.short_bound,
                knobs.racetrack_bins,
                knobs.nbd_epsilon,
                knobs.dilation,
            )?;
            Ok(out)
        })
        .collect();
    let mut expansion = Vec::with_capacity(parts.iter().filter_map(|p| p.as_ref().ok()).map(Vec::len).sum());
    for part in parts {
        expansion.extend(part?);
    }
    Ok(expansion)
}

/// Rounds a CRI expansion onto the integer support the Denning recursion wants,
/// merging duplicates and sorting by reuse interval.
///
/// `denning::MissRatioCurve::new` reads the *unaccounted* mass as compulsory
/// misses, so the portions must stay portions of all accesses here — they are
/// deliberately not renormalized to sum to one.
pub fn to_denning_support(mut expansion: Vec<(f64, f64)>) -> Vec<(isize, f64)> {
    expansion.retain(|(interval, weight)| interval.is_finite() && *weight > 0.0);
    // Stable, so equal intervals keep job order and the merged weights sum in
    // the same sequence as the sequential version: identical output.
    expansion.par_sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(isize, f64)> = vec![(0, 0.0)];
    for (interval, weight) in expansion {
        let rounded = interval.round().max(0.0) as isize;
        match out.last_mut() {
            Some((last, running)) if *last == rounded => *running += weight,
            _ => out.push((rounded, weight)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shared datum touched once per traversal, so every reuse of it is
    /// intercepted.
    fn shared(threads: u32) -> Sharing {
        Sharing::Shared {
            sharers: threads,
            lap: 0.0,
        }
    }

    fn expand_one(
        reuse_interval: f64,
        sharing: Sharing,
        threads: u32,
        short_bound: f64,
    ) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        expand(
            &mut out,
            reuse_interval,
            1.0,
            sharing,
            threads,
            short_bound,
            512,
            1e-12,
            DilationLaw::NegativeBinomial,
        )
        .expect("expands");
        out
    }

    fn mean(points: &[(f64, f64)]) -> f64 {
        let mass: f64 = points.iter().map(|(_, w)| *w).sum();
        points.iter().map(|(v, w)| v * w).sum::<f64>() / mass
    }

    fn mass(points: &[(f64, f64)]) -> f64 {
        points.iter().map(|(_, w)| *w).sum()
    }

    #[test]
    fn long_private_reuse_scales_by_the_thread_count() {
        for threads in [2u32, 4, 8, 64] {
            let points = expand_one(4096.0, Sharing::Private, threads, 64.0);
            assert_eq!(points.len(), 1, "a deterministic law is a single point");
            assert_eq!(points[0].0, f64::from(threads) * 4096.0);
        }
    }

    #[test]
    fn long_shared_reuse_preserves_the_sequential_mean() {
        // Reordering a trace cannot change how often a datum is touched per
        // access, so the mean reuse interval survives parallelization. The
        // racetrack's dilation and splitting must therefore cancel, for any
        // thread count and any sharing degree.
        for threads in [2u32, 4, 8, 64] {
            for sharers in [2u32, 3, threads] {
                let mut out = Vec::new();
                expand(&mut out, 4096.0, 1.0, Sharing::Shared { sharers, lap: 0.0 }, threads, 64.0, 512, 1e-12, DilationLaw::NegativeBinomial)
                    .expect("expands");
                let observed = mean(&out);
                assert!(
                    (observed - 4096.0).abs() < 4096.0 * 0.02,
                    "T={threads} s={sharers}: mean {observed} should stay 4096"
                );
            }
        }
    }

    #[test]
    fn long_shared_reuse_spans_up_to_the_dilated_window() {
        let points = expand_one(1000.0, shared(4), 4, 64.0);
        let largest = points.iter().map(|(v, _)| *v).fold(0.0, f64::max);
        assert!(
            largest > 3000.0 && largest <= 4000.0,
            "support should reach toward s*r = 4000, got {largest}"
        );
    }

    #[test]
    fn a_low_sharing_degree_lengthens_reuse_relative_to_all_threads_sharing() {
        // A stencil halo is shared by a couple of neighbours, so its window is
        // split far less than a matrix every thread sweeps. Treating it as
        // T-way shared would under-predict the reuse interval.
        let mut halo = Vec::new();
        expand(&mut halo, 4096.0, 1.0, Sharing::Shared { sharers: 2, lap: 0.0 }, 16, 64.0, 512, 1e-12, DilationLaw::NegativeBinomial)
            .expect("expands");
        let mut everyone = Vec::new();
        expand(&mut everyone, 4096.0, 1.0, Sharing::Shared { sharers: 16, lap: 0.0 }, 16, 64.0, 512, 1e-12, DilationLaw::NegativeBinomial)
            .expect("expands");
        let widest = |points: &[(f64, f64)]| points.iter().map(|(v, _)| *v).fold(0.0, f64::max);
        assert!(
            widest(&halo) < widest(&everyone),
            "a 2-way shared window spans less than a 16-way one"
        );
    }

    #[test]
    fn short_private_reuse_uses_the_negative_binomial() {
        for threads in [2u32, 4, 8] {
            let points = expand_one(2.0, Sharing::Private, threads, 64.0);
            let observed = mean(&points);
            let expected = 2.0 * f64::from(threads);
            assert!(
                (observed - expected).abs() < expected * 0.01,
                "T={threads}: mean {observed} should be r*T = {expected}"
            );
        }
    }

    #[test]
    fn a_short_shared_reuse_that_covers_a_traversal_is_intercepted() {
        // Small shared arrays sweep in fewer accesses than the Chernoff bound,
        // so the paper's short/long split would dilate a reuse that is in fact
        // certain to be intercepted. Measured, that costs 0.44 mean miss-ratio
        // error against 0.0024 for the racetrack.
        for threads in [2u32, 4, 8] {
            let sharing = Sharing::Shared {
                sharers: threads,
                lap: 64.0,
            };
            let points = expand_one(64.0, sharing, threads, 77.7);
            let observed = mean(&points);
            assert!(
                (observed - 64.0).abs() < 64.0 * 0.05,
                "T={threads}: mean {observed} should stay at r = 64, not r*T"
            );
        }
    }

    #[test]
    fn a_sub_traversal_shared_reuse_only_dilates() {
        // Two back-to-back accesses to one cache block. The datum is shared, but
        // no other thread will reach it inside a window of one access, so the
        // window dilates rather than being cut short. Charging it an
        // interception was measurably worse across the validation matrix.
        for threads in [2u32, 4, 8] {
            let sharing = Sharing::Shared {
                sharers: threads,
                lap: 4096.0,
            };
            let points = expand_one(1.0, sharing, threads, 77.7);
            let observed = mean(&points);
            let expected = f64::from(threads);
            assert!(
                (observed - expected).abs() < expected * 0.05,
                "T={threads}: mean {observed} should dilate to r*T = {expected}"
            );
        }
    }

    #[test]
    fn every_expansion_preserves_its_weight() {
        for sharing in [Sharing::Private, shared(8)] {
            for reuse_interval in [1.0, 2.0, 63.0, 4096.0] {
                let points = expand_one(reuse_interval, sharing, 8, 64.0);
                assert!(
                    (mass(&points) - 1.0).abs() < 1e-9,
                    "{sharing:?} r={reuse_interval}: mass {} is not 1",
                    mass(&points)
                );
            }
        }
    }

    #[test]
    fn a_single_thread_leaves_reuse_untouched() {
        let points = expand_one(4096.0, shared(1), 1, 64.0);
        assert_eq!(points, vec![(4096.0, 1.0)]);
    }

    #[test]
    fn the_racetrack_law_is_not_the_undilated_one() {
        // Guards the exact defect that made the previous port under-predict:
        // sampling [0, r] with no dilation puts the mean at r/s.
        let points = expand_one(4096.0, shared(8), 8, 64.0);
        assert!(
            mean(&points) > 4096.0 / 8.0 * 4.0,
            "an undilated racetrack would have mean 512"
        );
    }

    fn expand_law(
        reuse_interval: f64,
        threads: u32,
        short_bound: f64,
        dilation: DilationLaw,
    ) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        expand(
            &mut out,
            reuse_interval,
            1.0,
            Sharing::Private,
            threads,
            short_bound,
            512,
            1e-12,
            dilation,
        )
        .expect("expands");
        out
    }

    #[test]
    fn the_gamma_dilation_matches_the_negative_binomial_it_replaces() {
        // Same law, continuous: both must put the mean at r*T and the variance
        // near r*T*(T-1).
        for threads in [2u32, 8, 64] {
            for r in [4.0, 64.0, 512.0] {
                // The negative binomial's first term is `(1/T)^r`, which
                // underflows to zero once `r * log10(T)` passes about 308 -- at
                // 64 threads that is a window of only 170 accesses. Past there
                // it cannot be expanded at all, which is one reason the Gamma
                // is worth having. Compare only where both laws can run.
                if r * f64::from(threads).log10() > 300.0 {
                    continue;
                }
                let gamma = expand_law(r, threads, f64::INFINITY, DilationLaw::Gamma);
                let nbd = expand_law(r, threads, f64::INFINITY, DilationLaw::NegativeBinomial);
                let observed = mean(&gamma);
                let expected = r * f64::from(threads);
                assert!(
                    (observed - expected).abs() < expected * 0.02,
                    "T={threads} r={r}: gamma mean {observed} should be r*T = {expected}"
                );
                assert!(
                    (observed - mean(&nbd)).abs() < expected * 0.02,
                    "T={threads} r={r}: gamma and negative binomial should agree"
                );
            }
        }
    }

    #[test]
    fn the_gamma_dilation_never_falls_below_the_window_it_dilates() {
        // CRI = r + X, so the window always holds this thread's own r accesses.
        // An unshifted Gamma puts mass below r, which costs a factor of thirty
        // on short reuse.
        for threads in [2u32, 8] {
            for r in [1.0, 2.0, 64.0] {
                let points = expand_law(r, threads, f64::INFINITY, DilationLaw::Gamma);
                let smallest = points.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
                assert!(
                    smallest >= r - 1e-9,
                    "T={threads} r={r}: support reaches {smallest}, below the window itself"
                );
            }
        }
    }

    #[test]
    fn the_hybrid_takes_each_law_where_it_is_better() {
        // Below the bound it must be the negative binomial exactly; above it,
        // the Gamma rather than the point mass the negative binomial falls back
        // to.
        let bound = 64.0;
        assert_eq!(
            expand_law(2.0, 8, bound, DilationLaw::Hybrid),
            expand_law(2.0, 8, bound, DilationLaw::NegativeBinomial),
        );
        let long = expand_law(4096.0, 8, bound, DilationLaw::Hybrid);
        assert!(
            long.len() > 1,
            "a long window should keep its spread, not collapse to one point"
        );
        assert_eq!(long, expand_law(4096.0, 8, bound, DilationLaw::Gamma));
    }

    #[test]
    fn short_bound_follows_theorem_3_1() {
        let bound = ShortBound::default().value().expect("valid constants");
        assert!(bound > 1.0, "bound {bound} should exceed a trivial interval");
        // Tighter concentration (c1 -> 0.5, c2 -> 1) demands a longer interval.
        let tighter = ShortBound {
            c1: 0.9,
            c2: 1.1,
            epsilon: 1e-3,
        }
        .value()
        .expect("valid constants");
        assert!(
            tighter > bound,
            "tighter constants {tighter} should exceed looser {bound}"
        );
    }

    #[test]
    fn short_bound_rejects_constants_outside_the_theorem() {
        assert!(ShortBound { c1: 0.4, c2: 2.0, epsilon: 1e-3 }.value().is_err());
        assert!(ShortBound { c1: 0.6, c2: 0.9, epsilon: 1e-3 }.value().is_err());
        assert!(ShortBound { c1: 0.6, c2: 2.0, epsilon: 2.0 }.value().is_err());
    }

    #[test]
    fn denning_support_merges_and_sorts() {
        let support = to_denning_support(vec![(3.4, 0.1), (3.2, 0.2), (10.0, 0.3)]);
        assert_eq!(support[0], (0, 0.0), "leading zero entry is kept");
        assert!(
            support
                .windows(2)
                .all(|pair| pair[0].0 <= pair[1].0),
            "support must be sorted"
        );
        let three = support.iter().find(|(v, _)| *v == 3).expect("merged bucket");
        assert!((three.1 - 0.3).abs() < 1e-12, "3.4 and 3.2 both round to 3");
    }

    #[test]
    fn denning_support_does_not_renormalize() {
        // The gap below one is the compulsory-miss mass; renormalizing it away
        // would report a program with no cold misses.
        let support = to_denning_support(vec![(4.0, 0.25), (8.0, 0.25)]);
        let total: f64 = support.iter().map(|(_, w)| *w).sum();
        assert!((total - 0.5).abs() < 1e-12, "mass {total} should stay 0.5");
    }
}
