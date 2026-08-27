//! Ground-truth behaviour of each cell of PACT'24 Table 1.
//!
//! | | Short RI | Long RI |
//! |---|---|---|
//! | Data sharing | NB distribution | Racetrack model |
//! | No sharing | NB distribution | Scale by thread count |
//!
//! These tests pin what the sampler actually observes, so the analytical model
//! can be checked against measurement rather than against a reading of the
//! paper. Two of them encode findings that contradict the earlier port of this
//! model, and are the reason it predicted far too few misses:
//!
//!  * a long *private* reuse dilates to `T * PRI`, not to `PRI / T`;
//!  * a long *shared* reuse follows `T * PRI * X`, not `PRI * X`.
//!
//! Both were measured under [`Scheduler::Uniform`], which *is* the model's own
//! Assumption 1, so a disagreement here is an error in the model's algebra and
//! cannot be excused by the interleaving being unrealistic.

use std::num::NonZero;

use melior::ir::Module;
use parallel_sampler::interleave::Scheduler;
use parallel_sampler::interp::ChunkSize;
use parallel_sampler::measure::{Binner, Sampler, Sharing};
use parallel_sampler::{RunConfig, frontend, run};
use raffine::{Context, DominanceInfo};

const PRIVATE_LONG: &str = include_str!("../misc/cell_private_long.mlir");
const SHARED_LONG: &str = include_str!("../misc/cell_shared_long.mlir");
const SHORT: &str = include_str!("../misc/cell_short.mlir");

fn sample(source: &str, threads: u32) -> Sampler {
    let config = RunConfig {
        symbols: Vec::new(),
        block_size: NonZero::new(1).expect("non-zero"),
        threads: NonZero::new(threads).expect("non-zero"),
        parallel_depth: Some(0),
        chunk: ChunkSize::Auto,
        scheduler: Scheduler::Uniform,
        seed: 1,
        binner: Binner::default(),
        track_distance: false,
    };
    let context = Context::new();
    let module = Module::parse(context.mlir_context(), source).expect("parses");
    let dom = DominanceInfo::new(&module);
    let tree = frontend::extract_target(&module, None, &context, &dom).expect("builds");
    run(tree, &config).expect("samples")
}

/// Mean concurrent reuse interval among reuses whose private interval falls in
/// the most common bin, and the fraction of all reuses that crossed threads.
fn mean_cri_at_dominant_pri(sampler: &Sampler) -> (f64, f64, f64) {
    let binner = sampler.binner();
    let mut by_pri: Vec<(u32, u64)> = Vec::new();
    let mut total = 0u64;
    let mut shared = 0u64;
    for (key, count) in sampler.joint() {
        total += count;
        if key.sharing == Sharing::Shared {
            shared += count;
        }
        match by_pri.iter_mut().find(|(bin, _)| *bin == key.pri_bin) {
            Some((_, running)) => *running += count,
            None => by_pri.push((key.pri_bin, *count)),
        }
    }
    let (dominant, _) = by_pri
        .iter()
        .copied()
        .max_by_key(|(_, count)| *count)
        .expect("at least one reuse");

    let mut weight = 0f64;
    let mut weighted = 0f64;
    for (key, count) in sampler.joint() {
        if key.pri_bin != dominant {
            continue;
        }
        weight += *count as f64;
        weighted += binner.representative(key.cri_bin) * *count as f64;
    }
    (
        binner.representative(dominant),
        weighted / weight,
        shared as f64 / total as f64,
    )
}

/// Fraction of reuse mass at or below `bound`, among cross-thread reuses.
fn shared_cdf(sampler: &Sampler, bound: f64) -> f64 {
    let binner = sampler.binner();
    let mut total = 0f64;
    let mut below = 0f64;
    for (key, count) in sampler.joint() {
        if key.sharing != Sharing::Shared {
            continue;
        }
        total += *count as f64;
        if binner.representative(key.cri_bin) <= bound {
            below += *count as f64;
        }
    }
    below / total
}

#[test]
fn private_long_reuse_scales_by_the_thread_count() {
    for threads in [2u32, 4, 8] {
        let sampler = sample(PRIVATE_LONG, threads);
        let (pri, cri, shared) = mean_cri_at_dominant_pri(&sampler);
        assert_eq!(
            shared, 0.0,
            "rows are indexed by the parallel induction variable, so no reuse can cross threads"
        );
        let expected = f64::from(threads) * pri;
        assert!(
            (cri - expected).abs() < expected * 0.05,
            "T={threads}: measured E[CRI]={cri:.1}, scale-by-T predicts {expected:.1}"
        );
        // The law the previous port applied here instead. Its mean is PRI / T
        // against a truth of T * PRI, so it is short by exactly T^2 -- 4x at
        // two threads, 64x at eight. Asserted explicitly so the gap cannot
        // quietly reappear.
        let racetrack_mean = pri / f64::from(threads);
        let shortfall = cri / racetrack_mean;
        let expected_shortfall = f64::from(threads) * f64::from(threads);
        assert!(
            (shortfall - expected_shortfall).abs() < expected_shortfall * 0.05,
            "T={threads}: a racetrack law predicts {racetrack_mean:.1} against a measured \
             {cri:.1}, a factor of {shortfall:.1}; expected T^2 = {expected_shortfall:.0}"
        );
    }
}

#[test]
fn shared_long_reuse_follows_the_racetrack_with_the_interleaving_dilation() {
    for threads in [2u32, 4, 8] {
        let sampler = sample(SHARED_LONG, threads);
        let (pri, cri, shared) = mean_cri_at_dominant_pri(&sampler);
        assert!(
            shared > 0.9,
            "T={threads}: every thread sweeps the same array, but only {:.1}% of reuse crossed \
             threads",
            shared * 100.0
        );
        // CRI = T * PRI * X with X ~ (T-1)(1-x)^(T-2) has E[X] = 1/T, so the
        // dilation and the splitting cancel: E[CRI] = PRI, independent of T.
        assert!(
            (cri - pri).abs() < pri * 0.05,
            "T={threads}: measured E[CRI]={cri:.1}, racetrack predicts PRI={pri:.1}"
        );
        // Without the dilation the mean would be PRI/T, which at T=8 is off by
        // nearly an order of magnitude.
        let as_shipped = pri / f64::from(threads);
        assert!(
            cri > as_shipped * 1.5,
            "T={threads}: an undilated racetrack predicts {as_shipped:.1}, measured {cri:.1}"
        );
    }
}

#[test]
fn shared_long_reuse_matches_the_racetrack_distribution_not_just_its_mean() {
    // F(c) = 1 - (1 - c / (T * R))^(T - 1) for the dilated racetrack.
    for threads in [2u32, 4, 8] {
        let sampler = sample(SHARED_LONG, threads);
        let (pri, _, _) = mean_cri_at_dominant_pri(&sampler);
        let span = f64::from(threads) * pri;
        let mut worst = 0f64;
        for step in 1..20 {
            let bound = span * f64::from(step) / 20.0;
            let predicted = 1.0 - (1.0 - bound / span).powi(threads as i32 - 1);
            worst = worst.max((shared_cdf(&sampler, bound) - predicted).abs());
        }
        assert!(
            worst < 0.10,
            "T={threads}: racetrack CDF deviates by {worst:.3} at worst"
        );
    }
}

#[test]
fn short_reuse_has_the_negative_binomial_mean() {
    for threads in [2u32, 4, 8] {
        let sampler = sample(SHORT, threads);
        let (pri, cri, shared) = mean_cri_at_dominant_pri(&sampler);
        assert_eq!(shared, 0.0, "the two elements are private to each thread");
        assert!((pri - 2.0).abs() < 1e-9, "the kernel ping-pongs, so PRI is 2");
        // X ~ NB(r, 1/T) has mean r(T-1), so CRI = r + X has mean rT.
        let expected = pri * f64::from(threads);
        assert!(
            (cri - expected).abs() < expected * 0.05,
            "T={threads}: measured E[CRI]={cri:.2}, NBD predicts {expected:.2}"
        );
    }
}

#[test]
fn the_scheduler_assumption_is_what_produces_these_laws() {
    // Under strict lockstep the racetrack collapses: threads never drift apart,
    // so a shared reuse is always cut at the same point instead of at a
    // uniformly random one. This is the assumption the laws rest on, made
    // visible.
    let uniform = sample(SHARED_LONG, 4);
    let lockstep = {
        let config = RunConfig {
            symbols: Vec::new(),
            block_size: NonZero::new(1).expect("non-zero"),
            threads: NonZero::new(4).expect("non-zero"),
            parallel_depth: Some(0),
            chunk: ChunkSize::Auto,
            scheduler: Scheduler::RoundRobin,
            seed: 1,
            binner: Binner::default(),
            track_distance: false,
        };
        let context = Context::new();
        let module = Module::parse(context.mlir_context(), SHARED_LONG).expect("parses");
        let dom = DominanceInfo::new(&module);
        let tree = frontend::extract_target(&module, None, &context, &dom).expect("builds");
        run(tree, &config).expect("samples")
    };

    let spread = |sampler: &Sampler| {
        sampler
            .joint()
            .keys()
            .filter(|key| key.sharing == Sharing::Shared)
            .map(|key| key.cri_bin)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    assert!(
        spread(&uniform) > spread(&lockstep),
        "uniform interleaving must spread shared reuse across more intervals than lockstep"
    );
}
