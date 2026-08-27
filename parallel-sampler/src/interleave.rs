//! Deterministic merging of per-thread access streams.
//!
//! Reproducibility is the whole point of this crate, so no real threads run
//! here. The T streams are merged by a seeded PRNG into one global order, which
//! means a run is a pure function of (program, symbols, thread count, chunk,
//! scheduler, seed). Two machines, or the same machine a year apart, produce
//! byte-identical histograms.
//!
//! The scheduler is pluggable because the model under test *assumes* a
//! particular interleaving. [`Scheduler::Uniform`] is the paper's Assumption 1
//! (statistical uniform interleaving) and is therefore the model's own oracle:
//! if prediction and sampler disagree under `Uniform`, the disagreement is in
//! the model's algebra, not in its assumptions. The other schedulers exist to
//! probe what happens when the assumption is false.

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::interp::{Cursor, Emitted};

/// How the next thread to step is chosen.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Scheduler {
    /// Uniformly at random among threads that still have work — the paper's
    /// Assumption 1. Once a thread drains, the choice narrows to the rest,
    /// which is what a real execution's tail looks like.
    Uniform,
    /// Strict lockstep. The degenerate case the Chernoff bound of Theorem 3.1
    /// says a long reuse interval converges to.
    RoundRobin,
    /// Threads advance at different average speeds. Probes the "same average
    /// speed" half of the racetrack assumption.
    SkewedSpeed { rates: Vec<f64> },
    /// A thread runs a geometric number of accesses before yielding. Probes the
    /// assumption that interleaving happens per access, which no real core does.
    Burst { mean_run: f64 },
}

impl Scheduler {
    fn label(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::RoundRobin => "round_robin",
            Self::SkewedSpeed { .. } => "skewed_speed",
            Self::Burst { .. } => "burst",
        }
    }
}

impl std::fmt::Display for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Merges per-thread cursors into a single deterministic access order.
pub struct Interleaver<'a, 'r> {
    cursors: Vec<Cursor<'a, 'r>>,
    /// One access of lookahead per thread, so the scheduler always picks among
    /// threads that genuinely have work left.
    pending: Vec<Option<Emitted>>,
    live: Vec<u32>,
    rng: ChaCha8Rng,
    scheduler: Scheduler,
    round_robin: usize,
    burst: Option<(u32, u64)>,
}

impl<'a, 'r> Interleaver<'a, 'r> {
    pub fn new(cursors: Vec<Cursor<'a, 'r>>, scheduler: Scheduler, seed: u64) -> Result<Self> {
        let mut interleaver = Self {
            pending: vec![None; cursors.len()],
            live: Vec::with_capacity(cursors.len()),
            cursors,
            rng: ChaCha8Rng::seed_from_u64(seed),
            scheduler,
            round_robin: 0,
            burst: None,
        };
        for tid in 0..interleaver.cursors.len() {
            interleaver.pending[tid] = interleaver.cursors[tid].next_access()?;
            if interleaver.pending[tid].is_some() {
                interleaver.live.push(tid as u32);
            }
        }
        Ok(interleaver)
    }

    /// Yields the next `(thread, access)` in the merged order.
    pub fn next_access(&mut self) -> Result<Option<(u32, Emitted)>> {
        if self.live.is_empty() {
            return Ok(None);
        }
        let position = self.choose();
        let tid = self.live[position];
        let Some(access) = self.pending[tid as usize].take() else {
            // A live thread always has a buffered access; reaching here would
            // mean the live list and the buffer disagreed.
            unreachable!("live thread {tid} had no pending access");
        };
        match self.cursors[tid as usize].next_access()? {
            Some(next) => self.pending[tid as usize] = Some(next),
            None => {
                self.live.swap_remove(position);
                // `swap_remove` moved another thread into this slot, so a
                // round-robin cursor parked here must not skip it.
                if self.round_robin > self.live.len() {
                    self.round_robin = 0;
                }
                if self.burst.is_some_and(|(current, _)| current == tid) {
                    self.burst = None;
                }
            }
        }
        Ok(Some((tid, access)))
    }

    /// Picks an index into `live`.
    fn choose(&mut self) -> usize {
        match &self.scheduler {
            Scheduler::Uniform => self.rng.random_range(0..self.live.len()),
            Scheduler::RoundRobin => {
                let position = self.round_robin % self.live.len();
                self.round_robin = position + 1;
                position
            }
            Scheduler::SkewedSpeed { rates } => {
                let weights: Vec<f64> = self
                    .live
                    .iter()
                    .map(|tid| rates.get(*tid as usize).copied().unwrap_or(1.0).max(0.0))
                    .collect();
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    return self.rng.random_range(0..self.live.len());
                }
                let mut target = self.rng.random::<f64>() * total;
                for (position, weight) in weights.iter().enumerate() {
                    target -= weight;
                    if target <= 0.0 {
                        return position;
                    }
                }
                self.live.len() - 1
            }
            Scheduler::Burst { mean_run } => {
                if let Some((tid, remaining)) = self.burst
                    && remaining > 0
                    && let Some(position) = self.live.iter().position(|live| *live == tid)
                {
                    self.burst = Some((tid, remaining - 1));
                    return position;
                }
                let position = self.rng.random_range(0..self.live.len());
                let run = sample_geometric(&mut self.rng, *mean_run);
                self.burst = Some((self.live[position], run.saturating_sub(1)));
                position
            }
        }
    }
}

/// Draws a run length with the given mean from a geometric distribution.
fn sample_geometric(rng: &mut ChaCha8Rng, mean_run: f64) -> u64 {
    let mean_run = mean_run.max(1.0);
    if mean_run == 1.0 {
        return 1;
    }
    let p = 1.0 / mean_run;
    let uniform: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let draw = (uniform.ln() / (1.0 - p).ln()).floor();
    (draw as u64).saturating_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometric_mean(mean_run: f64) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let samples = 200_000;
        let total: u64 = (0..samples)
            .map(|_| sample_geometric(&mut rng, mean_run))
            .sum();
        total as f64 / f64::from(samples)
    }

    #[test]
    fn geometric_run_lengths_hit_their_mean() {
        for mean in [2.0, 8.0, 64.0] {
            let observed = geometric_mean(mean);
            assert!(
                (observed - mean).abs() < mean * 0.05,
                "mean {mean} sampled as {observed}"
            );
        }
    }

    #[test]
    fn a_mean_run_of_one_never_bursts() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..100 {
            assert_eq!(sample_geometric(&mut rng, 1.0), 1);
        }
    }

    #[test]
    fn scheduler_labels_are_stable_for_manifests() {
        assert_eq!(Scheduler::Uniform.to_string(), "uniform");
        assert_eq!(Scheduler::Burst { mean_run: 4.0 }.to_string(), "burst");
    }
}
