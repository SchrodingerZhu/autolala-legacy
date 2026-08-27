//! A reproducible parallel-locality sampler.
//!
//! Given the same affine MLIR the symbolic analyzer consumes, this crate
//! produces ground-truth reuse statistics for a loop nest parallelized by
//! `schedule(static, chunk)`: reuse-interval and reuse-distance histograms, and
//! the joint law of private versus concurrent reuse interval, split by whether
//! the reuse crossed a thread boundary.
//!
//! Every run is a pure function of its configuration. Thread interleaving is
//! simulated by a seeded PRNG rather than by real threads, so results are
//! byte-identical across machines and repeat runs — which is what makes it
//! usable as a regression oracle for the analytical model.

use std::num::NonZero;

use anyhow::{Result, bail};
use raffine::tree::Tree;

pub mod address;
pub mod affine_eval;
pub mod frontend;
pub mod interleave;
pub mod interp;
pub mod measure;
pub mod report;

use interleave::{Interleaver, Scheduler};
use interp::{ChunkSize, Cursor, Partition, ReferenceTable};
use measure::{Binner, Sampler};

/// Everything that determines a run's output.
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// Concrete values for the program's symbols, indexed by symbol number.
    pub symbols: Vec<i64>,
    /// Cache block size in array elements, applied to the innermost dimension.
    pub block_size: NonZero<usize>,
    pub threads: NonZero<u32>,
    /// Nesting depth of the loop to parallelize. `None` runs the nest
    /// sequentially, which is the mode used to cross-check against the
    /// analyzer's own reuse-interval distribution.
    pub parallel_depth: Option<usize>,
    pub chunk: ChunkSize,
    pub scheduler: Scheduler,
    pub seed: u64,
    pub binner: Binner,
    /// Track exact reuse distance as well as reuse interval. Costs memory
    /// proportional to the working set.
    pub track_distance: bool,
}

/// Runs one sample to completion.
pub fn run<'a>(tree: &'a Tree<'a>, config: &RunConfig) -> Result<Sampler> {
    let table = ReferenceTable::build(tree)?;
    if table.reference_count() == 0 {
        bail!("the target loop nest contains no memory accesses");
    }
    if table.max_rank() > address::MAX_ARRAY_DIMS {
        bail!(
            "the kernel has an array of rank {}, above the supported maximum of {}",
            table.max_rank(),
            address::MAX_ARRAY_DIMS
        );
    }

    let threads = match config.parallel_depth {
        Some(_) => config.threads.get(),
        // Sequential mode is one stream; a thread count would be meaningless.
        None => 1,
    };

    let mut cursors = Vec::with_capacity(threads as usize);
    for tid in 0..threads {
        let partition = config.parallel_depth.map(|depth| Partition {
            depth,
            tid,
            threads: config.threads,
            chunk: config.chunk,
        });
        cursors.push(Cursor::new(
            tree,
            &table,
            config.symbols.clone(),
            config.block_size,
            partition,
        ));
    }

    let mut sampler = Sampler::new(
        threads as usize,
        table.reference_count(),
        config.binner,
        config.track_distance,
    );
    let mut interleaver = Interleaver::new(cursors, config.scheduler.clone(), config.seed)?;
    while let Some((tid, access)) = interleaver.next_access()? {
        sampler.observe(tid, access);
    }
    Ok(sampler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::Module;
    use raffine::{Context, DominanceInfo};

    /// Runs a snippet of MLIR through the sampler.
    fn sample(source: &str, config: &RunConfig) -> Sampler {
        let context = Context::new();
        let module = Module::parse(context.mlir_context(), source).expect("parses");
        let dom = DominanceInfo::new(&module);
        let tree = frontend::extract_target(&module, None, &context, &dom).expect("builds");
        run(tree, config).expect("samples")
    }

    fn config(threads: u32, parallel_depth: Option<usize>) -> RunConfig {
        RunConfig {
            symbols: Vec::new(),
            block_size: NonZero::new(1).expect("non-zero"),
            threads: NonZero::new(threads).expect("non-zero"),
            parallel_depth,
            chunk: ChunkSize::Auto,
            scheduler: Scheduler::Uniform,
            seed: 0xC0FFEE,
            binner: Binner::default(),
            track_distance: true,
        }
    }

    const SWEEP: &str = r#"
module {
  func.func @sweep(%A: memref<?xf64>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 4 {
        %0 = affine.load %A[%j] : memref<?xf64>
      }
    }
    return
  }
}
"#;

    #[test]
    fn sequential_sweep_has_the_expected_access_count_and_reuse() {
        let sampler = sample(SWEEP, &config(1, None));
        assert_eq!(sampler.total_accesses(), 32);
        // Four distinct elements are cold; every other access reuses at
        // interval 4.
        assert_eq!(sampler.cold_accesses(), 4);
        let cri: Vec<_> = sampler.cri().iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(cri, vec![(4, 28)]);
    }

    #[test]
    fn parallelizing_the_outer_loop_preserves_the_total_access_count() {
        for threads in [1u32, 2, 4, 8] {
            let sampler = sample(SWEEP, &config(threads, Some(0)));
            assert_eq!(
                sampler.total_accesses(),
                32,
                "thread count {threads} changed the work"
            );
        }
    }

    #[test]
    fn runs_are_reproducible_across_repeats() {
        let config = config(4, Some(0));
        let first = sample(SWEEP, &config);
        let second = sample(SWEEP, &config);
        assert_eq!(first.cri(), second.cri());
        assert_eq!(first.joint(), second.joint());
    }

    #[test]
    fn a_different_seed_changes_the_interleaving() {
        let mut config = config(4, Some(0));
        let first = sample(SWEEP, &config);
        config.seed = 12345;
        let second = sample(SWEEP, &config);
        assert_ne!(
            first.cri(),
            second.cri(),
            "two seeds produced the same interleaving"
        );
        assert_eq!(first.total_accesses(), second.total_accesses());
    }

    #[test]
    fn threads_sharing_one_array_produce_shared_reuses() {
        // Every thread sweeps the same four elements of A, so most reuse is
        // cross-thread. This is the racetrack cell of the model's table.
        let sampler = sample(SWEEP, &config(4, Some(0)));
        let shared: u64 = sampler
            .per_reference()
            .iter()
            .map(|stats| stats.shared_reuses)
            .sum();
        assert!(shared > 0, "no cross-thread reuse was observed");
    }

    #[test]
    fn a_thread_private_array_produces_no_shared_reuse() {
        // Indexing by the parallel induction variable makes each thread's data
        // disjoint: the "no sharing" row of the model's table.
        const PRIVATE: &str = r#"
module {
  func.func @private(%A: memref<?x?xf64>) {
    affine.for %i = 0 to 8 {
      affine.for %r = 0 to 3 {
        affine.for %j = 0 to 4 {
          %0 = affine.load %A[%i, %j] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
"#;
        let sampler = sample(PRIVATE, &config(4, Some(0)));
        let shared: u64 = sampler
            .per_reference()
            .iter()
            .map(|stats| stats.shared_reuses)
            .sum();
        assert_eq!(shared, 0, "disjoint data produced a cross-thread reuse");
    }

    #[test]
    fn affine_if_guards_are_honored() {
        const GUARDED: &str = r#"
module {
  func.func @guarded(%A: memref<?xf64>) {
    affine.for %i = 0 to 10 {
      affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
        %0 = affine.load %A[%i] : memref<?xf64>
      }
    }
    return
  }
}
"#;
        let sampler = sample(GUARDED, &config(1, None));
        assert_eq!(sampler.total_accesses(), 5);
    }

    #[test]
    fn a_nest_without_accesses_is_rejected() {
        const EMPTY: &str = r#"
module {
  func.func @empty() {
    affine.for %i = 0 to 4 {
    }
    return
  }
}
"#;
        let context = Context::new();
        let module = Module::parse(context.mlir_context(), EMPTY).expect("parses");
        let dom = DominanceInfo::new(&module);
        let tree = frontend::extract_target(&module, None, &context, &dom).expect("builds");
        let error = run(tree, &config(1, None)).expect_err("no accesses");
        assert!(error.to_string().contains("no memory accesses"), "{error}");
    }
}
