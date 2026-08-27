//! `psample` — reproducible parallel locality sampling over affine MLIR.

use std::num::NonZero;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use melior::ir::Module;
use palc::Parser;
use parallel_sampler::frontend::{self, parse_symbol_binding};
use parallel_sampler::interleave::Scheduler;
use parallel_sampler::interp::ChunkSize;
use parallel_sampler::measure::Binner;
use parallel_sampler::report::{Manifest, Report, digest_bytes};
use parallel_sampler::{RunConfig, run};
use raffine::{Context, DominanceInfo};

#[derive(Debug, Parser)]
struct Options {
    /// Affine MLIR file to sample.
    #[arg(short, long)]
    input: PathBuf,

    /// Write the JSON report here instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Function to analyze. Defaults to the first `func.func` in the module.
    #[arg(short = 'f', long)]
    target_function: Option<String>,

    /// Bind a program symbol, as `s0=512`. Repeatable; every symbol the
    /// program reads needs one.
    #[arg(long = "symbol")]
    symbols: Vec<String>,

    /// Cache block size in array elements, applied to the innermost dimension
    /// exactly as the analyzer's polyhedral encoding applies it.
    #[arg(short = 'b', long, default_value = "1")]
    block_size: NonZero<usize>,

    /// Number of cache sets, recorded in the manifest for set-associative
    /// comparisons.
    #[arg(short = 's', long, default_value = "1")]
    num_sets: NonZero<usize>,

    /// Thread count. Ignored without `--parallel-loop-depth`.
    #[arg(short = 't', long, default_value = "1")]
    threads: NonZero<u32>,

    /// Nesting depth of the loop to parallelize, outermost being 0. Omit to
    /// sample the nest sequentially.
    #[arg(short = 'p', long)]
    parallel_loop_depth: Option<usize>,

    /// `schedule(static, chunk)` chunk size. Defaults to OpenMP's contiguous
    /// blocks of `ceil(trip / threads)`; pass 1 for round-robin.
    #[arg(short = 'c', long)]
    chunk: Option<NonZero<i64>>,

    /// Thread interleaving model: `uniform` (the analytical model's own
    /// assumption), `round-robin`, `burst:<mean run>`, or
    /// `skewed:<rate>,<rate>,..`.
    #[arg(long, default_value = "uniform")]
    scheduler: String,

    /// PRNG seed. Runs with the same seed are byte-identical.
    #[arg(long, default_value = "0")]
    seed: u64,

    /// Skip exact reuse-distance tracking, which costs memory proportional to
    /// the working set.
    #[arg(long)]
    no_reuse_distance: bool,

    /// Reuse values below this are histogrammed exactly; larger ones are
    /// binned logarithmically.
    #[arg(long, default_value = "256")]
    exact_below: u64,

    /// Logarithmic bins per octave above `--exact-below`.
    #[arg(long, default_value = "32")]
    bins_per_octave: NonZero<u32>,
}

fn parse_scheduler(text: &str) -> Result<Scheduler> {
    let (kind, argument) = match text.split_once(':') {
        Some((kind, argument)) => (kind, Some(argument)),
        None => (text, None),
    };
    match kind.trim() {
        "uniform" => Ok(Scheduler::Uniform),
        "round-robin" | "round_robin" | "rr" => Ok(Scheduler::RoundRobin),
        "burst" => {
            let Some(argument) = argument else {
                bail!("`burst` needs a mean run length, as `burst:8`");
            };
            let mean_run: f64 = argument
                .trim()
                .parse()
                .with_context(|| format!("`{argument}` is not a mean run length"))?;
            if mean_run < 1.0 || mean_run.is_nan() {
                bail!("burst mean run length must be at least 1, found {mean_run}");
            }
            Ok(Scheduler::Burst { mean_run })
        }
        "skewed" | "skewed-speed" => {
            let Some(argument) = argument else {
                bail!("`skewed` needs per-thread rates, as `skewed:1,2,1,1`");
            };
            let rates = argument
                .split(',')
                .map(|rate| {
                    rate.trim()
                        .parse::<f64>()
                        .with_context(|| format!("`{rate}` is not a rate"))
                })
                .collect::<Result<Vec<f64>>>()?;
            if rates.iter().any(|rate| *rate < 0.0) {
                bail!("skewed-speed rates must be non-negative");
            }
            if rates.iter().sum::<f64>() <= 0.0 {
                bail!("skewed-speed rates must not all be zero");
            }
            Ok(Scheduler::SkewedSpeed { rates })
        }
        other => bail!(
            "unknown scheduler `{other}`; expected uniform, round-robin, burst:<n>, or skewed:<rates>"
        ),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let options = Options::parse();

    let source = std::fs::read_to_string(&options.input)
        .with_context(|| format!("reading {}", options.input.display()))?;
    let input_digest = digest_bytes(source.as_bytes());

    let context = Context::new();
    let module = Module::parse(context.mlir_context(), &source)
        .ok_or_else(|| anyhow::anyhow!("{} is not valid MLIR", options.input.display()))?;
    let dom = DominanceInfo::new(&module);
    let tree = frontend::extract_target(
        &module,
        options.target_function.as_deref(),
        &context,
        &dom,
    )?;

    let facts = frontend::facts(tree);
    let mut symbols = vec![0i64; facts.symbols];
    let mut bound = vec![false; facts.symbols];
    for binding in &options.symbols {
        let (index, value) = parse_symbol_binding(binding)?;
        if index >= facts.symbols {
            bail!(
                "symbol s{index} does not exist; the program reads {} symbol(s)",
                facts.symbols
            );
        }
        symbols[index] = value;
        bound[index] = true;
    }
    if let Some(missing) = bound.iter().position(|bound| !bound) {
        bail!(
            "symbol s{missing} has no binding; pass `--symbol s{missing}=<value>` \
             (the program reads {} symbol(s))",
            facts.symbols
        );
    }

    if let Some(depth) = options.parallel_loop_depth
        && depth > facts.max_depth
    {
        bail!(
            "--parallel-loop-depth {depth} is deeper than the nest, whose loops are at depths 0..={}",
            facts.max_depth
        );
    }
    if options.parallel_loop_depth.is_none() && options.threads.get() > 1 {
        bail!("--threads has no effect without --parallel-loop-depth");
    }

    let config = RunConfig {
        symbols,
        block_size: options.block_size,
        threads: options.threads,
        parallel_depth: options.parallel_loop_depth,
        chunk: match options.chunk {
            Some(chunk) => ChunkSize::Fixed(chunk),
            None => ChunkSize::Auto,
        },
        scheduler: parse_scheduler(&options.scheduler)?,
        seed: options.seed,
        binner: Binner::new(options.exact_below, options.bins_per_octave),
        track_distance: !options.no_reuse_distance,
    };

    let sampler = run(tree, &config)?;
    if sampler.overflowed() {
        tracing::warn!(
            "the reuse-distance index overflowed; the distance histogram is incomplete"
        );
    }

    let table = parallel_sampler::interp::ReferenceTable::build(tree)?;
    let manifest = Manifest::new(
        options.input.display().to_string(),
        input_digest,
        options.target_function.clone(),
        options.num_sets,
        &config,
    );
    let report = Report::build(manifest, &sampler, table.memref_names().to_vec());

    let json = serde_json::to_string_pretty(&report)?;
    match &options.output {
        Some(path) => std::fs::write(path, json)
            .with_context(|| format!("writing {}", path.display()))?,
        None => println!("{json}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_spellings_parse() {
        assert_eq!(parse_scheduler("uniform").expect("parses"), Scheduler::Uniform);
        assert_eq!(parse_scheduler("rr").expect("parses"), Scheduler::RoundRobin);
        assert_eq!(
            parse_scheduler("burst:8").expect("parses"),
            Scheduler::Burst { mean_run: 8.0 }
        );
        assert_eq!(
            parse_scheduler("skewed:1,2.5").expect("parses"),
            Scheduler::SkewedSpeed {
                rates: vec![1.0, 2.5]
            }
        );
    }

    #[test]
    fn bad_schedulers_are_rejected() {
        assert!(parse_scheduler("nope").is_err());
        assert!(parse_scheduler("burst").is_err());
        assert!(parse_scheduler("burst:0.5").is_err());
        assert!(parse_scheduler("skewed:0,0").is_err());
        assert!(parse_scheduler("skewed:-1,2").is_err());
    }
}
