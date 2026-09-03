//! Miss ratio and miss count at one cache size.
//!
//! Plain mode reads a report that already carries a miss-ratio curve (the
//! analyzer's `--json` output or `assoc-conv`'s) and interpolates it. With
//! `--symbol` it instead instantiates the report's symbolic distribution at
//! the given parameter values -- the derivation is done once, symbolically;
//! every (input size, cache size) query is this command.

mod symbolic;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context as _, Result};
use palc::Parser;
use serde::Deserialize;
use splines::Key;

#[derive(Debug, Clone, Deserialize)]
struct JsonInput {
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
}

#[derive(Debug, Clone, Deserialize)]
struct MissRatioCurve {
    turning_points: Box<[f64]>,
    miss_ratio: Box<[f64]>,
}

#[derive(Deserialize)]
struct SymbolicInput {
    symbolic: symbolic::SymbolicDistribution,
}

#[derive(Debug, Clone, Parser)]
struct Option {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,
    /// Target cache size
    #[arg(short, long)]
    cache_size: u64,
    /// Target block size
    #[arg(short, long)]
    block_size: u64,
    /// Instantiate the report's symbolic distribution at this parameter
    /// value (`p0=256`; repeat once per parameter). Without it the report's
    /// own miss-ratio curve is used.
    #[arg(short, long)]
    symbol: Vec<String>,
    /// Set associativity applied after instantiation (1 = fully associative).
    #[arg(short, long, default_value = "1")]
    assoc: usize,
}

/// Miss count at `target` blocks, by Catmull-Rom interpolation of the curve's
/// turning points, as the plain mode has always done.
fn interpolate(turning_points: &[f64], miss_ratio: &[f64], total: f64, target: f64) -> f64 {
    let sequence = turning_points
        .iter()
        .zip(miss_ratio.iter())
        .map(|(k, v)| Key::new(*k, *v * total, splines::Interpolation::CatmullRom))
        .collect::<Vec<_>>();
    splines::Spline::from_vec(sequence)
        .clamped_sample(target)
        .unwrap_or(f64::NAN)
}

fn main() -> Result<()> {
    let option = Option::parse();
    let program_name = option
        .input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let target = (option.cache_size / option.block_size) as f64;

    let (sample, total) = if option.symbol.is_empty() {
        let json_input: JsonInput =
            simd_json::from_reader(std::fs::File::open(&option.input)?)
                .context("failed to parse JSON input")?;
        let total: f64 = json_input
            .total_count
            .trim_end_matches(" R")
            .parse()
            .context("invalid total count")?;
        let curve = json_input.miss_ratio_curve;
        (interpolate(&curve.turning_points, &curve.miss_ratio, total, target), total)
    } else {
        let mut values = HashMap::new();
        for binding in &option.symbol {
            let (name, value) = symbolic::parse_symbol_binding(binding)?;
            values.insert(name, value);
        }
        let text = std::fs::read_to_string(&option.input)?;
        let input: SymbolicInput = serde_json::from_str(&text)
            .context("input has no `symbolic` section; derive it with `analyzer --json`")?;
        let instantiated = symbolic::instantiate(&input.symbolic, &values, option.assoc)?;
        eprintln!(
            "instantiated {} reuse intervals from {} points: histogram {:.3}s, curve {:.3}s",
            instantiated.support,
            instantiated.points,
            instantiated.histogram_seconds,
            instantiated.curve_seconds
        );
        let curve = instantiated.curve;
        (
            interpolate(curve.turning_points(), curve.miss_ratio(), instantiated.total, target),
            instantiated.total,
        )
    };
    println!(
        "{program_name},{},{},{},{}",
        sample.round(),
        total.round(),
        sample / total,
        option.cache_size,
    );
    Ok(())
}
