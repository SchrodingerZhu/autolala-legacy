//! Serializable results plus the manifest that makes a run reproducible.
//!
//! Every number a run produces is emitted alongside the full configuration that
//! produced it and a digest of the histograms. Re-running with the manifest must
//! reproduce the digest; if it does not, something non-deterministic crept into
//! the sampler and the comparison against the model is worthless.

use std::num::NonZero;

use serde::{Deserialize, Serialize};

use crate::RunConfig;
use crate::interp::ChunkSize;
use crate::measure::{Histogram, ReferenceStats, Sampler, Sharing};

/// The exact configuration of a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub tool_version: String,
    pub input: String,
    /// Digest of the MLIR source, so a rerun cannot silently use a changed
    /// kernel.
    pub input_digest: String,
    pub target_function: Option<String>,
    pub symbols: Vec<i64>,
    pub block_size: usize,
    pub num_sets: usize,
    pub threads: u32,
    pub parallel_depth: Option<usize>,
    pub chunk: String,
    pub scheduler: crate::interleave::Scheduler,
    pub seed: u64,
}

/// One bucket of a histogram, with the value its bin stands for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bucket {
    pub bin: u32,
    pub value: f64,
    pub count: u64,
}

/// One cell of the joint `(PRI, CRI)` table.
///
/// `pri` is `null` when this thread had never touched the datum before, which
/// happens only for cross-thread reuse and which the model must account for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointRow {
    pub reference: u32,
    pub pri: Option<f64>,
    pub cri: f64,
    pub sharing: Sharing,
    pub count: u64,
}

/// Miss-ratio curves over a shared grid of cache sizes.
///
/// Two curves are reported because a model can be wrong in two independent
/// places. `exact` is ground truth from the measured reuse-distance histogram.
/// `from_reuse_interval` runs the same Denning recursion the analytical model
/// uses, but over the *measured* reuse intervals. Comparing an analytical curve
/// against `from_reuse_interval` isolates the error of the CRI model; comparing
/// `from_reuse_interval` against `exact` isolates the error of the
/// reuse-interval-to-reuse-distance conversion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissRatioCurves {
    pub cache_sizes: Vec<f64>,
    pub exact: Option<Vec<f64>>,
    pub from_reuse_interval: Vec<f64>,
}

/// A complete sampler run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub manifest: Manifest,
    pub total_accesses: u64,
    pub cold_accesses: u64,
    /// Set when the reuse-distance index overflowed, meaning `reuse_distance`
    /// is incomplete and must not be read as ground truth.
    pub reuse_distance_truncated: bool,
    pub concurrent_reuse_interval: Vec<Bucket>,
    pub private_reuse_interval: Vec<Bucket>,
    pub reuse_distance: Option<Vec<Bucket>>,
    pub joint: Vec<JointRow>,
    pub miss_ratio_curves: MissRatioCurves,
    pub per_reference: Vec<ReferenceStats>,
    pub memref_names: Vec<String>,
    /// Digest over the histograms; equal digests mean equal runs.
    pub digest: String,
}

impl Manifest {
    pub fn new(
        input: String,
        input_digest: String,
        target_function: Option<String>,
        num_sets: NonZero<usize>,
        config: &RunConfig,
    ) -> Self {
        Self {
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            input,
            input_digest,
            target_function,
            symbols: config.symbols.clone(),
            block_size: config.block_size.get(),
            num_sets: num_sets.get(),
            threads: config.threads.get(),
            parallel_depth: config.parallel_depth,
            chunk: match config.chunk {
                ChunkSize::Auto => "auto".to_string(),
                ChunkSize::Fixed(chunk) => chunk.get().to_string(),
            },
            scheduler: config.scheduler.clone(),
            seed: config.seed,
        }
    }
}

impl Report {
    pub fn build(manifest: Manifest, sampler: &Sampler, memref_names: Vec<String>) -> Self {
        let binner = sampler.binner();
        let buckets = |histogram: &Histogram| {
            let mut items: Vec<Bucket> = histogram
                .iter()
                .map(|(bin, count)| Bucket {
                    bin: *bin,
                    value: binner.representative(*bin),
                    count: *count,
                })
                .collect();
            items.sort_by(|a, b| a.bin.cmp(&b.bin));
            items
        };

        let mut joint: Vec<JointRow> = sampler
            .joint()
            .iter()
            .map(|(key, count)| JointRow {
                reference: key.reference,
                pri: (key.pri_bin != u32::MAX).then(|| binner.representative(key.pri_bin)),
                cri: binner.representative(key.cri_bin),
                sharing: key.sharing,
                count: *count,
            })
            .collect();
        joint.sort_by(|a, b| {
            a.reference
                .cmp(&b.reference)
                .then(a.sharing.cmp(&b.sharing))
                .then(a.pri.partial_cmp(&b.pri).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.cri.partial_cmp(&b.cri).unwrap_or(std::cmp::Ordering::Equal))
        });

        let concurrent_reuse_interval = buckets(sampler.cri());
        let private_reuse_interval = buckets(sampler.pri());
        let reuse_distance = sampler.reuse_distance().map(buckets);
        let miss_ratio_curves = miss_ratio_curves(sampler, &concurrent_reuse_interval);

        let digest = digest_of(
            sampler.total_accesses(),
            sampler.cold_accesses(),
            &concurrent_reuse_interval,
            &private_reuse_interval,
            reuse_distance.as_deref(),
        );

        Self {
            manifest,
            total_accesses: sampler.total_accesses(),
            cold_accesses: sampler.cold_accesses(),
            reuse_distance_truncated: sampler.overflowed(),
            concurrent_reuse_interval,
            private_reuse_interval,
            reuse_distance,
            joint,
            miss_ratio_curves,
            per_reference: sampler.per_reference().to_vec(),
            memref_names,
            digest,
        }
    }
}

/// Builds both curves over a logarithmic grid of cache sizes.
///
/// The grid runs to the largest observed reuse interval, past which the curve
/// is flat at the compulsory-miss ratio, and carries four points per octave --
/// enough to place each step without making the report enormous.
fn miss_ratio_curves(sampler: &Sampler, cri: &[Bucket]) -> MissRatioCurves {
    let largest = cri.iter().map(|bucket| bucket.value).fold(1.0, f64::max);
    let octaves = largest.log2().ceil().max(1.0) as u32;
    let mut cache_sizes = vec![0.0];
    for step in 0..=octaves * 4 {
        cache_sizes.push(2f64.powf(f64::from(step) / 4.0));
    }

    let curve = denning::MissRatioCurve::new(&sampler.denning_support());
    let from_reuse_interval = cache_sizes
        .iter()
        .map(|size| curve.miss_ratio_at(*size))
        .collect();
    let exact = cache_sizes
        .iter()
        .map(|size| sampler.exact_miss_ratio_at(*size))
        .collect::<Option<Vec<f64>>>();

    MissRatioCurves {
        cache_sizes,
        exact,
        from_reuse_interval,
    }
}

/// FNV-1a over the run's histograms.
///
/// Hand-rolled rather than pulled from a crate because it must stay stable
/// forever: a digest that changes when a dependency bumps its hasher would
/// invalidate every stored result.
fn digest_of(
    total: u64,
    cold: u64,
    cri: &[Bucket],
    pri: &[Bucket],
    distance: Option<&[Bucket]>,
) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut feed = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    feed(total);
    feed(cold);
    for section in [Some(cri), Some(pri), distance] {
        // Distinguish "absent" from "empty" so a run with distance tracking off
        // cannot collide with one where nothing was recorded.
        feed(section.map_or(u64::MAX, |items| items.len() as u64));
        for bucket in section.unwrap_or(&[]) {
            feed(u64::from(bucket.bin));
            feed(bucket.count);
        }
    }
    format!("{hash:016x}")
}

/// FNV-1a over arbitrary bytes, used for the input digest.
pub fn digest_bytes(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bucket(bin: u32, count: u64) -> Bucket {
        Bucket {
            bin,
            value: f64::from(bin),
            count,
        }
    }

    #[test]
    fn digest_changes_with_the_histogram() {
        let base = digest_of(10, 2, &[bucket(1, 3)], &[], None);
        assert_ne!(base, digest_of(10, 2, &[bucket(1, 4)], &[], None));
        assert_ne!(base, digest_of(10, 2, &[bucket(2, 3)], &[], None));
        assert_ne!(base, digest_of(11, 2, &[bucket(1, 3)], &[], None));
    }

    #[test]
    fn absent_and_empty_distance_sections_differ() {
        let absent = digest_of(1, 0, &[], &[], None);
        let empty = digest_of(1, 0, &[], &[], Some(&[]));
        assert_ne!(absent, empty);
    }

    #[test]
    fn digest_is_stable_for_equal_input() {
        let first = digest_of(7, 1, &[bucket(3, 9)], &[bucket(2, 4)], Some(&[bucket(1, 1)]));
        let second = digest_of(7, 1, &[bucket(3, 9)], &[bucket(2, 4)], Some(&[bucket(1, 1)]));
        assert_eq!(first, second);
    }

    #[test]
    fn input_digest_distinguishes_sources() {
        assert_ne!(digest_bytes(b"module {}"), digest_bytes(b"module { }"));
    }
}
