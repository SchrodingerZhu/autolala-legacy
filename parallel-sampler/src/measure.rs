//! Reuse measurement over an interleaved access stream.
//!
//! The sampler tracks the private and the concurrent view of every reuse *at the
//! same instant*: when thread `t` touches datum `d`, we know both the global
//! time since `d` was last touched by anyone (its CRI) and the thread-local time
//! since `t` itself last touched it (its PRI). Recording the pair, rather than
//! two independent histograms, is what lets the joint law `CRI | PRI = r` be
//! measured — and that conditional law is exactly what the model's Table-1 cells
//! claim to predict.
//!
//! Keeping both views in one pass also means no per-access history is stored:
//! memory is proportional to the working set, not to the trace length.

use std::num::NonZero;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::address::DataId;
use crate::interp::Emitted;

/// Whether a reuse crossed a thread boundary.
///
/// This is the *dynamic* truth about data sharing: the previous access to this
/// datum came from another thread. The model's static classification (does the
/// reference's subscript mention the parallel induction variable) is a
/// prediction of this, and the two are compared during validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sharing {
    /// The previous access came from the same thread.
    Private,
    /// The previous access came from a different thread.
    Shared,
}

/// Bins reuse values so joint histograms stay bounded.
///
/// Small values are kept exactly — the NBD cell of Table 1 lives entirely down
/// there, and rounding it away would hide precisely the disagreement we are
/// hunting. Larger values fall into logarithmic bins, which is the right
/// resolution for the racetrack and scale-by-T cells since their error is
/// multiplicative.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Binner {
    exact_below: u64,
    per_octave: u32,
}

impl Default for Binner {
    fn default() -> Self {
        Self {
            exact_below: 256,
            per_octave: 32,
        }
    }
}

impl Binner {
    pub fn new(exact_below: u64, per_octave: NonZero<u32>) -> Self {
        Self {
            exact_below,
            per_octave: per_octave.get(),
        }
    }

    /// Maps a reuse value to its bin index.
    pub fn bin(&self, value: u64) -> u32 {
        if value < self.exact_below {
            return value as u32;
        }
        let octave = (value as f64 / self.exact_below as f64).log2();
        let offset = (octave * f64::from(self.per_octave)).floor() as u64;
        (self.exact_below + offset) as u32
    }

    /// A representative value for a bin, used when comparing against a model
    /// distribution. Exact bins report themselves; log bins report the
    /// geometric midpoint of their range.
    pub fn representative(&self, bin: u32) -> f64 {
        let bin = u64::from(bin);
        if bin < self.exact_below {
            return bin as f64;
        }
        let offset = (bin - self.exact_below) as f64;
        let per_octave = f64::from(self.per_octave);
        let low = self.exact_below as f64 * (offset / per_octave).exp2();
        let high = self.exact_below as f64 * ((offset + 1.0) / per_octave).exp2();
        (low * high).sqrt()
    }
}

/// A sparse histogram over bin indices.
pub type Histogram = FxHashMap<u32, u64>;

/// Key of the joint `(PRI, CRI, sharing)` table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointKey {
    pub reference: u32,
    pub pri_bin: u32,
    pub cri_bin: u32,
    pub sharing: Sharing,
}

/// What the global (interleaved) view knows about a datum.
#[derive(Clone, Copy, Debug)]
struct GlobalEntry {
    time: u64,
    tid: u32,
    /// Position of this datum's marker in the reuse-distance index.
    slot: u32,
}

/// Exact LRU reuse distance: the number of *distinct* data touched since a
/// datum was last accessed.
///
/// Implemented as a Fenwick tree of "most recent access" markers over a
/// monotonically growing timeline. The timeline is compacted once it outgrows
/// the live marker count, so memory tracks the working set rather than the trace
/// length — a 512-cubed matmul would otherwise want gigabytes of index.
#[derive(Debug)]
struct ReuseDistance {
    tree: Vec<u32>,
    used: u32,
    live: u32,
}

impl ReuseDistance {
    fn new() -> Self {
        Self {
            tree: vec![0; 1025],
            used: 0,
            live: 0,
        }
    }

    fn capacity(&self) -> u32 {
        (self.tree.len() - 1) as u32
    }

    fn add(&mut self, mut index: u32, delta: i64) {
        while index <= self.capacity() {
            let slot = &mut self.tree[index as usize];
            *slot = (i64::from(*slot) + delta) as u32;
            index += index & index.wrapping_neg();
        }
    }

    /// Number of markers in `1..=index`.
    fn prefix(&self, mut index: u32) -> u32 {
        let mut total = 0;
        while index > 0 {
            total += self.tree[index as usize];
            index -= index & index.wrapping_neg();
        }
        total
    }

    /// Retires the marker at `slot` and returns the number of distinct data
    /// touched after it.
    fn distance_from(&mut self, slot: u32) -> u64 {
        let after = self.live - self.prefix(slot);
        self.add(slot, -1);
        self.live -= 1;
        u64::from(after)
    }

    /// Places a marker at the newest timeline position.
    fn touch(&mut self) -> u32 {
        if self.used >= self.capacity() {
            return u32::MAX;
        }
        self.used += 1;
        self.live += 1;
        self.add(self.used, 1);
        self.used
    }

    /// True once the timeline is more sparse than dense and should be rebuilt.
    fn needs_compaction(&self) -> bool {
        self.used >= self.capacity() || self.used > 2 * self.live.max(512)
    }

    /// Rebuilds the timeline densely. Callers must renumber their stored slots
    /// using the returned mapping order.
    fn compact(&mut self, slots: &mut [u32]) {
        slots.sort_unstable();
        let live = slots.len() as u32;
        let capacity = (live.saturating_mul(4)).max(1024);
        self.tree = vec![0; capacity as usize + 1];
        self.used = live;
        self.live = live;
        for index in 1..=live {
            self.add(index, 1);
        }
    }
}

/// Per-reference tallies, so error can be attributed to the access site the
/// model classified rather than only to the kernel as a whole.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReferenceStats {
    /// Which array this access site targets, as an index into the report's
    /// `memref_names`. Attributing an error to a reference is only useful if
    /// you can tell which array it reads.
    pub memref: Option<u32>,
    pub accesses: u64,
    pub cold: u64,
    pub private_reuses: u64,
    pub shared_reuses: u64,
}

/// Accumulates every statistic the validation needs from one interleaved run.
#[derive(Debug)]
pub struct Sampler {
    binner: Binner,
    clock: u64,
    global_last: FxHashMap<DataId, GlobalEntry>,
    private_last: Vec<FxHashMap<DataId, u64>>,
    private_clock: Vec<u64>,
    cri: Histogram,
    pri: Histogram,
    reuse_distance: Histogram,
    joint: FxHashMap<JointKey, u64>,
    per_reference: Vec<ReferenceStats>,
    total_accesses: u64,
    cold_accesses: u64,
    distance: Option<ReuseDistance>,
    overflowed: bool,
}

impl Sampler {
    pub fn new(threads: usize, references: usize, binner: Binner, track_distance: bool) -> Self {
        Self {
            binner,
            clock: 0,
            global_last: FxHashMap::default(),
            private_last: vec![FxHashMap::default(); threads],
            private_clock: vec![0; threads],
            cri: Histogram::default(),
            pri: Histogram::default(),
            reuse_distance: Histogram::default(),
            joint: FxHashMap::default(),
            per_reference: vec![ReferenceStats::default(); references],
            total_accesses: 0,
            cold_accesses: 0,
            distance: track_distance.then(ReuseDistance::new),
            overflowed: false,
        }
    }

    /// Records one access from thread `tid`.
    pub fn observe(&mut self, tid: u32, access: Emitted) {
        self.clock += 1;
        self.total_accesses += 1;
        let thread = tid as usize;
        self.private_clock[thread] += 1;
        let private_now = self.private_clock[thread];

        if let Some(stats) = self.per_reference.get_mut(access.reference as usize) {
            stats.accesses += 1;
            stats.memref = Some(access.data.memref.0);
        }

        let private_previous = self.private_last[thread].insert(access.data, private_now);
        let previous = self.global_last.get(&access.data).copied();

        let slot = match &mut self.distance {
            Some(distance) => {
                if let Some(entry) = previous
                    && entry.slot != u32::MAX
                {
                    let value = distance.distance_from(entry.slot);
                    *self.reuse_distance.entry(self.binner.bin(value)).or_default() += 1;
                }
                let slot = distance.touch();
                if slot == u32::MAX {
                    self.overflowed = true;
                }
                slot
            }
            None => u32::MAX,
        };

        self.global_last.insert(
            access.data,
            GlobalEntry {
                time: self.clock,
                tid,
                slot,
            },
        );

        let Some(entry) = previous else {
            self.cold_accesses += 1;
            if let Some(stats) = self.per_reference.get_mut(access.reference as usize) {
                stats.cold += 1;
            }
            self.maybe_compact();
            return;
        };

        let sharing = if entry.tid == tid {
            Sharing::Private
        } else {
            Sharing::Shared
        };
        if let Some(stats) = self.per_reference.get_mut(access.reference as usize) {
            match sharing {
                Sharing::Private => stats.private_reuses += 1,
                Sharing::Shared => stats.shared_reuses += 1,
            }
        }

        let cri = self.clock - entry.time;
        let cri_bin = self.binner.bin(cri);
        *self.cri.entry(cri_bin).or_default() += 1;

        // A shared reuse can have no private predecessor at all: this thread may
        // never have touched the datum. Those samples still belong in the joint
        // table, under the sentinel PRI bin, because the model must predict them
        // too.
        let pri_bin = match private_previous {
            Some(previous) => {
                let pri = private_now - previous;
                let bin = self.binner.bin(pri);
                *self.pri.entry(bin).or_default() += 1;
                bin
            }
            None => u32::MAX,
        };

        *self
            .joint
            .entry(JointKey {
                reference: access.reference,
                pri_bin,
                cri_bin,
                sharing,
            })
            .or_default() += 1;

        self.maybe_compact();
    }

    /// Rebuilds the reuse-distance timeline when it has grown too sparse,
    /// renumbering every stored slot to match.
    fn maybe_compact(&mut self) {
        let Some(distance) = &mut self.distance else {
            return;
        };
        if !distance.needs_compaction() {
            return;
        }
        let mut slots: Vec<u32> = self
            .global_last
            .values()
            .filter(|entry| entry.slot != u32::MAX)
            .map(|entry| entry.slot)
            .collect();
        distance.compact(&mut slots);
        let renumbered: FxHashMap<u32, u32> = slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (*slot, index as u32 + 1))
            .collect();
        for entry in self.global_last.values_mut() {
            if entry.slot != u32::MAX
                && let Some(new_slot) = renumbered.get(&entry.slot)
            {
                entry.slot = *new_slot;
            }
        }
    }

    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    pub fn cold_accesses(&self) -> u64 {
        self.cold_accesses
    }

    pub fn binner(&self) -> Binner {
        self.binner
    }

    pub fn cri(&self) -> &Histogram {
        &self.cri
    }

    pub fn pri(&self) -> &Histogram {
        &self.pri
    }

    pub fn reuse_distance(&self) -> Option<&Histogram> {
        self.distance.as_ref().map(|_| &self.reuse_distance)
    }

    pub fn joint(&self) -> &FxHashMap<JointKey, u64> {
        &self.joint
    }

    pub fn per_reference(&self) -> &[ReferenceStats] {
        &self.per_reference
    }

    /// True if the reuse-distance index ran out of room, which would make the
    /// distance histogram silently incomplete.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The measured CRI histogram as `(interval, portion of all accesses)`,
    /// ready for `denning::MissRatioCurve::new`.
    ///
    /// Portions are of *all* accesses, not just warm ones: the Denning
    /// recursion reads the missing mass as compulsory misses, so renormalizing
    /// here would report a program that never misses cold.
    pub fn denning_support(&self) -> Vec<(isize, f64)> {
        let total = self.total_accesses as f64;
        let mut support: Vec<(isize, f64)> = vec![(0, 0.0)];
        let mut points: Vec<(isize, f64)> = self
            .cri
            .iter()
            .map(|(bin, count)| {
                (
                    self.binner.representative(*bin).round() as isize,
                    *count as f64 / total,
                )
            })
            .collect();
        points.sort_by_key(|(interval, _)| *interval);
        for (interval, portion) in points {
            match support.last_mut() {
                Some((last, running)) if *last == interval => *running += portion,
                _ => support.push((interval, portion)),
            }
        }
        support
    }

    /// Exact fully-associative LRU miss ratio at `cache_size` blocks, from the
    /// measured reuse-distance histogram.
    ///
    /// An access hits exactly when fewer than `cache_size` distinct data were
    /// touched since it last ran, so this is ground truth rather than a model.
    /// Returns `None` when distance tracking was disabled or overflowed, since
    /// a partial histogram would understate the miss ratio.
    pub fn exact_miss_ratio_at(&self, cache_size: f64) -> Option<f64> {
        if self.distance.is_none() || self.overflowed {
            return None;
        }
        let misses: u64 = self
            .reuse_distance
            .iter()
            .filter(|(bin, _)| self.binner.representative(**bin) >= cache_size)
            .map(|(_, count)| *count)
            .sum();
        Some((misses + self.cold_accesses) as f64 / self.total_accesses as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::MemrefKey;

    fn datum(index: i64) -> DataId {
        DataId::new(MemrefKey(0), &[index], NonZero::new(1).expect("non-zero")).expect("fits")
    }

    fn access(index: i64) -> Emitted {
        Emitted {
            reference: 0,
            data: datum(index),
            is_write: false,
        }
    }

    fn sorted(histogram: &Histogram) -> Vec<(u32, u64)> {
        let mut items: Vec<_> = histogram.iter().map(|(k, v)| (*k, *v)).collect();
        items.sort_unstable();
        items
    }

    #[test]
    fn exact_bins_are_the_identity_below_the_threshold() {
        let binner = Binner::default();
        for value in [1u64, 7, 255] {
            assert_eq!(binner.bin(value), value as u32);
            assert_eq!(binner.representative(binner.bin(value)), value as f64);
        }
    }

    #[test]
    fn log_bins_are_monotone_and_bracket_their_representative() {
        let binner = Binner::default();
        let mut previous = binner.bin(255);
        for value in [256u64, 300, 512, 4096, 1 << 30] {
            let bin = binner.bin(value);
            assert!(bin >= previous, "bin {bin} went backwards at {value}");
            let representative = binner.representative(bin);
            assert!(
                representative > 0.0 && (representative / value as f64).abs() < 2.0,
                "value {value} -> {representative}"
            );
            previous = bin;
        }
    }

    #[test]
    fn sequential_trace_measures_reuse_interval_and_distance() {
        // "0 1 2 2 1 0": classic example. RI of the second 0 is 5, and its
        // reuse distance (distinct data in between) is 3.
        let mut sampler = Sampler::new(1, 1, Binner::default(), true);
        for index in [0, 1, 2, 2, 1, 0] {
            sampler.observe(0, access(index));
        }
        assert_eq!(sampler.total_accesses(), 6);
        assert_eq!(sampler.cold_accesses(), 3);
        assert_eq!(sorted(sampler.cri()), vec![(1, 1), (3, 1), (5, 1)]);
        let distances = sampler.reuse_distance().expect("tracked");
        assert_eq!(sorted(distances), vec![(0, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn single_thread_reuse_is_always_private_with_pri_equal_to_cri() {
        let mut sampler = Sampler::new(1, 1, Binner::default(), false);
        for index in [0, 1, 0] {
            sampler.observe(0, access(index));
        }
        let joint: Vec<_> = sampler
            .joint()
            .iter()
            .map(|(key, count)| (*key, *count))
            .collect();
        assert_eq!(joint.len(), 1);
        let (key, count) = joint[0];
        assert_eq!(count, 1);
        assert_eq!(key.sharing, Sharing::Private);
        assert_eq!(key.pri_bin, key.cri_bin);
    }

    #[test]
    fn cross_thread_reuse_is_shared_and_keeps_both_views() {
        // t0:d0, t1:d1, t1:d0, t0:d0. Both reuses of d0 cross a thread
        // boundary, and the last one is the interesting case: globally it is a
        // *shared* reuse at CRI 1, but in t0's own stream it is a private reuse
        // at PRI 1. Recording only one of the two views would lose that.
        let mut sampler = Sampler::new(2, 1, Binner::default(), false);
        sampler.observe(0, access(0));
        sampler.observe(1, access(1));
        sampler.observe(1, access(0));
        sampler.observe(0, access(0));

        let mut joint: Vec<_> = sampler
            .joint()
            .iter()
            .map(|(key, count)| (*key, *count))
            .collect();
        joint.sort_by_key(|(key, _)| *key);

        assert!(
            joint.iter().all(|(key, _)| key.sharing == Sharing::Shared),
            "every reuse of d0 followed an access by the other thread: {joint:?}"
        );
        assert_eq!(
            joint.iter().map(|(_, count)| *count).sum::<u64>(),
            2,
            "two reuses of d0"
        );

        // t1 had never touched d0, so its reuse carries no private view.
        let orphan = joint
            .iter()
            .find(|(key, _)| key.pri_bin == u32::MAX)
            .expect("t1's reuse has no private predecessor");
        assert_eq!(orphan.0.cri_bin, 2, "d0 was last touched two accesses ago");

        // t0 had, so its reuse carries both.
        let paired = joint
            .iter()
            .find(|(key, _)| key.pri_bin != u32::MAX)
            .expect("t0's reuse has a private predecessor");
        assert_eq!(paired.0.pri_bin, 1, "t0's previous access was its own last");
        assert_eq!(paired.0.cri_bin, 1, "globally it was the immediately prior access");
    }

    #[test]
    fn a_shared_reuse_with_no_private_predecessor_uses_the_sentinel() {
        let mut sampler = Sampler::new(2, 1, Binner::default(), false);
        sampler.observe(0, access(0));
        sampler.observe(1, access(0));
        let (key, _) = sampler.joint().iter().next().expect("one reuse");
        assert_eq!(key.sharing, Sharing::Shared);
        assert_eq!(key.pri_bin, u32::MAX, "thread 1 had never seen the datum");
    }

    #[test]
    fn compaction_preserves_reuse_distances() {
        // Enough distinct data and reuses to force several rebuilds.
        let mut sampler = Sampler::new(1, 1, Binner::default(), true);
        for _ in 0..40 {
            for index in 0..2000 {
                sampler.observe(0, access(index));
            }
        }
        assert!(!sampler.overflowed(), "timeline should have been rebuilt");
        let distances = sampler.reuse_distance().expect("tracked");
        // Every warm access sweeps the same 2000-element cycle, so every reuse
        // distance is exactly 1999.
        let observed = sorted(distances);
        assert_eq!(observed.len(), 1, "{observed:?}");
        assert_eq!(observed[0].1, 39 * 2000);
        assert!(
            (sampler.binner().representative(observed[0].0) - 1999.0).abs() < 1999.0 * 0.05,
            "bin {} represents {}",
            observed[0].0,
            sampler.binner().representative(observed[0].0)
        );
    }
}
