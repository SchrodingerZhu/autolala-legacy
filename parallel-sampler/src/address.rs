//! Data identity for the sampler.
//!
//! This must agree, element for element, with the polyhedral access relation
//! built by `analyzer/src/isl.rs::get_access_map_impl`. That function encodes an
//! access as the range tuple
//!
//! ```text
//! (memref_id, 0, .., 0, idx_0, .., idx_{n-2}, floor(idx_{n-1} / block_size))
//! ```
//!
//! i.e. the leading zeros pad every access out to `max_array_dim` coordinates,
//! and the *innermost* array dimension alone is divided by the block size. If
//! the sampler linearized addresses instead (the more obvious choice), the two
//! sides would disagree on every kernel whose rows are not a multiple of the
//! block size, and no amount of model fixing would make them line up.

use std::num::NonZero;

/// Maximum array rank the sampler supports. Padding to a fixed width keeps
/// [`DataId`] `Copy` and hashable without an allocation per access.
pub const MAX_ARRAY_DIMS: usize = 6;

/// Identifies which memref an access targets.
///
/// `raffine` hands back either a numbered local memref or a named global; both
/// are interned to a dense `u32` by the interpreter so the hot path compares
/// integers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MemrefKey(pub u32);

/// A cache-block-granularity datum: the unit whose reuse we track.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DataId {
    pub memref: MemrefKey,
    /// Right-aligned coordinates; unused leading slots are zero, mirroring the
    /// zero padding in `get_access_map_impl`.
    pub coords: [i64; MAX_ARRAY_DIMS],
}

impl DataId {
    /// Applies the block-size quotient to the innermost dimension and packs the
    /// coordinates, exactly as the polyhedral encoding does.
    ///
    /// Returns `None` if the access has more dimensions than [`MAX_ARRAY_DIMS`].
    pub fn new(memref: MemrefKey, indices: &[i64], block_size: NonZero<usize>) -> Option<Self> {
        if indices.len() > MAX_ARRAY_DIMS {
            return None;
        }
        let block_size = block_size.get() as i64;
        let mut coords = [0i64; MAX_ARRAY_DIMS];
        let offset = MAX_ARRAY_DIMS - indices.len();
        for (slot, &index) in coords[offset..].iter_mut().zip(indices) {
            *slot = index;
        }
        if let Some(last) = coords.last_mut()
            && block_size > 1
        {
            // `div_euclid` rather than `/`: isl's floor is a true floor, and a
            // negative subscript (a stencil halo, say) would otherwise round
            // toward zero and merge two distinct blocks.
            *last = last.div_euclid(block_size);
        }
        Some(Self { memref, coords })
    }

    /// Cache set this datum maps to under `num_sets` sets.
    ///
    /// Mirrors the `set_tag` dimension of `get_access_map_impl`, which tags on
    /// `(idx_{n-1} / block_size) mod num_sets` — the set index is taken from the
    /// blocked innermost coordinate only.
    pub fn set_index(&self, num_sets: NonZero<usize>) -> usize {
        let num_sets = num_sets.get() as i64;
        self.coords[MAX_ARRAY_DIMS - 1].rem_euclid(num_sets) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: usize) -> NonZero<usize> {
        NonZero::new(value).expect("non-zero")
    }

    #[test]
    fn block_size_divides_only_the_innermost_dimension() {
        let a = DataId::new(MemrefKey(0), &[3, 8], nz(4)).expect("fits");
        let b = DataId::new(MemrefKey(0), &[3, 11], nz(4)).expect("fits");
        let c = DataId::new(MemrefKey(0), &[4, 8], nz(4)).expect("fits");
        assert_eq!(a, b, "8 and 11 share block 2 of the same row");
        assert_ne!(a, c, "the outer dimension is never blocked");
    }

    #[test]
    fn distinct_memrefs_never_alias() {
        let a = DataId::new(MemrefKey(0), &[1, 1], nz(1)).expect("fits");
        let b = DataId::new(MemrefKey(1), &[1, 1], nz(1)).expect("fits");
        assert_ne!(a, b);
    }

    #[test]
    fn padding_keeps_ranks_apart() {
        let vector = DataId::new(MemrefKey(0), &[5], nz(1)).expect("fits");
        let matrix = DataId::new(MemrefKey(0), &[0, 5], nz(1)).expect("fits");
        // Leading zero padding makes A[5] and A[0][5] the same tuple, which is
        // exactly what the polyhedral encoding does; a kernel mixing ranks on
        // one memref would need `max_array_dim` disambiguation there too.
        assert_eq!(vector, matrix);
    }

    #[test]
    fn negative_subscripts_floor_toward_negative_infinity() {
        let a = DataId::new(MemrefKey(0), &[0, -1], nz(4)).expect("fits");
        let b = DataId::new(MemrefKey(0), &[0, -4], nz(4)).expect("fits");
        let c = DataId::new(MemrefKey(0), &[0, 0], nz(4)).expect("fits");
        assert_eq!(a.coords[MAX_ARRAY_DIMS - 1], -1);
        assert_eq!(b.coords[MAX_ARRAY_DIMS - 1], -1);
        assert_ne!(a, c);
    }

    #[test]
    fn rank_beyond_the_cap_is_rejected() {
        assert!(DataId::new(MemrefKey(0), &[0; MAX_ARRAY_DIMS + 1], nz(1)).is_none());
    }

    #[test]
    fn set_index_uses_the_blocked_innermost_coordinate() {
        let datum = DataId::new(MemrefKey(0), &[7, 20], nz(4)).expect("fits");
        // 20 / 4 = 5, and 5 mod 4 = 1.
        assert_eq!(datum.set_index(nz(4)), 1);
    }
}
