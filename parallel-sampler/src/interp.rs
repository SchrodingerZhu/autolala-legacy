//! A resumable interpreter over `raffine`'s affine loop tree.
//!
//! The tree walk mirrors `cachegrind-runner/src/main.rs::emit_tree`, but instead
//! of emitting C it yields one access at a time. Laziness matters: a 512-cubed
//! matmul is half a billion accesses, and the sampler runs `threads + 1` of
//! these streams concurrently, so nothing may be materialized.
//!
//! Rather than recursing, the cursor keeps an explicit frame stack, which is
//! what makes it suspendable between accesses.

use std::num::NonZero;

use anyhow::{Result, bail};
use raffine::tree::{Tree, ValID};
use rustc_hash::FxHashMap;

use crate::address::{DataId, MemrefKey};
use crate::affine_eval::{Env, eval_integer_set, eval_lower_bound, eval_map, eval_upper_bound};

/// One memory access produced by the interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emitted {
    /// Static access site, numbered in tree order. Reuse is attributed per
    /// reference so each Table-1 cell can be checked against the references the
    /// model assigned to it.
    pub reference: u32,
    pub data: DataId,
    pub is_write: bool,
}

/// How the parallel loop's iterations are handed to threads.
///
/// Only `schedule(static, chunk)` is modeled: iteration `k` (normalized to
/// `0..trip`) belongs to thread `(k / chunk) mod threads`. `Auto` reproduces
/// OpenMP's default `schedule(static)` contiguous blocks; `Fixed(1)` is the
/// round-robin the previous attempt hard-wired. There is no dynamic or guided
/// schedule and no work stealing, so thread ownership stays a pure function of
/// the iteration index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkSize {
    Auto,
    Fixed(NonZero<i64>),
}

impl ChunkSize {
    fn resolve(self, trip: i64, threads: i64) -> i64 {
        match self {
            Self::Fixed(chunk) => chunk.get(),
            // ceil(trip / threads), never zero even for an empty loop.
            Self::Auto => (trip + threads - 1).div_euclid(threads).max(1),
        }
    }
}

/// Assigns one thread's share of the parallel loop.
#[derive(Clone, Copy, Debug)]
pub struct Partition {
    /// Nesting depth of the parallel loop. `raffine` numbers induction
    /// variables by depth (see [`Env`]), so this is also the loop's `IVar` id.
    pub depth: usize,
    pub tid: u32,
    pub threads: NonZero<u32>,
    pub chunk: ChunkSize,
}

/// Static per-access-site information, shared by every thread's cursor.
///
/// Built once by walking the tree, so reference numbering and memref interning
/// are identical across threads and across runs.
#[derive(Debug, Default)]
pub struct ReferenceTable {
    sites: FxHashMap<usize, u32>,
    memrefs: FxHashMap<usize, MemrefKey>,
    names: Vec<String>,
    max_rank: usize,
}

impl ReferenceTable {
    pub fn build(tree: &Tree<'_>) -> Result<Self> {
        let mut table = Self::default();
        table.visit(tree)?;
        Ok(table)
    }

    fn visit(&mut self, tree: &Tree<'_>) -> Result<()> {
        match tree {
            Tree::For { body, .. } => self.visit(body),
            Tree::Block(stmts) => {
                for stmt in stmts.iter() {
                    self.visit(stmt)?;
                }
                Ok(())
            }
            Tree::If { then, r#else, .. } => {
                self.visit(then)?;
                if let Some(otherwise) = r#else {
                    self.visit(otherwise)?;
                }
                Ok(())
            }
            Tree::Access { memref, map, .. } => {
                let site = tree as *const Tree<'_> as usize;
                let next = self.sites.len() as u32;
                self.sites.insert(site, next);
                self.max_rank = self.max_rank.max(map.num_results());
                let name = match memref {
                    ValID::Memref(id) => format!("m{id}"),
                    ValID::Global(name) => name.to_string(),
                    other => bail!("access targets {other}, which is not a memref"),
                };
                // `insert` would *renumber* a memref seen a second time, and
                // renumbering is not harmless: two arrays can end up sharing a
                // key, which silently merges them into one datum and makes
                // every reuse statistic wrong. A stencil reading one array five
                // times and writing another once collapses exactly that way.
                let key = memref_key(memref);
                let next_key = self.memrefs.len() as u32;
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    self.memrefs.entry(key)
                {
                    slot.insert(MemrefKey(next_key));
                    self.names.push(name);
                }
                Ok(())
            }
        }
    }

    fn reference_of(&self, tree: &Tree<'_>) -> Result<u32> {
        match self.sites.get(&(tree as *const Tree<'_> as usize)) {
            Some(id) => Ok(*id),
            None => bail!("access site was not registered by the reference pre-pass"),
        }
    }

    fn memref_of(&self, memref: &ValID) -> Result<MemrefKey> {
        match self.memrefs.get(&memref_key(memref)) {
            Some(key) => Ok(*key),
            None => bail!("memref {memref} was not registered by the reference pre-pass"),
        }
    }

    /// Number of distinct static access sites.
    pub fn reference_count(&self) -> usize {
        self.sites.len()
    }

    /// Display name of each interned memref, indexed by [`MemrefKey`].
    pub fn memref_names(&self) -> &[String] {
        &self.names
    }

    /// Largest array rank seen, used to reject kernels the fixed-width
    /// [`DataId`] cannot represent before a run starts.
    pub fn max_rank(&self) -> usize {
        self.max_rank
    }
}

/// Stable identity for a memref across the whole tree.
///
/// `ValID::Global` interns its name through `ustr`, so the pointer is already a
/// unique key; local memrefs are numbered by `raffine`. The two spaces are kept
/// apart by tagging the global's pointer.
fn memref_key(memref: &ValID) -> usize {
    match memref {
        ValID::Memref(id) => *id << 1,
        ValID::Global(name) => (name.as_ptr() as usize) | 1,
        _ => usize::MAX,
    }
}

enum Frame<'a> {
    Block {
        stmts: &'a [&'a Tree<'a>],
        next: usize,
    },
    For {
        body: &'a Tree<'a>,
        ivar: usize,
        current: i64,
        upper: i64,
        step: i64,
    },
    /// The parallel loop, walked over only the iterations this thread owns.
    OwnedFor {
        body: &'a Tree<'a>,
        ivar: usize,
        lower: i64,
        step: i64,
        trip: i64,
        chunk: i64,
        threads: i64,
        chunk_index: i64,
        offset: i64,
    },
}

/// A suspendable walk over one thread's share of the loop nest.
pub struct Cursor<'a, 'r> {
    stack: Vec<Frame<'a>>,
    env: Env,
    scratch: Vec<i64>,
    table: &'r ReferenceTable,
    block_size: NonZero<usize>,
    partition: Option<Partition>,
    root: Option<&'a Tree<'a>>,
}

impl<'a, 'r> Cursor<'a, 'r> {
    /// Creates a cursor over the whole nest (`partition == None`) or over one
    /// thread's iterations of the parallel loop.
    pub fn new(
        tree: &'a Tree<'a>,
        table: &'r ReferenceTable,
        symbols: Vec<i64>,
        block_size: NonZero<usize>,
        partition: Option<Partition>,
    ) -> Self {
        Self {
            stack: Vec::new(),
            env: Env::new(symbols),
            scratch: Vec::new(),
            table,
            block_size,
            partition,
            root: Some(tree),
        }
    }

    /// Yields the next access, or `None` once this thread's stream is drained.
    pub fn next_access(&mut self) -> Result<Option<Emitted>> {
        if let Some(root) = self.root.take()
            && let Some(access) = self.enter(root)?
        {
            return Ok(Some(access));
        }
        loop {
            let Self { stack, env, .. } = self;
            let Some(frame) = stack.last_mut() else {
                return Ok(None);
            };
            let node = match frame {
                Frame::Block { stmts, next } => {
                    if *next >= stmts.len() {
                        stack.pop();
                        continue;
                    }
                    let node = stmts[*next];
                    *next += 1;
                    node
                }
                Frame::For {
                    body,
                    ivar,
                    current,
                    upper,
                    step,
                } => {
                    if *current >= *upper {
                        stack.pop();
                        continue;
                    }
                    env.set_ivar(*ivar, *current);
                    *current += *step;
                    *body
                }
                Frame::OwnedFor {
                    body,
                    ivar,
                    lower,
                    step,
                    trip,
                    chunk,
                    threads,
                    chunk_index,
                    offset,
                } => {
                    let Some(index) =
                        next_owned_index(trip, *chunk, *threads, chunk_index, offset)
                    else {
                        stack.pop();
                        continue;
                    };
                    env.set_ivar(*ivar, *lower + index * *step);
                    *body
                }
            };
            if let Some(access) = self.enter(node)? {
                return Ok(Some(access));
            }
        }
    }

    /// Descends into a node: pushes a frame for a construct, or produces the
    /// access for a leaf.
    fn enter(&mut self, node: &'a Tree<'a>) -> Result<Option<Emitted>> {
        match node {
            Tree::Block(stmts) => {
                if !stmts.is_empty() {
                    self.stack.push(Frame::Block { stmts, next: 0 });
                }
                Ok(None)
            }
            Tree::For {
                lower_bound,
                upper_bound,
                lower_bound_operands,
                upper_bound_operands,
                step,
                ivar,
                body,
            } => {
                let ValID::IVar(ivar) = *ivar else {
                    bail!("affine.for is driven by {ivar}, which is not an induction variable");
                };
                let step = *step as i64;
                if step <= 0 {
                    bail!("affine.for step must be positive, found {step}");
                }
                let lower = eval_lower_bound(lower_bound, lower_bound_operands, &self.env)?;
                let upper = eval_upper_bound(upper_bound, upper_bound_operands, &self.env)?;
                match self.partition {
                    Some(partition) if partition.depth == ivar => {
                        let threads = i64::from(partition.threads.get());
                        let trip = (upper - lower + step - 1).div_euclid(step).max(0);
                        let chunk = partition.chunk.resolve(trip, threads);
                        self.stack.push(Frame::OwnedFor {
                            body,
                            ivar,
                            lower,
                            step,
                            trip,
                            chunk,
                            threads,
                            chunk_index: i64::from(partition.tid),
                            offset: 0,
                        });
                    }
                    _ => self.stack.push(Frame::For {
                        body,
                        ivar,
                        current: lower,
                        upper,
                        step,
                    }),
                }
                Ok(None)
            }
            Tree::If {
                condition,
                operands,
                then,
                r#else,
            } => {
                let taken = eval_integer_set(condition, operands, &self.env)?;
                match (taken, r#else) {
                    (true, _) => self.enter(then),
                    (false, Some(otherwise)) => self.enter(otherwise),
                    (false, None) => Ok(None),
                }
            }
            Tree::Access {
                memref,
                map,
                operands,
                is_write,
            } => {
                eval_map(map, operands, &self.env, &mut self.scratch)?;
                let key = self.table.memref_of(memref)?;
                let Some(data) = DataId::new(key, &self.scratch, self.block_size) else {
                    bail!(
                        "access has rank {}, above the sampler's supported maximum of {}",
                        self.scratch.len(),
                        crate::address::MAX_ARRAY_DIMS
                    );
                };
                Ok(Some(Emitted {
                    reference: self.table.reference_of(node)?,
                    data,
                    is_write: *is_write,
                }))
            }
        }
    }
}

/// Advances a `schedule(static, chunk)` walk to this thread's next normalized
/// iteration index, stepping over the chunks owned by other threads.
fn next_owned_index(
    trip: &i64,
    chunk: i64,
    threads: i64,
    chunk_index: &mut i64,
    offset: &mut i64,
) -> Option<i64> {
    loop {
        let start = chunk_index.checked_mul(chunk)?;
        if start >= *trip {
            return None;
        }
        let end = (start + chunk).min(*trip);
        let index = start + *offset;
        if index >= end {
            *chunk_index += threads;
            *offset = 0;
            continue;
        }
        *offset += 1;
        return Some(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(trip: i64, chunk: i64, threads: i64, tid: i64) -> Vec<i64> {
        let mut chunk_index = tid;
        let mut offset = 0;
        let mut out = Vec::new();
        while let Some(index) = next_owned_index(&trip, chunk, threads, &mut chunk_index, &mut offset)
        {
            out.push(index);
        }
        out
    }

    #[test]
    fn chunk_one_is_round_robin() {
        assert_eq!(owned(10, 1, 4, 0), vec![0, 4, 8]);
        assert_eq!(owned(10, 1, 4, 1), vec![1, 5, 9]);
        assert_eq!(owned(10, 1, 4, 3), vec![3, 7]);
    }

    #[test]
    fn auto_chunk_gives_contiguous_blocks() {
        let chunk = ChunkSize::Auto.resolve(10, 4);
        assert_eq!(chunk, 3);
        assert_eq!(owned(10, chunk, 4, 0), vec![0, 1, 2]);
        assert_eq!(owned(10, chunk, 4, 1), vec![3, 4, 5]);
        assert_eq!(owned(10, chunk, 4, 3), vec![9]);
    }

    #[test]
    fn every_iteration_is_owned_exactly_once() {
        for &chunk in &[1i64, 2, 3, 5, 16] {
            for &threads in &[2i64, 3, 8] {
                let mut all: Vec<i64> = (0..threads)
                    .flat_map(|tid| owned(37, chunk, threads, tid))
                    .collect();
                all.sort_unstable();
                assert_eq!(
                    all,
                    (0..37).collect::<Vec<_>>(),
                    "chunk={chunk} threads={threads}"
                );
            }
        }
    }

    #[test]
    fn threads_beyond_the_work_get_nothing() {
        assert!(owned(2, ChunkSize::Auto.resolve(2, 8), 8, 5).is_empty());
    }

    #[test]
    fn repeated_memrefs_keep_one_stable_key_each() {
        // Regression: a memref revisited during the pre-pass used to be
        // renumbered to the current table length, so an array read five times
        // and an array read once both ended up as key 1 and merged.
        let mut table = ReferenceTable::default();
        let a = ValID::Memref(0);
        let b = ValID::Memref(1);
        for memref in [&a, &a, &a, &a, &a, &b] {
            let key = memref_key(memref);
            let next_key = table.memrefs.len() as u32;
            if let std::collections::hash_map::Entry::Vacant(slot) = table.memrefs.entry(key) {
                slot.insert(MemrefKey(next_key));
                table.names.push(memref.to_string());
            }
        }
        assert_eq!(table.memrefs.len(), 2);
        assert_eq!(table.names.len(), 2);
        let keys: std::collections::BTreeSet<_> =
            table.memrefs.values().map(|key| key.0).collect();
        assert_eq!(
            keys,
            [0, 1].into_iter().collect(),
            "two arrays must occupy two distinct keys"
        );
        // Names are indexed by key, so the two must stay in step.
        assert_eq!(table.memref_of(&a).expect("interned"), MemrefKey(0));
        assert_eq!(table.memref_of(&b).expect("interned"), MemrefKey(1));
    }

    #[test]
    fn empty_loop_yields_no_iterations() {
        assert!(owned(0, ChunkSize::Auto.resolve(0, 4), 4, 0).is_empty());
    }
}
