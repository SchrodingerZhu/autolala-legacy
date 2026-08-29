use anyhow::{Result, anyhow};
use indicatif::ParallelProgressIterator;
use melior::ir::attribute::{StringAttribute, TypeAttribute};
use melior::ir::r#type::{DimSize, MemRefType};
use melior::ir::{
    BlockLike, Module, OperationRef, RegionLike, ShapedTypeLike, ValueLike,
    operation::OperationLike,
};
use palc::Parser;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use raffine::affine::{AffineExpr, AffineMap};
use raffine::tree::{Tree, ValID};
use raffine::{Context, DominanceInfo};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use sysinfo::Components;

use tracing::{debug, info, trace, warn};

struct CoolDown {
    flag: AtomicBool,
    finished: AtomicBool,
    components: Mutex<Components>,
}

impl CoolDown {
    fn wait(&self) {
        while !self.flag.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
    fn finish(&self) {
        self.finished
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
    fn monitor_loop(&self) {
        let mut components = self.components.lock().unwrap();
        'waiting: loop {
            if self.finished.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            components.iter_mut().for_each(|c| c.refresh());
            let average = components
                .iter()
                .filter(|c| c.label().contains("Tccd") || c.label().contains("Tctl"))
                .filter_map(|c| c.temperature())
                .collect::<Vec<_>>();
            let average = average.iter().copied().sum::<f32>() / average.len() as f32;
            if average >= 70.0 {
                warn!(
                    "High temperature detected: {:.2}°C, cooling down...",
                    average
                );
                self.flag.store(false, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue 'waiting;
            }
            self.flag.store(true, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
}

struct CProgramEmitter<W: Write> {
    writer: BufWriter<W>,
    indent: usize,
    hashed: bool,
}

impl<W: Write> CProgramEmitter<W> {
    fn new(writer: W, hashed: bool) -> Self {
        CProgramEmitter {
            writer: BufWriter::new(writer),
            indent: 1,
            hashed,
        }
    }

    fn emit_indent(&mut self) -> Result<()> {
        for _ in 0..self.indent {
            write!(self.writer, "\t")?;
        }
        Ok(())
    }

    fn emit(mut self, tree: &Tree) -> Result<()> {
        writeln!(self.writer, "extern \"C\" void _start() {{")?;
        self.emit_tree(tree)?;
        self.emit_indent()?;

        // x86_64: exit(0) => eax=60 (sys_exit), edi=0, syscall
        #[cfg(target_arch = "x86_64")]
        {
            writeln!(
                self.writer,
                r#"asm volatile("xor %edi, %edi\n\tmov $60, %eax\n\tsyscall");"#
            )?;
        }

        // aarch64 (Linux): exit(0) => x8=93 (sys_exit), x0=0, svc #0
        #[cfg(target_arch = "aarch64")]
        {
            writeln!(
                self.writer,
                r#"asm volatile("mov x0, #0\n\tmov x8, #93\n\tsvc #0");"#
            )?;
        }

        Ok(writeln!(self.writer, "}}")?)
    }

    fn emit_tree(&mut self, tree: &Tree) -> Result<()> {
        match tree {
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
                    return Err(anyhow!("expected ivar in for loop, found: {ivar}"));
                };
                self.emit_indent()?;
                write!(self.writer, "for (int ivar_{ivar} = ")?;
                self.emit_affine_map(lower_bound, lower_bound_operands)?;
                write!(self.writer, "; ivar_{ivar} < ")?;
                self.emit_affine_map(upper_bound, upper_bound_operands)?;
                write!(self.writer, "; ivar_{ivar} += {step}",)?;
                writeln!(self.writer, ") {{")?;
                self.indent += 1;
                self.emit_tree(body)?;
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
            }
            Tree::Block(trees) => {
                self.emit_indent()?;
                writeln!(self.writer, "{{")?;
                self.indent += 1;
                for tree in trees.iter() {
                    self.emit_tree(tree)?;
                    writeln!(self.writer)?;
                }
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
            }
            Tree::Access {
                memref,
                map,
                operands,
                ..
            } => {
                self.emit_indent()?;
                self.emit_access(*memref, operands, *map)?;
            }
            Tree::If {
                condition,
                operands,
                then,
                r#else,
            } => {
                self.emit_indent()?;
                write!(self.writer, "if ")?;
                self.emit_integer_set(condition, operands)?;
                writeln!(self.writer, " {{")?;
                self.indent += 1;
                self.emit_tree(then)?;
                self.indent -= 1;
                self.emit_indent()?;
                writeln!(self.writer, "}}")?;
                if let Some(else_block) = r#else {
                    self.emit_indent()?;
                    writeln!(self.writer, "else {{")?;
                    self.indent += 1;
                    self.emit_tree(else_block)?;
                    self.indent -= 1;
                    self.emit_indent()?;
                    writeln!(self.writer, "}}")?;
                }
            }
        }
        Ok(())
    }

    fn emit_integer_set(
        &mut self,
        set: &raffine::affine::IntegerSet,
        operands: &[ValID],
    ) -> Result<()> {
        write!(self.writer, "(")?;
        for i in 0..set.num_constraints() {
            if i > 0 {
                write!(self.writer, " && ")?;
            }
            let expr = set.get_constraint(i as isize);
            write!(self.writer, "((")?;
            let operands = operands
                .iter()
                .map(|&operand| match operand {
                    ValID::IVar(x) | ValID::Symbol(x) => x,
                    ValID::Global(_) | ValID::Memref(_) => {
                        unreachable!("global/memref cannot be used as operand")
                    }
                })
                .collect::<Vec<_>>();
            self.emit_affine_expr(expr, &operands)?;
            if set.is_constraint_equal(i as isize) {
                write!(self.writer, ") == 0)")?;
            } else {
                write!(self.writer, ") >= 0)")?; // need to confirm this
            }
        }
        write!(self.writer, ")")?;
        Ok(())
    }

    fn emit_affine_map(
        &mut self,
        map: &raffine::affine::AffineMap,
        operands: &[ValID],
    ) -> Result<()> {
        let operands = operands
            .iter()
            .map(|&operand| match operand {
                ValID::IVar(x) | ValID::Symbol(x) => x,
                ValID::Global(_) | ValID::Memref(_) => {
                    unreachable!("global/memref cannot be used as operand")
                }
            })
            .collect::<Vec<_>>();
        for i in 0..map.num_results() {
            let expr = map
                .get_result_expr(i as isize)
                .ok_or_else(|| anyhow!("invalid affine map: result {i} does not exist in map"))?;
            if i > 0 {
                write!(self.writer, "][")?;
            }
            self.emit_affine_expr(expr, &operands)?;
        }
        Ok(())
    }

    fn emit_access(&mut self, array: ValID, operands: &[ValID], map: AffineMap) -> Result<()> {
        let name = array_c_name(array)
            .ok_or_else(|| anyhow!("expected memref/global in access, found: {array}"))?;
        if self.hashed {
            // Route the store through the block permutation: the array itself
            // is only used as a constant-folded address calculator.
            write!(
                self.writer,
                "{{ __access(__OFF_{name} + (unsigned long)((const volatile char *)&{name}["
            )?;
            self.emit_affine_map(&map, operands)?;
            write!(
                self.writer,
                r#"] - (const volatile char *)&{name})); __asm__ __volatile__ ("" ::: "memory"); }}"#
            )?;
            return Ok(());
        }
        write!(self.writer, "{{ {name}[")?;
        self.emit_affine_map(&map, operands)?;
        write!(
            self.writer,
            r#"] = 0; __asm__ __volatile__ ("" ::: "memory"); }}"#
        )?;
        Ok(())
    }

    fn emit_affine_expr(&mut self, expr: AffineExpr, operands: &[usize]) -> Result<()> {
        match expr.get_kind() {
            raffine::affine::AffineExprKind::Add
            | raffine::affine::AffineExprKind::FloorDiv
            | raffine::affine::AffineExprKind::Mul
            | raffine::affine::AffineExprKind::Mod => {
                let operator = match expr.get_kind() {
                    raffine::affine::AffineExprKind::Add => "+",
                    raffine::affine::AffineExprKind::FloorDiv => "/",
                    raffine::affine::AffineExprKind::Mul => "*",
                    raffine::affine::AffineExprKind::Mod => "%",
                    _ => unreachable!(),
                };
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("addition should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("addition should have rhs"))?;
                write!(self.writer, "(")?;
                self.emit_affine_expr(lhs, operands)?;
                write!(self.writer, " {operator}")?;
                self.emit_affine_expr(rhs, operands)?;
                write!(self.writer, ")")?;
            }

            raffine::affine::AffineExprKind::Dim | raffine::affine::AffineExprKind::Symbol => {
                let operand = expr
                    .get_position()
                    .ok_or_else(|| anyhow!("dimension expression should have position"))?
                    as usize;
                let target = *operands
                    .get(operand)
                    .ok_or_else(|| anyhow!("invalid operand index"))?;
                let prefix = if matches!(expr.get_kind(), raffine::affine::AffineExprKind::Symbol) {
                    "SYM"
                } else {
                    "ivar"
                };
                write!(self.writer, "{prefix}_{target}",)?;
            }
            raffine::affine::AffineExprKind::CeilDiv => {
                let lhs = expr
                    .get_lhs()
                    .ok_or_else(|| anyhow!("ceil division should have lhs"))?;
                let rhs = expr
                    .get_rhs()
                    .ok_or_else(|| anyhow!("ceil division should have rhs"))?;
                let decomposed = (lhs + lhs % rhs) / rhs;
                self.emit_affine_expr(decomposed, operands)?;
            }
            raffine::affine::AffineExprKind::Constant => {
                let value = expr
                    .get_value()
                    .ok_or_else(|| anyhow!("constant expression should have value"))?;
                if value < 0 {
                    write!(self.writer, "(")?;
                }
                write!(self.writer, "{value}")?;
                if value < 0 {
                    write!(self.writer, ")")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Parser)]
struct Cli {
    #[arg(long, short)]
    input: PathBuf,

    /// name of the target function to extract
    /// if not specified, the program will try to find first function
    #[arg(short = 'f', long)]
    target_function: Option<String>,

    // /// target affine loop attribute
    // /// if not specified, the program will try to find first affine loop in the function
    // #[arg(short = 'l', long)]
    // target_affine_loop: Option<String>,
    /// valgrind path
    #[arg(long, default_value = "valgrind")]
    valgrind_path: String,

    /// block size (D1)
    #[arg(long, default_value = "64", short = 'B')]
    d1_block_size: usize,

    /// associativity of the cache
    /// if not specified, the program will assume fully associative
    #[arg(long, short = 'A')]
    d1_associativity: Option<usize>,

    /// number of blocks upper bound
    #[arg(long, short = 'C')]
    d1_cache_size: usize,

    /// block size (LL)
    #[arg(long, default_value = "64", short = 'b')]
    ll_block_size: usize,

    /// associativity of the second level cache
    #[arg(long, short = 'a')]
    ll_associativity: usize,

    /// cache size of the second level cache
    #[arg(long, short = 'c')]
    ll_cache_size: usize,

    /// Do batched run from block size to cache size, stepping by block size
    #[arg(long)]
    batched: bool,

    /// Permute block addresses in the emitted program so cachegrind observes
    /// a uniform (hashed) set mapping instead of the biased modulo mapping
    #[arg(long)]
    hash: bool,

    /// Database file to store the results
    #[arg(long, short = 'd', default_value = "/tmp/cachegrind.db")]
    database: PathBuf,
}

#[derive(Debug, Clone)]
struct Record {
    program: String,
    d1_cache_size: usize,
    d1_associativity: usize,
    d1_block_size: usize,
    ll_associativity: usize,
    ll_cache_size: usize,
    ll_block_size: usize,
    d1_miss_count: usize,
    ll_miss_count: usize,
    total_access: usize,
    process_time: usize,
    hashed: bool,
}

fn extract_target<'ctx>(
    module: &'ctx Module<'ctx>,
    options: &Cli,
    context: &'ctx Context,
    dom: &'ctx DominanceInfo<'ctx>,
) -> anyhow::Result<&'ctx Tree<'ctx>> {
    let body = module.body();
    fn locate_function<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        options: &'_ Cli,
        conti: F,
    ) -> anyhow::Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            return Err(anyhow!("No operation found"));
        };
        if op.name().as_string_ref().as_str()? == "func.func" {
            if let Some(name) = options.target_function.as_deref() {
                let sym_name = op.attribute("sym_name")?;
                debug!("Checking function: {}", sym_name);
                if sym_name.to_string().trim_matches('"') == name {
                    debug!("Found target function: {}", name);
                    return conti(op);
                }
            } else {
                return conti(op);
            }
        }
        locate_function(op.next_in_block(), options, conti)
    }
    // fn locate_loop<'a, 'b, F>(
    //     cursor: Option<OperationRef<'a, 'b>>,
    //     options: &'_ Cli,
    //     conti: F,
    // ) -> anyhow::Result<&'a Tree<'a>>
    // where
    //     F: for<'c> FnOnce(OperationRef<'a, 'c>) -> anyhow::Result<&'a Tree<'a>>,
    // {
    //     let Some(op) = cursor else {
    //         return Err(anyhow!("No operation found"));
    //     };
    //     if op.name().as_string_ref().as_str()? == "affine.for" {
    //         if let Some(name) = options.target_affine_loop.as_deref() {
    //             if op.has_attribute(name) {
    //                 debug!("Found target affine loop: {}", name);
    //                 return conti(op);
    //             }
    //         } else {
    //             return conti(op);
    //         }
    //     }
    //     locate_loop(op.next_in_block(), options, conti)
    // }

    let cursor = body.first_operation();
    locate_function(cursor, options, move |func| {
        Ok(context.build_func_tree(func, dom, false)?)
    })
}

impl Record {
    fn insert(&self, database: &Pool<SqliteConnectionManager>) {
        while pool_get_retry(database)
            .execute(
                r#"INSERT INTO records (
                    program,
                    d1_block_size,
                    d1_associativity,
                    d1_cache_size,
                    ll_block_size,
                    ll_associativity,
                    ll_cache_size,
                    d1_miss_count,
                    ll_miss_count,
                    total_access,
                    process_time,
                    hashed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(program, d1_block_size, d1_associativity, d1_cache_size, ll_block_size, ll_associativity, ll_cache_size, hashed) DO UPDATE SET
                    d1_miss_count = excluded.d1_miss_count,
                    ll_miss_count = excluded.ll_miss_count,
                    total_access = excluded.total_access,
                    process_time = excluded.process_time"#,
                rusqlite::params![
                    self.program,
                    self.d1_block_size,
                    self.d1_associativity,
                    self.d1_cache_size,
                    self.ll_block_size,
                    self.ll_associativity,
                    self.ll_cache_size,
                    self.d1_miss_count,
                    self.ll_miss_count,
                    self.total_access,
                    self.process_time,
                    self.hashed,
                ],
            )
            .is_err()
        {}
    }
}

fn pool_get_retry(
    pool: &Pool<SqliteConnectionManager>,
) -> r2d2::PooledConnection<SqliteConnectionManager> {
    loop {
        match pool.get() {
            Ok(conn) => return conn,
            Err(e) => {
                debug!("Failed to get connection from pool: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

struct GlobalArrayDeclaration {
    name: String,
    size: Box<[usize]>,
}

fn array_c_name(array: ValID) -> Option<String> {
    match array {
        ValID::Memref(id) => Some(format!("ARRAY_{id}")),
        ValID::Global(name) => Some(name.to_string()),
        _ => None,
    }
}

fn collect_accessed_arrays(tree: &Tree, out: &mut Vec<String>) {
    match tree {
        Tree::For { body, .. } => collect_accessed_arrays(body, out),
        Tree::Block(children) => {
            for child in children.iter() {
                collect_accessed_arrays(child, out);
            }
        }
        Tree::If { then, r#else, .. } => {
            collect_accessed_arrays(then, out);
            if let Some(r#else) = r#else {
                collect_accessed_arrays(r#else, out);
            }
        }
        Tree::Access { memref, .. } => {
            if let Some(name) = array_c_name(*memref)
                && !out.contains(&name)
            {
                out.push(name);
            }
        }
    }
}

/// Parse the module's `simulation.dims` attribute:
/// `"ARRAY_0:200x220;ARRAY_1:200x240"` — a `;`-separated list of
/// `NAME:D0xD1x...` entries describing arrays of `double`.
fn parse_simulation_dims(raw: &str) -> Result<Vec<GlobalArrayDeclaration>> {
    let mut out = Vec::new();
    for entry in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, dims) = entry.split_once(':').ok_or_else(|| {
            anyhow!("simulation.dims entry `{entry}` is not of the form NAME:D0xD1x...")
        })?;
        let size = dims
            .split('x')
            .map(|d| d.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("simulation.dims entry `{entry}` has a non-numeric dimension"))?;
        if size.is_empty() || size.iter().any(|&d| d == 0) {
            return Err(anyhow!("simulation.dims entry `{entry}` has an empty dimension"));
        }
        out.push(GlobalArrayDeclaration {
            name: name.trim().to_string(),
            size: size.into_boxed_slice(),
        });
    }
    Ok(out)
}

/// Emit the `--hash` support code: per-array logical byte offsets, one
/// block-aligned arena, and a bijective permutation of block numbers so
/// every power-of-two set count sees a uniform set mapping. Block identity
/// is preserved exactly (distinct logical blocks map to distinct arena
/// blocks), so fully-associative behavior is unchanged.
fn emit_hash_preamble<W: Write>(
    w: &mut W,
    declared: &[GlobalArrayDeclaration],
    accessed: &[String],
    block_size: usize,
) -> Result<()> {
    for name in accessed {
        if !declared.iter().any(|a| a.name == *name) {
            return Err(anyhow!(
                "--hash needs the size of every accessed array, but `{name}` is not declared \
                 via memref.global, memref.alloc, or the module's `simulation.dims` attribute"
            ));
        }
    }
    let block = block_size as u64;
    let log2b = block.trailing_zeros();
    let mut offsets = Vec::new();
    let mut next = 0u64;
    for array in declared {
        let bytes = array.size.iter().product::<usize>() as u64 * 8;
        offsets.push((array.name.as_str(), next));
        next += bytes.div_ceil(block) * block;
    }
    let total_blocks = (next >> log2b).max(1);
    // k-bit domain of the permutation; low bits are the (hashed) set index.
    let k = total_blocks.next_power_of_two().trailing_zeros();
    let mask = (1u64 << k) - 1;
    // `x ^= x >> 0` is the zero map, not a bijection — clamp shifts to >= 1.
    let s1 = (k / 2).max(1);
    let s2 = (k / 3).max(1);
    let arena = (mask + 1) << log2b;
    writeln!(w, "\n// --hash: uniform set-index permutation over {total_blocks} logical blocks")?;
    for (name, off) in &offsets {
        writeln!(w, "static const unsigned long __OFF_{name} = {off}UL;")?;
    }
    writeln!(w, "alignas(4096) static volatile unsigned char __ARENA[{arena}UL];")?;
    // Force-inlined pure register arithmetic: a call would push a return
    // address and any spill would touch the stack, and cachegrind counts
    // both as data references. The chain below keeps <= 3 values live.
    writeln!(
        w,
        "static inline __attribute__((always_inline)) unsigned long __perm(unsigned long x) {{"
    )?;
    writeln!(w, "\tx ^= x >> {s1}; x = (x * 0x9E3779B97F4A7C15UL) & {mask}UL;")?;
    writeln!(w, "\tx ^= x >> {s2}; x = (x * 0xBF58476D1CE4E5B9UL) & {mask}UL;")?;
    writeln!(w, "\tx ^= x >> {s1}; return x;")?;
    writeln!(w, "}}")?;
    writeln!(
        w,
        "static inline __attribute__((always_inline)) void __access(unsigned long a) {{"
    )?;
    writeln!(
        w,
        "\t__ARENA[(__perm(a >> {log2b}) << {log2b}) | (a & {offmask}UL)] = 0;",
        offmask = block - 1
    )?;
    writeln!(w, "}}")?;
    Ok(())
}

fn extract_global_arrays(module: &Module) -> anyhow::Result<Vec<GlobalArrayDeclaration>> {
    let mut arrays = vec![];
    let body = module.body();
    let op = body.first_operation();
    fn collect_arrays(
        operation: OperationRef,
        arrays: &mut Vec<GlobalArrayDeclaration>,
    ) -> anyhow::Result<()> {
        if operation.name().as_string_ref().as_str()? == "memref.global" {
            let sym_name = operation.attribute("sym_name")?;
            let sym_name = StringAttribute::try_from(sym_name)?
                .to_string()
                .trim_matches('"')
                .to_string();
            let type_attr = operation.attribute("type")?;
            let type_attr = TypeAttribute::try_from(type_attr)?;
            let memref_type = type_attr.value();
            let memref_type = MemRefType::try_from(memref_type)?;
            let rank = memref_type.rank();
            let mut shape = Vec::with_capacity(rank);
            for dim in 0..rank {
                let DimSize::Static(size) = memref_type.dim_size(dim)? else {
                    return Err(anyhow!("dynamic dimension size is not supported"));
                };
                shape.push(size as usize);
            }
            arrays.push(GlobalArrayDeclaration {
                name: sym_name,
                size: shape.into_boxed_slice(),
            });
        }
        if let Some(next_op) = operation.next_in_block() {
            collect_arrays(next_op, arrays)?;
        }
        Ok(())
    }
    if let Some(op) = op {
        collect_arrays(op, &mut arrays)?;
    }
    Ok(arrays)
}
fn extract_func_alloc_arrays(module: &Module) -> anyhow::Result<Vec<GlobalArrayDeclaration>> {
    let body = module.body();
    fn extract_func_op(operation: OperationRef) -> anyhow::Result<Vec<GlobalArrayDeclaration>> {
        if operation.name().as_string_ref().as_str()? == "func.func" {
            return collect_allocs(operation);
        }
        if let Some(next_op) = operation.next_in_block() {
            extract_func_op(next_op)
        } else {
            Err(anyhow!("No function found"))
        }
    }
    extract_func_op(
        body.first_operation()
            .ok_or_else(|| anyhow!("No operation found"))?,
    )
}
fn collect_allocs(func: OperationRef) -> anyhow::Result<Vec<GlobalArrayDeclaration>> {
    let mut arrays = vec![];
    let body = func
        .region(0)?
        .first_block()
        .ok_or_else(|| anyhow::anyhow!("No block found"))?;
    let op = body.first_operation();
    fn collect_arrays(
        operation: OperationRef,
        arrays: &mut Vec<GlobalArrayDeclaration>,
        mut allocated: usize,
    ) -> anyhow::Result<()> {
        if operation.name().as_string_ref().as_str()? == "memref.alloc" {
            let type_attr = operation.result(0)?.r#type();
            let memref_type = MemRefType::try_from(type_attr)?;
            let rank = memref_type.rank();
            let mut shape = Vec::with_capacity(rank);
            for dim in 0..rank {
                let DimSize::Static(size) = memref_type.dim_size(dim)? else {
                    return Err(anyhow!("dynamic dimension size is not supported"));
                };
                shape.push(size as usize);
            }
            arrays.push(GlobalArrayDeclaration {
                name: format!("ARRAY_{}", allocated),
                size: shape.into_boxed_slice(),
            });
            allocated += 1;
        }
        if let Some(next_op) = operation.next_in_block() {
            collect_arrays(next_op, arrays, allocated)?;
        }
        Ok(())
    }
    if let Some(op) = op {
        collect_arrays(op, &mut arrays, 0)?;
    }
    Ok(arrays)
}

/// cachegrind requires each cache's set count (size / (associativity *
/// block_size)) to be a power of two. Aborts with a message naming the
/// offending cache and parameters if it is not — a non-power-of-two config
/// makes valgrind refuse to run, and a silently-recorded zero row is worse
/// than a loud failure.
fn check_cache_geometry(name: &str, size: usize, assoc: usize, block: usize) {
    assert!(assoc > 0 && block > 0, "{name} cache: assoc and block must be > 0");
    let denom = assoc * block;
    assert!(
        size % denom == 0,
        "{name} cache size {size} is not a multiple of assoc*block ({assoc}*{block}={denom})"
    );
    let sets = size / denom;
    assert!(
        sets.is_power_of_two(),
        "{name} cache set count {sets} = {size}/({assoc}*{block}) is not a power of two;          cachegrind will refuse this config. Pick a size/assoc/block whose set count is a          power of two, or run a valid config and derive this associativity with assoc-conv."
    );
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let rcontext = Context::new();
    let args = Cli::parse();
    let source = std::fs::read_to_string(&args.input).unwrap();
    let module = Module::parse(rcontext.mlir_context(), &source).unwrap();
    let dom = DominanceInfo::new(&module);
    let target_tree = extract_target(&module, &args, &rcontext, &dom).unwrap();
    trace!("Extracted target tree: {:#}", target_tree);
    let workdir = tempfile::tempdir().unwrap();
    let program_path = workdir.path().join("program.cxx");
    let mut program_file = std::fs::File::create(&program_path).unwrap();
    if let Ok(attr) = module.as_operation().attribute("simulation.prologue") {
        let string = attr.to_string();
        let unescaped = unescaper::unescape(string.trim_matches('"')).unwrap();
        write!(program_file, "// Simulation Prologue:\n{unescaped}\n\n").unwrap();
    } else {
        trace!("No simulation prologue found");
    }
    let mut global_arrays = extract_global_arrays(&module).unwrap();
    global_arrays.extend(extract_func_alloc_arrays(&module).unwrap());
    for array in &global_arrays {
        write!(program_file, "double {}[", array.name).unwrap();
        for (i, dim) in array.size.iter().enumerate() {
            if i > 0 {
                write!(program_file, "][").unwrap();
            }
            write!(program_file, "{dim}").unwrap();
        }
        writeln!(program_file, "];").unwrap();
    }
    if let Ok(attr) = module.as_operation().attribute("simulation.dims") {
        let string = attr.to_string();
        let dims_arrays = parse_simulation_dims(string.trim().trim_matches('"')).unwrap();
        for array in &dims_arrays {
            // volatile, matching the declarations legacy prologues carried
            write!(program_file, "volatile double {}[", array.name).unwrap();
            for (i, dim) in array.size.iter().enumerate() {
                if i > 0 {
                    write!(program_file, "][").unwrap();
                }
                write!(program_file, "{dim}").unwrap();
            }
            writeln!(program_file, "];").unwrap();
        }
        global_arrays.extend(dims_arrays);
    } else {
        trace!("No simulation.dims found");
    }
    if args.hash {
        assert!(
            args.d1_block_size == args.ll_block_size,
            "--hash requires equal D1/LL block sizes (got {} and {})",
            args.d1_block_size,
            args.ll_block_size
        );
        assert!(
            args.d1_block_size.is_power_of_two() && args.d1_block_size <= 4096,
            "--hash requires a power-of-two block size <= 4096"
        );
        let mut accessed = Vec::new();
        collect_accessed_arrays(target_tree, &mut accessed);
        emit_hash_preamble(
            &mut program_file,
            &global_arrays,
            &accessed,
            args.d1_block_size,
        )
        .unwrap();
    }
    let emitter = CProgramEmitter::new(program_file, args.hash);
    emitter.emit(target_tree).unwrap();
    info!("C program emitted:{}", {
        std::fs::read_to_string(&program_path).unwrap()
    });
    let output_path = workdir.path().join("test.exe");
    std::process::Command::new("clang++")
        .arg(&program_path)
        .arg("-o")
        .arg(&output_path)
        .args([
            "-static",
            "-nostdlib",
            "-fno-stack-protector",
            "-fno-pic",
            // -Os starves the register allocator once --hash adds the
            // permutation arithmetic, spilling to the stack — and cachegrind
            // counts every spill as a data reference. -O2 keeps the access
            // path stackless.
            if args.hash { "-O2" } else { "-Os" },
            "-ffreestanding",
        ])
        .current_dir(workdir.path())
        .status()
        .expect("Failed to compile C program");
    let manager = r2d2_sqlite::SqliteConnectionManager::file(&args.database);
    let pool = r2d2::Pool::new(manager).unwrap();
    pool.get()
        .unwrap()
        .execute(
            r#"CREATE TABLE IF NOT EXISTS records (
            program TEXT NOT NULL,
            d1_block_size INTEGER NOT NULL,
            d1_associativity INTEGER NOT NULL,
            d1_cache_size INTEGER NOT NULL,
            ll_block_size INTEGER NOT NULL,
            ll_associativity INTEGER NOT NULL,
            ll_cache_size INTEGER NOT NULL,
            d1_miss_count INTEGER NOT NULL,
            ll_miss_count INTEGER NOT NULL,
            total_access INTEGER NOT NULL,
            process_time INTEGER NOT NULL,
            hashed INTEGER NOT NULL,
            PRIMARY KEY (program, d1_block_size, d1_associativity, d1_cache_size, ll_block_size, ll_associativity, ll_cache_size, hashed)
        )"#,
            (),
        )
        .unwrap();
    // CREATE TABLE IF NOT EXISTS silently keeps an old schema; without this
    // probe, the insert-retry loop below would spin forever on a database
    // created before the `hashed` column existed.
    {
        let conn = pool.get().unwrap();
        let mut stmt = conn.prepare("PRAGMA table_info(records)").unwrap();
        let has_hashed = stmt
            .query_map((), |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(Result::ok)
            .any(|name| name == "hashed");
        assert!(
            has_hashed,
            "database {} has a pre-`hashed` records table; point --database at a new file",
            args.database.display()
        );
    }

    let program = args
        .input
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let range = if args.batched {
        if let Some(associativity) = args.d1_associativity {
            let factor = args.d1_block_size * associativity;
            let mut cache_sizes = vec![];
            let mut exp = 1;
            while factor * exp <= args.d1_cache_size {
                cache_sizes.push(factor * exp);
                exp *= 2;
            }
            cache_sizes
        } else {
            let block_size = args.d1_block_size;
            let cache_size = args.d1_cache_size;
            (block_size..=cache_size)
                .step_by(block_size)
                .collect::<Vec<_>>()
        }
    } else {
        vec![args.d1_cache_size]
    };
    let cool_down = CoolDown {
        flag: AtomicBool::new(false),
        finished: AtomicBool::new(false),
        components: Mutex::new(Components::new_with_refreshed_list()),
    };
    std::thread::scope(|s| {
        s.spawn(|| cool_down.monitor_loop());
        range.into_par_iter().progress().for_each(|cache_size| {
            cool_down.wait();
            let associativity = args
                .d1_associativity
                .unwrap_or_else(|| cache_size / args.d1_block_size);
            // cachegrind requires each cache's set count — size/(assoc*block) —
            // to be a power of two, and refuses to run otherwise. Catch that
            // here with a message naming the parameter, rather than letting
            // valgrind fail and recording a row of zeros that reads as a real
            // (all-hit) measurement downstream.
            check_cache_geometry("D1", cache_size, associativity, args.d1_block_size);
            check_cache_geometry(
                "LL",
                args.ll_cache_size,
                args.ll_associativity,
                args.ll_block_size,
            );
            let d1_string = format!(
                "--D1={},{},{}",
                cache_size, associativity, args.d1_block_size,
            );
            let ll_string = format!(
                "--LL={},{},{}",
                args.ll_cache_size, args.ll_associativity, args.ll_block_size,
            );
            let start = std::time::Instant::now();
            let raw = std::process::Command::new(&args.valgrind_path)
                .arg("--tool=cachegrind")
                .arg("--cache-sim=yes")
                .arg("-v")
                .arg(d1_string)
                .arg(ll_string)
                .arg(&output_path)
                .current_dir(workdir.path())
                .output()
                .unwrap();
            let process_time = start.elapsed().as_nanos() as usize;
            let output = String::from_utf8_lossy(&raw.stderr);
            info!("Valgrind output:\n{output}");
            // Fail closed: a non-zero valgrind exit, or output that never
            // yields the summary lines, must abort — never persist a
            // zero-filled row that is indistinguishable from a real all-hit
            // result. "did not measure" and "measured zero" are different facts.
            if !raw.status.success() {
                panic!(
                    "valgrind failed for {program} (exit {:?}); not recording. \
                     stderr:\n{output}",
                    raw.status.code()
                );
            }
            let mut total_access = None;
            let mut d1_miss_count = None;
            let mut ll_miss_count = None;
            for line in output.lines() {
                if line.contains("D refs:") {
                    if let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next()) {
                        total_access = value.trim().replace(",", "").parse().ok();
                    }
                } else if line.contains("D1  misses:") {
                    if let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next()) {
                        d1_miss_count = value.trim().replace(",", "").parse().ok();
                    }
                } else if line.contains("LLd misses:")
                    && let Some(value) = line.split(':').nth(1).and_then(|s| s.split('(').next())
                {
                    ll_miss_count = value.trim().replace(",", "").parse().ok();
                }
            }
            let (total_access, d1_miss_count, ll_miss_count) =
                match (total_access, d1_miss_count, ll_miss_count) {
                    (Some(t), Some(d), Some(l)) => (t, d, l),
                    _ => panic!(
                        "valgrind produced no parseable cache summary for {program}; \
                         not recording. stderr:\n{output}"
                    ),
                };

            let record = Record {
                program: program.clone(),
                d1_cache_size: cache_size,
                d1_associativity: associativity,
                d1_block_size: args.d1_block_size,
                ll_associativity: args.ll_associativity,
                ll_cache_size: args.ll_cache_size,
                ll_block_size: args.ll_block_size,
                d1_miss_count,
                ll_miss_count,
                total_access,
                process_time,
                hashed: args.hash,
            };
            record.insert(&pool);
        });
        cool_down.finish();
    });
}
