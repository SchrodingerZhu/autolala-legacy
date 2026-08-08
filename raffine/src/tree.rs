use std::{
    cell::{Cell, RefCell},
    collections::hash_map::Entry,
    fmt::Write,
};

use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, OperationRef, RegionLike, Value, ValueLike};
use rustc_hash::FxHashMapRand;
pub use ustr::Ustr;

use crate::{
    Context, DominanceInfo,
    affine::{AffineMap, IntegerSet},
    cxx::{AccessId, defined_in_any_loop},
};

#[derive(Debug)]
pub enum Tree<'a> {
    For {
        lower_bound: AffineMap<'a>,
        upper_bound: AffineMap<'a>,
        lower_bound_operands: &'a [ValID],
        upper_bound_operands: &'a [ValID],
        step: isize,
        ivar: ValID,
        body: &'a Tree<'a>,
    },
    Block(&'a [&'a Tree<'a>]),
    Access {
        memref: ValID,
        map: AffineMap<'a>,
        operands: &'a [ValID],
        is_write: bool,
    },
    If {
        condition: IntegerSet<'a>,
        operands: &'a [ValID],
        then: &'a Tree<'a>,
        r#else: Option<&'a Tree<'a>>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum ValID {
    IVar(usize),
    Symbol(usize),
    Memref(usize),
    Global(Ustr),
}

impl std::fmt::Display for ValID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValID::IVar(id) => write!(f, "i{id}"),
            ValID::Symbol(id) => write!(f, "s{id}"),
            ValID::Memref(id) => write!(f, "m{id}"),
            ValID::Global(name) => write!(f, "{}", name),
        }
    }
}

struct TranslationContext<'a, 'b> {
    ivar_counter: Cell<usize>,
    symbol_counter: Cell<usize>,
    memref_counter: Cell<usize>,
    values: RefCell<FxHashMapRand<usize, ValID>>,
    toplevel: OperationRef<'a, 'b>,
    dom_info: &'a DominanceInfo<'a>,
    symbolic_memref: bool,
}

impl<'a, 'b> TranslationContext<'a, 'b> {
    fn new(
        toplevel: OperationRef<'a, 'b>,
        dom_info: &'a DominanceInfo<'a>,
        symbolic_memref: bool,
    ) -> Self {
        Self {
            values: RefCell::new(FxHashMapRand::default()),
            ivar_counter: Cell::new(0),
            symbol_counter: Cell::new(0),
            memref_counter: Cell::new(0),
            toplevel,
            dom_info,
            symbolic_memref,
        }
    }
    fn get_affine_operand<'c>(&self, value: Value<'a, 'c>) -> ValID
    where
        'b: 'c,
    {
        let mut values = self.values.borrow_mut();
        let id = value.to_raw().ptr as usize;
        match values.entry(id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                if self.dom_info.properly_dominates(value, self.toplevel)
                    || !defined_in_any_loop(value)
                {
                    let id = self.symbol_counter.get();
                    self.symbol_counter.set(id + 1);
                    let value = ValID::Symbol(id);
                    entry.insert(value);
                    value
                } else {
                    let id = self.ivar_counter.get();
                    self.ivar_counter.set(id + 1);
                    let value = ValID::IVar(id);
                    entry.insert(value);
                    value
                }
            }
        }
    }
    fn get_memref(&self, memref: AccessId) -> ValID {
        match memref {
            AccessId::Local(memref) => {
                let mut values = self.values.borrow_mut();
                match values.entry(memref) {
                    Entry::Occupied(entry) => *entry.get(),
                    Entry::Vacant(entry) => {
                        let id = self.memref_counter.get();
                        self.memref_counter.set(id + 1);
                        let value = ValID::Memref(id);
                        entry.insert(value);
                        value
                    }
                }
            }
            AccessId::Global(name) => {
                let unique_name = Ustr::from(name.to_str().unwrap());
                if self.symbolic_memref {
                    let mut values = self.values.borrow_mut();
                    let id = unique_name.as_ptr() as usize;
                    match values.entry(id) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            let id = self.memref_counter.get();
                            self.memref_counter.set(id + 1);
                            let value = ValID::Memref(id);
                            entry.insert(value);
                            value
                        }
                    }
                } else {
                    ValID::Global(Ustr::from(name.to_str().unwrap()))
                }
            }
        }
    }
    fn loop_scope<R>(&self, f: impl FnOnce(&Self) -> R) -> R {
        let current_ivar = self.ivar_counter.get();
        let res = f(self);
        self.ivar_counter.set(current_ivar);
        res
    }
}

struct VarArrayDisplay<'a>(&'a [ValID]);
impl<'a> std::fmt::Display for VarArrayDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }
        f.write_char('[')?;
        for (i, id) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{id}")?;
        }
        f.write_char(']')
    }
}

impl std::fmt::Display for Tree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tree::For {
                lower_bound,
                upper_bound,
                step,
                body,
                lower_bound_operands,
                upper_bound_operands,
                ivar,
            } => write!(
                f,
                "for {} := {}{} to {}{} step {} {}",
                ivar,
                lower_bound,
                VarArrayDisplay(lower_bound_operands),
                upper_bound,
                VarArrayDisplay(upper_bound_operands),
                step,
                body
            ),

            Tree::Block(body) => write!(
                f,
                "{{ {} }}",
                body.iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Tree::Access {
                map,
                is_write: true,
                memref,
                operands,
            } => write!(f, "store {}{} @ {}", map, VarArrayDisplay(operands), memref),
            Tree::Access {
                map,
                is_write: false,
                memref,
                operands,
            } => write!(f, "load {}{} @ {}", map, VarArrayDisplay(operands), memref),
            Tree::If {
                condition,
                then,
                r#else,
                operands,
            } => write!(
                f,
                "if {}{} {{ {} }} else {{ {} }}",
                condition,
                VarArrayDisplay(operands),
                then,
                r#else.map_or("None".to_string(), |e| e.to_string())
            ),
        }
    }
}

impl Context {
    fn build_if<'a>(
        &'a self,
        condition: IntegerSet<'a>,
        then: &'a Tree<'a>,
        r#else: Option<&'a Tree<'a>>,
        operands: &'a [ValID],
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::If {
            condition,
            then,
            r#else,
            operands,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn build_for<'a>(
        &'a self,
        lower_bound: AffineMap<'a>,
        lower_bound_operands: &'a [ValID],
        upper_bound: AffineMap<'a>,
        upper_bound_operands: &'a [ValID],
        step: isize,
        body: &'a Tree<'a>,
        ivar: ValID,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::For {
            lower_bound,
            upper_bound,
            step,
            body,
            lower_bound_operands,
            upper_bound_operands,
            ivar,
        })
    }
    fn build_block<'a, I>(&'a self, body: I) -> &'a Tree<'a>
    where
        I: IntoIterator<Item = &'a Tree<'a>>,
        I::IntoIter: ExactSizeIterator,
    {
        let inner = self.arena.alloc_slice_fill_iter(body);
        self.arena.alloc(Tree::Block(inner))
    }

    fn build_access<'a>(
        &'a self,
        memref: ValID,
        map: AffineMap<'a>,
        operands: &'a [ValID],
        is_write: bool,
    ) -> &'a Tree<'a> {
        self.arena.alloc(Tree::Access {
            memref,
            map,
            is_write,
            operands,
        })
    }

    pub fn build_tree<'a, 'b>(
        &'a self,
        entry: OperationRef<'a, 'b>,
        dom_info: &'a DominanceInfo<'a>,
        symbolic_memref: bool,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        let ctx = TranslationContext::new(entry, dom_info, symbolic_memref);
        self.build_tree_from_loop(entry, &ctx)
    }

    pub fn build_func_tree<'a, 'b>(
        &'a self,
        entry: OperationRef<'a, 'b>,
        dom_info: &'a DominanceInfo<'a>,
        symbolic_memref: bool,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        let ctx = TranslationContext::new(entry, dom_info, symbolic_memref);
        let region = entry.region(0)?;
        let body = region
            .first_block()
            .ok_or(crate::Error::InvalidLoopNest("invalid function body"))?;
        self.build_tree_from_block(body, &ctx)
    }

    fn build_tree_from_loop<'a, 'b, 'c: 'b>(
        &'a self,
        entry: OperationRef<'a, 'b>,
        ctx: &TranslationContext<'a, 'c>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from loop: {}", entry);
        ctx.loop_scope(|ctx| {
            let lower_bound = crate::cxx::for_op_get_lower_bound_map(entry)?;
            let upper_bound = crate::cxx::for_op_get_upper_bound_map(entry)?;
            let lower_bound_operands = crate::cxx::for_op_get_lower_bound_operands(entry)?;
            let upper_bound_operands = crate::cxx::for_op_get_upper_bound_operands(entry)?;
            let ivar = crate::cxx::for_op_get_induction_variable(entry)?;
            let lower_bound_operands = self.arena.alloc_slice_fill_iter(
                lower_bound_operands
                    .iter()
                    .map(|v| ctx.get_affine_operand(*v)),
            );
            let upper_bound_operands = self.arena.alloc_slice_fill_iter(
                upper_bound_operands
                    .iter()
                    .map(|v| ctx.get_affine_operand(*v)),
            );
            let ivar = ctx.get_affine_operand(ivar);
            let step = crate::cxx::for_op_get_step(entry)?;
            let region = entry.region(0)?;
            let body = region
                .first_block()
                .ok_or(crate::Error::InvalidLoopNest("invalid loop body"))?;
            if body.next_in_region().is_some() {
                return Err(crate::Error::InvalidLoopNest("invalid loop body"));
            }
            let body = self.build_tree_from_block(body, ctx)?;
            Ok(self.build_for(
                lower_bound,
                lower_bound_operands,
                upper_bound,
                upper_bound_operands,
                step,
                body,
                ivar,
            ))
        })
    }

    fn build_tree_from_load_store<'a, 'b, 'c: 'b>(
        &'a self,
        is_write: bool,
        entry: OperationRef<'a, 'b>,
        ctx: &TranslationContext<'a, 'c>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from load: {}", entry);
        let memref = crate::cxx::load_store_op_get_access_id(entry)?;
        let memref = ctx.get_memref(memref);
        let map = crate::cxx::load_store_op_get_access_map(entry)?;
        let operands = crate::cxx::load_store_op_get_affine_operands(entry)?;
        let operands = self
            .arena
            .alloc_slice_fill_iter(operands.iter().map(|v| ctx.get_affine_operand(*v)));
        Ok(self.build_access(memref, map, operands, is_write))
    }

    fn build_tree_from_if<'a, 'b, 'c: 'b>(
        &'a self,
        entry: OperationRef<'a, 'b>,
        ctx: &TranslationContext<'a, 'c>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from if: {}", entry);
        let condition = crate::cxx::if_op_get_condition(entry)?;
        let then_block = crate::cxx::if_op_get_then_block(entry)?;
        let then = self.build_tree_from_block(then_block, ctx)?;
        let operands = crate::cxx::if_op_get_condition_operands(entry)?;
        let operands = self
            .arena
            .alloc_slice_fill_iter(operands.iter().map(|v| ctx.get_affine_operand(*v)));
        let r#else = if let Some(else_block) = crate::cxx::if_op_get_else_block(entry)? {
            Some(self.build_tree_from_block(else_block, ctx)?)
        } else {
            None
        };
        Ok(self.build_if(condition, then, r#else, operands))
    }

    fn build_tree_from_block<'a, 'b, 'c: 'b>(
        &'a self,
        entry: BlockRef<'a, 'b>,
        ctx: &TranslationContext<'a, 'c>,
    ) -> Result<&'a Tree<'a>, crate::Error> {
        tracing::trace!("building tree from block: {}", entry);
        let mut subtrees = Vec::new();
        fn collect_op<'a, 'b, 'c: 'b>(
            this: &'a Context,
            op: OperationRef<'a, 'b>,
            subtrees: &mut Vec<&'a Tree<'a>>,
            ctx: &TranslationContext<'a, 'c>,
        ) -> Result<(), crate::Error> {
            let name = op.name();
            let name = name
                .as_string_ref()
                .as_str()
                .map_err(|_| crate::Error::InvalidLoopNest("invalid operation name"))?;
            'dispatch: {
                let res = match name {
                    "affine.for" => this.build_tree_from_loop(op, ctx)?,
                    "affine.load" => this.build_tree_from_load_store(false, op, ctx)?,
                    "affine.store" => this.build_tree_from_load_store(true, op, ctx)?,
                    "affine.if" => this.build_tree_from_if(op, ctx)?,
                    _ => {
                        tracing::trace!("ignored operation: {}", op);
                        break 'dispatch;
                    }
                };
                subtrees.push(res);
            }
            if let Some(next) = op.next_in_block() {
                return collect_op(this, next, subtrees, ctx);
            }
            Ok(())
        }
        if let Some(op) = entry.first_operation() {
            collect_op(self, op, &mut subtrees, ctx)?;
        }
        if subtrees.len() == 1 {
            return Ok(subtrees[0]);
        }
        let block = self.build_block(subtrees);
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Module, RegionLike};

    use crate::Context;

    #[test]
    fn build_tree() {
        _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .try_init();
        let context = Context::new();
        let module = r#"
 module {
  func.func @stencil_kernel(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
    affine.for %i = 1 to 99 {
      affine.for %j = 1 to 99 {
        // Load the neighboring values and the center value
        %top    = affine.load %A[%i - 1, %j] : memref<100x100xf32>
        %bottom = affine.load %A[%i + 1, %j] : memref<100x100xf32>
        %left   = affine.load %A[%i, %j - 1] : memref<100x100xf32>
        %right  = affine.load %A[%i, %j + 1] : memref<100x100xf32>
        %center = affine.load %A[%i, %j] : memref<100x100xf32>

        // Perform the sum of the loaded values
        %sum1 = arith.addf %top, %bottom : f32
        %sum2 = arith.addf %left, %right : f32
        %sum3 = arith.addf %sum1, %sum2 : f32
        %sum4 = arith.addf %sum3, %center : f32

        // Compute the average (sum / 5.0)
        %c5 = arith.constant 5.0 : f32
        %avg = arith.divf %sum4, %c5 : f32

        // Store the result in the output array
        affine.store %avg, %B[%i, %j] : memref<100x100xf32>
      }
    } { slap.extract }
    return
  }
}
"#;
        let module = Module::parse(context.mlir_context(), module).unwrap();
        tracing::debug!("Parsed module: {}", module.body().to_string());
        let body = module.body();
        let op = body.first_operation().unwrap();
        let body = op.region(0).unwrap();
        let body = body.first_block().unwrap();
        let first_op = body.first_operation().unwrap();
        println!("First operation: {}", first_op);
        let dom = crate::DominanceInfo::new(&module);
        let tree = context.build_tree(first_op, &dom, false).unwrap();
        tracing::debug!("Tree: {:#}", tree);
    }

    #[test]
    fn symbolic_matmul() {
        _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .try_init();
        let context = Context::new();
        let module = r#"
        module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to %arg3 {
      affine.for %arg7 = 0 to %arg5 {
        affine.for %arg8 = 0 to %arg4 {
          %0 = affine.load %arg0[%arg6, %arg8] : memref<?x?xf32>
          %1 = affine.load %arg1[%arg8, %arg7] : memref<?x?xf32>
          %2 = affine.load %arg2[%arg6, %arg7] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg6, %arg7] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
"#;
        let module = Module::parse(context.mlir_context(), module).unwrap();
        tracing::debug!("Parsed module: {}", module.body().to_string());
        let body = module.body();
        let op = body.first_operation().unwrap();
        let body = op.region(0).unwrap();
        let body = body.first_block().unwrap();
        let first_op = body.first_operation().unwrap();
        println!("First operation: {}", first_op);
        let dom = crate::DominanceInfo::new(&module);
        let tree = context.build_tree(first_op, &dom, false).unwrap();
        tracing::debug!("Tree: {:#}", tree);
    }
}
