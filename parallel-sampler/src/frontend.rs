//! MLIR ingestion, shared with the rest of the workspace.
//!
//! The sampler deliberately reads the *same* MLIR the analyzer reads, through
//! the same `raffine` tree builder, so a disagreement between sampler and model
//! can only come from the modeling — never from the two having been handed
//! different programs.

use anyhow::{Result, bail};
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, Module, OperationRef};
use raffine::tree::{Tree, ValID};
use raffine::{Context, DominanceInfo};

/// Builds the loop tree for the target function.
///
/// Mirrors `cachegrind-runner/src/main.rs::extract_target`: the first
/// `func.func` is used unless a name is given.
pub fn extract_target<'ctx>(
    module: &'ctx Module<'ctx>,
    target_function: Option<&str>,
    context: &'ctx Context,
    dom: &'ctx DominanceInfo<'ctx>,
) -> Result<&'ctx Tree<'ctx>> {
    fn locate<'a, 'b, F>(
        cursor: Option<OperationRef<'a, 'b>>,
        target_function: Option<&str>,
        conti: F,
    ) -> Result<&'a Tree<'a>>
    where
        F: for<'c> FnOnce(OperationRef<'a, 'c>) -> Result<&'a Tree<'a>>,
    {
        let Some(op) = cursor else {
            match target_function {
                Some(name) => bail!("no function named `{name}` in the module"),
                None => bail!("the module contains no `func.func`"),
            }
        };
        if op.name().as_string_ref().as_str()? == "func.func" {
            match target_function {
                None => return conti(op),
                Some(name) => {
                    let sym_name = op.attribute("sym_name")?;
                    if sym_name.to_string().trim_matches('"') == name {
                        return conti(op);
                    }
                }
            }
        }
        locate(op.next_in_block(), target_function, conti)
    }

    let cursor = module.body().first_operation();
    locate(cursor, target_function, move |func| {
        Ok(context.build_func_tree(func, dom, false)?)
    })
}

/// Static facts about a loop tree, gathered in one walk.
#[derive(Debug, Default, Clone)]
pub struct TreeFacts {
    /// Number of distinct `ValID::Symbol` slots the program reads. Every one
    /// needs a `--symbol` binding before the tree can be interpreted.
    pub symbols: usize,
    /// Deepest loop nesting level. `raffine` numbers induction variables by
    /// depth, so a valid `--parallel-loop-depth` is `0..=max_depth`.
    pub max_depth: usize,
}

/// Collects the symbol count and nesting depth of a tree.
pub fn facts(tree: &Tree<'_>) -> TreeFacts {
    fn note(operands: &[ValID], facts: &mut TreeFacts) {
        for operand in operands {
            if let ValID::Symbol(index) = operand {
                facts.symbols = facts.symbols.max(index + 1);
            }
        }
    }

    fn walk(tree: &Tree<'_>, depth: usize, facts: &mut TreeFacts) {
        match tree {
            Tree::For {
                lower_bound_operands,
                upper_bound_operands,
                body,
                ..
            } => {
                note(lower_bound_operands, facts);
                note(upper_bound_operands, facts);
                facts.max_depth = facts.max_depth.max(depth);
                walk(body, depth + 1, facts);
            }
            Tree::Block(stmts) => {
                for stmt in stmts.iter() {
                    walk(stmt, depth, facts);
                }
            }
            Tree::If {
                operands,
                then,
                r#else,
                ..
            } => {
                note(operands, facts);
                walk(then, depth, facts);
                if let Some(otherwise) = r#else {
                    walk(otherwise, depth, facts);
                }
            }
            Tree::Access { operands, .. } => note(operands, facts),
        }
    }

    let mut facts = TreeFacts::default();
    walk(tree, 0, &mut facts);
    facts
}

/// Parses a `name=value` symbol binding.
///
/// Accepts both the `raffine` display form (`s0=512`) and a bare position
/// (`0=512`), because the symbol numbering is assigned by first use during the
/// tree walk and has no relation to the MLIR argument order.
pub fn parse_symbol_binding(text: &str) -> Result<(usize, i64)> {
    let Some((name, value)) = text.split_once('=') else {
        bail!("symbol binding `{text}` is not of the form `s<index>=<value>`");
    };
    let index = name.trim().trim_start_matches('s');
    let Ok(index) = index.parse::<usize>() else {
        bail!("symbol binding `{text}` has a non-numeric index `{name}`");
    };
    let Ok(value) = value.trim().parse::<i64>() else {
        bail!("symbol binding `{text}` has a non-integer value `{value}`");
    };
    Ok((index, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_bindings_accept_both_spellings() {
        assert_eq!(parse_symbol_binding("s2=512").expect("parses"), (2, 512));
        assert_eq!(parse_symbol_binding("0 = 64").expect("parses"), (0, 64));
    }

    #[test]
    fn malformed_symbol_bindings_are_rejected() {
        assert!(parse_symbol_binding("s0").is_err());
        assert!(parse_symbol_binding("sx=1").is_err());
        assert!(parse_symbol_binding("s0=abc").is_err());
    }
}
