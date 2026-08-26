//! Direct C input via Polygeist.
//!
//! `cgeist` (from PATH) lowers C to affine-dialect MLIR, but it is built
//! against an older LLVM than this crate links, so its output cannot be fed
//! to our parser as-is. The bridge:
//!
//! 1. `cgeist <input> --function=<target> -S --raise-scf-to-affine` — C to
//!    affine MLIR, in Polygeist's (older-LLVM) syntax;
//! 2. `polygeist-opt --canonicalize` — canonicalize inside the producing
//!    toolchain;
//! 3. textually drop the top-level `module attributes { … }` dict, whose
//!    `dlti.dl_spec` uses a `vector<2xi32>` shape the analyzer's newer MLIR
//!    parser rejects — this has to happen *before* parsing (see
//!    `strip_module_attributes`);
//! 4. parse here with unregistered dialects allowed, then strip every
//!    remaining dialect-qualified discardable attribute (`llvm.linkage`,
//!    `polygeist.*`, …) so what's left is bare func/affine/memref/arith IR.
//!
//! Attributes are stripped, foreign *operations* are not: if a Polygeist
//! dialect op survives into the target loop nest the analysis will reject it
//! later with a clear error, which is preferable to silently ignoring it.

use std::error::Error;
use std::path::Path;
use std::process::{Command, Stdio};

use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, Module, OperationRef, RegionLike};

/// Attribute namespaces that survive stripping. `slap` is the analyzer's own
/// loop-marking namespace; everything else dialect-qualified goes.
const KEPT_NAMESPACES: &[&str] = &["slap"];

/// Runs `cgeist | polygeist-opt` (both resolved from PATH) over a C source
/// file and returns affine-dialect MLIR text with Polygeist's module-level
/// data-layout attributes removed.
pub fn run_polygeist(input: &Path, target_function: Option<&str>) -> Result<String, Box<dyn Error>> {
    let function = target_function.unwrap_or("*");
    let cgeist = Command::new("cgeist")
        .arg(input)
        .arg(format!("--function={function}"))
        .arg("-S")
        .arg("--raise-scf-to-affine")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to spawn `cgeist` (is Polygeist on PATH?): {e}"))?;

    let opt = Command::new("polygeist-opt")
        .arg("--canonicalize")
        .stdin(cgeist.stdout.ok_or("cgeist produced no stdout")?)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("failed to spawn `polygeist-opt` (is Polygeist on PATH?): {e}"))?;

    if !opt.status.success() {
        return Err(format!("polygeist-opt exited with {}", opt.status).into());
    }
    Ok(strip_module_attributes(&String::from_utf8(opt.stdout)?))
}

/// Removes the `attributes { … }` dictionary from the top-level `module`.
///
/// Polygeist (built on an older LLVM) stamps the module with a `dlti.dl_spec`
/// whose `dl_entry` values are `vector<2xi32>` — a shape the analyzer's newer
/// MLIR parser rejects outright (it wants dense i64), so it must go *before*
/// parsing rather than via the post-parse attribute strip. The analyzer
/// ignores module attributes entirely, so dropping the whole dict is safe.
/// Balanced-brace scan from the `attributes {`, because the dict spans one
/// long line with nested `<…>`/`[…]` but a single `{…}` level.
fn strip_module_attributes(mlir: &str) -> String {
    const NEEDLE: &str = "module attributes {";
    let Some(start) = mlir.find(NEEDLE) else {
        return mlir.to_string();
    };
    let open = start + NEEDLE.len() - 1; // index of the '{'
    let bytes = mlir.as_bytes();
    let mut depth = 0usize;
    let mut close = None;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    match close {
        // Replace `module attributes { … }` with `module`, keeping the region
        // brace that follows.
        Some(end) => format!("{}module{}", &mlir[..start], &mlir[end + 1..]),
        None => mlir.to_string(),
    }
}

/// Removes every dialect-qualified discardable attribute (except the
/// analyzer's own namespaces) from `op` and, recursively, everything nested
/// inside it. Runs on the module operation to cover `llvm.data_layout`-style
/// module attributes as well.
pub fn strip_foreign_attributes(op: OperationRef) {
    let mut doomed = Vec::new();
    for index in 0..op.discardable_attribute_count() {
        if let Ok((name, _)) = op.discardable_attribute_at(index) {
            let name = name.as_string_ref().as_str().unwrap_or_default().to_string();
            if let Some((namespace, _)) = name.split_once('.') {
                if !KEPT_NAMESPACES.contains(&namespace) {
                    doomed.push(name);
                }
            }
        }
    }
    for name in doomed {
        // SAFETY: removing a discardable attribute never invalidates the
        // operation; the raw call sidesteps melior's mutable-traversal
        // plumbing, which cannot hand out mutable refs during a walk.
        unsafe {
            mlir_sys::mlirOperationRemoveDiscardableAttributeByName(
                op.to_raw(),
                mlir_sys::MlirStringRef {
                    data: name.as_ptr() as *const _,
                    length: name.len(),
                },
            );
        }
    }

    for region in op.regions() {
        let mut block = region.first_block();
        while let Some(b) = block {
            let mut nested = b.first_operation();
            while let Some(n) = nested {
                strip_foreign_attributes(n);
                nested = n.next_in_block();
            }
            block = b.next_in_region();
        }
    }
}

/// Strips the whole module in place (module operation included).
pub fn strip_module(module: &Module) {
    strip_foreign_attributes(module.as_operation());
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::Module;

    #[test]
    fn strips_module_attribute_dict() {
        let src = "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>>, llvm.target_triple = \"aarch64\"} {\n  func.func @f() { return }\n}\n";
        let out = super::strip_module_attributes(src);
        assert!(out.starts_with("module {"), "{out}");
        assert!(!out.contains("dl_spec"), "{out}");
        assert!(out.contains("func.func @f"), "{out}");
    }

    #[test]
    fn strips_polygeist_attributes() {
        let context = melior::Context::new();
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        context.set_allow_unregistered_dialects(true);
        let source = r#"
module attributes {llvm.data_layout = "e-m:e", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @kernel(%arg0: memref<16xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %i = 0 to 16 {
      %v = affine.load %arg0[%i] : memref<16xi32>
      affine.store %v, %arg0[%i] : memref<16xi32>
    } {slap.extract}
    return
  }
}
"#;
        let module = Module::parse(&context, source).expect("parse");
        strip_module(&module);
        let printed = module.as_operation().to_string();
        assert!(!printed.contains("llvm.data_layout"), "{printed}");
        assert!(!printed.contains("llvm.linkage"), "{printed}");
        assert!(!printed.contains("llvm.target_triple"), "{printed}");
        assert!(printed.contains("slap.extract"), "{printed}");
    }
}
