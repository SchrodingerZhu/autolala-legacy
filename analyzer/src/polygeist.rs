//! Direct C input via Polygeist.
//!
//! `cgeist` (from PATH) lowers C to affine-dialect MLIR, but it is built
//! against an older LLVM than this crate links, so its output cannot be fed
//! to our parser as-is. The bridge:
//!
//! 1. `cgeist <input> --function=<target> -S --raise-scf-to-affine` — C to
//!    affine MLIR, printed in Polygeist's (older) syntax;
//! 2. `polygeist-opt --canonicalize --mlir-print-op-generic` — canonicalize
//!    *inside the producing toolchain*, then print in the generic op form,
//!    which is far more stable across MLIR versions than the pretty forms;
//! 3. parse here with unregistered dialects allowed, then strip every
//!    dialect-qualified discardable attribute Polygeist left behind
//!    (`llvm.linkage`, `polygeist.*`, `dlti.*`, the module data layout, …)
//!    so what remains is bare func/affine/memref/arith IR the analyzer
//!    understands.
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
/// file and returns generic-form MLIR text.
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
        .arg("--allow-unregistered-dialect")
        .arg("--mlir-print-op-generic")
        .stdin(cgeist.stdout.ok_or("cgeist produced no stdout")?)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("failed to spawn `polygeist-opt` (is Polygeist on PATH?): {e}"))?;

    if !opt.status.success() {
        return Err(format!("polygeist-opt exited with {}", opt.status).into());
    }
    Ok(String::from_utf8(opt.stdout)?)
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
                mlir_sys::mlirStringRefCreate(name.as_ptr() as *const _, name.len()),
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
