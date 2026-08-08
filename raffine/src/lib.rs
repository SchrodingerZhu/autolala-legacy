use melior::Context as MLIRContext;
pub mod affine;
mod cxx;
pub mod tree;
pub use cxx::DominanceInfo;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MLIR internal error: {0}")]
    MeliorError(#[from] melior::Error),
    #[error("invalid loop nest from MLIR code ({0})")]
    InvalidLoopNest(&'static str),
    #[error("invalid ffi operation ({0})")]
    FFIError(#[from] ::cxx::Exception),
}

pub fn initialize_mlir_context() -> MLIRContext {
    let context = melior::Context::new();
    let registery = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registery);
    context.append_dialect_registry(&registery);
    context.load_all_available_dialects();
    tracing::debug!(
        "Loaded all available dialects ({} in total)",
        context.loaded_dialect_count()
    );
    context
}

pub struct Context {
    mlir_context: MLIRContext,
    arena: bumpalo::Bump,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        let mlir_context = initialize_mlir_context();
        let arena = bumpalo::Bump::new();
        Self {
            mlir_context,
            arena,
        }
    }

    pub fn mlir_context(&self) -> &MLIRContext {
        &self.mlir_context
    }

    pub fn arena(&self) -> &bumpalo::Bump {
        &self.arena
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::{BlockLike, Module, RegionLike};
    use melior::ir::operation::OperationLike;

    #[test]
    fn mlir_loads_affine_dialect() {
        _ = tracing_subscriber::fmt::try_init();
        let context = super::initialize_mlir_context();
        let dialect = context.get_or_load_dialect("affine");
        tracing::debug!("Loaded dialect: {}", dialect.namespace().unwrap());
        assert_eq!(dialect.namespace().unwrap(), "affine");
    }

    #[test]
    fn mlir_parse_module() {
        let context = super::initialize_mlir_context();
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
        let module = Module::parse(&context, module).unwrap();
        tracing::debug!("Parsed module: {}", module.body().to_string());
        let body = module.body();
        let op = body.first_operation().unwrap();
        let body = op.region(0).unwrap();
        let body = body.first_block().unwrap();
        let first_op = body.first_operation().unwrap();
        println!("First operation: {}", first_op);
    }

    #[test]
    fn create_dominance_info() {
        let context = super::initialize_mlir_context();
        let module = r#"
        module {
            func.func @test_func() {
                func.return
            }
        }
        "#;
        let module = Module::parse(&context, module).unwrap();
        let _dominance_info = crate::DominanceInfo::new(&module);
    }
}
