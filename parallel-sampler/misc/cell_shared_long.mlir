// "Data sharing / long RI" cell of PACT'24 Table 1.
// B's subscript does not mention the parallel induction variable, so every
// thread sweeps the same 256 elements in the same order -- the racetrack
// picture exactly. Within a thread, PRI = 256.
// The racetrack model predicts CRI = T * PRI * X with X ~ (T-1)(1-x)^(T-2).
module {
  func.func @cell_shared_long(%B: memref<?xf64>) {
    affine.for %i = 0 to 2048 {
      affine.for %j = 0 to 256 {
        %0 = affine.load %B[%j] : memref<?xf64>
      }
    }
    return
  }
}
