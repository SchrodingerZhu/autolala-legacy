// "No sharing / long RI" cell of PACT'24 Table 1.
// The parallel loop's induction variable indexes the outer array dimension, so
// each thread owns a disjoint set of rows and never reuses another thread's
// data. Within a thread, A[i][j] is reused once per sweep of j: PRI = 256.
// The model predicts CRI = T * PRI (scale by thread count).
module {
  func.func @cell_private_long(%A: memref<?x?xf64>) {
    affine.for %i = 0 to 256 {
      affine.for %r = 0 to 8 {
        affine.for %j = 0 to 256 {
          %0 = affine.load %A[%i, %j] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
