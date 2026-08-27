// Five-point stencil, parallel over the row loop. Neighbouring rows are read by
// adjacent threads, so sharing appears only at chunk boundaries -- the case a
// purely syntactic "does the subscript mention the parallel variable" test gets
// wrong, since every subscript here does mention it.
module {
  func.func @jacobi_2d(%A: memref<?x?xf64>, %B: memref<?x?xf64>) {
    affine.for %i = 1 to 127 {
      affine.for %j = 1 to 127 {
        %0 = affine.load %A[%i, %j] : memref<?x?xf64>
        %1 = affine.load %A[%i - 1, %j] : memref<?x?xf64>
        %2 = affine.load %A[%i + 1, %j] : memref<?x?xf64>
        %3 = affine.load %A[%i, %j - 1] : memref<?x?xf64>
        %4 = affine.load %A[%i, %j + 1] : memref<?x?xf64>
        affine.store %4, %B[%i, %j] : memref<?x?xf64>
      }
    }
    return
  }
}
