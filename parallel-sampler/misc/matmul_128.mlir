// Matrix multiply, parallel over i. Exercises all four Table-1 cells at once:
// C[i][j] is a short private reuse, A[i][k] a long private one, and B[k][j] is
// read by every thread, so it is the long shared (racetrack) case.
module {
  func.func @matmul(%C: memref<?x?xf64>, %A: memref<?x?xf64>, %B: memref<?x?xf64>) {
    affine.for %i = 0 to 128 {
      affine.for %j = 0 to 128 {
        affine.for %k = 0 to 128 {
          %0 = affine.load %C[%i, %j] : memref<?x?xf64>
          %1 = affine.load %A[%i, %k] : memref<?x?xf64>
          %2 = affine.load %B[%k, %j] : memref<?x?xf64>
          affine.store %2, %C[%i, %j] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
