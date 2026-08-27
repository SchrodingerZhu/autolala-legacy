// Symmetric rank-k update, parallel over i. The triangular `j <= i` bound makes
// the per-thread work uneven, which stresses the assumption that all threads
// run at the same average speed.
module {
  func.func @syrk(%C: memref<?x?xf64>, %A: memref<?x?xf64>) {
    affine.for %i = 0 to 96 {
      affine.for %j = 0 to 96 {
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%i, %j) {
          affine.for %k = 0 to 96 {
            %0 = affine.load %C[%i, %j] : memref<?x?xf64>
            %1 = affine.load %A[%i, %k] : memref<?x?xf64>
            %2 = affine.load %A[%j, %k] : memref<?x?xf64>
            affine.store %2, %C[%i, %j] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
