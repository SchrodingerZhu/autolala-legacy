module attributes { "simulation.dims" = "ARRAY_0:400x400" } {
  func.func @lu_decomposition(%A: memref<?x?xf32>) {
    // Iterate over columns (k-th column of A)
    affine.for %k = 0 to 400 {
      // Update the upper triangular part (U) in the k-th row
      affine.for %j = affine_map<(d0) -> (d0)> (%k) to 400 {
        // A[k, j] remains as U[k, j] for k-th row
        %A_kj = affine.load %A[%k, %j] : memref<?x?xf32>
        affine.store %A_kj, %A[%k, %j] : memref<?x?xf32>
      }

      // Update the lower triangular part (L) in the k-th column
      affine.for %i = affine_map<(d0) -> (d0 + 1)> (%k) to 400 {
        // Compute L[i, k] = A[i, k] / U[k, k]
        %A_ik = affine.load %A[%i, %k] : memref<?x?xf32>
        %A_kk = affine.load %A[%k, %k] : memref<?x?xf32>
        %L_ik = arith.divf %A_ik, %A_kk : f32
        affine.store %L_ik, %A[%i, %k] : memref<?x?xf32>

        // Update the rest of the A matrix using the new L and U values
        affine.for %j = affine_map<(d0) -> (d0 + 1)> (%k) to 400 {
          // A[i, j] -= L[i, k] * U[k, j]
          %A_ij = affine.load %A[%i, %j] : memref<?x?xf32>
          %U_kj = affine.load %A[%k, %j] : memref<?x?xf32>
          %L_ik_new = affine.load %A[%i, %k] : memref<?x?xf32>
          %product = arith.mulf %L_ik_new, %U_kj : f32
          %A_ij_new = arith.subf %A_ij, %product : f32
          affine.store %A_ij_new, %A[%i, %j] : memref<?x?xf32>
        }
      }
    } { slap.extract }
    return
  }
}
