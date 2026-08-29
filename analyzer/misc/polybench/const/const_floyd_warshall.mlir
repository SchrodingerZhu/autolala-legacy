module attributes { "simulation.dims" = "ARRAY_0:500x500" } {
  func.func @kernel_floyd_warshall(%path: memref<500x500xf32>) {
    
    // Triple nested loop: k, i, j all from 0 to 500
    affine.for %k = 0 to 500 {
      affine.for %i = 0 to 500 {
        affine.for %j = 0 to 500 {
          // Load path[i][j], path[i][k], path[k][j]
          %path_ij = affine.load %path[%i, %j] : memref<500x500xf32>
          %path_ik = affine.load %path[%i, %k] : memref<500x500xf32>
          %path_kj = affine.load %path[%k, %j] : memref<500x500xf32>
          
          // Compute path[i][k] + path[k][j]
          %sum = arith.addf %path_ik, %path_kj : f32
          
          // Compare: path[i][j] < path[i][k] + path[k][j]
          %cond = arith.cmpf olt, %path_ij, %sum : f32
          
          // Select minimum: path[i][j] < sum ? path[i][j] : sum
          %min_val = arith.select %cond, %path_ij, %sum : f32
          
          // Store result back to path[i][j]
          affine.store %min_val, %path[%i, %j] : memref<500x500xf32>
        }
      }
    }
    
    return
  }
}
