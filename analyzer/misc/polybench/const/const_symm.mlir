module attributes {
  "simulation.dims" = "ARRAY_0:200x240;ARRAY_1:200x240;ARRAY_2:200x240"
} {
  func.func @kernel_symm(%C: memref<200x240xf64>, %A: memref<200x200xf64>, %B: memref<200x240xf64>) {
    %c0 = arith.constant 0.0 : f64
    %alpha = arith.constant 1.0 : f64
    %beta = arith.constant 1.0 : f64

    affine.for %i = 0 to 200 {
      affine.for %j = 0 to 240 {
        // temp2 carried by iter_args instead of alloca
        %temp2 = affine.for %k = 0 to affine_map<(d0) -> (d0)>(%i) iter_args(%acc = %c0) -> (f64) {
          // C[k][j] += alpha * B[i][j] * A[i][k]
          %C_kj = affine.load %C[%k, %j] : memref<200x240xf64>
          %B_ij = affine.load %B[%i, %j] : memref<200x240xf64>
          %A_ik = affine.load %A[%i, %k] : memref<200x200xf64>
          %prod1 = arith.mulf %alpha, %B_ij : f64
          %prod2 = arith.mulf %prod1, %A_ik : f64
          %new_C_kj = arith.addf %C_kj, %prod2 : f64
          affine.store %new_C_kj, %C[%k, %j] : memref<200x240xf64>

          // temp2 += B[k][j] * A[i][k]
          %B_kj = affine.load %B[%k, %j] : memref<200x240xf64>
          %prod3 = arith.mulf %B_kj, %A_ik : f64
          %acc_new = arith.addf %acc, %prod3 : f64
          affine.yield %acc_new : f64
        }

        // C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2
        %C_ij = affine.load %C[%i, %j] : memref<200x240xf64>
        %B_ij = affine.load %B[%i, %j] : memref<200x240xf64>
        %A_ii = affine.load %A[%i, %i] : memref<200x200xf64>

        %term1 = arith.mulf %beta, %C_ij : f64
        %prod4 = arith.mulf %alpha, %B_ij : f64
        %term2 = arith.mulf %prod4, %A_ii : f64
        %term3 = arith.mulf %alpha, %temp2 : f64

        %sum1 = arith.addf %term1, %term2 : f64
        %result = arith.addf %sum1, %term3 : f64

        affine.store %result, %C[%i, %j] : memref<200x240xf64>
      }
    }
    { slap.extract }
    return
  }
}
