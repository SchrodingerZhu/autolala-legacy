module attributes { "simulation.dims" = "ARRAY_0:200x220;ARRAY_1:200x240;ARRAY_2:240x220" }{
  func.func @kernel_gemm(%C: memref<200x220xf32>, 
                         %A: memref<200x240xf32>, 
                         %B: memref<240x220xf32>) {
    %alpha = arith.constant 1.0 : f32
    %beta = arith.constant 1.0 : f32
    // Main computation: C := alpha*A*B + beta*C
    affine.for %i = 0 to 200 {
      
      // First inner loop: C[i][j] *= beta
      affine.for %j = 0 to 220 {
        %0 = affine.load %C[%i, %j] : memref<200x220xf32>
        %1 = arith.mulf %0, %beta : f32
        affine.store %1, %C[%i, %j] : memref<200x220xf32>
      }
      
      // Second inner loop nest: C[i][j] += alpha * A[i][k] * B[k][j]
      affine.for %k = 0 to 240 {
        affine.for %j = 0 to 220 {
          %2 = affine.load %C[%i, %j] : memref<200x220xf32>
          %3 = affine.load %A[%i, %k] : memref<200x240xf32>
          %4 = affine.load %B[%k, %j] : memref<240x220xf32>
          %5 = arith.mulf %alpha, %3 : f32
          %6 = arith.mulf %5, %4 : f32
          %7 = arith.addf %2, %6 : f32
          affine.store %7, %C[%i, %j] : memref<200x220xf32>
        }
      }
    }
     { slap.extract }
    return
  }
}
