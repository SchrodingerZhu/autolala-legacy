module attributes { "simulation.dims" = "ARRAY_0:390;ARRAY_1:390x410;ARRAY_2:410;ARRAY_3:410" } {
  func.func @kernel_atax(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %tmp: memref<?xf64>) {
   affine.for %loop_once = 0 to 1 {
      %c0 = arith.constant 0.0 : f64

      // for (i = 0; i < 390; i++)
      affine.for %i = 0 to 390 {
        // tmp[i] = 0.0;
        affine.store %c0, %tmp[%i] : memref<?xf64>

        // for (j = 0; j < 410; j++)
        //   tmp[i] = tmp[i] + A[i][j] * x[j];
        affine.for %j = 0 to 410 {
          %tmp_i = affine.load %tmp[%i] : memref<?xf64>
          %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
          %x_j = affine.load %x[%j] : memref<?xf64>
          %prod = arith.mulf %A_ij, %x_j : f64
          %new_tmp_i = arith.addf %tmp_i, %prod : f64
          affine.store %new_tmp_i, %tmp[%i] : memref<?xf64>
        }

        // for (j = 0; j < 410; j++)
        //   y[j] = y[j] + A[i][j] * tmp[i];
        affine.for %j = 0 to 410 {
          %y_j = affine.load %y[%j] : memref<?xf64>
          %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
          %tmp_i = affine.load %tmp[%i] : memref<?xf64>
          %prod = arith.mulf %A_ij, %tmp_i : f64
          %new_y_j = arith.addf %y_j, %prod : f64
          affine.store %new_y_j, %y[%j] : memref<?xf64>
        }
      } 
   }
    return
  }
}
