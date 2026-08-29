module attributes {"simulation.dims" = "ARRAY_0:200x240;ARRAY_1:200x200"}
{
    func.func @kernel_trmm(%A: memref<200x200xf64>, %B: memref<200x240xf64>) {
            %alpha = arith.constant 1.0 : f64
            
            // for (i = 0; i < 200; i++)
            affine.for %i = 0 to 200 {
                // for (j = 0; j < 240; j++)
                affine.for %j = 0 to 240 {
                    // for (k = i+1; k < 200; k++) - using affine_map for dependent loop
                    affine.for %k = affine_map<(d0) -> (d0 + 1)> (%i) to 200 {
                        // B[i][j] += A[k][i] * B[k][j];
                        %B_ij = affine.load %B[%i, %j] : memref<200x240xf64>
                        %A_ki = affine.load %A[%k, %i] : memref<200x200xf64>
                        %B_kj = affine.load %B[%k, %j] : memref<200x240xf64>
                        %prod = arith.mulf %A_ki, %B_kj : f64
                        %new_B_ij = arith.addf %B_ij, %prod : f64
                        affine.store %new_B_ij, %B[%i, %j] : memref<200x240xf64>
                    }
                    
                    // B[i][j] = alpha * B[i][j];
                    %B_ij_final = affine.load %B[%i, %j] : memref<200x240xf64>
                    %result = arith.mulf %alpha, %B_ij_final : f64
                    affine.store %result, %B[%i, %j] : memref<200x240xf64>
                }
            }
            func.return
        }
        
}
