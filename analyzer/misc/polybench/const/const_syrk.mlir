module attributes {"simulation.dims" = "ARRAY_0:240x240;ARRAY_1:240x200"} {
    func.func @kernel_syrk(%C: memref<240x240xf64>, %A: memref<240x200xf64>) {
        %alpha = arith.constant 1.0 : f64
        %beta = arith.constant 1.0 : f64
        
        // for (i = 0; i < 240; i++)
        affine.for %i = 0 to 240 {
            // for (j = 0; j <= i; j++) - using affine_map for dependent loop
            affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                // C[i][j] *= beta;
                %C_ij = affine.load %C[%i, %j] : memref<240x240xf64>
                %new_C_ij = arith.mulf %C_ij, %beta : f64
                affine.store %new_C_ij, %C[%i, %j] : memref<240x240xf64>
            }
            
            // for (k = 0; k < 200; k++)
            affine.for %k = 0 to 200 {
                // for (j = 0; j <= i; j++) - using affine_map for dependent loop
                affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)> (%i) {
                    // C[i][j] += alpha * A[i][k] * A[j][k];
                    %C_ij = affine.load %C[%i, %j] : memref<240x240xf64>
                    %A_ik = affine.load %A[%i, %k] : memref<240x200xf64>
                    %A_jk = affine.load %A[%j, %k] : memref<240x200xf64>
                    %prod1 = arith.mulf %alpha, %A_ik : f64
                    %prod2 = arith.mulf %prod1, %A_jk : f64
                    %new_C_ij = arith.addf %C_ij, %prod2 : f64
                    affine.store %new_C_ij, %C[%i, %j] : memref<240x240xf64>
                }
            }
        } { slap.extract }
        
        return
    }
}

