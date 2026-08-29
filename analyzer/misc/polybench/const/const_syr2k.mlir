module attributes {"simulation.dims" = "ARRAY_0:240x240;ARRAY_1:240x200;ARRAY_2:240x200"} {
    func.func @kernel_syr2k(%C: memref<240x240xf64>, %A: memref<240x200xf64>, %B: memref<240x200xf64>) {
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
                    // C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
                    %C_ij = affine.load %C[%i, %j] : memref<240x240xf64>
                    %A_jk = affine.load %A[%j, %k] : memref<240x200xf64>
                    %B_ik = affine.load %B[%i, %k] : memref<240x200xf64>
                    %B_jk = affine.load %B[%j, %k] : memref<240x200xf64>
                    %A_ik = affine.load %A[%i, %k] : memref<240x200xf64>
                    
                    // First term: A[j][k]*alpha*B[i][k]
                    %prod1 = arith.mulf %A_jk, %alpha : f64
                    %term1 = arith.mulf %prod1, %B_ik : f64
                    
                    // Second term: B[j][k]*alpha*A[i][k]
                    %prod2 = arith.mulf %B_jk, %alpha : f64
                    %term2 = arith.mulf %prod2, %A_ik : f64
                    
                    // Sum both terms
                    %sum_terms = arith.addf %term1, %term2 : f64
                    %new_C_ij = arith.addf %C_ij, %sum_terms : f64
                    affine.store %new_C_ij, %C[%i, %j] : memref<240x240xf64>
                }
            }
        } { slap.extract }
        
        return
    }
}
