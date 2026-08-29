module attributes { "simulation.dims" = "ARRAY_0:180x190;ARRAY_1:180x200;ARRAY_2:200x190;ARRAY_3:190x210;ARRAY_4:190x220;ARRAY_5:220x210;ARRAY_6:180x210" } {
    func.func @kernel_3mm(
        %E: memref<?x?xf64>, 
        %A: memref<?x?xf64>, 
        %B: memref<?x?xf64>, 
        %F: memref<?x?xf64>, 
        %C: memref<?x?xf64>, 
        %D: memref<?x?xf64>, 
        %G: memref<?x?xf64>
    ) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64
            
            // E := A*B
            // for (i = 0; i < ni; i++)
            affine.for %i = 0 to 180 {
                // for (j = 0; j < nj; j++)
                affine.for %j = 0 to 190 {
                    // E[i][j] = 0.0;
                    affine.store %c0, %E[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < nk; k++)
                    affine.for %k = 0 to 200 {
                        // E[i][j] += A[i][k] * B[k][j];
                        %E_ij = affine.load %E[%i, %j] : memref<?x?xf64>
                        %A_ik = affine.load %A[%i, %k] : memref<?x?xf64>
                        %B_kj = affine.load %B[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %A_ik, %B_kj : f64
                        %new_E_ij = arith.addf %E_ij, %prod : f64
                        affine.store %new_E_ij, %E[%i, %j] : memref<?x?xf64>
                    }
                }
            }
            
            // F := C*D
            // for (i = 0; i < nj; i++)
            affine.for %i = 0 to 190 {
                // for (j = 0; j < nl; j++)
                affine.for %j = 0 to 210 {
                    // F[i][j] = 0.0;
                    affine.store %c0, %F[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < nm; k++)
                    affine.for %k = 0 to 220 {
                        // F[i][j] += C[i][k] * D[k][j];
                        %F_ij = affine.load %F[%i, %j] : memref<?x?xf64>
                        %C_ik = affine.load %C[%i, %k] : memref<?x?xf64>
                        %D_kj = affine.load %D[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %C_ik, %D_kj : f64
                        %new_F_ij = arith.addf %F_ij, %prod : f64
                        affine.store %new_F_ij, %F[%i, %j] : memref<?x?xf64>
                    }
                }
            }
            
            // G := E*F
            // for (i = 0; i < ni; i++)
            affine.for %i = 0 to 180 {
                // for (j = 0; j < nl; j++)
                affine.for %j = 0 to 210 {
                    // G[i][j] = 0.0;
                    affine.store %c0, %G[%i, %j] : memref<?x?xf64>
                    
                    // for (k = 0; k < nj; k++)
                    affine.for %k = 0 to 190 {
                        // G[i][j] += E[i][k] * F[k][j];
                        %G_ij = affine.load %G[%i, %j] : memref<?x?xf64>
                        %E_ik = affine.load %E[%i, %k] : memref<?x?xf64>
                        %F_kj = affine.load %F[%k, %j] : memref<?x?xf64>
                        %prod = arith.mulf %E_ik, %F_kj : f64
                        %new_G_ij = arith.addf %G_ij, %prod : f64
                        affine.store %new_G_ij, %G[%i, %j] : memref<?x?xf64>
                    }
                }
            } 
        }
        return
    }
}
