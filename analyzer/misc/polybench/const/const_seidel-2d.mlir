module attributes { "simulation.dims" = "ARRAY_0:400x400"} 
{
    func.func @kernel_seidel_2d(%A: memref<400x400xf64>) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %divisor = arith.constant 9.0 : f64
        
        // for (t = 0; t <= 99; t++)
        affine.for %t = 0 to 100 {
            // for (i = 1; i <= 398; i++)
            affine.for %i = 1 to 399 {
                // for (j = 1; j <= 398; j++)
                affine.for %j = 1 to 399 {
                    // Load all 9 neighboring values
                    %A_i_minus_1_j_minus_1 = affine.load %A[%i - 1, %j - 1] : memref<400x400xf64>
                    %A_i_minus_1_j = affine.load %A[%i - 1, %j] : memref<400x400xf64>
                    %A_i_minus_1_j_plus_1 = affine.load %A[%i - 1, %j + 1] : memref<400x400xf64>
                    %A_i_j_minus_1 = affine.load %A[%i, %j - 1] : memref<400x400xf64>
                    %A_i_j = affine.load %A[%i, %j] : memref<400x400xf64>
                    %A_i_j_plus_1 = affine.load %A[%i, %j + 1] : memref<400x400xf64>
                    %A_i_plus_1_j_minus_1 = affine.load %A[%i + 1, %j - 1] : memref<400x400xf64>
                    %A_i_plus_1_j = affine.load %A[%i + 1, %j] : memref<400x400xf64>
                    %A_i_plus_1_j_plus_1 = affine.load %A[%i + 1, %j + 1] : memref<400x400xf64>
                    
                    // Sum all 9 values
                    %sum1 = arith.addf %A_i_minus_1_j_minus_1, %A_i_minus_1_j : f64
                    %sum2 = arith.addf %sum1, %A_i_minus_1_j_plus_1 : f64
                    %sum3 = arith.addf %sum2, %A_i_j_minus_1 : f64
                    %sum4 = arith.addf %sum3, %A_i_j : f64
                    %sum5 = arith.addf %sum4, %A_i_j_plus_1 : f64
                    %sum6 = arith.addf %sum5, %A_i_plus_1_j_minus_1 : f64
                    %sum7 = arith.addf %sum6, %A_i_plus_1_j : f64
                    %sum8 = arith.addf %sum7, %A_i_plus_1_j_plus_1 : f64
                    
                    // Divide by 9.0
                    %result = arith.divf %sum8, %divisor : f64
                    
                    // Store the result
                    affine.store %result, %A[%i, %j] : memref<400x400xf64>
                }
            }
        } { slap.extract }
        
        return
    }
}
