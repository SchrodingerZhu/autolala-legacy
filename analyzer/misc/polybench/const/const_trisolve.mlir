module attributes {"simulation.dims" = "ARRAY_0:640;ARRAY_1:640;ARRAY_2:640x640" }
 {
    func.func @kernel_trisolv(%L: memref<400x400xf64>, %x: memref<400xf64>, %b: memref<400xf64>) {
        
        // for (i = 0; i < 400; i++)
        affine.for %i = 0 to 400 {
            // x[i] = b[i];
            %b_i = affine.load %b[%i] : memref<400xf64>
            affine.store %b_i, %x[%i] : memref<400xf64>
            
            // for (j = 0; j < i; j++) - using affine_map for dependent loop
            affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i) {
                // x[i] -= L[i][j] * x[j];
                %x_i = affine.load %x[%i] : memref<400xf64>
                %L_ij = affine.load %L[%i, %j] : memref<400x400xf64>
                %x_j = affine.load %x[%j] : memref<400xf64>
                %prod = arith.mulf %L_ij, %x_j : f64
                %new_x_i = arith.subf %x_i, %prod : f64
                affine.store %new_x_i, %x[%i] : memref<400xf64>
            }
            
            // x[i] = x[i] / L[i][i];
            %x_i_final = affine.load %x[%i] : memref<400xf64>
            %L_ii = affine.load %L[%i, %i] : memref<400x400xf64>
            %result = arith.divf %x_i_final, %L_ii : f64
            affine.store %result, %x[%i] : memref<400xf64>
        } 
        
        return
    }
}
