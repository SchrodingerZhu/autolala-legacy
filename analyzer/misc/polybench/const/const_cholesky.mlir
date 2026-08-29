module attributes { "simulation.dims" = "ARRAY_0:400x400" }{
    func.func @cholesky(%A: memref<400x400xf64>) {
        affine.for %i = 0 to 400 {
            // j < i case
            affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i)  {
                affine.for %k = 0 to affine_map<(d0) -> (d0)> (%j)  {
                    // A[i][j] -= A[i][k] * A[j][k];
                    %a_ik = affine.load %A[%i, %k] : memref<400x400xf64>
                    %a_jk = affine.load %A[%j, %k] : memref<400x400xf64>
                    %temp = arith.mulf %a_ik, %a_jk : f64
                    %a_ij = affine.load %A[%i, %j] : memref<400x400xf64>
                    %new_value_ij = arith.subf %a_ij, %temp : f64
                    affine.store %new_value_ij, %A[%i, %j] : memref<400x400xf64>
                }
                // A[i][j] /= A[j][j];
                %a_jj = affine.load %A[%j, %j] : memref<400x400xf64>
                %a_ij = affine.load %A[%i, %j] : memref<400x400xf64>
                %new_value_ij_div = arith.divf %a_ij, %a_jj : f64
                affine.store %new_value_ij_div, %A[%i, %j] : memref<400x400xf64>
            }

            // i == j case
            affine.for %k = 0 to affine_map<(d0) -> (d0)> (%i)  {
                // A[i][i] -= A[i][k] * A[i][k];
                %a_ik = affine.load %A[%i, %k] : memref<400x400xf64>
                %temp = arith.mulf %a_ik, %a_ik : f64
                %a_ii = affine.load %A[%i, %i] : memref<400x400xf64>
                %new_value_ii = arith.subf %a_ii, %temp : f64
                affine.store %new_value_ii, %A[%i, %i] : memref<400x400xf64>
            }

            // A[i][i] = SQRT_FUN(A[i][i]);
            %a_ii_final = affine.load %A[%i, %i] : memref<400x400xf64>
            %sqrt_value = math.sqrt %a_ii_final : f64
            affine.store %sqrt_value, %A[%i, %i] : memref<400x400xf64>
        } { slap.extract }
        return
    } 
}
