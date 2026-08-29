module attributes { "simulation.dims" = "ARRAY_0:390;ARRAY_1:390;ARRAY_2:410;ARRAY_3:410x390;ARRAY_4:410" } {
    // Constants
    // Function definition
    func.func @bicg(%argA: memref<410x390xf64>, %argS: memref<390xf64>, %argR: memref<410xf64>, %argQ: memref<390xf64>, %argP: memref<410xf64>) {
        %c0 = arith.constant 0.0 : f64
        // Outer loop: for (i = 0; i < N; i++)
        affine.for %i = 0 to 410 {
            // Initialize q[i] to 0
            affine.store %c0, %argQ[%i] : memref<390xf64>
            // Inner loop: for (j = 0; j < M; j++)
            affine.for %j = 0 to 390 {
                // s[j] = s[j] + r[i] * A[i][j];
                %s_j = affine.load %argS[%j] : memref<390xf64>
                %r_i = affine.load %argR[%i] : memref<410xf64>
                %a_ij = affine.load %argA[%i, %j] : memref<410x390xf64>
                %prod_r_a = arith.mulf %r_i, %a_ij : f64
                %new_s_j = arith.addf %s_j, %prod_r_a : f64
                affine.store %new_s_j, %argS[%j] : memref<390xf64>
                // q[i] = q[i] + A[i][j] * p[j];
                %q_i = affine.load %argQ[%i] : memref<390xf64>
                %p_j = affine.load %argP[%j] : memref<410xf64>
                %prod_a_p = arith.mulf %a_ij, %p_j : f64
                %new_q_i = arith.addf %q_i, %prod_a_p : f64
                affine.store %new_q_i, %argQ[%i] : memref<390xf64>
            }
        } { slap.extract }
    return
    }
}
