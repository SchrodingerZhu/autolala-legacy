module attributes { "simulation.dims" = "ARRAY_0:16x16;ARRAY_1:512x512;ARRAY_2:497x497" } {
  func.func @conv2d_kernel(%input: memref<512x512xf64>, %filter: memref<16x16xf64>, %output: memref<497x497xf64>) {
    // Loop over the output matrix dimensions (497x497)
    affine.for %i = 0 to 497 {
      affine.for %j = 0 to 497 {
        // Use affine.parallel to accumulate values into %acc using iter_args
        %zero = arith.constant 0.0 : f64
        %acc = affine.for %fi = 0 to 16 iter_args(%acc = %zero) -> (f64) {
          %acc_inner = affine.for %fj = 0 to 16 iter_args(%acc_inner = %acc) -> (f64) {
            // Load filter value
            %filter_val = affine.load %filter[%fi, %fj] : memref<16x16xf64>

            // Load corresponding input value from the input matrix
            %input_val = affine.load %input[%i + %fi, %j + %fj] : memref<512x512xf64>

            // Multiply input value with filter value
            %prod = arith.mulf %input_val, %filter_val : f64

            // Add product to the accumulator
            %new_acc = arith.addf %acc_inner, %prod : f64
            affine.yield %new_acc : f64
          }
          affine.yield %acc_inner : f64
        }

        // Store the accumulated result in the output matrix
        affine.store %acc, %output[%i, %j] : memref<497x497xf64>
      }
    } { slap.extract }
    return
  }
}
