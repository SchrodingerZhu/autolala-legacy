module 
   attributes { "simulation.dims" = "ARRAY_0:400x400;ARRAY_1:400;ARRAY_2:400;ARRAY_3:400;ARRAY_4:400;ARRAY_5:400;ARRAY_6:400;ARRAY_7:400;ARRAY_8:400" } {
  func.func @kernel_gemver(%A: memref<400x400xf64>,
                           %u1: memref<400xf64>, %v1: memref<400xf64>,
                           %u2: memref<400xf64>, %v2: memref<400xf64>,
                           %w: memref<400xf64>, %x: memref<400xf64>,
                           %y: memref<400xf64>, %z: memref<400xf64>) {
    %alpha = arith.constant 1.0 : f64
    %beta = arith.constant 1.0 : f64
    affine.for %loop_once = 0 to 1 {
    // First loop nest: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    affine.for %i = 0 to 400 {
      affine.for %j = 0 to 400 {
        %0 = affine.load %A[%i, %j] : memref<400x400xf64>
        %1 = affine.load %u1[%i] : memref<400xf64>
        %2 = affine.load %v1[%j] : memref<400xf64>
        %3 = affine.load %u2[%i] : memref<400xf64>
        %4 = affine.load %v2[%j] : memref<400xf64>
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.mulf %3, %4 : f64
        %7 = arith.addf %0, %5 : f64
        %8 = arith.addf %7, %6 : f64
        affine.store %8, %A[%i, %j] : memref<400x400xf64>
      }
    }
    
    // Second loop nest: x[i] = x[i] + beta * A[j][i] * y[j]
    affine.for %i = 0 to 400 {
      affine.for %j = 0 to 400 {
        %9 = affine.load %x[%i] : memref<400xf64>
        %10 = affine.load %A[%j, %i] : memref<400x400xf64>
        %11 = affine.load %y[%j] : memref<400xf64>
        %12 = arith.mulf %beta, %10 : f64
        %13 = arith.mulf %12, %11 : f64
        %14 = arith.addf %9, %13 : f64
        affine.store %14, %x[%i] : memref<400xf64>
      }
    }
    
    // Third loop: x[i] = x[i] + z[i]
    affine.for %i = 0 to 400 {
      %15 = affine.load %x[%i] : memref<400xf64>
      %16 = affine.load %z[%i] : memref<400xf64>
      %17 = arith.addf %15, %16 : f64
      affine.store %17, %x[%i] : memref<400xf64>
    }
    
    // Fourth loop nest: w[i] = w[i] + alpha * A[i][j] * x[j]
    affine.for %i = 0 to 400 {
      affine.for %j = 0 to 400 {
        %18 = affine.load %w[%i] : memref<400xf64>
        %19 = affine.load %A[%i, %j] : memref<400x400xf64>
        %20 = affine.load %x[%j] : memref<400xf64>
        %21 = arith.mulf %alpha, %19 : f64
        %22 = arith.mulf %21, %20 : f64
        %23 = arith.addf %18, %22 : f64
        affine.store %23, %w[%i] : memref<400xf64>
      }
    }
    }{ slap.extract }
    return
  }
}
