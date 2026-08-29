module 
attributes { "simulation.dims" = "ARRAY_0:400;ARRAY_1:400x400;ARRAY_2:400;ARRAY_3:400;ARRAY_4:400" }
{
  func.func @kernel_mvt(%x1: memref<400xf64>, %x2: memref<400xf64>, 
                        %y_1: memref<400xf64>, %y_2: memref<400xf64>, 
                        %A: memref<400x400xf64>) {
    affine.for %loop_once = 0 to 1 {
    // First loop nest: x1[i] = x1[i] + A[i][j] * y_1[j]
    affine.for %i = 0 to 400 {
      affine.for %j = 0 to 400 {
        %0 = affine.load %x1[%i] : memref<400xf64>
        %1 = affine.load %A[%i, %j] : memref<400x400xf64>
        %2 = affine.load %y_1[%j] : memref<400xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %x1[%i] : memref<400xf64>
      }
    }
    
    // Second loop nest: x2[i] = x2[i] + A[j][i] * y_2[j]
    affine.for %i = 0 to 400 {
      affine.for %j = 0 to 400 {
        %5 = affine.load %x2[%i] : memref<400xf64>
        %6 = affine.load %A[%j, %i] : memref<400x400xf64>
        %7 = affine.load %y_2[%j] : memref<400xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %5, %8 : f64
        affine.store %9, %x2[%i] : memref<400xf64>
      }
    }
    } { slap.extract }
    return
  }
}
