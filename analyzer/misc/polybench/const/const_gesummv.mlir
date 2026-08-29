module 
 attributes { "simulation.dims" = "ARRAY_0:256;ARRAY_1:256;ARRAY_2:256x256;ARRAY_3:256;ARRAY_4:256x256" }
{
  func.func @kernel_gesummv( %A: memref<256x256xf64>,
                            %B: memref<256x256xf64>,
                            %tmp: memref<256xf64>,
                            %x: memref<256xf64>,
                            %y: memref<256xf64>) {
    %c0_f64 = arith.constant 0.0 : f64
    %alpha = arith.constant 1.0 : f64
    %beta = arith.constant 1.0 : f64
    affine.for %i = 0 to 256 {
      // Initialize tmp[i] = 0.0 and y[i] = 0.0
      affine.store %c0_f64, %tmp[%i] : memref<256xf64>
      affine.store %c0_f64, %y[%i] : memref<256xf64>
      
      // Inner loop: compute tmp[i] and y[i] accumulations
      affine.for %j = 0 to 256 {
        // tmp[i] = A[i][j] * x[j] + tmp[i]
        %0 = affine.load %tmp[%i] : memref<256xf64>
        %1 = affine.load %A[%i, %j] : memref<256x256xf64>
        %2 = affine.load %x[%j] : memref<256xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %3, %0 : f64
        affine.store %4, %tmp[%i] : memref<256xf64>
        
        // y[i] = B[i][j] * x[j] + y[i]
        %5 = affine.load %y[%i] : memref<256xf64>
        %6 = affine.load %B[%i, %j] : memref<256x256xf64>
        %7 = affine.load %x[%j] : memref<256xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %8, %5 : f64
        affine.store %9, %y[%i] : memref<256xf64>
      }
      
      // Final computation: y[i] = alpha * tmp[i] + beta * y[i]
      %10 = affine.load %tmp[%i] : memref<256xf64>
      %11 = affine.load %y[%i] : memref<256xf64>
      %12 = arith.mulf %alpha, %10 : f64
      %13 = arith.mulf %beta, %11 : f64
      %14 = arith.addf %12, %13 : f64
      affine.store %14, %y[%i] : memref<256xf64>
    }
     { slap.extract }
    return
  }
}
