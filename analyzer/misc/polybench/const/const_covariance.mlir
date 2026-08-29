module 
  attributes {
    "simulation.dims" = "ARRAY_0:256;ARRAY_1:65536;ARRAY_2:65536"
  }
{
  func.func @kernel_covariance(
      %data     : memref<62560xf64>,   // 256*256
      %cov      : memref<57600xf64>,   // 256*256
      %mean     : memref<256xf64>,
      %float_n  : f64) {

    %c0   = arith.constant 0.0 : f64
    %c1   = arith.constant 1.0 : f64

    affine.for %loop_once = 0 to 1 {

      // ── Step 1: mean[j] = Σ_i data[i,j] / float_n ──
      affine.for %j = 0 to 256 {
        affine.store %c0, %mean[%j] : memref<256xf64>

        affine.for %i = 0 to 256 {
          %sum = affine.load %mean[%j] : memref<256xf64>
          %val = affine.load %data[%i * 256 + %j] : memref<62560xf64>
          %new = arith.addf %sum, %val : f64
          affine.store %new, %mean[%j] : memref<256xf64>
        }

        %tot  = affine.load %mean[%j] : memref<256xf64>
        %norm = arith.divf %tot, %float_n : f64
        affine.store %norm, %mean[%j] : memref<256xf64>
      }

      // ── Step 2: center data ──
      affine.for %i = 0 to 256 {
        affine.for %j = 0 to 256 {
          %val = affine.load %data[%i * 256 + %j] : memref<62560xf64>
          %m   = affine.load %mean[%j] : memref<256xf64>
          %c   = arith.subf %val, %m : f64
          affine.store %c, %data[%i * 256 + %j] : memref<62560xf64>
        }
      }

      // ── Step 3: covariance ──
      %fnm1 = arith.subf %float_n, %c1 : f64

      affine.for %i = 0 to 256 {
        affine.for %j = affine_map<(d0)->(d0)>(%i) to 256 {

          // cov[i,j] 归零
          affine.store %c0, %cov[%i * 256 + %j] : memref<57600xf64>

          affine.for %k = 0 to 256 {
            %a = affine.load %data[%k * 256 + %i] : memref<62560xf64>
            %b = affine.load %data[%k * 256 + %j] : memref<62560xf64>
            %p = arith.mulf %a, %b : f64

            %sum = affine.load %cov[%i * 256 + %j] : memref<57600xf64>
            %new = arith.addf %sum, %p : f64
            affine.store %new, %cov[%i * 256 + %j] : memref<57600xf64>
          }

          %acc = affine.load %cov[%i * 256 + %j] : memref<57600xf64>
          %nrm = arith.divf %acc, %fnm1 : f64
          affine.store %nrm, %cov[%i * 256 + %j] : memref<57600xf64>

          // 对称赋值 cov[j,i] = cov[i,j]
          affine.store %nrm, %cov[%j * 256 + %i] : memref<57600xf64>
        }
      }
    }
    return
  }
}
