module
attributes { "simulation.dims" = "ARRAY_0:240;ARRAY_1:62400;ARRAY_2:240;ARRAY_3:57600" }
{
  func.func @kernel_correlation(
      %data   : memref<62400xf64>,   // 260 × 240
      %corr   : memref<57600xf64>,   // 240 × 240
      %mean   : memref<240xf64>,
      %stddev : memref<240xf64>,
      %float_n: f64) {

    affine.for %dummy = 0 to 1 {
      %c0  = arith.constant 0.0 : f64
      %c1  = arith.constant 1.0 : f64
      %eps = arith.constant 0.1 : f64

      // ───── Step 1: mean[j] = Σ_i data[i,j] / float_n ─────
      affine.for %j = 0 to 240 {
        affine.store %c0, %mean[%j] : memref<240xf64>
        affine.for %i = 0 to 260 {
          %sum = affine.load %mean[%j]           : memref<240xf64>
          %val = affine.load %data[%i * 240 + %j] : memref<62400xf64>
          %new = arith.addf %sum, %val : f64
          affine.store %new, %mean[%j] : memref<240xf64>
        }
        %tot  = affine.load %mean[%j] : memref<240xf64>
        %norm = arith.divf %tot, %float_n : f64
        affine.store %norm, %mean[%j] : memref<240xf64>
      }

      // ───── Step 2: stddev[j] = √( Σ_i (data[i,j]−mean[j])² / float_n ) ─────
      affine.for %j = 0 to 240 {
        affine.store %c0, %stddev[%j] : memref<240xf64>
        affine.for %i = 0 to 260 {
          %acc   = affine.load %stddev[%j] : memref<240xf64>
          %val   = affine.load %data[%i * 240 + %j] : memref<62400xf64>
          %meanj = affine.load %mean[%j] : memref<240xf64>
          %diff  = arith.subf %val, %meanj : f64
          %sq    = arith.mulf %diff, %diff : f64
          %new   = arith.addf %acc, %sq : f64
          affine.store %new, %stddev[%j] : memref<240xf64>
        }
        %var   = affine.load %stddev[%j] : memref<240xf64>
        %varn  = arith.divf %var, %float_n : f64
        %sd    = math.sqrt %varn : f64
        %cmp   = arith.cmpf ole, %sd, %eps : f64
        %sd_ok = arith.select %cmp, %c1, %sd : f64
        affine.store %sd_ok, %stddev[%j] : memref<240xf64>
      }

      // ───── Step 3: center & scale data ─────
      %sqrtN = math.sqrt %float_n : f64
      affine.for %i = 0 to 260 {
        affine.for %j = 0 to 240 {
          %val   = affine.load %data[%i * 240 + %j] : memref<62400xf64>
          %meanj = affine.load %mean[%j] : memref<240xf64>
          %sdj   = affine.load %stddev[%j] : memref<240xf64>
          %cent  = arith.subf %val, %meanj : f64
          %den   = arith.mulf %sqrtN, %sdj : f64
          %norm  = arith.divf %cent, %den : f64
          affine.store %norm, %data[%i * 240 + %j] : memref<62400xf64>
        }
      }

      // ───── Step 4: correlation matrix ─────
      affine.for %i = 0 to 239 {
        affine.store %c1, %corr[%i * 240 + %i] : memref<57600xf64>

        affine.for %j = affine_map<(d0)->(d0 + 1)>(%i) to 240 {
          affine.store %c0, %corr[%i * 240 + %j] : memref<57600xf64>
          affine.for %k = 0 to 260 {
            %sum = affine.load %corr[%i * 240 + %j] : memref<57600xf64>
            %a   = affine.load %data[%k * 240 + %i] : memref<62400xf64>
            %b   = affine.load %data[%k * 240 + %j] : memref<62400xf64>
            %p   = arith.mulf %a, %b : f64
            %new = arith.addf %sum, %p : f64
            affine.store %new, %corr[%i * 240 + %j] : memref<57600xf64>
          }
          %val = affine.load %corr[%i * 240 + %j] : memref<57600xf64>
          affine.store %val, %corr[%j * 240 + %i] : memref<57600xf64>
        }
      }
      affine.store %c1, %corr[239 * 240 + 239] : memref<57600xf64>
    }
    return
  }
}
