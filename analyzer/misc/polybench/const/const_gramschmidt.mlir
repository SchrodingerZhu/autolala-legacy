module attributes { "simulation.dims" = "ARRAY_0:200x240;ARRAY_1:240x240;ARRAY_2:200x240" } {
  func.func @gramschmidt(%A: memref<?x?xf64>,
                         %R: memref<?x?xf64>,
                         %Q: memref<?x?xf64>) {
    // 常量 0.0f
    %cst0 = arith.constant 0.0 : f64

    // k 循环：列索引
    affine.for %k = 0 to 240 {
      // ─────── 1. 计算列范数 nrm = Σ_i A[i,k]² ───────
      %nrm = affine.for %i = 0 to 128 iter_args(%acc = %cst0) -> (f64) {
        %Aik = affine.load %A[%i, %k] : memref<?x?xf64>
        %sq  = arith.mulf %Aik, %Aik : f64
        %sum = arith.addf %acc, %sq  : f64
        affine.yield %sum : f64
      }
      // R[k,k] = sqrt(nrm)
      %Rkk = math.sqrt %nrm : f64
      affine.store %Rkk, %R[%k, %k] : memref<?x?xf64>

      // ─────── 2. 归一化得到 Q[:,k] ───────
      affine.for %i = 0 to 200 {
        %Aik = affine.load %A[%i, %k] : memref<?x?xf64>
        %Qik = arith.divf %Aik, %Rkk   : f64
        affine.store %Qik, %Q[%i, %k] : memref<?x?xf64>
      }

      // ─────── 3. 处理后续列 j = k+1 … N-1 ───────
      affine.for %j = affine_map<(d0)->(d0 + 1)>(%k) to 240 {
        // 3-a) R[k,j] = Σ_i Q[i,k] * A[i,j]
        %Rkj = affine.for %i = 0 to 200 iter_args(%acc2 = %cst0) -> (f64) {
          %Qik = affine.load %Q[%i, %k] : memref<?x?xf64>
          %Aij = affine.load %A[%i, %j] : memref<?x?xf64>
          %prd = arith.mulf %Qik, %Aij  : f64
          %sum = arith.addf %acc2, %prd : f64
          affine.yield %sum : f64
        }
        affine.store %Rkj, %R[%k, %j] : memref<?x?xf64>

        // 3-b) A[:,j] ← A[:,j] − Q[:,k] * R[k,j]
        affine.for %i = 0 to 200 {
          %Aij_old = affine.load %A[%i, %j] : memref<?x?xf64>
          %Qik     = affine.load %Q[%i, %k] : memref<?x?xf64>
          %Rkj_val = affine.load %R[%k, %j] : memref<?x?xf64>
          %prd2    = arith.mulf %Qik, %Rkj_val : f64
          %Aij_new = arith.subf %Aij_old, %prd2 : f64
          affine.store %Aij_new, %A[%i, %j] : memref<?x?xf64>
        }
      }
    }
    func.return
  }
}
