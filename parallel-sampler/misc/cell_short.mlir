// "Short RI" row of PACT'24 Table 1 (both columns use the NB distribution).
// Each thread ping-pongs between two private elements, so PRI = 2 -- far below
// the Chernoff-Hoeffding bound of Theorem 3.1, where the CRI distribution has
// not yet concentrated on its mean and the negative binomial matters.
// The model predicts CRI = 2 + X with X ~ NB(2, 1/T).
module {
  func.func @cell_short(%A: memref<?x?xf64>) {
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 512 {
        %0 = affine.load %A[%i, 0] : memref<?x?xf64>
        %1 = affine.load %A[%i, 1] : memref<?x?xf64>
      }
    }
    return
  }
}
