// Shared data that no thread reuses on its own.
//
// A[i][j] is read as the centre of row i and as the neighbour of row i+1, so
// with chunk-1 scheduling exactly two threads touch it and *neither* touches it
// twice. The thread-local reuse interval therefore does not exist, while the
// sequential one does (2*M, two accesses per iteration over M columns).
//
// This is what separates "the racetrack consumes a private reuse interval" from
// "the racetrack consumes a sequential reuse interval". In the kernels where
// both are defined they coincide, so only a case like this can tell them apart.
module {
  func.func @shared_no_private(%A: memref<?x?xf64>) {
    affine.for %i = 1 to 257 {
      affine.for %j = 0 to 256 {
        %0 = affine.load %A[%i, %j] : memref<?x?xf64>
        %1 = affine.load %A[%i - 1, %j] : memref<?x?xf64>
      }
    }
    return
  }
}
