// RUN: sblp-opt --optimize-loops %s | FileCheck %s

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

#matmul_trait = {
  doc = "C(m, n) += A(m, k) * B(k, n)",
  indexing_maps = #matmul_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @matmul(%A: tensor<1024x256xf32>, %B: tensor<256x1024xf32>, %C: tensor<1024x1024xf32>) {
    %result = linalg.generic #matmul_trait
      ins(%A, %B : tensor<1024x256xf32>, tensor<256x1024xf32>)
      outs(%C : tensor<1024x1024xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<1024x1024xf32>
    return
  }
  // CHECK:      func.func @matmul(
  // CHECK-SAME:       %[[arg0:[^:]+]]: memref<1024x256xf32, strided<[?, ?], offset: ?>>,
  // CHECK-SAME:       %[[arg1:[^:]+]]: memref<256x1024xf32, strided<[?, ?], offset: ?>>,
  // CHECK-SAME:       %[[arg2:[^:]+]]: memref<1024x1024xf32, strided<[?, ?], offset: ?>>) {
  // CHECK:        scf.for
  // CHECK:          scf.for
  // CHECK:           memref.subview
  // CHECK:           scf.for
  // CHECK:             scf.for
  // CHECK:               scf.for
  // CHECK:                 memref.load
  // CHECK:                 arith.mulf
  // CHECK:                 arith.addf
  // CHECK:        return
}
