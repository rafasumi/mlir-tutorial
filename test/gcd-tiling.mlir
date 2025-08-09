// RUN: ./build/bin/sblp-opt --gcd-tiling %s | FileCheck %s

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

  // This function can be tiled because all tensors have static shapes.
  func.func @tileable_matmul(%A: tensor<1024x256xf32>, %B: tensor<256x1024xf32>, %C: tensor<1024x1024xf32>) {
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
  // CHECK:     func.func @tileable_matmul
  // CHECK:       %[[C:.*]] = arith.constant 0
  // CHECK-NEXT:  %[[C:.*]] = arith.constant 1024
  // CHECK-NEXT:  %[[C:.*]] = arith.constant 256
  // CHECK-NEXT:  %[[C:.*]] = arith.constant 0
  // CHECK-NEXT:  %[[C:.*]] = arith.constant 1024
  // CHECK-NEXT:  %[[C:.*]] = arith.constant 256
  // CHECK-NEXT:  scf.for
  // CHECK-NEXT:    scf.for
  // CHECK:           scf.yield
  // CHECK:         scf.yield
  // CHECK:       return

  // This function cannot be tiled because it uses tensors with dynamic shapes.
  func.func @untileable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) {
    %result = linalg.generic #matmul_trait
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return
  }
  // CHECK:     func.func @untileable_matmul
  // CHECK-NOT:   scf.for
  // CHECK-NOT:     scf.for
  // CHECK-NOT:       scf.yield
  // CHECK-NOT:     scf.yield
  // CHECK:       return
}
