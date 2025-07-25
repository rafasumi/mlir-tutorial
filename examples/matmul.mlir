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
  llvm.mlir.global private constant @float_fmt("%f\20\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @endl("\n") {addr_space = 0 : i32}

  llvm.func @printf(!llvm.ptr, ...) -> i32

  func.func @matmul_generic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = linalg.generic #matmul_trait
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  func.func @dump(%M: tensor<?x?xf32>) {
    %num_fmt = llvm.mlir.addressof @float_fmt : !llvm.ptr
    %endl_str = llvm.mlir.addressof @endl : !llvm.ptr

    %0 = index.constant 0
    %step = index.constant 1
    %m = index.constant 2
    %n = index.constant 2
    scf.for %i = %0 to %m step %step {
      scf.for %j = %0 to %n step %step {
        %el = tensor.extract %M[%i, %j] : tensor<?x?xf32>
        %double = arith.extf %el : f32 to f64
        llvm.call @printf(%num_fmt, %double) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      }
      llvm.call @printf(%endl_str) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    return
  }

  func.func @main() {
    %a_const = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b_const = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    %c_const = arith.constant dense<0.0> : tensor<2x2xf32>

    %a_dyn = tensor.cast %a_const : tensor<2x2xf32> to tensor<?x?xf32>
    %b_dyn = tensor.cast %b_const : tensor<2x2xf32> to tensor<?x?xf32>
    %c_dyn = tensor.cast %c_const : tensor<2x2xf32> to tensor<?x?xf32>

    %result = func.call @matmul_generic(%a_dyn, %b_dyn, %c_dyn) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

    func.call @dump(%result) : (tensor<?x?xf32>) -> ()

    return
  }
}
