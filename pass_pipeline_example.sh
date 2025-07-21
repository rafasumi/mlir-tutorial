# TODO: Implement this as a standalone pass pipeline in C++
./build/matmul-opt examples/matmul.mlir \
--one-shot-bufferize="bufferize-function-boundaries" \
--convert-linalg-to-loops \
--convert-scf-to-cf \
--convert-cf-to-llvm \
--convert-arith-to-llvm \
--convert-func-to-llvm \
--convert-index-to-llvm \
--finalize-memref-to-llvm \
--reconcile-unrealized-casts | \
mlir-runner -e main -entry-point-result=void -shared-libs=/home/rafael/llvm-20.1.4/build/lib/libmlir_runner_utils.so

# Observations:
#   - This pass pipeline does not include optimizations, we would have to add them.
#   - --convert-linalg-to-loops can be replaced with --convert-linalg-to-affine-loops to use affine before scf.
#     The lowering will be a bit different, but it will allow for using affine optimizations (which I think is
#     desired).
#   - We can compile the MLIR file to a .so shared library file, instead of running it directly. That way, we
#     don't need a main function in the matmul.mlir example. Instead we can use a Python file as a "driver" to
#     call the generated function (and maybe even run some benchmarks if we're feeling bold).
