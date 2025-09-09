#!/bin/bash

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

./build/bin/sblp-opt examples/matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --expand-strided-metadata \
  --lower-affine \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts | \
mlir-runner -e main -entry-point-result=void -shared-libs=$LLVM_BUILD_DIR/lib/libmlir_runner_utils.so
