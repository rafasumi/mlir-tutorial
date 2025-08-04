#!/bin/bash

ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")/.."
TMP_DIR="$ROOT_DIR/tmp"
BUILD_DIR="$ROOT_DIR/build"
EXAMPLES_DIR="$ROOT_DIR/examples"

gen_dylib () {
    $BUILD_DIR/bin/sblp-opt --optimize-matmul="$1" $EXAMPLES_DIR/matmul_lib.mlir -o $TMP_DIR/matmul.llvm.mlir
    mlir-translate $TMP_DIR/matmul.llvm.mlir -mlir-to-llvmir -o $TMP_DIR/matmul.ll
    llc -filetype=obj --relocation-model=pic $TMP_DIR/matmul.ll -o $TMP_DIR/matmul.o
    clang -shared -o $TMP_DIR/libmatmul$2.dylib $TMP_DIR/matmul.o
}

gen_dylib "" baseline
gen_dylib "enable-loop-unrolling" unroll
