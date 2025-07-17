#!/usr/bin/env bash

LLVM_BUILD_DIR="$HOME/llvm-20.1.4/build"

mkdir -p build
BUILD_DIR=$(realpath -L ./build)

CC=$LLVM_BUILD_DIR/bin/clang
CXX=$LLVM_BUILD_DIR/bin/clang++

cmake -S . -B $BUILD_DIR -G "Ninja"               \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm   \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir   \
    -DCMAKE_C_COMPILER=$CC                        \
    -DCMAKE_CXX_COMPILER=$CXX                     \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .

ninja -j$(nproc) -C $BUILD_DIR