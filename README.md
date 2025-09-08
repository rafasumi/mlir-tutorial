# MLIR: From High Level to LLVM with Neural Networks and Optimizations
This repository contains supporting material for the tutorial which was presented in the 29th Brazilian Symposium on Programming Languages (SBLP 2025).

This tutorial is a brief introduction to MLIR. You can expect to learn basic concepts, such as dialects, operations, and passes. For more in-depth information about MLIR, we recommend checking out the resources in the [official MLIR page](https://mlir.llvm.org/docs/Tutorials/).

## Contents
### [`StrengthReductionPass`](https://github.com/rafasumi/mlir-tutorial/blob/main/lib/Transforms/StrengthReduction.cpp)
This is a very simple pass which replaces multiplications by constant integer powers of two with an arithmetic shift left. In compiler construction, this kind of optimization which replaces an operation by a cheaper equivalent is called [strength reduction](https://en.wikipedia.org/wiki/Strength_reduction#Other_strength_reduction_operations).

The purpose of this pass is to demonstrate how to use MLIR's [pattern rewriting infrastructure](https://mlir.llvm.org/docs/PatternRewriter/), which is a very useful method for applying transformations based on pattern matching.

### [`GCDTilingPass`](https://github.com/rafasumi/mlir-tutorial/blob/main/lib/Transforms/GCDTiling.cpp)
This is a more involved pass, which applies [loop tiling](https://www.intel.com/content/www/us/en/developer/articles/technical/loop-optimizations-where-blocks-are-required.html) to all [`linalg`](https://mlir.llvm.org/docs/Dialects/Linalg/#rationale) operations present in the IR. The pass will compute the tile size as the greatest common divisor of the dimensions of all operands for a given `linalg` op.

This pass illustrates many useful concepts for MLIR transformations, such as IR traversal and rewriting.

### [Pipelines](https://github.com/rafasumi/mlir-tutorial/blob/main/lib/Transforms/Pipeline.cpp)
We have included two examples of pass pipelines. Pipelines are simply sequences of passes which are applied in order, using the output of a pass as the input of the subsequent one.

The `optimize-loops` pipeline applies a few optimizations and transformations that are useful for `linalg` ops, such as bufferization, loop tiling and [loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling). The `mlir-to-llvm` pipeline applies a sequence of conversions passes to lower the MLIR example with `linalg` operations to the `llvm` dialect, which can be later translated to LLVM IR. 

### [`sblp-opt`](https://github.com/rafasumi/mlir-tutorial/blob/main/tools/sblp-opt.cpp)
This tool is an extension of the [`mlir-opt`](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) binary which adds all passes and pipelines implemented for the tutorial. `*-opt` tools are usually used in MLIR-based projects as the entry point for running transformations, optimizations, and lowerings.

### [Test Infrastructure](https://github.com/rafasumi/mlir-tutorial/tree/main/test)
We have included an example of a test infrastructure which follows the [guidelines](https://llvm.org/docs/TestingGuide.html) for LLVM-based projects. These tests use `llvm-lit` and `FileCheck` to verify that `sblp-opt` outputs the IR as expected when applying different transformations. This is not crucial for learning MLIR and thus is only a complement to the tutorial. However, testing is essential to the work of a compiler enginner so it's good to have a context of how to test MLIR-based projects.

## Dependencies
We haven't tested running this code on Windows, therefore we recommend using Linux or macOS. These are the software requirements for running the code in this tutorial:

| Dependency | Version   | Installation Link                                                   |
|------------|-----------|---------------------------------------------------------------------|
| LLVM       | >= 21     | [llvm.org](https://llvm.org/docs/CMake.html)                        |
| Python     | >= 3.10   | [python.org](https://www.python.org/downloads/release/python-3100/) |
| CMake      | >= 3.20   | [cmake.org](https://cmake.org/install/)                             |
| Ninja      | >= 1.10   | [ninja-build GitHub](https://github.com/ninja-build/ninja/releases) |

When building LLVM, be sure to enable the MLIR project in the `LLVM_ENABLE_PROJECTS` CMake variable. Here are some commands that can be used:
```bash
git clone -b llvmorg-21.1.0 https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON 
ninja
```
For more information on how to build LLVM, you can check [Prof. Fernando Pereira's video](https://www.youtube.com/watch?v=l0LI_7KeFtw) on the topic.

## Build
This repository includes an utility script to easily build the MLIR tutorial project. To use it, you must set the `LLVM_BUILD_DIR` variable to the directory where you have built LLVM. Here's an example:
```bash
export LLVM_BUILD_DIR=path/to/llvm-project/build
./scripts/build.sh
```

This will output the `sblp-opt` binary in the `./build/bin` directory.

## Paper
TODO
