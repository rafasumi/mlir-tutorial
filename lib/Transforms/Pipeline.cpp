#include "Transforms/Pipeline.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
struct MatmulOptimizationOptions
    : public PassPipelineOptions<MatmulOptimizationOptions> {
  Option<bool> enableUnrolling{
      *this, "enable-loop-unrolling",
      llvm::cl::desc("Whether to use the loop unrolling optimization"),
      llvm::cl::init(false)};
};

void addMatmulOptimizationPipeline(OpPassManager &pm,
                                   const MatmulOptimizationOptions &options) {
  // Bufferization (tensor->memref)
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));

  // Lowering linalg dialect
  if (options.enableUnrolling) {
    // Use affine optimizations
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    // Unrolling is one of many affine optimizations that could be applied
    pm.addPass(affine::createLoopUnrollAndJamPass());
    pm.addPass(createLowerAffinePass());
  } else {
    // Convert linalg straight to SCF
    pm.addPass(createConvertLinalgToLoopsPass());
  }

  // Lowering to LLVM
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}
} // namespace

namespace sblp {

void registerMatmulOptimizationPipeline() {
  PassPipelineRegistration<MatmulOptimizationOptions>(
      "optimize-matmul", "Optimize simple matrix multiplication",
      addMatmulOptimizationPipeline);
}

} // namespace sblp
