#include "Transforms/Pipeline.h"
#include "Transforms/StrengthReduction.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  sblp::registerStrengthReductionPass();
  sblp::registerLoopOptimizationPipeline();
  sblp::registerMLIRToLLVMPipeline();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return failed(mlir::MlirOptMain(argc, argv, "SBLP 2025 MLIR Tutorial test\n",
                                  registry));
}
