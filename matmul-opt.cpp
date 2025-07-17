#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::arith::ArithDialect>();

  return failed(mlir::MlirOptMain(argc, argv, "SBLP 2025 MLIR Tutorial test\n",
                                  registry));
}
