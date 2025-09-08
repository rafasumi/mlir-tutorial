#ifndef SBLP_INCLUDE_TRANSFORMS_GCDTILING_H
#define SBLP_INCLUDE_TRANSFORMS_GCDTILING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

namespace sblp {

class GCDTilingPass
    : public mlir::PassWrapper<GCDTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
private:
  int64_t computeTilingSize(mlir::linalg::LinalgOp op);
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "gcd-tiling"; }

  llvm::StringRef getDescription() const final {
    return "Tile loops by a factor of the greatest common divisor between "
           "tensor dimensions.";
  }
};

void registerGCDTilingPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createGCDTilingPass();

} // namespace sblp

#endif // SBLP_INCLUDE_TRANSFORMS_GCDTILING_H
