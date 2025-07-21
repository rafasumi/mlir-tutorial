#ifndef SBLP_INCLUDE_TRANSFORM_STRENGTHREDUCTION_H
#define SBLP_INCLUDE_TRANSFORM_STRENGTHREDUCTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace sblp {

class StrengthReductionPass
    : public mlir::PassWrapper<StrengthReductionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "strength-reduction"; }

  llvm::StringRef getDescription() const final {
    return "Replace math.ipowi operation when it uses powers of 2.";
  }
};

void registerStrengthReductionPass();

} // namespace sblp

#endif // LIB_TRANSFORM_ARITH_STRENGTHREDUCTION_H
