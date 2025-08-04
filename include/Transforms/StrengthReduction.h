#ifndef SBLP_INCLUDE_TRANSFORMS_STRENGTHREDUCTION_H
#define SBLP_INCLUDE_TRANSFORMS_STRENGTHREDUCTION_H

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
    return "Replace arith.muli operation with shift left when applicable.";
  }
};

void registerStrengthReductionPass();

} // namespace sblp

#endif // SBLP_INCLUDE_TRANSFORMS_STRENGTHREDUCTION_H
