#include "Transforms/StrengthReduction.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace sblp {

struct SimplifyPowerOp : public OpRewritePattern<math::IPowIOp> {
  SimplifyPowerOp(MLIRContext *context)
      : OpRewritePattern<math::IPowIOp>(context) {}

  llvm::LogicalResult
  matchAndRewrite(math::IPowIOp op, PatternRewriter &rewriter) const override {
    // Check if either operand is a constant *and* is a power of two

    // Replace op with arith::MulIOp

    return llvm::success();
  }
};

void StrengthReductionPass::runOnOperation() {}

void registerStrengthReductionPass() {
  mlir::PassRegistration<StrengthReductionPass>();
}

} // namespace sblp
