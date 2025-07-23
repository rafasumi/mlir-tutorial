#include "Transforms/StrengthReduction.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

namespace sblp {

struct SimplifyPowerOp : public OpRewritePattern<arith::MulIOp> {
  SimplifyPowerOp(MLIRContext *context)
      : OpRewritePattern<arith::MulIOp>(context) {}

  bool isPowerOfTwo(const llvm::APInt &c) const {
    auto val = c.getSExtValue();
    return (val & (val - 1)) == 0;
  }

  void createShiftLeft(PatternRewriter &rewriter, Operation *op, Value newLhs,
                       Value constantValue,
                       const llvm::APInt &constantOperand) const {
    unsigned shiftOffset = constantOperand.countTrailingZeros();
    auto type = constantValue.getType();
    auto attr = rewriter.getIntegerAttr(type, shiftOffset);

    auto loc = constantValue.getLoc();
    auto offsetOp = rewriter.create<arith::ConstantOp>(loc, type, attr);

    auto shiftOp =
        rewriter.create<arith::ShLIOp>(op->getLoc(), newLhs, offsetOp);
    rewriter.replaceOp(op, shiftOp);
  }

  llvm::LogicalResult
  matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    llvm::APInt constantOperand;

    if (matchPattern(lhs, m_ConstantInt(&constantOperand)) &&
        isPowerOfTwo(constantOperand)) {
      // LHS is a constant operand and a power of two
      // Resulting op will be:
      //    arith.shli rhs log(lhs)
      createShiftLeft(rewriter, op, rhs, lhs, constantOperand);
      return llvm::success();
    }

    if (matchPattern(rhs, m_ConstantInt(&constantOperand)) &&
        isPowerOfTwo(constantOperand)) {
      // RHS is a constant operand and a power of two
      // Resulting op will be:
      //    arith.shli lhs log(rhs)
      createShiftLeft(rewriter, op, lhs, rhs, constantOperand);
      return llvm::success();
    }

    // Neither operand is a constant power of two
    return llvm::success();
  }
};

void StrengthReductionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SimplifyPowerOp>(&getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

void registerStrengthReductionPass() {
  mlir::PassRegistration<StrengthReductionPass>();
}

} // namespace sblp
