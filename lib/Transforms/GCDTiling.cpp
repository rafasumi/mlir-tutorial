#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "Transforms/GCDTiling.h"

using namespace mlir;

namespace {
int64_t greatestCommonDivisor(int64_t lhs, int64_t rhs) {
  return rhs == 0 ? lhs : greatestCommonDivisor(rhs, lhs % rhs);
}
} // namespace

namespace sblp {

int64_t GCDTilingPass::computeTilingSize(linalg::LinalgOp op) {
  std::vector<int64_t> collectedDims;

  // Collect the dimensions from all operands.
  for (auto operand : op->getOperands()) {
    auto operandType = llvm::dyn_cast<ShapedType>(operand.getType());
    llvm::ArrayRef<int64_t> shape = operandType.getShape();
    for (int64_t dim : shape) {
      collectedDims.push_back(dim);
    }
  }

  // Compute the greatest common divisor between dimensions.
  int64_t tileSize = collectedDims[0];
  for (size_t idx = 1; idx < collectedDims.size(); idx++) {
    tileSize = greatestCommonDivisor(tileSize, collectedDims[idx]);
  }

  return tileSize;
}

void GCDTilingPass::runOnOperation() {
  auto F = getOperation();
  std::vector<linalg::LinalgOp> ops;

  F.walk([&](linalg::LinalgOp op) {
    bool tilable = true;

    // Only tile operations that take tensors with static shapes for operands.
    for (auto operand : op->getOperands()) {
      if (auto operandType = llvm::dyn_cast<ShapedType>(operand.getType())) {
        tilable &= ShapedType::isStaticShape(operandType.getShape());
      } else {
        tilable = false;
      }
    }

    if (tilable)
      ops.push_back(op);
  });

  IRRewriter rewriter(F.getContext());

  for (auto op : ops) {
    linalg::LinalgTilingOptions options;

    auto tileSize = computeTilingSize(op);

    options.setTileSizes({tileSize, tileSize});
    options.setLoopType(linalg::LinalgTilingLoopType::Loops);

    auto tiledOp = tileLinalgOp(rewriter, op, options);

    if (failed(tiledOp)) {
      break;
    }

    for (size_t idx = 0; idx < op->getNumResults(); idx++) {
      op->getResults()[idx].replaceAllUsesWith(tiledOp->tensorResults[idx]);
    }
    op->erase();
  }
}

void registerGCDTilingPass() { PassRegistration<GCDTilingPass>(); }

std::unique_ptr<OperationPass<func::FuncOp>> createGCDTilingPass() {
  return std::make_unique<GCDTilingPass>();
}

} // namespace sblp
