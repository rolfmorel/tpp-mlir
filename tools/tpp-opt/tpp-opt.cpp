//===- tpp-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <iostream>

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"

#define DEBUG_TYPE "horrible-nasty-hack"

using namespace mlir;
using namespace mlir::affine;

Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source, int64_t dim) {
  if (llvm::isa<UnrankedMemRefType, MemRefType>(source.getType()))
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (llvm::isa<UnrankedTensorType, RankedTensorType>(source.getType()))
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value source,
                               int64_t dim) {
  auto shapedType = llvm::cast<ShapedType>(source.getType());
  if (!shapedType.hasRank() || shapedType.isDynamicDim(dim))
    return createOrFoldDimOp(b, loc, source, dim);
  return b.getIndexAttr(shapedType.getDimSize(dim));
}

// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<OpFoldResult> tileSizes) : tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZeroIndex(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(cast<AffineConstantExpr>(expr.getRHS()).getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled = false;
  ArrayRef<OpFoldResult> tileSizes;
};

static bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

///// Returns a memref.subview or a tensor.extract_slice based on the type of
///the
///// `source`.
// static Operation *getSlice(OpBuilder &b, Location loc, Value source,
//                            ArrayRef<OpFoldResult> offsets,
//                            ArrayRef<OpFoldResult> sizes,
//                            ArrayRef<OpFoldResult> strides) {
//   return TypeSwitch<Type, Operation *>(source.getType())
//       .Case<VectorType>([&](VectorType t) -> Operation * {
//         return b.create<vector::ExtractStridedSliceOp>(loc, source, offsets,
//                                                        sizes, strides);
//       })
//       .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
//         return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
//                                                 strides);
//       })
//       .Case<MemRefType>([&](MemRefType type) -> Operation * {
//         return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
//                                            strides);
//       })
//       .Default([&](Type t) -> Operation * { return nullptr; });
// }

/// A struct containg offsets-sizes-strides arguments of the tiled shape.
struct SliceParameters {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

Operation *materializeTiledShape(OpBuilder &builder, Location loc,
                                 Value valueToTile,
                                 const SliceParameters &sliceParams) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  auto *sliceOp =
      TypeSwitch<ShapedType, Operation *>(shapedType)
          .Case([&](MemRefType) {
            return builder.create<memref::SubViewOp>(
                loc, valueToTile, sliceParams.offsets, sliceParams.sizes,
                sliceParams.strides);
          })
          .Case([&](RankedTensorType) {
            return builder.create<tensor::ExtractSliceOp>(
                loc, valueToTile, sliceParams.offsets, sliceParams.sizes,
                sliceParams.strides);
          })
          .Case([&](VectorType) {
            valueToTile.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>().dump();

            SmallVector<int64_t> offsets;
            for (OpFoldResult offset : sliceParams.offsets) {
              std::cerr << "offset ";
              offset.dump();
              if (auto offsetAttr = offset.dyn_cast<Attribute>()) {
                if (auto offsetIntAttr = dyn_cast<IntegerAttr>(offsetAttr)) {
                  offsets.push_back(offsetIntAttr.getInt());
                }
              } else if (auto offsetValue = offset.dyn_cast<Value>()) {
                offsets.push_back(1);
                std::cerr << " - I am so sorry - ";
              }
            }
            std::cerr << "\n ";
            SmallVector<int64_t> sizes;
            for (OpFoldResult size : sliceParams.sizes) {
              std::cerr << "size ";
              size.dump();
              if (auto sizeAttr = dyn_cast<Attribute>(size)) {
                if (auto sizeIntAttr = dyn_cast<IntegerAttr>(sizeAttr)) {
                  sizes.push_back(sizeIntAttr.getInt());
                }
              }
            }
            std::cerr << "\n ";
            SmallVector<int64_t> strides;
            for (OpFoldResult  stride : sliceParams.strides) {
              std::cerr << "stride ";
              stride.dump();
              if (auto strideAttr = dyn_cast<Attribute>(stride)) {
                if (auto strideIntAttr = dyn_cast<IntegerAttr>(strideAttr)) {
                  strides.push_back(strideIntAttr.getInt());
                }
              }
            }
            std::cerr << "\n ";
            return builder.create<vector::ExtractStridedSliceOp>(
                loc, valueToTile, offsets, sizes, strides);
          })
          .Default([](ShapedType) -> Operation * {
            llvm_unreachable("Unexpected shaped type");
          });
  return sliceOp;
}

SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Compute offsets/sizes/strides for the tile.
  SliceParameters sliceParams;
  sliceParams.offsets.reserve(rank);
  sliceParams.sizes.reserve(rank);
  sliceParams.strides.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      sliceParams.offsets.push_back(builder.getIndexAttr(0));
      OpFoldResult dim = createFoldedDimOp(builder, loc, valueToTile, r);
      sliceParams.sizes.push_back(dim);
      sliceParams.strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: submap: " << m << "\n");
    IRRewriter rewriter(builder);
    // The offset of the slice is m(lbs) - m(0).
    SmallVector<Attribute> zeros(lbs.size(), rewriter.getIndexAttr(0));
    SmallVector<Attribute> mAtZero;
    [[maybe_unused]] auto res = m.constantFold(zeros, mAtZero);
    assert(succeeded(res) && "affine_map must be evaluatable (not symbols)");
    int64_t mAtZeroInt =
        cast<IntegerAttr>(mAtZero[0]).getValue().getSExtValue();
    OpFoldResult offset = makeComposedFoldedAffineApply(
        rewriter, loc, m.getResult(0) - mAtZeroInt, lbs);
    sliceParams.offsets.push_back(offset);

    OpFoldResult closedIntSize =
        makeComposedFoldedAffineApply(rewriter, loc, m, subShapeSizes);
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    OpFoldResult size =
        makeComposedFoldedAffineApply(rewriter, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: new offset: " << offset << "\n");
    sliceParams.strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
      sliceParams.sizes.push_back(size);
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    std::optional<int64_t> sizeCst = getConstantIntValue(size);
    auto hasTileSizeOne = sizeCst && *sizeCst == 1;
    auto dividesEvenly = sizeCst && !ShapedType::isDynamic(shapeSize) &&
                         ((shapeSize % *sizeCst) == 0);
    if (!hasTileSizeOne && !dividesEvenly) {
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: shapeSize=" << shapeSize
                              << ", size: " << size
                              << ": make sure in bound with affine.min\n");

      AffineExpr dim0, dim1, dim2;
      MLIRContext *context = builder.getContext();
      bindDims(context, dim0, dim1, dim2);

      // Get the dimension size for this dimension. We need to first calculate
      // the max index and then plus one. This is important because for
      // convolution ops, we have its input window dimension's affine map of the
      // form `(d0 * s0 + d1)`, where `d0`/`d1 is an output/filter window
      // dimension and `s0` is stride. Directly use the dimension size of
      // output/filer window dimensions will cause incorrect calculation.
      AffineMap minusOneMap = AffineMap::inferFromExprList(
                                  {ArrayRef<AffineExpr>{dim0 - 1}}, context)
                                  .front();
      AffineMap plusOneMap = AffineMap::inferFromExprList(
                                 {ArrayRef<AffineExpr>{dim0 + 1}}, context)
                                 .front();
      SmallVector<OpFoldResult> maxIndices =
          llvm::to_vector(llvm::map_range(ubs, [&](OpFoldResult ub) {
            return mlir::affine::makeComposedFoldedAffineApply(
                rewriter, loc, minusOneMap, {ub});
          }));
      OpFoldResult maxIndex =
          makeComposedFoldedAffineApply(rewriter, loc, m, maxIndices);
      OpFoldResult d =
          makeComposedFoldedAffineApply(rewriter, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}}, context)
                             .front();
      size =
          makeComposedFoldedAffineMin(rewriter, loc, minMap, {size, d, offset});
    }
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    sliceParams.sizes.push_back(size);
  }
  return sliceParams;
}

//Operation *makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
//                          ArrayRef<OpFoldResult> tileSizes, AffineMap map,
//                          ArrayRef<OpFoldResult> lbs,
//                          ArrayRef<OpFoldResult> ubs,
//                          ArrayRef<OpFoldResult> subShapeSizes,
//                          bool omitPartialTileCheck) {
//  SliceParameters sliceParams =
//      computeSliceParameters(builder, loc, valueToTile, tileSizes, map, lbs,
//                             ubs, subShapeSizes, omitPartialTileCheck);
//  return materializeTiledShape(builder, loc, valueToTile, sliceParams);
//}

SmallVector<OpFoldResult> computeTileOffsets(OpBuilder &b, Location loc,
                                             ArrayRef<OpFoldResult> ivs,
                                             ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> offsets;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for loop#" << idx << "\n");
    bool isTiled = !isZeroIndex(tileSizes[idx]);
    offsets.push_back(isTiled ? ivs[idxIvs++] : b.getIndexAttr(0));
    LLVM_DEBUG(llvm::dbgs()
               << "computeTileOffsets: " << offsets.back() << "\n");
  }
  return offsets;
}

SmallVector<OpFoldResult> computeTileSizes(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           ArrayRef<OpFoldResult> sizeBounds) {
  SmallVector<OpFoldResult> sizes;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZeroIndex(tileSizes[idx]);
    // Before composing, we need to make range a closed interval.
    OpFoldResult size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    IRRewriter rewriter(b);
    sizes.push_back(makeComposedFoldedAffineApply(rewriter, loc, d0 - 1, size));
    LLVM_DEBUG(llvm::dbgs() << "computeTileSizes: " << sizes.back() << "\n");
  }
  return sizes;
}

SmallVector<Type> getTensorOutputTypes(vector::ContractionOp op,
                                       ValueRange operands) {
  return {op.getResultType()};
  // if (op.hasPureBufferSemantics())
  //   return {};
  // return llvm::to_vector(
  //     llvm::map_range(op.getDpsInitsMutable(), [&](OpOperand &opOperand) {
  //       return operands[opOperand.getOperandNumber()].getType();
  //     }));
}

// RM: unused by rest of code here
// SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
//                                     LinalgOp op, ValueRange operands,
//                                     ValueRange results) {
//   if (op.hasPureBufferSemantics())
//     return {};
//   SmallVector<Value> tensorResults;
//   tensorResults.reserve(results.size());
//   // Insert a insert_slice for each output tensor.
//   unsigned resultIdx = 0;
//   for (OpOperand &opOperand : op.getDpsInitsMutable()) {
//     // TODO: use an interface/adaptor to avoid leaking position in
//     // `tiledOperands`.
//     Value outputTensor = operands[opOperand.getOperandNumber()];
//     if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>())
//     {
//       Value inserted = builder.create<tensor::InsertSliceOp>(
//           loc, sliceOp.getSource().getType(), results[resultIdx],
//           sliceOp.getSource(), sliceOp.getOffsets(), sliceOp.getSizes(),
//           sliceOp.getStrides(), sliceOp.getStaticOffsets(),
//           sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
//       tensorResults.push_back(inserted);
//     } else {
//       tensorResults.push_back(results[resultIdx]);
//     }
//     ++resultIdx;
//   }
//   return tensorResults;
// }

SmallVector<std::optional<SliceParameters>> computeAllSliceParameters(
    OpBuilder &builder, Location loc, vector::ContractionOp contractOp,
    ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds,
    bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroIndex(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             contractOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<std::optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  bool dpsInit[3] = {false, false, true};
  for (auto [opOperand, val, map, isDpsInit] :
       llvm::zip(contractOp->getOpOperands(), valuesToTile,
                 contractOp.getIndexingMapsArray(), dpsInit)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    // AffineMap map = contractOp.getMatchingIndexingMap(&opOperand);
    //  Use `opOperand` as is if it is not tiled and not an output tensor.
    //  Having an extract/insert slice pair for all output tensors simplifies
    //  follow up transformations such as padding and bufferization since the
    //  extract/insert slice pairs make the accessed iteration argument
    //  subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) &&
        !(isa<VectorType>(operandType) && isDpsInit)) {
      allSliceParams.push_back(std::nullopt);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

SmallVector<Value>
makeTiledShapes(OpBuilder &builder, Location loc,
                vector::ContractionOp contractOp, ValueRange valuesToTile,
                ArrayRef<OpFoldResult> ivs, ArrayRef<OpFoldResult> tileSizes,
                ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck) {
  SmallVector<std::optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, contractOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
                  ->getResult(0)
            : valueToTile);
  }
  return tiledShapes;
}

// RM: ignore for now
// void offsetIndices(OpBuilder &b, LinalgOp linalgOp,
//                    ArrayRef<OpFoldResult> offsets) {
//   IRRewriter rewriter(b);
//   offsetIndices(rewriter, linalgOp, offsets);
// }
// void offsetIndices(RewriterBase &b, LinalgOp linalgOp,
//                    ArrayRef<OpFoldResult> offsets) {
//   if (!linalgOp.hasIndexSemantics())
//     return;

//   for (IndexOp indexOp : linalgOp.getBlock()->getOps<IndexOp>()) {
//     if (indexOp.getDim() >= offsets.size() || !offsets[indexOp.getDim()])
//       continue;
//     OpBuilder::InsertionGuard guard(b);
//     b.setInsertionPointAfter(indexOp);
//     AffineExpr index, offset;
//     bindDims(b.getContext(), index, offset);
//     OpFoldResult applied = makeComposedFoldedAffineApply(
//         b, indexOp.getLoc(), index + offset,
//         {getAsOpFoldResult(indexOp.getResult()), offsets[indexOp.getDim()]});
//     Value materialized =
//         getValueOrCreateConstantIndexOp(b, indexOp.getLoc(), applied);
//     b.replaceUsesWithIf(indexOp, materialized, [&](OpOperand &use) {
//       return use.getOwner() != materialized.getDefiningOp();
//     });
//   }
// }

struct VectorContractionOpTiling
    : public TilingInterface::ExternalModel<VectorContractionOpTiling,
                                            mlir::vector::ContractionOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    // assert(false);
    auto contractOp = cast<vector::ContractionOp>(op);
    SmallVector<utils::IteratorType> vecUsingUtilsIterType;
    for (auto it : contractOp.getIteratorTypesArray()) {
      vecUsingUtilsIterType.push_back(it == vector::IteratorType::parallel
                                          ? utils::IteratorType::parallel
                                          : utils::IteratorType::reduction);
    }
    return vecUsingUtilsIterType;
  }

  SmallVector<Range> getIterationDomain(Operation *op,
                                        OpBuilder &builder) const {
    auto contractOp = cast<vector::ContractionOp>(op);
    SmallVector<int64_t> iterationBounds;
    contractOp.getIterationBounds(iterationBounds);

    SmallVector<Range> loopRanges(iterationBounds.size());
    Location loc = op->getLoc();
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (auto dim : llvm::seq<int64_t>(0, iterationBounds.size())) {
      loopRanges[dim].offset = zero; // FIXME: not always the case?
      Value bound =
          builder.create<arith::ConstantIndexOp>(loc, iterationBounds[dim]);
      loopRanges[dim].size = bound;
      loopRanges[dim].stride = one; // FIXME: not always the case?
    }
    return loopRanges;

    // return {};
    // ReifiedRankedShapedTypeDims reifiedShapes;
    //(void)reifyResultShapes(b, op, reifiedShapes);
    // OpFoldResult zero = b.getIndexAttr(0);
    // OpFoldResult one = b.getIndexAttr(1);
    //// Initialize all the ranges to {zero, one, one}. All the `ub`s are
    //// overwritten.
    // SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    // for (const auto &ub : enumerate(reifiedShapes[0]))
    //   loopRanges[ub.index()].size = ub.value();
    // return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    Location loc = op->getLoc();
    auto contractOp = cast<vector::ContractionOp>(op);
    SmallVector<Value> valuesToTile = op->getOperands();
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, contractOp, valuesToTile, offsets, sizes, {}, true);

    LLVM_DEBUG(llvm::dbgs() << "passed makeTiledShape\n");
    SmallVector<Operation *> generatedSlices = llvm::map_to_vector(
        llvm::make_filter_range(
            tiledOperands,
            [](Value v) -> bool {
              return isa_and_nonnull<tensor::ExtractSliceOp, memref::SubViewOp,
                                     vector::ExtractStridedSliceOp>(
                  v.getDefiningOp());
            }),
        [](Value v) -> Operation * { return v.getDefiningOp(); });
    LLVM_DEBUG(llvm::dbgs() << "passed extracting generatedSlices\n");

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(contractOp, tiledOperands);

    Operation *tiledOp = clone(b, contractOp, tiledOperands[2].getType(), tiledOperands);
    // offsetIndices(b, cast<LinalgOp>(tiledOp), offsets); // RM: ignore for now
    LLVM_DEBUG(llvm::dbgs() << "passed cloning\n");

    op->getParentOfType<ModuleOp>().dump();
    return TilingResult{
        {tiledOp}, SmallVector<Value>(tiledOp->getResults()), generatedSlices};
    return {};
    // FailureOr<TilingResult> result =
    //     tensor::bubbleUpVectorContractSlice(b, cast<PadOp>(op), offsets,
    //     sizes);
    // if (failed(result))
    //   return failure();
    // return result.value();
  }

  /// RM: this is a helper
  /// Utility to fetch the offsets and sizes when applied as per the indexing
  /// map of the linalg op. This helps in fusing the linalg op as a consumer of
  /// a given slice op.
  void
  getMappedOffsetAndSize(vector::ContractionOp contractOp, OpBuilder &b,
                         AffineMap indexingMap, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         SmallVectorImpl<OpFoldResult> &mappedOffsets,
                         SmallVectorImpl<OpFoldResult> &mappedSizes) const {
    // unsigned numLoops = linalgOp.getNumLoops();
    unsigned numLoops = contractOp.getIteratorTypesArray().size();
    auto tilingInterfaceOp = cast<TilingInterface>(contractOp.getOperation());
    mappedOffsets.resize(numLoops);
    mappedSizes.resize(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto &&[index, value] : llvm::enumerate(iterationDomain)) {
        mappedOffsets[index] = value.offset;
        mappedSizes[index] = value.size;
      }
    }
    for (const auto &&[index, value] :
         llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition = cast<AffineDimExpr>(value).getPosition();
      mappedOffsets[dimPosition] = offsets[index];
      mappedSizes[dimPosition] = sizes[index];
    }
  }

  /// Method to return the position of the result tile computed by the tiled
  /// operation.
  LogicalResult getIterationDomainTileFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto contractOp = cast<vector::ContractionOp>(op);

    // Check that the indexing map used for the operand is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the operand to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap = contractOp.getIndexingMapsArray()[operandNumber];

    // AffineMap indexingMap =
    //     contractOp.getMatchingIndexingMap(&op->getOpOperand(operandNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitError()
             << "unhandled get iter domain position when operand is not "
                "accessed using a permuted projection";
    }

    getMappedOffsetAndSize(contractOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  /// Return the details of the output tile generated by the tiled
  /// implementation.
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    Location loc = op->getLoc();
    vector::ContractionOp contractOp = cast<vector::ContractionOp>(op);

    AffineExpr d0;
    bindDims(b.getContext(), d0);
    SmallVector<OpFoldResult> subShapeSizes =
        llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
          return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
        }));

    OpOperand *outOperand = &contractOp->getOpOperand(2);
    auto indexingMap = contractOp.getIndexingMapsArray()[2];
    SliceParameters sliceParams = computeSliceParameters(
        b, loc, outOperand->get(), sizes, indexingMap, offsets,
        /*ubs*/ {}, subShapeSizes, true);
    resultOffsets = sliceParams.offsets;
    resultSizes = sliceParams.sizes;
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    auto contractOp = cast<vector::ContractionOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap = contractOp.getIndexingMapsArray()[2 + resultNumber];
    //        contractOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitOpError(
          "unhandled tiled implementation generation when result is not "
          "accessed using a permuted projection");
    }

    getMappedOffsetAndSize(contractOp, b, indexingMap, offsets, sizes,
                           iterDomainOffsets, iterDomainSizes);
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(getIterationDomainTileFromResultTile(
            op, b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
      return failure();
    }
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, mappedOffsets, mappedSizes);

    if (failed(tilingResult))
      return failure();

    if (tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return TilingResult{
        tilingResult->tiledOps,
        SmallVector<Value>{tilingResult->tiledValues[resultNumber]},
        tilingResult->generatedSlices};
  }

  /// Method to generate the tiled implementation of an operation from the tile
  /// of the operand.
  FailureOr<TilingResult> getTiledImplementationFromOperandTile(
      Operation *op, OpBuilder &b, unsigned operandNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(getIterationDomainTileFromOperandTile(
            op, b, operandNumber, offsets, sizes, mappedOffsets,
            mappedSizes))) {
      return failure();
    }
    return getTiledImplementation(op, b, mappedOffsets, mappedSizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &builder,
                                             Location loc,
                                             ValueRange ivs) const {
    // auto linalgOp = cast<LinalgOp>(op);
    // if (!linalgOp.hasPureBufferSemantics())
    return op->emitOpError("expected operation to have buffer semantics");

    // SmallVector<Value> indexedValues;
    // indexedValues.reserve(linalgOp->getNumOperands());
    // Location linalgOpLoc = op->getLoc();
    ///// Load the data corresponding to the block arguments that
    ///// represent input operands.
    // for (OpOperand &operand : linalgOp->getOpOperands()) {
    //   if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
    //     indexedValues.push_back(nullptr);
    //     continue;
    //   }
    //   if (linalgOp.isScalar(&operand)) {
    //     indexedValues.push_back(operand.get());
    //     continue;
    //   }
    //   SmallVector<Value> indices = getIndicesForAccess(
    //       builder, linalgOpLoc, linalgOp.getMatchingIndexingMap(&operand),
    //       ivs);
    //   Value load =
    //       builder.create<memref::LoadOp>(linalgOpLoc, operand.get(),
    //       indices);
    //   indexedValues.push_back(load);
    // }

    ///// Inline the op payload and store the result.
    // return inlinePayload(builder, linalgOp, ivs, indexedValues);
  }
};

void vectorRegisterTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::vector::VectorDialect *dialect) {
        mlir::vector::ContractionOp::attachInterface<VectorContractionOpTiling>(
            *ctx);
      });
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::tpp::registerTppCompilerPasses();
  mlir::tpp::registerTppPassBundlePasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::check::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::perf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tpp::registerTestStructuralMatchers();
  mlir::tpp::registerTestForToForAllRewrite();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated.
  registerAllDialects(registry);
  // unclear whether vector dialect must be made to promise tilinginterface on
  // vector.contract
  vectorRegisterTilingInterfaceExternalModels(registry);
  mlir::registerAllExtensions(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  registerAllToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TPP optimizer driver\n", registry));
}
