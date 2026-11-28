//===- SliceOp.cpp - Cut one accelerator operation out of a linalg graph --===//
//
// gvn-slice-op{index=k layout=<json>}, on the output of gvn-insert-op-checks
// (an accelerator that takes whole linalg operations from DDR: Gemmini): the
// module keeps one public function, gvn_slice, which computes the operation
// of check k alone. Its tensor operands are read from the external byte
// array gvn_in at the offsets the external word array gvn_table gives (one
// slot each), its result is written into the external byte array gvn_out.
// An operand defined by an operation without operands (a constant, an empty
// tensor) is computed in place instead. A dynamic operand's sizes follow its
// offset in gvn_table.
//
// With parts=P > 1 (an operation too long for one simulation), the slice
// computes the first tile of the operation only: its parallel loops cut into
// P (TilingInterface), the one cutting the most operand bytes (or the loop
// first=l, its size possibly dynamic) first down to one iteration, then the
// next; the operands extracted from their whole values.
//
// The layout file lists the operands read from gvn_in (their linalg value,
// `line:column` in the linalg input or arg:<n> for an argument of the graph,
// their slot and type), the output (the value of check k, or its first
// tile) and the tiles: the parts
// and the loops cut in order, by the dimension of the value each is and its
// extent (-1 when dynamic).
//
//===----------------------------------------------------------------------===//

#include "GvnOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace {
// The external arrays are declared with this many elements; the harness
// defines them with their real sizes.
constexpr int64_t externalSize = int64_t(1) << 31;

std::string elementName(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(64))
    return "i64";
  llvm::report_fatal_error("gvn-slice-op: unsupported element type");
}

std::string typeName(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

// `line:column` of the first location of a value in the linalg input, or
// arg:<n> for an argument of the graph.
std::string valueId(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return "arg:" + std::to_string(argument.getArgNumber());
  Location loc = value.getDefiningOp()->getLoc();
  while (auto fused = dyn_cast<FusedLoc>(loc))
    loc = fused.getLocations().front();
  auto file = dyn_cast<FileLineColLoc>(loc);
  if (!file)
    llvm::report_fatal_error("gvn-slice-op: a value has no linalg location");
  return std::to_string(file.getLine()) + ":" +
         std::to_string(file.getColumn());
}

class SliceOpPass : public PassWrapper<SliceOpPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SliceOpPass)
  SliceOpPass() = default;
  SliceOpPass(const SliceOpPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "gvn-slice-op"; }
  StringRef getDescription() const final {
    return "Keep one checked accelerator operation, reading its operands from "
           "gvn_in";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  Option<unsigned> index{*this, "index", llvm::cl::desc("The check to keep")};
  Option<std::string> layout{*this, "layout",
                             llvm::cl::desc("The layout file to write")};
  Option<int64_t> parts{*this, "parts",
                        llvm::cl::desc("The tiles of the operation")};
  Option<int64_t> first{
      *this, "first",
      llvm::cl::desc("The loop cut first (-1: the one that cuts the most "
                     "operand bytes)"),
      llvm::cl::init(-1)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::gvn::CheckOp> checks;
    module.walk([&](::gvn::CheckOp check) { checks.push_back(check); });
    if (index >= checks.size())
      llvm::report_fatal_error("gvn-slice-op: no such check");
    ::gvn::CheckOp check = checks[index];
    Operation *op = check.getValue().getDefiningOp();
    auto outputType = cast<RankedTensorType>(check.getValue().getType());

    OpBuilder builder(module.getContext());
    Location loc = op->getLoc();
    builder.setInsertionPointToEnd(module.getBody());
    for (auto [name, element] :
         {std::pair<StringRef, Type>{"gvn_in", builder.getI8Type()},
          {"gvn_out", builder.getI8Type()},
          {"gvn_table", builder.getI64Type()}})
      memref::GlobalOp::create(
          builder, loc, name, builder.getStringAttr("public"),
          MemRefType::get({externalSize}, element), Attribute(), false,
          builder.getI64IntegerAttr(64));
    auto slice = func::FuncOp::create(builder, loc, "gvn_slice",
                                      builder.getFunctionType({}, {}));
    builder.setInsertionPointToStart(slice.addEntryBlock());
    auto global = [&](StringRef name, Type element) {
      return memref::GetGlobalOp::create(
          builder, loc, MemRefType::get({externalSize}, element), name);
    };

    IRMapping mapping;
    llvm::json::Array entries;
    int64_t slot = 0;
    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand))
        continue;
      Operation *producer = operand.getDefiningOp();
      if (producer && producer->getNumOperands() == 0) {
        mapping.map(operand, builder.clone(*producer)->getResult(0));
        continue;
      }
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type) {
        op->emitOpError("has an operand that is no tensor");
        llvm::report_fatal_error("gvn-slice-op: malformed slice");
      }
      auto word = [&](int64_t at) {
        return arith::IndexCastOp::create(
            builder, loc, builder.getIndexType(),
            memref::LoadOp::create(
                builder, loc, global("gvn_table", builder.getI64Type()),
                ValueRange{arith::ConstantIndexOp::create(builder, loc, at)}));
      };
      // Its offset, then its dynamic sizes.
      Value offset = word(slot);
      SmallVector<Value> sizes;
      for (int64_t dynamic = 0; dynamic < type.getNumDynamicDims(); ++dynamic)
        sizes.push_back(word(slot + 1 + dynamic));
      Value view = memref::ViewOp::create(
          builder, loc, MemRefType::get(type.getShape(), type.getElementType()),
          global("gvn_in", builder.getI8Type()), offset, sizes);
      mapping.map(operand, bufferization::ToTensorOp::create(builder, loc, type,
                                                             view, true));
      entries.push_back(
          llvm::json::Object{{"id", valueId(operand)},
                             {"slot", slot},
                             {"type", typeName(type)},
                             {"element", elementName(type.getElementType())}});
      slot += 1 + type.getNumDynamicDims();
    }
    Operation *computed = builder.clone(*op, mapping);
    Value result = computed->getResult(0);
    llvm::json::Object tiles{{"parts", parts.getValue()}};
    auto structured = cast<linalg::LinalgOp>(computed);
    SmallVector<int64_t> extents = structured.getStaticLoopRanges();
    SmallVector<utils::IteratorType> iterators =
        structured.getIteratorTypesArray();
    // The parallel loops by the operand bytes each cuts, the most first
    // (the columns of a weight, the rows of a long input): a tile moves
    // and computes a part of the operation's work. A design whose cost is
    // elsewhere names the loop to cut first, which may have a dynamic size
    // (the tokens). The parts cut the first loop, down to one iteration,
    // then the next: cut = min(left, extent), size = ceil(extent / cut),
    // left = ceil(left / cut), at run time for a dynamic extent.
    SmallVector<std::pair<int64_t, int64_t>> ranked;
    for (int64_t l = 0; l < int64_t(extents.size()); ++l) {
      if (iterators[l] != utils::IteratorType::parallel ||
          (l != first && extents[l] < 2))
        continue;
      int64_t bytes = 0;
      for (OpOperand &operand : computed->getOpOperands()) {
        auto type = cast<RankedTensorType>(operand.get().getType());
        if (type.hasStaticShape() &&
            structured.getMatchingIndexingMap(&operand).isFunctionOfDim(l))
          bytes += type.getNumElements() * type.getElementTypeBitWidth() / 8;
      }
      ranked.push_back({bytes, l});
    }
    llvm::stable_sort(ranked, [&](auto a, auto b) {
      return (a.second == first) != (b.second == first) ? a.second == first
                                                        : a.first > b.first;
    });
    AffineMap map = structured.getIndexingMapMatchingResult(
        cast<OpResult>(computed->getResult(0)));
    // The loops cut, by the dimension of the value each is (its extent, -1
    // when dynamic): the driver computes the tiles from them.
    llvm::json::Array loops;
    for (auto [bytes, l] : ranked) {
      int64_t dim = -1;
      for (unsigned r = 0; r < map.getNumResults(); ++r)
        if (map.getResult(r) == builder.getAffineDimExpr(l))
          dim = r;
      loops.push_back(llvm::json::Object{
          {"dim", dim},
          {"extent", ShapedType::isDynamic(extents[l]) ? -1 : extents[l]}});
    }
    tiles["loops"] = std::move(loops);
    if (parts > 1) {
      SmallVector<OpFoldResult> offsets(extents.size(),
                                        builder.getIndexAttr(0));
      // The loops not cut keep their sizes, dynamic ones too.
      SmallVector<OpFoldResult> sizes =
          llvm::map_to_vector(structured.createLoopRanges(builder, loc),
                              [](Range range) { return range.size; });
      Value left = arith::ConstantIndexOp::create(builder, loc, parts);
      for (auto [bytes, l] : ranked) {
        Value extent = getValueOrCreateConstantIndexOp(builder, loc, sizes[l]);
        Value cut = builder.createOrFold<arith::MinUIOp>(loc, left, extent);
        sizes[l] = builder.createOrFold<arith::CeilDivUIOp>(loc, extent, cut);
        left = builder.createOrFold<arith::CeilDivUIOp>(loc, left, cut);
      }
      for (OpFoldResult &size : sizes)
        if (std::optional<int64_t> known = getConstantIntValue(size))
          size = builder.getIndexAttr(*known);
      FailureOr<TilingResult> tiled =
          cast<TilingInterface>(computed).getTiledImplementation(
              builder, offsets, sizes);
      if (failed(tiled))
        llvm::report_fatal_error("gvn-slice-op: the operation does not tile");
      // The tile's operands are views of the whole values; the accelerator
      // takes contiguous ones, as in the graph: each one cut is copied.
      Operation *tile = tiled->tiledOps[0];
      builder.setInsertionPoint(tile);
      for (OpOperand &operand : tile->getOpOperands())
        if (auto view = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
            view && view.getType() != view.getSourceType())
          operand.set(bufferization::AllocTensorOp::create(
              builder, loc, view.getType(), ValueRange{}, view.getResult()));
      result = tiled->tiledValues[0];
      computed->erase();
      outputType = cast<RankedTensorType>(result.getType());
      builder.setInsertionPointToEnd(&slice.front());
    }
    SmallVector<Value> outSizes;
    for (int64_t dim = 0; dim < outputType.getRank(); ++dim)
      if (outputType.isDynamicDim(dim))
        outSizes.push_back(tensor::DimOp::create(builder, loc, result, dim));
    Value out = memref::ViewOp::create(
        builder, loc,
        MemRefType::get(outputType.getShape(), outputType.getElementType()),
        global("gvn_out", builder.getI8Type()),
        arith::ConstantIndexOp::create(builder, loc, 0), outSizes);
    auto store = bufferization::MaterializeInDestinationOp::create(builder, loc,
                                                                   result, out);
    store.setRestrict(true);
    store.setWritable(true);
    func::ReturnOp::create(builder, loc);

    // Nothing else of the graph stays.
    for (Operation &other : llvm::make_early_inc_range(*module.getBody()))
      if (auto function = dyn_cast<func::FuncOp>(other);
          function && function != slice)
        function.erase();

    std::error_code error;
    llvm::raw_fd_ostream file(layout, error);
    if (error)
      llvm::report_fatal_error("gvn-slice-op: cannot write the layout file");
    file << llvm::formatv(
                "{0:2}",
                llvm::json::Value(llvm::json::Object{
                    {"index", index.getValue()},
                    {"operation", op->getName().getStringRef()},
                    {"entries", std::move(entries)},
                    {"table_words", slot},
                    {"tiles", std::move(tiles)},
                    {"output",
                     llvm::json::Object{
                         {"id", check.getId().str()},
                         {"type", typeName(outputType)},
                         {"element",
                          elementName(outputType.getElementType())}}}}))
         << "\n";
  }
};
} // namespace

namespace gvn {
void registerSliceOpPass() { PassRegistration<SliceOpPass>(); }
} // namespace gvn
