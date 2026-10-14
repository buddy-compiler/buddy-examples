if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()
if(NOT EXISTS "${BUCKYBALL_THREAD_PB}")
  message(FATAL_ERROR "missing thread.pb: '${BUCKYBALL_THREAD_PB}'")
endif()

get_filename_component(BB_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BB_COMMON "${BB_REPO_ROOT}/stack/compiler/src")
set(BB_GENERATOR "${BB_REPO_ROOT}/stack/compiler/scripts/thread_to_compiler.py")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(bb_generate OUT)
  execute_process(
    COMMAND
      "${Python3_EXECUTABLE}" "${BB_GENERATOR}" --repo "${BB_REPO_ROOT}" --thread-pb "${BUCKYBALL_THREAD_PB}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${BB_GENERATOR} ${ARGN} failed:\n${output}${error}")
  endif()
  string(STRIP "${output}" output)
  string(REPLACE "\n" ";" output "${output}")
  set(${OUT} "${output}" PARENT_SCOPE)
endfunction()

bb_generate(_
  --registry-out "${CMAKE_CURRENT_BINARY_DIR}/BuckyballTargetRegistry.inc"
  --dialect-td-out "${CMAKE_CURRENT_BINARY_DIR}/BuckyballUnion.td"
  --hooks-out "${CMAKE_CURRENT_BINARY_DIR}/BuckyballBallLoweringHooks.inc"
  --pass-decls-out "${CMAKE_CURRENT_BINARY_DIR}/BuckyballBallPassesDecl.inc"
  --pass-registration-out "${CMAKE_CURRENT_BINARY_DIR}/BuckyballBallPassesRegistration.inc"
)
foreach(
  kind
  thread
  dialect-dir
  legalize
  assign
  bank_ssa
  tile
  passes
)
  bb_generate(BB_${kind} --print-sources ${kind})
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${BUCKYBALL_THREAD_PB}" "${BB_GENERATOR}" ${BB_thread})

set(
  BB_IR_SOURCES
  "${BB_COMMON}/Dialect/Buckyball/IR/BuckyballDialect.cpp"
  "${BB_COMMON}/Target/BuckyballTargetRegistry.cpp"
)
set(
  BB_TRANSFORMS_SOURCES
  "${BB_COMMON}/Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.cpp"
  "${BB_COMMON}/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp"
  ${BB_legalize}
)
file(GLOB BB_LOWER_SOURCES "${BB_COMMON}/Transforms/*.cpp" "${BB_COMMON}/Conversion/LowerBuckyball/*.cpp")
list(APPEND BB_LOWER_SOURCES ${BB_assign} ${BB_bank_ssa} ${BB_passes})
file(GLOB BB_TILE_SOURCES "${BB_COMMON}/Conversion/LowerTileToBuckyball/*.cpp")
list(APPEND BB_TILE_SOURCES ${BB_tile})
set(BB_TRANSLATION_SOURCES "${BB_COMMON}/Target/LLVMIR/Dialect/Buckyball/BuckyballToLLVMIRTranslation.cpp")

foreach(dir ${BB_dialect-dir})
  include_directories("${dir}")
  file(GLOB ir "${dir}/IR/*.cpp")
  list(APPEND BB_IR_SOURCES ${ir})
endforeach()

# A Core source replaces the common source at the same relative path.
foreach(source ${BB_thread})
  string(REGEX REPLACE "^.*/compiler/src/" "" rel "${source}")
  string(REGEX REPLACE "/compiler/src/.*$" "/compiler/src" BB_CORE_SRC "${source}")
  if(rel MATCHES "^Dialect/Buckyball/IR/")
    set(lib BB_IR_SOURCES)
  elseif(rel MATCHES "^Dialect/Buckyball/Transforms/")
    set(lib BB_TRANSFORMS_SOURCES)
  elseif(rel MATCHES "^Conversion/LowerTileToBuckyball/")
    set(lib BB_TILE_SOURCES)
  elseif(rel MATCHES "^(Conversion/LowerBuckyball|Transforms)/")
    set(lib BB_LOWER_SOURCES)
  else()
    message(FATAL_ERROR "unsupported Core compiler source: ${source}")
  endif()
  list(REMOVE_ITEM ${lib} "${BB_COMMON}/${rel}")
  list(APPEND ${lib} "${source}")
endforeach()
if(BB_CORE_SRC)
  include_directories(BEFORE "${BB_CORE_SRC}")
endif()

include_directories("${BB_COMMON}")
include_directories("${BB_COMMON}/Dialect")
include_directories("${BB_COMMON}/Dialect/Buckyball")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_BINARY_DIR}/BuckyballUnion.td")
mlir_tablegen(Buckyball.h.inc -gen-op-decls)
mlir_tablegen(Buckyball.cpp.inc -gen-op-defs)
mlir_tablegen(BuckyballTypes.h.inc -gen-typedef-decls -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=buckyball)
mlir_tablegen(BuckyballDialect.h.inc -gen-dialect-decls -dialect=buckyball)
mlir_tablegen(BuckyballDialect.cpp.inc -gen-dialect-defs -dialect=buckyball)
add_public_tablegen_target(MLIRBuckyballIncGen)
mlir_tablegen(BuckyballConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyBuckyballConversionsIncGen)

add_mlir_dialect_library(BuddyBuckyball
  ${BB_IR_SOURCES}
  LINK_LIBS PUBLIC MLIRIR MLIRLLVMDialect MLIRBufferizationDialect MLIRMemRefDialect
)
target_include_directories(BuddyBuckyball PUBLIC "${BB_COMMON}"
                                                 "${BB_COMMON}/Dialect"
                                                 "${BB_COMMON}/Dialect/Buckyball"
                                                 "${CMAKE_CURRENT_BINARY_DIR}"
)

add_mlir_dialect_library(BuddyBuckyballTransforms
  ${BB_TRANSFORMS_SOURCES}
  LINK_LIBS PUBLIC BuddyBuckyball MLIRIR MLIRLLVMCommonConversion MLIRLLVMDialect
)

add_mlir_library(LowerBuckyballPass
  ${BB_LOWER_SOURCES}
  LINK_LIBS PUBLIC BuddyBuckyballTransforms MLIRLinalgDialect MLIRLinalgTransforms MLIRTosaDialect
)

add_mlir_library(LowerTileToBuckyballPass
  ${BB_TILE_SOURCES}
  LINK_LIBS PUBLIC BuddyTile BuddyBuckyball MLIRLinalgDialect
)

add_mlir_translation_library(BuddyBuckyballToLLVMIRTranslation
  ${BB_TRANSLATION_SOURCES}
  DEPENDS BuddyBuckyballConversionsIncGen intrinsics_gen
  LINK_COMPONENTS Core
  LINK_LIBS PUBLIC MLIRIR MLIRSupport BuddyBuckyball
)
