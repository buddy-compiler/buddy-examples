set(MODEL_MOBILENETV3_OUTPUT_DIR ${MODEL_MOBILENETV3_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.rax
         ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/params.f32
         ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/weights.i8
         ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/scales.bin
         ${MODEL_MOBILENETV3_OUTPUT_DIR}/forward.mlir
         ${MODEL_MOBILENETV3_OUTPUT_DIR}/subgraph0.mlir
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_MOBILENETV3_DIR}/buddy-mobilenetv3-import.py
          --output-dir ${MODEL_MOBILENETV3_OUTPUT_DIR}
          --trace
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(mobilenetv3-model-build DEPENDS
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/forward.mlir
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.rax
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/params.f32
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/weights.i8
  ${MODEL_MOBILENETV3_OUTPUT_DIR}/mobilenetv3.payload/scales.bin
)
