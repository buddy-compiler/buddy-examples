set(MODEL_RESNET18_OUTPUT_DIR ${MODEL_RESNET18_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.rax
         ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/params.f32
         ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/weights.i8
         ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/scales.bin
         ${MODEL_RESNET18_OUTPUT_DIR}/forward.mlir
         ${MODEL_RESNET18_OUTPUT_DIR}/subgraph0.mlir
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_RESNET18_DIR}/buddy-resnet-import.py
          --output-dir ${MODEL_RESNET18_OUTPUT_DIR}
          --trace
          --trace-config ${RESNET18_TRACE_CONFIG}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(resnet-model-build DEPENDS
  ${MODEL_RESNET18_OUTPUT_DIR}/forward.mlir
  ${MODEL_RESNET18_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.rax
  ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/params.f32
  ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/weights.i8
  ${MODEL_RESNET18_OUTPUT_DIR}/resnet18.payload/scales.bin
)
