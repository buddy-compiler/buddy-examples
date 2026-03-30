file(GLOB BUDDYNEXT_MLIR_FILES CONFIGURE_DEPENDS
  ${MODEL_BUDDYNEXT_DIR}/*.mlir
)

add_custom_target(buddynext-model-build DEPENDS
  ${BUDDYNEXT_MLIR_FILES}
)
