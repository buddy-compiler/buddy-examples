set(MODEL_CONFIG_ARGS)
if(DEFINED CHECKPOINT)
  list(APPEND MODEL_CONFIG_ARGS --checkpoint "${CHECKPOINT}")
endif()
set(MODEL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/model.toml ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py)
add_custom_command(
  OUTPUT ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
         ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
         ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
         ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_STABLEDIFFUSION_DIR}/import-stable-diffusion.py ${MODEL_CONFIG_ARGS}
          --output-dir ${MODEL_STABLEDIFFUSION_DIR}
          --trace
  DEPENDS ${MODEL_CONFIG_FILES}
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_target(stable-diffusion-model-build DEPENDS
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_text_encoder.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg1_text_encoder.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_unet.data
  ${MODEL_STABLEDIFFUSION_DIR}/arg0_vae.data
  ${MODEL_STABLEDIFFUSION_DIR}/forward_text_encoder.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_text_encoder.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/forward_unet.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_unet.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/forward_vae.mlir
  ${MODEL_STABLEDIFFUSION_DIR}/subgraph0_vae.mlir
)
