add_custom_command(
  OUTPUT ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
         ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
  COMMAND ${Python3_EXECUTABLE} ${MODEL_MINIMAX_H3_REF2VA_DIR}/import-minimax-h3-ref2va.py
          --output-dir ${MODEL_MINIMAX_H3_REF2VA_DIR}
          --trace
  COMMENT "Generating MiniMax-H3 Ref2VA MLIR and parameter files"
)

add_custom_target(minimax-h3-ref2va-model-build DEPENDS
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_text_encoder.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_transformer.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_visual_vae.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/arg0_audio_vae.data
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_text_encoder.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_text_encoder.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_transformer.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_transformer.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_visual_vae_encode.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_visual_vae_encode.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/forward_audio_vae.mlir
  ${MODEL_MINIMAX_H3_REF2VA_DIR}/subgraph0_audio_vae.mlir
)
