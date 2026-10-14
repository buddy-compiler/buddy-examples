set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

#===------------------------------------------------------------------===#
# Train LeNet and save its checkpoint.
#===------------------------------------------------------------------===#
add_custom_command(
  OUTPUT ${OUTPUT_DIR}/lenet-model.pth
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${PYTHON} prepare.py
            --output ${OUTPUT_DIR}/lenet-model.pth
            --data ${OUTPUT_DIR}/datasets
            --epochs 150
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/prepare.py
          ${CMAKE_CURRENT_SOURCE_DIR}/model.py
          ${CMAKE_CURRENT_SOURCE_DIR}/configs/importer-param.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Preparing LeNet checkpoint"
  VERBATIM
)
add_custom_target(lenet-prepare DEPENDS ${OUTPUT_DIR}/lenet-model.pth)
