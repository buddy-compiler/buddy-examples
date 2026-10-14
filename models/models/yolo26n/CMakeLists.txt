set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

add_custom_command(
  OUTPUT ${OUTPUT_DIR}/yolo26n.pt
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUTPUT_DIR} ${PYTHON} -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
  VERBATIM
)
add_custom_target(yolo26n-prepare DEPENDS ${OUTPUT_DIR}/yolo26n.pt)
