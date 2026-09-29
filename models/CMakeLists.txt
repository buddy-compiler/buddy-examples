cmake_minimum_required(VERSION 3.22)
project(ModelBuild LANGUAGES NONE)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include("${REPO_ROOT}/stack/serving/CMakeLists.txt")
