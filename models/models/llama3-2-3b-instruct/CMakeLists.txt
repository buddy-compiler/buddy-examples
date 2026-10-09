add_library(llama_model OBJECT main.cpp generation.cpp)
target_include_directories(llama_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(llama_model PUBLIC cxx_std_20)
