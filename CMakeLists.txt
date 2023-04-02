cmake_minimum_required(VERSION 3.8)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	message("cuda found")
else()
	message("cuda not found")
endif()
project(MyCudaProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "-gencode arch=compute_37,code=sm_37")

add_executable(main main.cu)
target_link_libraries(main -lcurand -lcublas -lcusolver)

target_compile_features(main PUBLIC cxx_std_11)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
