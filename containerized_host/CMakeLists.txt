cmake_minimum_required(VERSION 3.10)
project(cuda_ptx_runner LANGUAGES C)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags and options
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try the real path; if not found, fallback to stubs
find_library(MY_CUDA_LIBRARY
  NAMES cuda
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/cuda/lib64/stubs
)

if (NOT MY_CUDA_LIBRARY)
  message(FATAL_ERROR "Could not find libcuda.so (nor stub). 
                      Please install or provide the correct path.")
endif()

# Create shared library with explicit CUDA runtime linking
add_library(cuda_ptx_runner SHARED tiny_ptx.c)
target_link_libraries(cuda_ptx_runner 
  PRIVATE
    ${CUDA_LIBRARIES}
    ${MY_CUDA_LIBRARY}
)
set_target_properties(cuda_ptx_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Create executable for testing
add_executable(tiny_ptx tiny_ptx.c)
target_link_libraries(tiny_ptx
  PRIVATE
    ${CUDA_LIBRARIES}
    ${MY_CUDA_LIBRARY}
)
