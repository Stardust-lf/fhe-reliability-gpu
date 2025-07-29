cmake_minimum_required(VERSION 3.18)
project(BaseConvCUDA LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Define executable target, include both CUDA and CPU versions
add_executable(baseConv_cpp
    src/baseConv.cpp
)

add_executable(baseConv_cu
    src/baseConv.cu
)

# If you have headers, you can add an include directory
# target_include_directories(baseConv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link any additional libraries if needed
# target_link_libraries(baseConv PRIVATE <your_libs_here>)

# Set CUDA architectures if desired (e.g., sm_60, sm_70)
# set_target_properties(baseConv PROPERTIES CUDA_ARCHITECTURES "60;70;75")

# Optionally adjust optimization flags
target_compile_options(baseConv_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_options(baseConv_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
