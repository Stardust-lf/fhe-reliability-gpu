cmake_minimum_required(VERSION 3.18)
project(PhantomFHEExamples LANGUAGES CXX CUDA)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phantom-FHE include directories
set(PHANTOMFHE_INCLUDE ${CMAKE_SOURCE_DIR}/../phantom-fhe/include)
set(PHANTOMFHE_EXAMPLES ${CMAKE_SOURCE_DIR}/../phantom-fhe/examples)
include_directories(
    ${PHANTOMFHE_INCLUDE}
    ${PHANTOMFHE_EXAMPLES}
)

# Phantom-FHE library directory and linkage
link_directories(${CMAKE_SOURCE_DIR}/../phantom-fhe/build/lib)
find_library(PHANTOM_LIB
  NAMES Phantom
  PATHS ${CMAKE_SOURCE_DIR}/../phantom-fhe/build/lib
  REQUIRED
)

# List of example targets and their source files
set(MY_TARGETS
  ntt_test
  ntt_real_test
  dotprod_test
  dotprod_real_test
  gemm_test
)
set(MY_SOURCES
  ntt_test.cu
  ntt_real_test.cu
  dotprod_test.cu
  dotprod_real_test.cu
  naive_gemm_test.cu
)

# Create each executable
foreach(idx RANGE 0 4)
  list(GET MY_TARGETS ${idx} target)
  list(GET MY_SOURCES ${idx} src)

  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE
    ${PHANTOMFHE_INCLUDE}
    ${PHANTOMFHE_EXAMPLES}
  )
  target_link_libraries(${target} PRIVATE ${PHANTOM_LIB})
endforeach()
