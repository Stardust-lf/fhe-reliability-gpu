cmake_minimum_required(VERSION 3.10)

project(FHE_Profile_Framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/SEAL-4.1)
# include_directories(${PROJECT_SOURCE_DIR}/include/SEAL/native/src)
# include_directories(${PROJECT_SOURCE_DIR}/include/SEAL/native/tests)
# include_directories(${PROJECT_SOURCE_DIR}/include/SEAL/native/examples)

link_directories(${PROJECT_SOURCE_DIR}/lib)

set(SOURCES
    src/framewk.cpp
)

add_executable(profile_framewk ${SOURCES})

target_link_libraries(profile_framewk
    seal-4.1
    z
    zstd
)

message(STATUS "SEAL, Zlib, and Zstd linked successfully.")
