cmake_minimum_required(VERSION 3.10)

project(FHE_Profile_Framework)

set(CMAKE_BUILD_TYPE        Debug)
set(CMAKE_CXX_STANDARD      17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-pg -fno-omit-frame-pointer -O0")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-pg")

# --- 通用 Include 路径 ---
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/include/SEAL-4.1        # SEAL 头文件
    /usr/local/include/openfhe         # OpenFHE 头文件
)

# --- 通用 Library 路径 ---
link_directories(/usr/local/lib)       # SEAL 和 OpenFHE 库都在这里

# --- SEAL 应用程序列表 ---
add_executable(profile_framewk
    src/framewk.cpp
)
target_link_libraries(profile_framewk
    seal-4.1    # 系统安装的 libseal-4.1.so
    z
    zstd
)

add_executable(dotprod_bfv
    src/dotprod_bfv.cpp
)
target_link_libraries(dotprod_bfv
    seal-4.1
    z
    zstd
)

add_executable(dotprod_ckks
    src/dotprod_ckks.cpp
)
target_link_libraries(dotprod_ckks
    seal-4.1
    z
    zstd
)

add_executable(matmul_ckks
    src/matmul_ckks.cpp
)
target_link_libraries(matmul_ckks
    seal-4.1
    z
    zstd
)



