cmake_minimum_required(VERSION 3.12)
project(fast_join)

# --- Python & Pybind11 ---
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- mimalloc (Build from Subdirectory) ---
set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests")
set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as a static library")
set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as a static library")
add_subdirectory(mimalloc)

# --- Create Python Module Target ---
pybind11_add_module(fast_join hash_join.cpp) #<-- 我把你的文件名改成了hash_join.cpp

# --- Link Libraries and Set Include Directories ---

# ************************ 关键修复 ************************
# 明确地告诉 fast_join 目标去 mimalloc/include 目录寻找头文件
target_include_directories(fast_join PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include)
# **********************************************************

target_link_libraries(fast_join PRIVATE
    mimalloc-static
    pthread
    pybind11::module
    Python::Module
    OpenMP::OpenMP_CXX
)

# --- Compiler Flags for Performance ---
target_compile_features(fast_join PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(fast_join PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(fast_join PRIVATE -O3 -march=native -fvisibility=hidden)
endif()