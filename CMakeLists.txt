cmake_minimum_required(VERSION 3.12)
project(flash_join)

# --- Python & Pybind11 ---
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- mimalloc (Re-enabled with proper configuration) ---
set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests")
set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as a static library")
set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as a static library")
set(MI_OVERRIDE OFF CACHE BOOL "Don't override system malloc")
set(MI_MALLOC_OVERRIDE OFF CACHE BOOL "Don't override malloc globally")
add_subdirectory(mimalloc)

# --- Create Python Module Target ---
pybind11_add_module(flash_join SHARED hash_join.cpp)

# --- Link Libraries and Set Include Directories ---

target_include_directories(flash_join PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include)

target_link_libraries(flash_join PRIVATE mimalloc-static
    pthread
    pybind11::module
    Python::Module
    OpenMP::OpenMP_CXX
    )

# --- Compiler Flags for Performance ---
target_compile_features(flash_join PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(flash_join PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(flash_join PRIVATE -O3 -march=native -fvisibility=hidden)
endif()