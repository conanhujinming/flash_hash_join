cmake_minimum_required(VERSION 3.12)
project(flash_join)

# --- Python & Pybind11 ---
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# --- mimalloc (Re-enabled with proper configuration) ---
set(MI_BUILD_TESTS OFF CACHE BOOL "Disable mimalloc tests")
set(MI_BUILD_SHARED OFF CACHE BOOL "Build mimalloc as a static library")
set(MI_BUILD_STATIC ON CACHE BOOL "Build mimalloc as a static library")
set(MI_OVERRIDE ON CACHE BOOL "override system malloc")
set(MI_MALLOC_OVERRIDE ON CACHE BOOL "override malloc globally")
# For ASan to work correctly with mimalloc, we need to ensure mimalloc itself is also compiled with ASan.
# CMake will propagate flags like CMAKE_C_FLAGS and CMAKE_CXX_FLAGS to subdirectories.
add_subdirectory(mimalloc)

# --- Create Python Module Target ---
pybind11_add_module(flash_join SHARED hash_join.cpp)

# --- Link Libraries and Set Include Directories ---
target_include_directories(flash_join PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include)
target_link_libraries(flash_join PRIVATE 
    mimalloc-static
    pthread
    pybind11::module
    Python::Module
)

# --- Compiler Flags ---
target_compile_features(flash_join PRIVATE cxx_std_17)

# We check the build type. If it's a "Release" build (or empty), we use performance flags.
# Otherwise, we set up for debugging with ASan.
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for Release build")
    if (MSVC)
        target_compile_options(flash_join PRIVATE /O2 /arch:AVX2)
    else()
        target_compile_options(flash_join PRIVATE -O3 -march=native -fvisibility=hidden)
    endif()
else()
    message(STATUS "Configuring for Debug/ASan build")
    if (MSVC)
        # MSVC's ASan support is different. This example focuses on GCC/Clang.
        message(WARNING "AddressSanitizer is not configured for MSVC in this script.")
        target_compile_options(flash_join PRIVATE /Zi /Ob0) # Basic debug flags for MSVC
    else()
        # --- Flags for GCC/Clang with ASan and Debug Info ---
        message(STATUS "Enabling AddressSanitizer and debug flags.")
        # General debug flags
        target_compile_options(flash_join PRIVATE
            -g                 # Generate debug information
            -O1                # Use a low optimization level (-O0 can sometimes be too slow)
            -Wall              # Enable all warnings
            -Wextra            # Enable extra warnings
            -fno-omit-frame-pointer # Needed for reliable stack traces
            -march=native
        )
        # ASan specific flags for both compiler and linker
        # target_compile_options(flash_join PRIVATE -fsanitize=address)
        # target_link_options(flash_join PRIVATE -fsanitize=address)
    endif()
endif()