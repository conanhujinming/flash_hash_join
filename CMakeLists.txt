cmake_minimum_required(VERSION 3.20)

project(
    flash_join
    VERSION 0.2.0
    DESCRIPTION "Adaptive perfect and CedarDB-style unchained joins"
    LANGUAGES CXX
)

option(FLASH_JOIN_NATIVE_ARCH "Tune the extension for the build machine" ON)
option(FLASH_JOIN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(flash_join MODULE src/python_module.cpp)
target_compile_features(flash_join PRIVATE cxx_std_17)
target_include_directories(flash_join PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(flash_join PRIVATE /W4 /permissive-)
    if(FLASH_JOIN_WARNINGS_AS_ERRORS)
        target_compile_options(flash_join PRIVATE /WX)
    endif()
else()
    target_compile_options(flash_join PRIVATE -Wall -Wextra -Wpedantic)
    if(FLASH_JOIN_WARNINGS_AS_ERRORS)
        target_compile_options(flash_join PRIVATE -Werror)
    endif()

    # The CedarDB hash uses hardware CRC32C. A portable fallback remains
    # available, while release builds enable the native instruction set.
    if(FLASH_JOIN_NATIVE_ARCH)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            target_compile_options(flash_join PRIVATE -msse4.2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND NOT APPLE)
            target_compile_options(flash_join PRIVATE -march=armv8-a+crc)
        endif()
    endif()
endif()

set_target_properties(flash_join PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS flash_join LIBRARY DESTINATION . RUNTIME DESTINATION .)
