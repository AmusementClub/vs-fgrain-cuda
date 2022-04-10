cmake_minimum_required(VERSION 3.20)

project(FilmGrain LANGUAGES CXX CUDA)

add_library(fgrain_kernel OBJECT source/kernel.cu)

set_target_properties(fgrain_kernel PROPERTIES
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_library(fgrain_source OBJECT source/source.cpp)

target_include_directories(fgrain_source PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)

    if(VS_FOUND)
        message(STATUS "Found VapourSynth r${VS_VERSION}")
        target_include_directories(fgrain_source PRIVATE ${VS_INCLUDE_DIRS})
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")

    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(fgrain_source PRIVATE ${VS_INCLUDE_DIR})
endif()

set_target_properties(fgrain_source PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_library(fgrain_cuda SHARED)
target_link_libraries(fgrain_cuda fgrain_source fgrain_kernel)

if(VS_FOUND)
    cmake_path(APPEND install_dir ${VS_LIBDIR} vapoursynth)
    install(TARGETS fgrain_cuda LIBRARY DESTINATION ${install_dir})
else()
    if(WIN32)
        install(TARGETS fgrain_cuda RUNTIME)
    else()
        install(TARGETS fgrain_cuda LIBRARY)
    endif()
endif()
