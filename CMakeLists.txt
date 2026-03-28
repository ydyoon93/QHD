cmake_minimum_required(VERSION 3.20)
project(qhd_emhd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(AMReX_SPACEDIM 2 CACHE STRING "")
set(AMReX_MPI YES CACHE BOOL "")
set(AMReX_OMP YES CACHE BOOL "")
set(AMReX_GPU_BACKEND NONE CACHE STRING "")
set(AMReX_PLOTFILE_TOOLS NO CACHE BOOL "")
set(AMReX_FORTRAN NO CACHE BOOL "")
set(AMReX_FFT YES CACHE BOOL "" FORCE)
set(AMReX_HDF5 YES CACHE BOOL "" FORCE)
set(AMReX_PARTICLES NO CACHE BOOL "" FORCE)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if (AMReX_HDF5)
    find_program(HDF5_MPI_C_COMPILER h5pcc)
    if (HDF5_MPI_C_COMPILER AND NOT HDF5_C_COMPILER_EXECUTABLE)
        set(HDF5_C_COMPILER_EXECUTABLE "${HDF5_MPI_C_COMPILER}" CACHE FILEPATH "Parallel HDF5 compiler wrapper")
    endif()
    if (HDF5_C_COMPILER_EXECUTABLE)
        execute_process(
            COMMAND "${HDF5_C_COMPILER_EXECUTABLE}" -show
            OUTPUT_VARIABLE HDF5_WRAPPER_SHOW_OUTPUT
            ERROR_VARIABLE HDF5_WRAPPER_SHOW_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCHALL "(^| )-I[^ ]+" HDF5_WRAPPER_INCLUDE_FLAGS "${HDF5_WRAPPER_SHOW_OUTPUT}")
        foreach(_hdf5_include_flag IN LISTS HDF5_WRAPPER_INCLUDE_FLAGS)
            string(STRIP "${_hdf5_include_flag}" _hdf5_include_flag)
            string(REGEX REPLACE "^-I" "" _hdf5_include_dir "${_hdf5_include_flag}")
            list(APPEND HDF5_WRAPPER_INCLUDE_DIRS "${_hdf5_include_dir}")
        endforeach()
        list(REMOVE_DUPLICATES HDF5_WRAPPER_INCLUDE_DIRS)
    endif()
endif()

add_subdirectory(external/amrex)

if (AMReX_HDF5)
    set(AMREX_LIB_TARGET "amrex_${AMReX_SPACEDIM}d")
    if (TARGET ${AMREX_LIB_TARGET})
        if (HDF5_WRAPPER_INCLUDE_DIRS)
            # Ensure AMReX sees the HDF5 headers that match the linked parallel HDF5 library
            # even when another package manager has a different hdf5.h on the default include path.
            target_include_directories(${AMREX_LIB_TARGET} BEFORE PUBLIC ${HDF5_WRAPPER_INCLUDE_DIRS})
            foreach(_hdf5_include_dir IN LISTS HDF5_WRAPPER_INCLUDE_DIRS)
                target_compile_options(${AMREX_LIB_TARGET} BEFORE PUBLIC "-I${_hdf5_include_dir}")
            endforeach()
        endif()
    endif()
endif()

add_executable(qhd_emhd
    src/main.cpp
    src/Config.cpp
    src/VectorOps.cpp
    src/Simulation.cpp
)

target_include_directories(qhd_emhd PRIVATE include)
target_link_libraries(qhd_emhd PRIVATE AMReX::amrex)

if (pybind11_FOUND)
    target_link_libraries(qhd_emhd PRIVATE pybind11::embed)
else()
    file(GLOB PYBIND11_HINT_DIRS
        "${Python3_SITELIB}/pybind11/include"
        "${Python3_SITEARCH}/pybind11/include"
        "${Python3_SITELIB}/torch/include"
        "${Python3_SITEARCH}/torch/include"
        "/opt/homebrew/envs/*/lib/python*/site-packages/torch/include"
        "/opt/homebrew/lib/python*/site-packages/torch/include"
        "/usr/local/lib/python*/site-packages/torch/include"
        "/usr/local/lib/python*/dist-packages/torch/include"
    )
    list(REMOVE_DUPLICATES PYBIND11_HINT_DIRS)
    find_path(PYBIND11_INCLUDE_DIR pybind11/embed.h HINTS ${PYBIND11_HINT_DIRS})
    if (NOT PYBIND11_INCLUDE_DIR)
        message(FATAL_ERROR "pybind11 headers not found. Install pybind11 or provide a torch include tree with vendored pybind11.")
    endif()
    target_include_directories(qhd_emhd PRIVATE ${PYBIND11_INCLUDE_DIR})
    target_link_libraries(qhd_emhd PRIVATE Python3::Python)
endif()

if (MSVC)
    target_compile_options(qhd_emhd PRIVATE /W4)
else()
    target_compile_options(qhd_emhd PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()
