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

add_subdirectory(external/amrex)

add_executable(qhd_emhd
    src/main.cpp
    src/Config.cpp
    src/VectorOps.cpp
    src/Simulation.cpp
)

target_include_directories(qhd_emhd PRIVATE include)
target_link_libraries(qhd_emhd PRIVATE AMReX::amrex)

if (MSVC)
    target_compile_options(qhd_emhd PRIVATE /W4)
else()
    target_compile_options(qhd_emhd PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()
