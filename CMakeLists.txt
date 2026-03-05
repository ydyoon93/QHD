cmake_minimum_required(VERSION 3.20)
project(qhd_emhd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(SOURCES
    src/main.cpp
    src/Config.cpp
    src/Grid2D.cpp
    src/Field.cpp
    src/VectorOps.cpp
    src/Physics.cpp
    src/HelmholtzSolver.cpp
    src/HDF5Writer.cpp
    src/Simulation.cpp
)

add_executable(qhd_emhd ${SOURCES})

target_include_directories(qhd_emhd
    PRIVATE
        include
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(qhd_emhd
    PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        ${HDF5_LIBRARIES}
)

if (MSVC)
    target_compile_options(qhd_emhd PRIVATE /W4)
else()
    target_compile_options(qhd_emhd PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()
