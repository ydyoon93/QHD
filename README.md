# 2D Canonical Vorticity EMHD Solver (MPI + OpenMP)

This project provides a modular C++ simulation code for 2D canonical vorticity evolution in EMHD:

- `curl(B) = -u`
- `Q = curl(u) - B = laplacian(B) - B`
- `dQ/dt = curl(u x Q) + nu * laplacian(curl(u)) - eta * laplacian(B)`

## Features

- 2D distributed domain decomposition with MPI Cartesian topology
- OpenMP-parallelized stencil kernels
- Explicit RK4 time integration for `Q`
- Spectral Helmholtz solver for `laplacian(B) - B = Q` on periodic domains
- Double current-sheet initialization with a flux-function perturbation for reconnection studies
- HDF5 output (one file per rank per output step)

## Helmholtz Solver Notes

- The Helmholtz solve now uses a 2D FFT-based spectral inversion of the discrete operator.
- The current implementation requires `nx` and `ny` to be powers of two.
- `helmholtz_max_iter` and `helmholtz_tol` are kept in config files for backward compatibility, but are not used by the spectral solver.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Dependencies:

- MPI
- OpenMP
- HDF5 C library
- CMake >= 3.20

## Run

```bash
mpirun -np 4 ./build/qhd_emhd config/simulation.cfg
```

Output files:

- `output/step_XXXXXX_rank_YYYY.h5`

Each file stores local slab data and metadata attributes (`global_nx`, `global_ny`, local offsets, spacing, time, etc.).
