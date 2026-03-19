# EQHD: 2D Staggered EMHD Solver

This repository contains a 2D single-level EMHD solver built on AMReX.
It is aimed at users who want to:

- run periodic reconnection-style EMHD problems on a uniform grid
- initialize from analytic Python functions for `B` or `Q`
- test pressure-tensor closures, including Python-defined closures evaluated every timestep
- inspect output interactively or export an MP4 movie

The current code is:

- single-level only, with no AMR
- periodic in `x` and `y`
- 2D in space, with staggered storage for vector components
- FFT-based for the Helmholtz inversion used to recover `B` from `Q`

## Quick Start

Build:

```bash
cmake -S . -B build
cmake --build build -j
```

Run the short smoke test:

```bash
./build/qhd_emhd config/smoke.py
```

Run the main example:

```bash
./build/qhd_emhd config/simulation.py
```

You can also use the legacy `.cfg` files:

```bash
./build/qhd_emhd config/smoke.cfg
./build/qhd_emhd config/simulation.cfg
```

The solver deletes and recreates the chosen `output_dir` at the start of each run.

## What You Edit

The preferred input format is Python. The main example is `config/simulation.py`.

Typical top-level parameters are:

- `nx`, `ny`, `lx`, `ly`
- `dt`, `t_end`, `output_every`
- `nu`, `eta`
- `output_dir`

The solver uses the user-specified `dt` directly. Only the last step may be shortened so the run lands exactly on `t_end`.

## Initial Conditions

You can initialize either `B` or `Q` by defining Python functions:

- `Bx(x, y)`, `By(x, y)`, `Bz(x, y)`
- `Qx(x, y)`, `Qy(x, y)`, `Qz(x, y)`

If only `B` is provided, the solver computes `Q`.
If only `Q` is provided, the solver solves for `B`.

These functions are evaluated on the solver staggering:

- `Bx`, `Qx`: horizontal edges
- `By`, `Qy`: vertical edges
- `Bz`, `Qz`: cell faces

## Pressure Closure

If you do nothing, the solver uses the built-in symmetric viscous closure

- `P = -nu * (grad(u) + grad(u)^T)`

If you want to test a custom closure, define:

- `pressure_closure(ctx)`

in the Python namelist.

It can work in either of these forms:

- return a dict with `xx`, `xy`, `xz`, `yy`, `yz`, `zz`
- write into `ctx["p_out"]` in place and return `None`

For performance, you can also declare what your closure needs:

- `pressure_closure_dependencies = ("u",)`

Valid dependency names are `b`, `q`, `u`, and `p`.
If this is omitted, all four are provided.

`ctx` contains:

- `t`, `dt`, `step`
- `nx`, `ny`, `lx`, `ly`, `dx`, `dy`
- `p_out` as writable NumPy arrays
- `b`, `q`, `u`, `p` as NumPy-array dicts when requested by the dependency list
- `coords` containing staggered `x`/`y` meshes for each component location

The current embedded-Python path lets you write closures using normal NumPy operations on the global periodic arrays, including `np.roll` if that is what your closure assumes.

## Output

The solver writes AMReX plotfiles:

- `output/plt_XXXXXX`

The saved plotfile variables are cell-centered views of the staggered state:

- `Bx`, `By`, `Bz`
- `Ux`, `Uy`, `Uz`
- `Qx`, `Qy`, `Qz`
- `Pxx`, `Pxy`, `Pxz`, `Pyy`, `Pyz`, `Pzz`
- `TrP`

## Inspecting Results

Read a single field from a plotfile:

```bash
python3 scripts/read_output.py --output-dir output --field Pxy --step 0
```

Open the interactive viewer:

```bash
python3 scripts/view_output_slider.py --output-dir output
```

The viewer shows:

- `Qz` with in-plane `Q` flux contours
- `Bz` with in-plane `B` flux contours
- `Uz` with in-plane `u` flux contours
- `Pxy` with the current AMReX grid layout

Export an MP4 movie:

```bash
python3 scripts/make_output_movie.py --output-dir output --mp4 output.mp4 --fps 5
```

That is a good default for a run that writes about 100 frames.

## Build Requirements

The code expects:

- C++20
- AMReX, provided through the vendored `external/amrex` tree
- MPI
- OpenMP
- FFTW, because the Helmholtz inversion uses AMReX FFT on CPU
- Python 3 interpreter and development headers
- `pybind11` headers, either from a normal install or from a vendored include tree such as PyTorch

## Numerical Model

The solver evolves:

- `u = -curl(B)`
- `Q = laplacian(B) - B`
- `dQ/dt = curl(u x Q) - curl(div(P)) - eta laplacian(B)`

The magnetic field is recovered from `Q` by solving:

- `(I - laplacian) B = -Q`

with an AMReX FFT-based discrete Helmholtz solve on the periodic grid.

The vector staggering is:

- `x` components on horizontal edges
- `y` components on vertical edges
- `z` components on cell faces

The pressure tensor is stored in its six independent symmetric components and enters the `Q` equation through `curl(div(P))`.

## Current Limitations

- No AMR
- No user-selectable physical boundary conditions yet
- Periodic `x` and periodic `y` only
- Single-level uniform grid only
- The FFT Helmholtz path assumes a periodic domain

If you need non-periodic boundary conditions, the main required change is to replace the current FFT Helmholtz solve with a boundary-aware AMReX linear solve and then make ghost filling BC-aware on the staggered fields.
