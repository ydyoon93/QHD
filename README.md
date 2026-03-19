# 2D Staggered EMHD Solver (AMReX)

This code now runs a staggered single-level AMReX solver with:

- single-level layout only
- staggered vector storage:
  `x` on horizontal edges, `y` on vertical edges, `z` on cell faces
- periodic boundaries in `x` and `y`

The model is evolved with a cell-centered discretization of

- `u = -curl(B)`
- `Q = laplacian(B) - B`
- `dQ/dt = curl(u x Q) - curl(div(P)) - eta laplacian(B)`

The staggered curl operators preserve the discrete Stokes pairing exactly. By default
the pressure contribution uses the symmetric closure
`P = -nu * (grad(u) + grad(u)^T)`.

`B` is recovered from `Q` by solving

- `(I - laplacian) B = -Q`

component-by-component with AMReX's FFT-based discrete Helmholtz inversion on
periodic grids. On CPU builds this uses FFTW through AMReX.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The build expects OpenMP-enabled `fftw3` to be available because the Helmholtz
inversion now uses AMReX FFT on CPU and links the threaded FFTW backend when
`AMReX_OMP` is enabled.

If you use Python-defined pressure closures, the build also needs Python 3
development headers and `pybind11` headers. CMake first looks for a normal
`pybind11` package and otherwise falls back to any local include tree that
vendors the `pybind11` headers.

## Run

```bash
./build/qhd_emhd config/simulation.py
```

or

```bash
./build/qhd_emhd config/smoke.py
```

## Inputs

The preferred input format is Python. Top-level scalar values such as

- `nx`, `ny`, `lx`, `ly`
- `dt`, `t_end`, `output_every`
- `nu`, `eta`
- `output_dir`

are read directly.

The solver advances with the user-provided `dt`. Only the last step may be shortened
so the run lands exactly on `t_end`.

If the same Python file also defines any of

- `Bx(x, y)`, `By(x, y)`, `Bz(x, y)`
- `Qx(x, y)`, `Qy(x, y)`, `Qz(x, y)`

then it is used as the initialization namelist automatically.

Those functions are evaluated on the solver staggering:

- `Bx`, `Qx`: horizontal edges
- `By`, `Qy`: vertical edges
- `Bz`, `Qz`: cell faces

If only `B` is provided, the solver computes `Q`.
If only `Q` is provided, the solver solves for `B`.

To override the default viscous closure, define

- `pressure_closure(ctx)`

in the Python namelist. It may either return a dict with the six independent
symmetric components

- `xx`, `xy`, `xz`, `yy`, `yz`, `zz`

or write them in-place into `ctx["p_out"]` and return `None`.

The closure is evaluated every timestep after `u` is updated from the latest `B`.
The runtime now embeds Python with `pybind11`, so this happens in-process
without spawning a new `python3` process or writing temporary `.npy` files per
timestep.
`ctx` contains:

- `t`, `dt`, `step`
- `nx`, `ny`, `lx`, `ly`, `dx`, `dy`
- `p_out` as a dict of writable NumPy arrays for in-place output
- `b`, `q`, `u`, `p` as dicts of NumPy arrays when requested by the closure
- `coords` with staggered `x`/`y` meshes for `Bx`, `By`, `Bz`, `Qx`, `Qy`, `Qz`,
  `Ux`, `Uy`, `Uz`, `Pxx`, `Pxy`, `Pxz`, `Pyy`, `Pyz`, `Pzz`

If the namelist defines

- `pressure_closure_dependencies = ("u",)`

then only those input families are gathered into Python. Valid dependency names
are `b`, `q`, `u`, and `p`. If `pressure_closure_dependencies` is omitted, all
four are provided.

If `pressure_closure` is absent, the solver falls back to the built-in symmetric
viscous closure `P = -nu * (grad(u) + grad(u)^T)`.

## Output

The solver writes standard AMReX plotfiles:

- `output/plt_XXXXXX`

The solver writes cell-centered plotfile fields by averaging the staggered state:

- `Bx`, `By`, `Bz`
- `Ux`, `Uy`, `Uz`
- `Qx`, `Qy`, `Qz`
- `Pxx`, `Pxy`, `Pxz`, `Pyy`, `Pyz`, `Pzz`
- `TrP`

The viewer uses the saved `TrP` field when it is present and only falls back to
the old viscosity-based estimate for older plotfiles.
