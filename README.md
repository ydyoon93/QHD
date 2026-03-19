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

## Output

The solver writes standard AMReX plotfiles:

- `output/plt_XXXXXX`

The solver writes cell-centered plotfile fields by averaging the staggered state:

- `Bx`, `By`, `Bz`
- `Ux`, `Uy`, `Uz`
- `Qx`, `Qy`, `Qz`

The existing reader/viewer scripts still work on those plotfiles.
