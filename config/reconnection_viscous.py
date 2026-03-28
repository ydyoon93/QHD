import numpy as np

nx = 1024
ny = 1024
lx = 80.0
ly = 40.0
dx = lx / nx
dy = ly / ny

dt_cfl = (dx**-2 + dy**-2) ** (-0.5)
dt = dt_cfl * .5
t_end = 800.0
output_every = int(t_end / dt / 200)

nu = 1.0e-3
eta = 0

output_dir = "output"

B0 = 1.0
sigma = 2.0
psi0 = 0.1


def _sheet_phase(y):
    return (ly / (2.0 * np.pi * sigma)) * np.cos(2.0 * np.pi * y / ly)


def Bx(x, y):
    bx_eq = B0 * np.tanh(_sheet_phase(y))
    perturb = 2.0 * (2.0 * np.pi / ly) * psi0 * np.cos(2.0 * np.pi * (x - 0.25 * lx) / lx) * np.sin(
        4.0 * np.pi * (y - 0.25 * ly) / ly
    )
    return bx_eq + perturb


def By(x, y):
    return -psi0 * (2.0 * np.pi / lx) * np.sin(2.0 * np.pi * (x - 0.25 * lx) / lx) * np.cos(
        4.0 * np.pi * (y - 0.25 * ly) / ly
    )


def Bz(x, y):
    return B0 / np.cosh(_sheet_phase(y))


pressure_closure_dependencies = ("u",)


def pressure_closure(ctx):
    ux = ctx["u"]["x"]
    uy = ctx["u"]["y"]
    uz = ctx["u"]["z"]
    dx = ctx["dx"]
    dy = ctx["dy"]
    p_out = ctx["p_out"]

    p_out["xx"][:] = -2.0 * nu * (ux - np.roll(ux, 1, axis=1)) / dx
    p_out["xy"][:] = -nu * ((np.roll(ux, -1, axis=0) - ux) / dy + (np.roll(uy, -1, axis=1) - uy) / dx)
    p_out["xz"][:] = -nu * (uz - np.roll(uz, 1, axis=1)) / dx
    p_out["yy"][:] = -2.0 * nu * (uy - np.roll(uy, 1, axis=0)) / dy
    p_out["yz"][:] = -nu * (uz - np.roll(uz, 1, axis=0)) / dy
    p_out["zz"][:] = 0.0

    return None
