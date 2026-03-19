import numpy as np

nx = 1024
ny = 512
lx = 40.0
ly = 20.0
dx = lx / nx
dy = ly / ny

dt_cfl = (dx**-2 + dy**-2) ** (-0.5)
dt = dt_cfl * .5
t_end = 600.0
output_every = int(t_end / dt / 100)

nu = 5.0e-3
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
