import numpy as np

nx = 64
ny = 64
lx = 20.0
ly = 20.0

dt = 0.01
t_end = 0.05
output_every = 2

nu = 1.0e-3
eta = 1.0e-3

output_dir = "output_smoke_py"

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
