import numpy as np

nx = 256
ny = 256
lx = 40.0
ly = 40.0
dx = lx / nx
dy = ly / ny

dt_cfl = (dx**-2 + dy**-2) ** (-0.5)
dt = dt_cfl * .5
t_end = 500.0
output_every = int(t_end / dt / 100)

nu = 0.01
eta = 0
output_dir = "output_Hesse"

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
    return B0 / np.cosh(_sheet_phase(y)) * 0


pressure_closure_dependencies = ("u",)

T = 0.1
def pressure_closure(ctx):
    ux = ctx["u"]["x"]
    uy = ctx["u"]["y"]
    dx = ctx["dx"]
    dy = ctx["dy"]
    p_out = ctx["p_out"]

    # dux/dx and duy/dy both land on nodes.
    dux_dx_node = (np.roll(ux, -1, axis=1) - ux) / dx
    duy_dy_node = (np.roll(uy, -1, axis=0) - uy) / dy

    # Move node-centered derivatives to the tensor grids.
    # Pxz is on vertical edges: average in y.
    dux_dx_on_pxz = 0.5 * (dux_dx_node + np.roll(dux_dx_node, -1, axis=0))

    # Pyz is on horizontal edges: average in x.
    duy_dy_on_pyz = 0.5 * (duy_dy_node + np.roll(duy_dy_node, -1, axis=1))

    p_out["xx"][:] = T
    p_out["xy"][:] = 0
    p_out["xz"][:] = - nu * dux_dx_on_pxz
    p_out["yy"][:] = T
    p_out["yz"][:] = - nu * duy_dy_on_pyz
    p_out["zz"][:] = T

    return None

