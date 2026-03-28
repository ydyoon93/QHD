import numpy as np
from scipy.interpolate import RectBivariateSpline

nx = 512
ny = 512
lx = 80.0
ly = 80.0
dx = lx / nx
dy = ly / ny

dt_cfl = (dx**-2 + dy**-2) ** (-0.5)
dt = dt_cfl * .5
t_end = 500.0
output_every = max(1, int(t_end / dt / 200))

nu = 8.0e-3
eta = 0

output_dir = "output_turbulence"
B0 = 1.0

number_of_waves = 100
np.random.seed(0)
# Treat B0 as the overall fluctuation scale, not the per-mode scale.
mode_amplitude_std = B0 / np.sqrt(number_of_waves)
dB = np.random.normal(0, mode_amplitude_std, number_of_waves)
kx = 2 * np.pi * np.random.choice(np.arange(-5, 6), number_of_waves) / lx
ky = 2 * np.pi * np.random.choice(np.arange(-5, 6), number_of_waves) / ly
k = np.sqrt(kx**2 + ky**2)
k_hat = np.zeros((2, number_of_waves))
nonzero_k = k > 0.0
k_hat[0, nonzero_k] = kx[nonzero_k] / k[nonzero_k]
k_hat[1, nonzero_k] = ky[nonzero_k] / k[nonzero_k]
phi = np.random.uniform(0, 2 * np.pi, number_of_waves)

x_samples = np.linspace(0.0, lx, nx, endpoint=False)
y_samples = np.linspace(0.0, ly, ny, endpoint=False)
x_grid, y_grid = np.meshgrid(x_samples, y_samples, indexing="xy")
Bx_arr = np.zeros((ny, nx))
By_arr = np.zeros((ny, nx))

for n in range(number_of_waves):
    phase = x_grid * kx[n] + y_grid * ky[n] + phi[n]
    Bx_arr += -dB[n] * k_hat[1, n] * np.sin(phase)
    By_arr += dB[n] * k_hat[0, n] * np.sin(phase)

Bx_interp = RectBivariateSpline(x_samples, y_samples, Bx_arr.T)
By_interp = RectBivariateSpline(x_samples, y_samples, By_arr.T)


def Bx(x, y):
    return Bx_interp.ev(np.mod(x, lx), np.mod(y, ly))


def By(x, y):
    return By_interp.ev(np.mod(x, lx), np.mod(y, ly))


def Bz(x, y):
    return 0.0


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
