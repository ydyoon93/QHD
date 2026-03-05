#!/usr/bin/env python3
"""Interactive viewer: z-component colormap + in-plane flux contours."""

from __future__ import annotations

import argparse
import pathlib
import sys
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from read_output import discover_steps, files_for_step, read_global_field

VECTOR_FIELDS = {
    "Q": ("Qx", "Qy", "Qz"),
    "B": ("Bx", "By", "Bz"),
    "u": ("Ux", "Uy", "Uz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Directory with step_XXXXXX_rank_YYYY.h5 files")
    parser.add_argument("--start-step", type=int, default=None, help="Initial step (default: latest)")
    parser.add_argument("--levels", type=int, default=16, help="Number of contour levels per subplot")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for z-component background")
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Use fixed color/contour ranges per field from the first displayed step",
    )
    return parser.parse_args()


def safe_limits(arr: np.ndarray) -> tuple[float, float]:
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        eps = max(1.0e-12, abs(vmin) * 1.0e-6 + 1.0e-12)
        return vmin - eps, vmax + eps
    return vmin, vmax


def inplane_flux_function(fx: np.ndarray, fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Recover psi where F_perp ~= (dpsi/dy, -dpsi/dx) via periodic Poisson solve."""
    ny, nx = fx.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    fx_hat = np.fft.fft2(fx)
    fy_hat = np.fft.fft2(fy)

    dfx_dy_hat = (1j * ky[:, None]) * fx_hat
    dfy_dx_hat = (1j * kx[None, :]) * fy_hat
    rhs_hat = dfx_dy_hat - dfy_dx_hat

    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    psi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    mask = k2 > 0.0
    psi_hat[mask] = -rhs_hat[mask] / k2[mask]
    psi = np.fft.ifft2(psi_hat).real
    psi -= np.mean(psi)
    return psi


@lru_cache(maxsize=8)
def load_step(output_dir_str: str, step: int) -> tuple[dict[str, np.ndarray], dict]:
    output_dir = pathlib.Path(output_dir_str)
    step_files = files_for_step(output_dir, step)
    if not step_files:
        raise RuntimeError(f"No files found for step {step}")

    arrays: dict[str, np.ndarray] = {}
    meta = None

    for field_x, field_y, field_z in VECTOR_FIELDS.values():
        arr_x, meta_x = read_global_field(step_files, field_x)
        arr_y, meta_y = read_global_field(step_files, field_y)
        arr_z, meta_z = read_global_field(step_files, field_z)
        arrays[field_x] = arr_x
        arrays[field_y] = arr_y
        arrays[field_z] = arr_z
        if meta is None:
            meta = meta_x
        if (
            meta_x["global_nx"] != meta_y["global_nx"]
            or meta_x["global_ny"] != meta_y["global_ny"]
            or meta_x["global_nx"] != meta_z["global_nx"]
            or meta_x["global_ny"] != meta_z["global_ny"]
        ):
            raise RuntimeError("Inconsistent metadata between vector components")

    assert meta is not None
    return arrays, meta


def compute_fluxes(arrays: dict[str, np.ndarray], meta: dict) -> dict[str, np.ndarray]:
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    fluxes: dict[str, np.ndarray] = {}
    for name, (field_x, field_y, _) in VECTOR_FIELDS.items():
        fluxes[name] = inplane_flux_function(arrays[field_x], arrays[field_y], dx, dy)
    return fluxes


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    steps = discover_steps(output_dir)
    if not steps:
        print(f"No step files found in {output_dir}", file=sys.stderr)
        return 1

    if args.start_step is None:
        step_idx = len(steps) - 1
    else:
        if args.start_step not in steps:
            print(f"start-step={args.start_step} not available. Steps: {steps}", file=sys.stderr)
            return 1
        step_idx = steps.index(args.start_step)

    first_step = steps[step_idx]
    arrays, meta = load_step(str(output_dir), first_step)
    fluxes = compute_fluxes(arrays, meta)

    ny, nx = next(iter(fluxes.values())).shape
    x = (np.arange(nx) + 0.5) * float(meta["dx"])
    y = (np.arange(ny) + 0.5) * float(meta["dy"])

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 13.2))
    fig.subplots_adjust(left=0.1, right=0.93, top=0.94, bottom=0.14, hspace=0.28)

    images = {}
    colorbars = {}
    contour_limits = {}
    z_limits = {}
    for ax, name in zip(axes, ["Q", "B", "u"]):
        field_z = VECTOR_FIELDS[name][2]
        z = arrays[field_z]
        psi = fluxes[name]
        z_vmin, z_vmax = safe_limits(z)
        psi_vmin, psi_vmax = safe_limits(psi)
        z_limits[name] = (z_vmin, z_vmax)
        contour_limits[name] = (psi_vmin, psi_vmax)

        image = ax.imshow(
            z,
            origin="lower",
            cmap=args.cmap,
            interpolation="nearest",
            extent=(x[0], x[-1], y[0], y[-1]),
            vmin=z_vmin,
            vmax=z_vmax,
        )
        levels = np.linspace(psi_vmin, psi_vmax, args.levels)
        ax.contour(x, y, psi, levels=levels, linewidths=0.8, colors="black")
        ax.set_title(f"{field_z} + {name} flux contours")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_z)
        images[name] = image
        colorbars[name] = cbar

    if not args.fixed_scale:
        z_limits = None
        contour_limits = None

    def draw(index: int) -> None:
        step = steps[index]
        arrays_now, meta_now = load_step(str(output_dir), step)
        fluxes_now = compute_fluxes(arrays_now, meta_now)

        for ax, name in zip(axes, ["Q", "B", "u"]):
            psi = fluxes_now[name]
            z = arrays_now[VECTOR_FIELDS[name][2]]
            if contour_limits is None:
                psi_vmin, psi_vmax = safe_limits(psi)
            else:
                psi_vmin, psi_vmax = contour_limits[name]

            if z_limits is None:
                z_vmin, z_vmax = safe_limits(z)
            else:
                z_vmin, z_vmax = z_limits[name]

            levels = np.linspace(psi_vmin, psi_vmax, args.levels)

            ax.clear()
            images[name] = ax.imshow(
                z,
                origin="lower",
                cmap=args.cmap,
                interpolation="nearest",
                extent=(x[0], x[-1], y[0], y[-1]),
                vmin=z_vmin,
                vmax=z_vmax,
            )
            ax.contour(x, y, psi, levels=levels, linewidths=0.8, colors="black")
            field_z = VECTOR_FIELDS[name][2]
            ax.set_title(f"{field_z} + {name} flux contours")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            colorbars[name].update_normal(images[name])

        title.set_text(
            f"step={step}  time={meta_now['time']:.6g}  "
            f"grid={meta_now['global_nx']}x{meta_now['global_ny']}"
        )
        fig.canvas.draw_idle()

    title = fig.suptitle("", fontsize=12)
    draw(step_idx)

    slider_ax = fig.add_axes([0.14, 0.06, 0.72, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="step index",
        valmin=0,
        valmax=len(steps) - 1,
        valinit=step_idx,
        valstep=1,
    )

    def on_slider_change(val: float) -> None:
        draw(int(val))

    slider.on_changed(on_slider_change)

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
