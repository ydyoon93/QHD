#!/usr/bin/env python3
"""Interactive viewer for AMReX plotfiles with grid-layout panel."""

from __future__ import annotations

import argparse
import pathlib
import sys
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import numpy as np

from read_output import discover_steps, plotfile_for_step, read_global_field, read_grid_layout

VECTOR_FIELDS = {
    "Q": ("Qx", "Qy", "Qz"),
    "B": ("Bx", "By", "Bz"),
    "u": ("Ux", "Uy", "Uz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Directory with AMReX plotfiles `plt_XXXXXX` or `plt_XXXXXX.h5`")
    parser.add_argument("--start-step", type=int, default=None, help="Initial step (default: latest)")
    parser.add_argument("--levels", type=int, default=16, help="Number of contour levels per subplot")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for z-component background")
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Use fixed robust color/contour ranges scanned over all available steps",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=1.0,
        help="Clip this percentile from each tail when computing fixed global ranges (default: 1.0)",
    )
    return parser.parse_args()


def safe_limits(arr: np.ndarray) -> tuple[float, float]:
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        eps = max(1.0e-12, abs(vmin) * 1.0e-6 + 1.0e-12)
        return vmin - eps, vmax + eps
    return vmin, vmax


def robust_limits(arr: np.ndarray, outlier_percentile: float) -> tuple[float, float]:
    if outlier_percentile <= 0.0:
        return safe_limits(arr)
    if outlier_percentile >= 50.0:
        raise ValueError("outlier_percentile must be in [0, 50)")

    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0

    vmin = float(np.percentile(finite, outlier_percentile))
    vmax = float(np.percentile(finite, 100.0 - outlier_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return safe_limits(finite)
    if np.isclose(vmin, vmax):
        return safe_limits(finite)
    return vmin, vmax


def merge_limits(limits: list[tuple[float, float]]) -> tuple[float, float]:
    if not limits:
        return -1.0, 1.0
    vmin = min(pair[0] for pair in limits)
    vmax = max(pair[1] for pair in limits)
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
    plotfile_dir = plotfile_for_step(output_dir, step)
    if not plotfile_dir.exists():
        raise RuntimeError(f"No plotfile found for step {step}")

    arrays: dict[str, np.ndarray] = {}
    meta = read_grid_layout(plotfile_dir)

    for field_x, field_y, field_z in VECTOR_FIELDS.values():
        arr_x, meta_x = read_global_field(plotfile_dir, field_x)
        arr_y, meta_y = read_global_field(plotfile_dir, field_y)
        arr_z, meta_z = read_global_field(plotfile_dir, field_z)
        arrays[field_x] = arr_x
        arrays[field_y] = arr_y
        arrays[field_z] = arr_z
        meta["time"] = meta_x["time"]
        if (
            meta_x["global_nx"] != meta_y["global_nx"]
            or meta_x["global_ny"] != meta_y["global_ny"]
            or meta_x["global_nx"] != meta_z["global_nx"]
            or meta_x["global_ny"] != meta_z["global_ny"]
        ):
            raise RuntimeError("Inconsistent metadata between vector components")

    try:
        arrays["Pxy"], _ = read_global_field(plotfile_dir, "Pxy")
    except RuntimeError:
        pass

    return arrays, meta


def compute_fluxes(arrays: dict[str, np.ndarray], meta: dict) -> dict[str, np.ndarray]:
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    fluxes: dict[str, np.ndarray] = {}
    for name, (field_x, field_y, _) in VECTOR_FIELDS.items():
        fluxes[name] = inplane_flux_function(arrays[field_x], arrays[field_y], dx, dy)
    return fluxes


def compute_global_display_limits(
    output_dir: pathlib.Path,
    steps: list[int],
    outlier_percentile: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float]]:
    z_limits_per_step: dict[str, list[tuple[float, float]]] = {name: [] for name in VECTOR_FIELDS}
    contour_limits_per_step: dict[str, list[tuple[float, float]]] = {name: [] for name in VECTOR_FIELDS}
    pxy_limits_per_step: list[tuple[float, float]] = []

    for idx, step in enumerate(steps, start=1):
        arrays, meta = load_step(str(output_dir), step)
        if "Pxy" not in arrays:
            raise RuntimeError(f"Pxy is not available in plotfile for step {step}")
        fluxes = compute_fluxes(arrays, meta)

        for name in VECTOR_FIELDS:
            z = arrays[VECTOR_FIELDS[name][2]]
            psi = fluxes[name]
            z_limits_per_step[name].append(robust_limits(z, outlier_percentile))
            contour_limits_per_step[name].append(robust_limits(psi, outlier_percentile))

        pxy_limits_per_step.append(robust_limits(arrays["Pxy"], outlier_percentile))

        if idx == 1 or idx == len(steps) or idx % 10 == 0:
            print(f"[scan] {idx}/{len(steps)} steps", file=sys.stderr)

    return (
        {name: merge_limits(limits) for name, limits in z_limits_per_step.items()},
        {name: merge_limits(limits) for name, limits in contour_limits_per_step.items()},
        merge_limits(pxy_limits_per_step),
    )


def cell_center_axes(meta: dict) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    nx = int(meta["global_nx"])
    ny = int(meta["global_ny"])
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    prob_lo_x = float(meta["prob_lo"][0])
    prob_lo_y = float(meta["prob_lo"][1])
    prob_hi_x = float(meta["prob_hi"][0])
    prob_hi_y = float(meta["prob_hi"][1])
    x = prob_lo_x + (np.arange(nx) + 0.5) * dx
    y = prob_lo_y + (np.arange(ny) + 0.5) * dy
    extent = (prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y)
    return x, y, extent


def draw_grid_layout(
    ax: plt.Axes,
    meta: dict,
    pxy: np.ndarray,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    prob_lo_x, prob_lo_y = float(meta["prob_lo"][0]), float(meta["prob_lo"][1])
    prob_hi_x, prob_hi_y = float(meta["prob_hi"][0]), float(meta["prob_hi"][1])
    domain_width = prob_hi_x - prob_lo_x
    domain_height = prob_hi_y - prob_lo_y
    nx = int(meta["global_nx"])
    ny = int(meta["global_ny"])

    ax.clear()
    image = ax.imshow(
        pxy,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=(prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_patch(
        Rectangle(
            (prob_lo_x, prob_lo_y),
            domain_width,
            domain_height,
            fill=False,
            edgecolor="black",
            linewidth=1.4,
        )
    )

    level_colors = ["#4C78A8", "#E45756", "#54A24B", "#B279A2"]
    for level in meta["levels"]:
        color = level_colors[level["level"] % len(level_colors)]
        for box in level["boxes"]:
            x0 = prob_lo_x + domain_width * (box["lo_x"] / nx)
            y0 = prob_lo_y + domain_height * (box["lo_y"] / ny)
            width = domain_width * ((box["hi_x"] - box["lo_x"]) / nx)
            height = domain_height * ((box["hi_y"] - box["lo_y"]) / ny)
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2 if level["level"] > 0 else 1.0,
                )
            )

    ax.set_title("Pxy with current grid layout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(prob_lo_x, prob_hi_x)
    ax.set_ylim(prob_lo_y, prob_hi_y)
    ax.set_aspect("equal")
    ax.grid(False)

    legend_lines = [
        plt.Line2D([0], [0], color=level_colors[level["level"] % len(level_colors)], lw=1.5, label=f"Level {level['level']}")
        for level in meta["levels"]
    ]
    ax.legend(handles=legend_lines, loc="upper right", frameon=True, fontsize=8)
    return image


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    steps = discover_steps(output_dir)
    if not steps:
        print(f"No plotfiles found in {output_dir}", file=sys.stderr)
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
    if "Pxy" not in arrays:
        print(f"Pxy is not available in plotfile for step {first_step}", file=sys.stderr)
        return 1
    fluxes = compute_fluxes(arrays, meta)
    pxy = arrays["Pxy"]
    x, y, extent = cell_center_axes(meta)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.2))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.14, hspace=0.3, wspace=0.28)
    field_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    grid_ax = axes[1, 1]

    images = {}
    colorbars = {}
    contour_limits = {}
    z_limits = {}
    pxy_limits = safe_limits(pxy)
    for ax, name in zip(field_axes, ["Q", "B", "u"]):
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
            extent=extent,
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

    images["pxy"] = draw_grid_layout(
        grid_ax,
        meta,
        pxy,
        "viridis",
        vmin=pxy_limits[0],
        vmax=pxy_limits[1],
    )
    colorbars["pxy"] = fig.colorbar(images["pxy"], ax=grid_ax, fraction=0.046, pad=0.04)
    colorbars["pxy"].set_label("Pxy")

    if args.fixed_scale:
        z_limits, contour_limits, pxy_limits = compute_global_display_limits(
            output_dir,
            steps,
            args.outlier_percentile,
        )
    else:
        z_limits = None
        contour_limits = None
        pxy_limits = None

    def draw(index: int) -> None:
        step = steps[index]
        arrays_now, meta_now = load_step(str(output_dir), step)
        fluxes_now = compute_fluxes(arrays_now, meta_now)
        if "Pxy" not in arrays_now:
            raise RuntimeError(f"Pxy is not available in plotfile for step {step}")
        pxy_now = arrays_now["Pxy"]
        x_now, y_now, extent_now = cell_center_axes(meta_now)

        for ax, name in zip(field_axes, ["Q", "B", "u"]):
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
                extent=extent_now,
                vmin=z_vmin,
                vmax=z_vmax,
            )
            ax.contour(x_now, y_now, psi, levels=levels, linewidths=0.8, colors="black")
            field_z = VECTOR_FIELDS[name][2]
            ax.set_title(f"{field_z} + {name} flux contours")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            colorbars[name].update_normal(images[name])

        if pxy_limits is None:
            pxy_vmin, pxy_vmax = safe_limits(pxy_now)
        else:
            pxy_vmin, pxy_vmax = pxy_limits

        images["pxy"] = draw_grid_layout(
            grid_ax,
            meta_now,
            pxy_now,
            "viridis",
            vmin=pxy_vmin,
            vmax=pxy_vmax,
        )
        colorbars["pxy"].update_normal(images["pxy"])

        title.set_text(
            f"step={step}  time={meta_now['time']:.6g}  "
            f"grid={meta_now['global_nx']}x{meta_now['global_ny']}  levels={meta_now['finest_level'] + 1}"
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
