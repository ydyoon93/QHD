#!/usr/bin/env python3
"""Render an MP4 movie from AMReX plotfiles."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from read_output import discover_steps
from view_output_slider import (
    VECTOR_FIELDS,
    cell_center_axes,
    compute_global_display_limits,
    compute_fluxes,
    draw_grid_layout,
    load_step,
    safe_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Directory with AMReX plotfiles `plt_XXXXXX` or `plt_XXXXXX.h5`")
    parser.add_argument("--mp4", default=None, help="Output MP4 path (default: sibling of output dir)")
    parser.add_argument("--fps", type=float, default=10.0, help="Video frame rate")
    parser.add_argument("--start-step", type=int, default=None, help="First step to render")
    parser.add_argument("--end-step", type=int, default=None, help="Last step to render")
    parser.add_argument("--every", type=int, default=1, help="Render every Nth plotfile")
    parser.add_argument("--levels", type=int, default=16, help="Number of contour levels for Q/B/u panels")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for z-component background")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for MP4 encoding")
    parser.add_argument(
        "--global-scale",
        action="store_true",
        help="Deprecated alias for the default robust global scaling behavior",
    )
    parser.add_argument(
        "--dynamic-scale",
        action="store_true",
        help="Recompute color and contour limits per frame instead of using one robust global scale",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=1.0,
        help="Clip this percentile from each tail when computing global ranges (default: 1.0)",
    )
    return parser.parse_args()


def default_mp4_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir.resolve().parent / f"{output_dir.resolve().name}.mp4"


def select_steps(all_steps: list[int], start: int | None, end: int | None, every: int) -> list[int]:
    steps = [step for step in all_steps if (start is None or step >= start) and (end is None or step <= end)]
    return steps[::every]


def draw_frame(
    fig: plt.Figure,
    output_dir: pathlib.Path,
    step: int,
    levels: int,
    cmap: str,
    z_limits: dict[str, tuple[float, float]] | None,
    contour_limits: dict[str, tuple[float, float]] | None,
    pxy_limits: tuple[float, float] | None,
) -> None:
    arrays, meta = load_step(str(output_dir), step)
    if "Pxy" not in arrays:
        raise RuntimeError(f"Pxy is not available in plotfile for step {step}")
    fluxes = compute_fluxes(arrays, meta)
    pxy = arrays["Pxy"]
    x, y, extent = cell_center_axes(meta)

    fig.clear()
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.3, wspace=0.28)
    field_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    grid_ax = axes[1, 1]

    for ax, name in zip(field_axes, ["Q", "B", "u"]):
        field_z = VECTOR_FIELDS[name][2]
        z = arrays[field_z]
        psi = fluxes[name]

        if z_limits is None:
            z_vmin, z_vmax = safe_limits(z)
        else:
            z_vmin, z_vmax = z_limits[name]
        if contour_limits is None:
            psi_vmin, psi_vmax = safe_limits(psi)
        else:
            psi_vmin, psi_vmax = contour_limits[name]

        image = ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            extent=extent,
            vmin=z_vmin,
            vmax=z_vmax,
        )
        contour_levels = np.linspace(psi_vmin, psi_vmax, levels)
        ax.contour(x, y, psi, levels=contour_levels, linewidths=0.8, colors="black")
        ax.set_title(f"{field_z} + {name} flux contours")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_z)

    if pxy_limits is None:
        pxy_vmin, pxy_vmax = safe_limits(pxy)
    else:
        pxy_vmin, pxy_vmax = pxy_limits
    grid_image = draw_grid_layout(grid_ax, meta, pxy, "viridis", vmin=pxy_vmin, vmax=pxy_vmax)
    grid_cbar = fig.colorbar(grid_image, ax=grid_ax, fraction=0.046, pad=0.04)
    grid_cbar.set_label("Pxy")

    fig.suptitle(
        f"step={step}  time={meta['time']:.6g}  "
        f"grid={meta['global_nx']}x{meta['global_ny']}  levels={meta['finest_level'] + 1}",
        fontsize=12,
    )


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    all_steps = discover_steps(output_dir)
    if not all_steps:
        print(f"No plotfiles found in {output_dir}", file=sys.stderr)
        return 1

    steps = select_steps(all_steps, args.start_step, args.end_step, args.every)
    if not steps:
        print("No steps selected for movie generation", file=sys.stderr)
        return 1

    mp4_path = pathlib.Path(args.mp4) if args.mp4 is not None else default_mp4_path(output_dir)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    z_limits = None
    contour_limits = None
    pxy_limits = None
    if not args.dynamic_scale:
        z_limits, contour_limits, pxy_limits = compute_global_display_limits(
            output_dir,
            steps,
            args.outlier_percentile,
        )

    fig = plt.figure(figsize=(12.6, 10.2))
    writer = FFMpegWriter(fps=args.fps, codec="libx264")

    print(
        f"[movie] writing {len(steps)} frames from {output_dir} to {mp4_path} at {args.fps:g} fps",
        file=sys.stderr,
    )
    with writer.saving(fig, str(mp4_path), dpi=args.dpi):
        for idx, step in enumerate(steps, start=1):
            draw_frame(fig, output_dir, step, args.levels, args.cmap, z_limits, contour_limits, pxy_limits)
            writer.grab_frame()
            if idx == 1 or idx == len(steps) or idx % 10 == 0:
                print(f"[movie] rendered {idx}/{len(steps)} frames", file=sys.stderr)

    plt.close(fig)
    print(f"[movie] wrote {mp4_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
