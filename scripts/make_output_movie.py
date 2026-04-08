#!/usr/bin/env python3
"""Render an MP4 movie from AMReX plotfiles."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(tempfile.gettempdir()) / "eqhd-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from read_output import discover_steps
from view_output_slider import (
    B_CONTOUR_COLOR,
    CANONICAL_CONTOUR_COLOR,
    VECTOR_FIELDS,
    cell_center_axes,
    compute_current_fields,
    compute_global_display_limits,
    compute_fluxes,
    load_step,
    safe_limits,
)

FIELD_NAMES = ["Q", "B", "u"]


@dataclass
class PreparedFrame:
    step: int
    arrays: dict[str, np.ndarray]
    meta: dict
    fluxes: dict[str, np.ndarray]
    x: np.ndarray
    y: np.ndarray
    extent: tuple[float, float, float, float]


@dataclass
class PanelState:
    ax: plt.Axes
    image: object
    contour: object
    colorbar: object


@dataclass
class GridState:
    ax: plt.Axes
    image: object
    colorbar: object
    legend: object | None
    b_contour: object | None = None
    q_contour: object | None = None


@dataclass
class MovieFigureState:
    title: object
    panels: dict[str, PanelState]
    grid: GridState


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
        "--current-component",
        choices=["Jx", "Jy", "Jz", "Jmag"],
        default="Jz",
        help="Current-density quantity to show in the lower-right panel",
    )
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
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=1,
        help="Worker threads used when scanning global display limits",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Number of future frames to prepare in background while encoding (0 disables)",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="libx264 preset for ffmpeg; faster presets trade compression efficiency for speed",
    )
    return parser.parse_args()


def default_mp4_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir.resolve().parent / f"{output_dir.resolve().name}.mp4"


def select_steps(all_steps: list[int], start: int | None, end: int | None, every: int) -> list[int]:
    steps = [step for step in all_steps if (start is None or step >= start) and (end is None or step <= end)]
    return steps[::every]


def prepare_frame(output_dir_str: str, step: int, current_component: str) -> PreparedFrame:
    arrays, meta = load_step(output_dir_str, step)
    arrays = dict(arrays)
    arrays.update(compute_current_fields(arrays, meta))
    arrays["CurrentPanel"] = arrays[current_component]
    x, y, extent = cell_center_axes(meta)

    return PreparedFrame(
        step=step,
        arrays=arrays,
        meta=meta,
        fluxes=compute_fluxes(arrays, meta),
        x=x,
        y=y,
        extent=extent,
    )


def image_limits(arr: np.ndarray, limits: dict[str, tuple[float, float]] | None, name: str) -> tuple[float, float]:
    if limits is None:
        return safe_limits(arr)
    return limits[name]


def scalar_limits(arr: np.ndarray, limits: tuple[float, float] | None) -> tuple[float, float]:
    if limits is None:
        return safe_limits(arr)
    return limits


def current_panel_cmap(current_component: str, cmap: str) -> str:
    return "viridis" if current_component == "Jmag" else cmap


def remove_contour(contour: object) -> None:
    if contour is None:
        return
    if hasattr(contour, "remove"):
        contour.remove()
        return
    for collection in getattr(contour, "collections", []):
        collection.remove()


def field_extent(meta: dict) -> tuple[float, float, float, float]:
    prob_lo_x = float(meta["prob_lo"][0])
    prob_lo_y = float(meta["prob_lo"][1])
    prob_hi_x = float(meta["prob_hi"][0])
    prob_hi_y = float(meta["prob_hi"][1])
    return prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y


def grid_legend_handles(meta: dict) -> list[object]:
    handles = []
    handles.append(plt.Line2D([0], [0], color=B_CONTOUR_COLOR, lw=1.5, label="B flux"))
    handles.append(plt.Line2D([0], [0], color=CANONICAL_CONTOUR_COLOR, lw=1.5, label="Canonical flux"))
    return handles


def update_grid_panel(
    state: GridState,
    meta: dict,
    scalar: np.ndarray,
    scalar_label: str,
    x: np.ndarray,
    y: np.ndarray,
    b_psi: np.ndarray,
    q_psi: np.ndarray,
    levels: int,
    contour_limits: dict[str, tuple[float, float]] | None,
    limits: tuple[float, float] | None,
) -> None:
    prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y = field_extent(meta)
    scalar_vmin, scalar_vmax = scalar_limits(scalar, limits)

    state.image.set_data(scalar)
    state.image.set_extent((prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y))
    state.image.set_clim(scalar_vmin, scalar_vmax)

    if state.legend is not None:
        state.legend.remove()
    state.legend = state.ax.legend(handles=grid_legend_handles(meta), loc="upper right", frameon=True, fontsize=8)

    remove_contour(state.b_contour)
    remove_contour(state.q_contour)
    if contour_limits is None:
        b_vmin, b_vmax = safe_limits(b_psi)
        q_vmin, q_vmax = safe_limits(q_psi)
    else:
        b_vmin, b_vmax = contour_limits["B"]
        q_vmin, q_vmax = contour_limits["Q"]
    state.b_contour = state.ax.contour(
        x,
        y,
        b_psi,
        levels=np.linspace(b_vmin, b_vmax, levels),
        linewidths=0.8,
        colors=B_CONTOUR_COLOR,
    )
    state.q_contour = state.ax.contour(
        x,
        y,
        q_psi,
        levels=np.linspace(q_vmin, q_vmax, levels),
        linewidths=0.8,
        colors=CANONICAL_CONTOUR_COLOR,
    )

    state.ax.set_title(f"{scalar_label} + B/Q flux contours")
    state.ax.set_xlabel("x")
    state.ax.set_ylabel("y")
    state.ax.set_xlim(prob_lo_x, prob_hi_x)
    state.ax.set_ylim(prob_lo_y, prob_hi_y)
    state.ax.set_aspect("equal")
    state.ax.grid(False)
    state.colorbar.update_normal(state.image)


def update_field_panel(
    panel: PanelState,
    frame: PreparedFrame,
    name: str,
    levels: int,
    z_limits: dict[str, tuple[float, float]] | None,
    contour_limits: dict[str, tuple[float, float]] | None,
) -> None:
    field_z = VECTOR_FIELDS[name][2]
    z = frame.arrays[field_z]
    psi = frame.fluxes[name]
    z_vmin, z_vmax = image_limits(z, z_limits, name)
    psi_vmin, psi_vmax = image_limits(psi, contour_limits, name)

    panel.image.set_data(z)
    panel.image.set_extent(frame.extent)
    panel.image.set_clim(z_vmin, z_vmax)
    panel.ax.set_xlim(frame.extent[0], frame.extent[1])
    panel.ax.set_ylim(frame.extent[2], frame.extent[3])

    remove_contour(panel.contour)
    panel.contour = panel.ax.contour(
        frame.x,
        frame.y,
        psi,
        levels=np.linspace(psi_vmin, psi_vmax, levels),
        linewidths=0.8,
        colors="black",
    )
    panel.colorbar.update_normal(panel.image)


def frame_title(frame: PreparedFrame) -> str:
    return (
        f"step={frame.step}  time={frame.meta['time']:.6g}  "
        f"grid={frame.meta['global_nx']}x{frame.meta['global_ny']}  levels={frame.meta['finest_level'] + 1}"
    )


def build_movie_figure(
    fig: plt.Figure,
    frame: PreparedFrame,
    levels: int,
    cmap: str,
    current_component: str,
    z_limits: dict[str, tuple[float, float]] | None,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> MovieFigureState:
    fig.clear()
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.3, wspace=0.28)
    field_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    grid_ax = axes[1, 1]

    panels: dict[str, PanelState] = {}
    for ax, name in zip(field_axes, FIELD_NAMES):
        field_z = VECTOR_FIELDS[name][2]
        z = frame.arrays[field_z]
        psi = frame.fluxes[name]
        z_vmin, z_vmax = image_limits(z, z_limits, name)
        psi_vmin, psi_vmax = image_limits(psi, contour_limits, name)

        image = ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            extent=frame.extent,
            vmin=z_vmin,
            vmax=z_vmax,
        )
        contour = ax.contour(
            frame.x,
            frame.y,
            psi,
            levels=np.linspace(psi_vmin, psi_vmax, levels),
            linewidths=0.8,
            colors="black",
        )
        ax.set_title(f"{field_z} + {name} flux contours")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(field_z)
        panels[name] = PanelState(ax=ax, image=image, contour=contour, colorbar=colorbar)

    current = frame.arrays["CurrentPanel"]
    current_vmin, current_vmax = scalar_limits(current, current_limits)
    grid_image = grid_ax.imshow(
        current,
        origin="lower",
        cmap=current_panel_cmap(current_component, cmap),
        interpolation="nearest",
        extent=field_extent(frame.meta),
        vmin=current_vmin,
        vmax=current_vmax,
    )
    grid_colorbar = fig.colorbar(grid_image, ax=grid_ax, fraction=0.046, pad=0.04)
    grid_colorbar.set_label(current_component)
    grid_state = GridState(ax=grid_ax, image=grid_image, colorbar=grid_colorbar, legend=None)
    update_grid_panel(
        grid_state,
        frame.meta,
        current,
        current_component,
        frame.x,
        frame.y,
        frame.fluxes["B"],
        frame.fluxes["Q"],
        levels,
        contour_limits,
        current_limits,
    )

    title = fig.suptitle(frame_title(frame), fontsize=12)
    return MovieFigureState(title=title, panels=panels, grid=grid_state)


def update_movie_figure(
    state: MovieFigureState,
    frame: PreparedFrame,
    levels: int,
    current_component: str,
    z_limits: dict[str, tuple[float, float]] | None,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> None:
    for name in FIELD_NAMES:
        update_field_panel(state.panels[name], frame, name, levels, z_limits, contour_limits)
    update_grid_panel(
        state.grid,
        frame.meta,
        frame.arrays["CurrentPanel"],
        current_component,
        frame.x,
        frame.y,
        frame.fluxes["B"],
        frame.fluxes["Q"],
        levels,
        contour_limits,
        current_limits,
    )
    state.title.set_text(frame_title(frame))


def draw_frame(
    fig: plt.Figure,
    output_dir: pathlib.Path,
    step: int,
    levels: int,
    cmap: str,
    current_component: str,
    z_limits: dict[str, tuple[float, float]] | None,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> None:
    frame = prepare_frame(str(output_dir), step, current_component)
    build_movie_figure(fig, frame, levels, cmap, current_component, z_limits, contour_limits, current_limits)


def iter_prepared_frames(output_dir: pathlib.Path, steps: list[int], current_component: str, prefetch: int):
    output_dir_str = str(output_dir)
    if prefetch <= 0:
        for step in steps:
            yield prepare_frame(output_dir_str, step, current_component)
        return

    window = max(1, prefetch + 1)
    with ThreadPoolExecutor(max_workers=window) as executor:
        pending: dict[int, Future[PreparedFrame]] = {}
        submit_index = 0

        while submit_index < min(window, len(steps)):
            pending[submit_index] = executor.submit(prepare_frame, output_dir_str, steps[submit_index], current_component)
            submit_index += 1

        for current_index in range(len(steps)):
            frame = pending.pop(current_index).result()
            while submit_index < len(steps) and len(pending) < window:
                pending[submit_index] = executor.submit(prepare_frame, output_dir_str, steps[submit_index], current_component)
                submit_index += 1
            yield frame


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1
    if args.every <= 0:
        print("--every must be a positive integer", file=sys.stderr)
        return 1
    if args.levels < 2:
        print("--levels must be at least 2", file=sys.stderr)
        return 1
    if args.scan_workers < 1:
        print("--scan-workers must be at least 1", file=sys.stderr)
        return 1
    if args.prefetch < 0:
        print("--prefetch must be non-negative", file=sys.stderr)
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
    current_limits = None
    if not args.dynamic_scale:
        z_limits, contour_limits, current_limits = compute_global_display_limits(
            output_dir,
            steps,
            args.current_component,
            args.outlier_percentile,
            workers=min(args.scan_workers, len(steps)),
        )

    frame_iter = iter_prepared_frames(output_dir, steps, args.current_component, args.prefetch)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        print("No steps selected for movie generation", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(12.6, 10.2))
    state = build_movie_figure(fig, first_frame, args.levels, args.cmap, args.current_component, z_limits, contour_limits, current_limits)

    writer_kwargs = {"fps": args.fps, "codec": "libx264"}
    if args.ffmpeg_preset:
        writer_kwargs["extra_args"] = ["-preset", args.ffmpeg_preset]
    writer = FFMpegWriter(**writer_kwargs)

    print(
        f"[movie] writing {len(steps)} frames from {output_dir} to {mp4_path} at {args.fps:g} fps",
        file=sys.stderr,
    )
    with writer.saving(fig, str(mp4_path), dpi=args.dpi):
        writer.grab_frame()
        if len(steps) == 1:
            print("[movie] rendered 1/1 frames", file=sys.stderr)
        elif len(steps) > 1:
            print(f"[movie] rendered 1/{len(steps)} frames", file=sys.stderr)

        for idx, frame in enumerate(frame_iter, start=2):
            update_movie_figure(state, frame, args.levels, args.current_component, z_limits, contour_limits, current_limits)
            writer.grab_frame()
            if idx == len(steps) or idx % 10 == 0:
                print(f"[movie] rendered {idx}/{len(steps)} frames", file=sys.stderr)

    plt.close(fig)
    print(f"[movie] wrote {mp4_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
