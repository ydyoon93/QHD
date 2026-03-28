#!/usr/bin/env python3

import argparse
import pathlib
import runpy
import sys

import numpy as np


def evaluate_field(obj, x, y, shape, name):
    if obj is None:
        return None
    value = obj(x, y) if callable(obj) else obj
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float64)
    try:
        broadcast = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} returned shape {array.shape}, expected broadcastable to {shape}") from exc
    return np.array(broadcast, dtype=np.float64, copy=True)


def evaluate_component(funcs, name, x, y, shape):
    return evaluate_field(funcs.get(name), x, y, shape, name)


def component_meshes(nx, ny, dx, dy, staggered):
    x_cc = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_cc = (np.arange(ny, dtype=np.float64) + 0.5) * dy

    if not staggered:
        xg, yg = np.meshgrid(x_cc, y_cc, indexing="xy")
        return {
            "x1d": x_cc,
            "y1d": y_cc,
            "Bx": (xg, yg),
            "By": (xg, yg),
            "Bz": (xg, yg),
            "Qx": (xg, yg),
            "Qy": (xg, yg),
            "Qz": (xg, yg),
            "Ux": (xg, yg),
            "Uy": (xg, yg),
            "Uz": (xg, yg),
        }

    x_edge_x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_edge_x = np.arange(ny, dtype=np.float64) * dy
    x_edge_y = np.arange(nx, dtype=np.float64) * dx
    y_edge_y = (np.arange(ny, dtype=np.float64) + 0.5) * dy

    xgx, ygx = np.meshgrid(x_edge_x, y_edge_x, indexing="xy")
    xgy, ygy = np.meshgrid(x_edge_y, y_edge_y, indexing="xy")
    xgz, ygz = np.meshgrid(x_cc, y_cc, indexing="xy")
    return {
        "x1d": x_cc,
        "y1d": y_cc,
        "Bx": (xgx, ygx),
        "By": (xgy, ygy),
        "Bz": (xgz, ygz),
        "Qx": (xgx, ygx),
        "Qy": (xgy, ygy),
        "Qz": (xgz, ygz),
        "Ux": (xgx, ygx),
        "Uy": (xgy, ygy),
        "Uz": (xgz, ygz),
    }


def reconstruct_b_from_u(ux, uy, uz, dx, dy):
    ny, nx = ux.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx2d = kx[None, :]
    ky2d = ky[:, None]

    dx_backward = (1.0 - np.exp(-1j * kx2d * dx)) / dx
    dy_backward = (1.0 - np.exp(-1j * ky2d * dy)) / dy
    dx_forward = (np.exp(1j * kx2d * dx) - 1.0) / dx
    dy_forward = (np.exp(1j * ky2d * dy) - 1.0) / dy

    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    uz_hat = np.fft.fft2(uz)

    bz_hat = np.zeros_like(ux_hat, dtype=np.complex128)
    denom_bz = np.abs(dx_backward) ** 2 + np.abs(dy_backward) ** 2
    mask_bz = denom_bz > 0.0
    numer_bz = -np.conj(dy_backward) * ux_hat + np.conj(dx_backward) * uy_hat
    bz_hat[mask_bz] = numer_bz[mask_bz] / denom_bz[mask_bz]

    bx_hat = np.zeros_like(uz_hat, dtype=np.complex128)
    by_hat = np.zeros_like(uz_hat, dtype=np.complex128)
    denom_xy = np.abs(dx_forward) ** 2 + np.abs(dy_forward) ** 2
    mask_xy = denom_xy > 0.0
    numer_bx = np.conj(dy_forward) * uz_hat
    numer_by = -np.conj(dx_forward) * uz_hat
    bx_hat[mask_xy] = numer_bx[mask_xy] / denom_xy[mask_xy]
    by_hat[mask_xy] = numer_by[mask_xy] / denom_xy[mask_xy]

    bx = np.fft.ifft2(bx_hat).real
    by = np.fft.ifft2(by_hat).real
    bz = np.fft.ifft2(bz_hat).real
    return bx, by, bz


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate initial fields from a Python namelist.")
    parser.add_argument("--namelist", required=True)
    parser.add_argument("--nx", required=True, type=int)
    parser.add_argument("--ny", required=True, type=int)
    parser.add_argument("--lx", required=True, type=float)
    parser.add_argument("--ly", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--staggered", action="store_true", help="Sample x/y/z components on the solver staggered grid")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = args.lx / args.nx
    dy = args.ly / args.ny
    namelist_path = pathlib.Path(args.namelist).resolve()

    meshes = component_meshes(args.nx, args.ny, dx, dy, args.staggered)

    namespace = {
        "__file__": str(namelist_path),
        "np": np,
        "nx": args.nx,
        "ny": args.ny,
        "lx": args.lx,
        "ly": args.ly,
        "dx": dx,
        "dy": dy,
        "x1d": meshes["x1d"],
        "y1d": meshes["y1d"],
    }
    funcs = runpy.run_path(str(namelist_path), init_globals=namespace)

    fields = {
        "bx.npy": evaluate_component(funcs, "Bx", *meshes["Bx"], (args.ny, args.nx)),
        "by.npy": evaluate_component(funcs, "By", *meshes["By"], (args.ny, args.nx)),
        "bz.npy": evaluate_component(funcs, "Bz", *meshes["Bz"], (args.ny, args.nx)),
        "qx.npy": evaluate_component(funcs, "Qx", *meshes["Qx"], (args.ny, args.nx)),
        "qy.npy": evaluate_component(funcs, "Qy", *meshes["Qy"], (args.ny, args.nx)),
        "qz.npy": evaluate_component(funcs, "Qz", *meshes["Qz"], (args.ny, args.nx)),
        "ux.npy": evaluate_component(funcs, "Ux", *meshes["Ux"], (args.ny, args.nx)),
        "uy.npy": evaluate_component(funcs, "Uy", *meshes["Uy"], (args.ny, args.nx)),
        "uz.npy": evaluate_component(funcs, "Uz", *meshes["Uz"], (args.ny, args.nx)),
    }

    has_b = all(fields[name] is not None for name in ("bx.npy", "by.npy", "bz.npy"))
    has_q = all(fields[name] is not None for name in ("qx.npy", "qy.npy", "qz.npy"))
    has_u = all(fields[name] is not None for name in ("ux.npy", "uy.npy", "uz.npy"))
    if not has_b and not has_q and not has_u:
        raise ValueError("The Python namelist must define either Bx/By/Bz, Qx/Qy/Qz, or Ux/Uy/Uz")
    if any(fields[name] is not None for name in ("bx.npy", "by.npy", "bz.npy")) and not has_b:
        raise ValueError("If any of Bx, By, Bz is defined, all three must be defined")
    if any(fields[name] is not None for name in ("qx.npy", "qy.npy", "qz.npy")) and not has_q:
        raise ValueError("If any of Qx, Qy, Qz is defined, all three must be defined")
    if any(fields[name] is not None for name in ("ux.npy", "uy.npy", "uz.npy")) and not has_u:
        raise ValueError("If any of Ux, Uy, Uz is defined, all three must be defined")

    if has_u and not has_b and not has_q:
        bx, by, bz = reconstruct_b_from_u(fields["ux.npy"], fields["uy.npy"], fields["uz.npy"], dx, dy)
        fields["bx.npy"] = bx
        fields["by.npy"] = by
        fields["bz.npy"] = bz

    for filename, array in fields.items():
        if filename.startswith("u"):
            continue
        if array is not None:
            np.save(out_dir / filename, array)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
