#!/usr/bin/env python3
"""Read AMReX plotfiles and reconstruct 2D fields on the finest-level canvas."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

try:
    import h5py
except ImportError:
    h5py = None

import numpy as np

PLOTFILE_RE = re.compile(r"plt_(\d{6})(?:\.h5)?$")
BOX_RE = re.compile(r"\(\(([-\d]+),([-\d]+)\)\s+\(([-\d]+),([-\d]+)\)\s+\(([-\d]+),([-\d]+)\)\)")
FAB_RE = re.compile(r"FabOnDisk:\s+(\S+)\s+(\d+)")


@dataclass(frozen=True)
class FabRecord:
    lo_i: int
    lo_j: int
    hi_i: int
    hi_j: int
    filename: str
    offset: int


def _is_hdf5_plotfile(path: pathlib.Path) -> bool:
    return path.is_file() and path.suffix == ".h5"


def _require_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to read AMReX HDF5 plotfiles")


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _as_scalar_int(value: object) -> int:
    return int(np.asarray(value).item())


def _as_scalar_float(value: object) -> float:
    return float(np.asarray(value).item())


def _compound_box_to_dict(record: np.void) -> dict[str, int]:
    names = set(record.dtype.names or ())
    return {
        "lo_i": int(record["lo_i"]),
        "lo_j": int(record["lo_j"] if "lo_j" in names else 0),
        "hi_i": int(record["hi_i"]),
        "hi_j": int(record["hi_j"] if "hi_j" in names else 0),
    }


def discover_steps(output_dir: pathlib.Path) -> list[int]:
    steps: set[int] = set()
    for path in output_dir.glob("plt_*"):
        if not path.is_dir() and not _is_hdf5_plotfile(path):
            continue
        match = PLOTFILE_RE.fullmatch(path.name)
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def plotfile_for_step(output_dir: pathlib.Path, step: int) -> pathlib.Path:
    hdf5_path = output_dir / f"plt_{step:06d}.h5"
    if hdf5_path.exists():
        return hdf5_path
    return output_dir / f"plt_{step:06d}"


def _read_header_lines(plotfile_dir: pathlib.Path) -> list[str]:
    header_path = plotfile_dir / "Header"
    if not header_path.exists():
        raise RuntimeError(f"Missing plotfile header: {header_path}")
    return [line.rstrip() for line in header_path.read_text().splitlines() if line.strip()]


def _read_plotfile_metadata_directory(plotfile_dir: pathlib.Path) -> dict:
    lines = _read_header_lines(plotfile_dir)
    idx = 0

    version = lines[idx]
    idx += 1
    nvars = int(lines[idx])
    idx += 1
    varnames = [lines[idx + i] for i in range(nvars)]
    idx += nvars

    space_dim = int(lines[idx])
    idx += 1
    time = float(lines[idx])
    idx += 1
    finest_level = int(lines[idx])
    idx += 1

    prob_lo = tuple(float(x) for x in lines[idx].split())
    idx += 1
    prob_hi = tuple(float(x) for x in lines[idx].split())
    idx += 1

    ref_ratio = [int(x) for x in lines[idx].split()] if finest_level > 0 else []
    idx += 1 if finest_level > 0 else 0

    domain_matches = BOX_RE.findall(lines[idx])
    idx += 1
    if len(domain_matches) != finest_level + 1:
        raise RuntimeError(f"Failed to parse all level domains from {plotfile_dir / 'Header'}")
    domains = []
    for match in domain_matches:
        lo_i, lo_j, hi_i, hi_j = map(int, match[:4])
        domains.append({"lo_i": lo_i, "lo_j": lo_j, "hi_i": hi_i, "hi_j": hi_j})

    level_steps = [int(x) for x in lines[idx].split()]
    idx += 1
    if len(level_steps) != finest_level + 1:
        raise RuntimeError(f"Unexpected level-step line in {plotfile_dir / 'Header'}")

    cell_sizes = []
    for _ in range(finest_level + 1):
        cell_sizes.append(tuple(float(x) for x in lines[idx].split()))
        idx += 1

    coord_sys = int(lines[idx])
    idx += 1
    _bwidth = int(lines[idx])
    idx += 1

    levels = []
    for lev in range(finest_level + 1):
        lev_id, lev_ref, lev_time = lines[idx].split()
        idx += 1
        lev_step = int(lines[idx])
        idx += 1

        data_rel_path = None
        extent_lines: list[str] = []
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            if line.startswith("Level_"):
                data_rel_path = line
                break
            extent_lines.append(line)

        if data_rel_path is None:
            raise RuntimeError(f"Failed to locate level data path in {plotfile_dir / 'Header'}")

        levels.append(
            {
                "level": int(lev_id),
                "ref_ratio_to_next": int(lev_ref),
                "time": float(lev_time),
                "step": lev_step,
                "extent_lines": extent_lines,
                "data_rel_path": data_rel_path,
            }
        )

    finest_domain = domains[-1]
    global_nx = finest_domain["hi_i"] - finest_domain["lo_i"] + 1
    global_ny = finest_domain["hi_j"] - finest_domain["lo_j"] + 1

    return {
        "version": version,
        "nvars": nvars,
        "varnames": varnames,
        "space_dim": space_dim,
        "time": time,
        "step": level_steps[0],
        "prob_lo": prob_lo,
        "prob_hi": prob_hi,
        "coord_sys": coord_sys,
        "finest_level": finest_level,
        "ref_ratio": ref_ratio,
        "domains": domains,
        "cell_sizes": cell_sizes,
        "levels": levels,
        "global_nx": global_nx,
        "global_ny": global_ny,
        "dx": cell_sizes[-1][0],
        "dy": cell_sizes[-1][1],
    }


def _read_plotfile_metadata_hdf5(plotfile_path: pathlib.Path) -> dict:
    _require_h5py()
    with h5py.File(plotfile_path, "r") as h5:
        nvars = _as_scalar_int(h5.attrs["num_components"])
        finest_level = _as_scalar_int(h5.attrs["finest_level"])
        levels = []
        domains = []
        cell_sizes = []
        ref_ratio = []
        level_steps = []

        prob_lo: tuple[float, ...] | None = None
        prob_hi: tuple[float, ...] | None = None

        for lev in range(finest_level + 1):
            grp = h5[f"level_{lev}"]
            domain = _compound_box_to_dict(grp.attrs["prob_domain"])
            domains.append(domain)

            level_prob_lo = tuple(float(x) for x in np.asarray(grp.attrs["prob_lo"], dtype=np.float64))
            level_prob_hi = tuple(float(x) for x in np.asarray(grp.attrs["prob_hi"], dtype=np.float64))
            if prob_lo is None:
                prob_lo = level_prob_lo
                prob_hi = level_prob_hi

            dx_vec = tuple(float(x) for x in np.asarray(grp.attrs["Vec_dx"], dtype=np.float64))
            cell_sizes.append(dx_vec)

            level_ref_ratio = _as_scalar_int(grp.attrs["ref_ratio"])
            if lev < finest_level:
                ref_ratio.append(level_ref_ratio)

            level_step = _as_scalar_int(grp.attrs["steps"])
            level_steps.append(level_step)
            levels.append(
                {
                    "level": lev,
                    "ref_ratio_to_next": level_ref_ratio,
                    "time": _as_scalar_float(grp.attrs["time"]),
                    "step": level_step,
                    "extent_lines": [],
                    "data_rel_path": f"level_{lev}",
                }
            )

        if prob_lo is None or prob_hi is None:
            raise RuntimeError(f"No levels found in HDF5 plotfile {plotfile_path}")

        finest_domain = domains[-1]
        global_nx = finest_domain["hi_i"] - finest_domain["lo_i"] + 1
        global_ny = finest_domain["hi_j"] - finest_domain["lo_j"] + 1

        return {
            "version": _decode_string(h5.attrs["version_name"]),
            "nvars": nvars,
            "varnames": [_decode_string(h5.attrs[f"component_{i}"]) for i in range(nvars)],
            "space_dim": _as_scalar_int(h5.attrs["dim"]),
            "time": _as_scalar_float(h5.attrs["time"]),
            "step": level_steps[0],
            "prob_lo": prob_lo,
            "prob_hi": prob_hi,
            "coord_sys": _as_scalar_int(h5.attrs["coordinate_system"]),
            "finest_level": finest_level,
            "ref_ratio": ref_ratio,
            "domains": domains,
            "cell_sizes": cell_sizes,
            "levels": levels,
            "global_nx": global_nx,
            "global_ny": global_ny,
            "dx": cell_sizes[-1][0],
            "dy": cell_sizes[-1][1],
        }


def read_plotfile_metadata(plotfile_path: pathlib.Path) -> dict:
    if _is_hdf5_plotfile(plotfile_path):
        return _read_plotfile_metadata_hdf5(plotfile_path)
    return _read_plotfile_metadata_directory(plotfile_path)


def _parse_cell_header(cell_header_path: pathlib.Path) -> tuple[list[FabRecord], int]:
    lines = [line.strip() for line in cell_header_path.read_text().splitlines() if line.strip()]
    ncomp = int(lines[2])

    box_start = None
    box_end = None
    for i, line in enumerate(lines):
        if line.startswith("("):
            box_start = i
            break
    if box_start is None:
        raise RuntimeError(f"Failed to locate box list in {cell_header_path}")

    for i in range(box_start, len(lines)):
        if lines[i] == ")":
            box_end = i
            break
    if box_end is None:
        raise RuntimeError(f"Failed to terminate box list in {cell_header_path}")

    boxes: list[tuple[int, int, int, int]] = []
    for line in lines[box_start + 1 : box_end]:
        match = BOX_RE.search(line)
        if match is not None:
            lo_i, lo_j, hi_i, hi_j = map(int, match.groups()[:4])
            boxes.append((lo_i, lo_j, hi_i, hi_j))

    fabs: list[FabRecord] = []
    fab_lines = [line for line in lines[box_end + 1 :] if line.startswith("FabOnDisk:")]
    if len(fab_lines) != len(boxes):
        raise RuntimeError(f"Mismatch between boxes and FabOnDisk entries in {cell_header_path}")

    for box, fab_line in zip(boxes, fab_lines):
        match = FAB_RE.fullmatch(fab_line)
        if match is None:
            raise RuntimeError(f"Failed to parse FabOnDisk line in {cell_header_path}: {fab_line}")
        filename, offset = match.groups()
        fabs.append(FabRecord(*box, filename, int(offset)))

    return fabs, ncomp


def _read_fab(path: pathlib.Path, offset: int, ncomp: int, nx: int, ny: int) -> np.ndarray:
    with path.open("rb") as fh:
        fh.seek(offset)
        _fab_header = fh.readline()
        data = np.fromfile(fh, dtype=np.float64, count=ncomp * nx * ny)
    if data.size != ncomp * nx * ny:
        raise RuntimeError(f"Unexpected FAB payload size in {path}")
    return data.reshape((ncomp, ny, nx))


def _scale_to_finest(meta: dict, lev: int) -> int:
    scale = 1
    for rr in meta["ref_ratio"][lev:]:
        scale *= rr
    return scale


def read_grid_layout(plotfile_path: pathlib.Path) -> dict:
    meta = read_plotfile_metadata(plotfile_path)
    levels: list[dict] = []

    if _is_hdf5_plotfile(plotfile_path):
        _require_h5py()
        with h5py.File(plotfile_path, "r") as h5:
            for lev in range(meta["finest_level"] + 1):
                grp = h5[f"level_{lev}"]
                scale = _scale_to_finest(meta, lev)
                boxes = []
                for box_record in grp["boxes"][:]:
                    box = _compound_box_to_dict(box_record)
                    boxes.append(
                        {
                            "lo_i": box["lo_i"],
                            "lo_j": box["lo_j"],
                            "hi_i": box["hi_i"],
                            "hi_j": box["hi_j"],
                            "scale_to_finest": scale,
                            "lo_x": box["lo_i"] * scale,
                            "lo_y": box["lo_j"] * scale,
                            "hi_x": (box["hi_i"] + 1) * scale,
                            "hi_y": (box["hi_j"] + 1) * scale,
                        }
                    )
                levels.append({"level": lev, "boxes": boxes})
    else:
        for lev in range(meta["finest_level"] + 1):
            level_dir = plotfile_path / f"Level_{lev}"
            fabs, _ncomp = _parse_cell_header(level_dir / "Cell_H")
            scale = _scale_to_finest(meta, lev)
            boxes = []
            for fab in fabs:
                boxes.append(
                    {
                        "lo_i": fab.lo_i,
                        "lo_j": fab.lo_j,
                        "hi_i": fab.hi_i,
                        "hi_j": fab.hi_j,
                        "scale_to_finest": scale,
                        "lo_x": fab.lo_i * scale,
                        "lo_y": fab.lo_j * scale,
                        "hi_x": (fab.hi_i + 1) * scale,
                        "hi_y": (fab.hi_j + 1) * scale,
                    }
                )
            levels.append({"level": lev, "boxes": boxes})

    return {
        "global_nx": meta["global_nx"],
        "global_ny": meta["global_ny"],
        "dx": meta["dx"],
        "dy": meta["dy"],
        "prob_lo": meta["prob_lo"],
        "prob_hi": meta["prob_hi"],
        "finest_level": meta["finest_level"],
        "levels": levels,
    }


def _read_global_field_directory(plotfile_dir: pathlib.Path, field: str) -> tuple[np.ndarray, dict]:
    meta = read_plotfile_metadata(plotfile_dir)
    if field not in meta["varnames"]:
        raise RuntimeError(f"Field '{field}' not found in plotfile {plotfile_dir}")

    comp = meta["varnames"].index(field)
    out = np.zeros((meta["global_ny"], meta["global_nx"]), dtype=np.float64)

    for lev in range(meta["finest_level"] + 1):
        level_dir = plotfile_dir / f"Level_{lev}"
        fabs, ncomp = _parse_cell_header(level_dir / "Cell_H")
        if ncomp != meta["nvars"]:
            raise RuntimeError(f"Variable count mismatch in {level_dir / 'Cell_H'}")

        scale = _scale_to_finest(meta, lev)
        for fab in fabs:
            nx = fab.hi_i - fab.lo_i + 1
            ny = fab.hi_j - fab.lo_j + 1
            block = _read_fab(level_dir / fab.filename, fab.offset, ncomp, nx, ny)[comp]

            for j in range(ny):
                dst_j0 = (fab.lo_j + j) * scale
                dst_j1 = dst_j0 + scale
                for i in range(nx):
                    dst_i0 = (fab.lo_i + i) * scale
                    dst_i1 = dst_i0 + scale
                    out[dst_j0:dst_j1, dst_i0:dst_i1] = block[j, i]

    return out, meta


def _read_hdf5_block(grp: h5py.Group, comp: int, ncomp: int, start: int, end: int, nx: int, ny: int) -> np.ndarray:
    single_dataset_name = "data:datatype=0"
    if single_dataset_name in grp:
        block = np.asarray(grp[single_dataset_name][start:end], dtype=np.float64)
        expected = ncomp * nx * ny
        if block.size != expected:
            raise RuntimeError(
                f"Unexpected HDF5 dataset size in group {grp.name}: expected {expected} values, got {block.size}"
            )
        return block.reshape((ncomp, ny, nx))[comp]

    multi_dataset_name = f"data:datatype={comp}"
    if multi_dataset_name in grp:
        block = np.asarray(grp[multi_dataset_name][start:end], dtype=np.float64)
        expected = nx * ny
        if block.size != expected:
            raise RuntimeError(
                f"Unexpected HDF5 dataset size in group {grp.name}: expected {expected} values, got {block.size}"
            )
        return block.reshape((ny, nx))

    raise RuntimeError(f"No AMReX field dataset found for component {comp} in HDF5 group {grp.name}")


def _read_global_field_hdf5(plotfile_path: pathlib.Path, field: str) -> tuple[np.ndarray, dict]:
    _require_h5py()
    meta = read_plotfile_metadata(plotfile_path)
    if field not in meta["varnames"]:
        raise RuntimeError(f"Field '{field}' not found in plotfile {plotfile_path}")

    comp = meta["varnames"].index(field)
    out = np.zeros((meta["global_ny"], meta["global_nx"]), dtype=np.float64)

    with h5py.File(plotfile_path, "r") as h5:
        for lev in range(meta["finest_level"] + 1):
            grp = h5[f"level_{lev}"]
            boxes = grp["boxes"][:]
            offsets = np.asarray(grp["data:offsets=0"][:], dtype=np.int64)
            if len(offsets) != len(boxes) + 1:
                raise RuntimeError(f"Unexpected offsets size in HDF5 group {grp.name}")

            scale = _scale_to_finest(meta, lev)
            for idx, box_record in enumerate(boxes):
                box = _compound_box_to_dict(box_record)
                nx = box["hi_i"] - box["lo_i"] + 1
                ny = box["hi_j"] - box["lo_j"] + 1
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                block = _read_hdf5_block(grp, comp, meta["nvars"], start, end, nx, ny)

                for j in range(ny):
                    dst_j0 = (box["lo_j"] + j) * scale
                    dst_j1 = dst_j0 + scale
                    for i in range(nx):
                        dst_i0 = (box["lo_i"] + i) * scale
                        dst_i1 = dst_i0 + scale
                        out[dst_j0:dst_j1, dst_i0:dst_i1] = block[j, i]

    return out, meta


def read_global_field(plotfile_path: pathlib.Path, field: str) -> tuple[np.ndarray, dict]:
    if _is_hdf5_plotfile(plotfile_path):
        return _read_global_field_hdf5(plotfile_path, field)
    return _read_global_field_directory(plotfile_path, field)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory with AMReX plotfiles `plt_XXXXXX` or `plt_XXXXXX.h5`",
    )
    parser.add_argument("--step", type=int, default=None, help="Step to read (default: latest)")
    parser.add_argument(
        "--field",
        default="Bz",
        choices=["Bx", "By", "Bz", "Ux", "Uy", "Uz", "Qx", "Qy", "Qz", "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz", "TrP"],
        help="Field component to reconstruct",
    )
    parser.add_argument("--save-npy", default=None, help="Optional output .npy path")
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit")

    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    steps = discover_steps(output_dir)
    if not steps:
        print(f"No plotfiles found in {output_dir}", file=sys.stderr)
        return 1

    if args.list_steps:
        print("Available steps:", " ".join(str(s) for s in steps))
        return 0

    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        print(f"Step {step} not found. Available: {steps}", file=sys.stderr)
        return 1

    plotfile_path = plotfile_for_step(output_dir, step)
    arr, meta = read_global_field(plotfile_path, args.field)

    print(f"Read field={args.field} at step={step} from plotfile {plotfile_path.name}")
    print(
        f"Grid={meta['global_nx']}x{meta['global_ny']}  "
        f"dx={meta['dx']:.6g} dy={meta['dy']:.6g}  time={meta['time']:.6g}  levels={meta['finest_level'] + 1}"
    )
    print(
        f"Stats: min={arr.min():.6e} max={arr.max():.6e} "
        f"mean={arr.mean():.6e} rms={np.sqrt(np.mean(arr * arr)):.6e}"
    )

    if args.save_npy:
        out_path = pathlib.Path(args.save_npy)
        np.save(out_path, arr)
        print(f"Saved array to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
