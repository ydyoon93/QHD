#!/usr/bin/env python3

import argparse
import pathlib
import runpy
import sys

import numpy as np


SCALAR_KEYS = (
    "nx",
    "ny",
    "lx",
    "ly",
    "dt",
    "t_end",
    "output_every",
    "nu",
    "eta",
    "init_b0",
    "init_sigma",
    "init_perturbation",
    "init_python_namelist",
    "output_dir",
)


def normalize_value(name, value, base_dir):
    if isinstance(value, pathlib.Path):
        value = str(value)
    if isinstance(value, str) and name == "init_python_namelist":
        path = pathlib.Path(value)
        if not path.is_absolute():
            return str((base_dir / path).resolve())
        return str(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Python QHD input file into cfg syntax.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()
    output_path = pathlib.Path(args.output)

    namespace = {
        "__file__": str(input_path),
        "np": np,
    }
    user_ns = runpy.run_path(str(input_path), init_globals=namespace)

    lines = []
    for key in SCALAR_KEYS:
        if key not in user_ns:
            continue
        value = normalize_value(key, user_ns[key], input_path.parent)
        lines.append(f"{key} = {int(value) if isinstance(value, bool) else value}")

    has_field_functions = any(callable(user_ns.get(name)) or name in user_ns for name in ("Bx", "By", "Bz", "Qx", "Qy", "Qz"))
    if has_field_functions and "init_python_namelist" not in user_ns:
        lines.append(f"init_python_namelist = {input_path}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
