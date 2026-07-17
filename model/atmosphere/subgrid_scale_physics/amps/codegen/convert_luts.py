#!/usr/bin/env python
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline converter: AMPS_DATA Fortran lookup-table files -> a single
packaged `amps_luts.npz` archive consumed at runtime by
`core/lookup_tables.py` via `importlib.resources`.

Parses the ASCII LUT files per the exact reader-subroutine formats
transcribed VERBATIM in docs/superpowers/facts/m1/lut-files.md ("F3" in
comments below): RDCETB (mod_amps_utility.F90:96-294, F3 SS2.1) and RDSTTB
(mod_amps_utility.F90:333-348, F3 SS2.3). RDAPTB is DEAD CODE per F3 SS2.2
(a bare `return` before any I/O statement) -- `ap_act.bin.dat` (7.9 MB on
disk, which alone would blow this task's <5MB packaged-artifact budget) is
therefore deliberately NOT read or converted.

This script is NOT part of the installed `icon4py-atmosphere-amps` package
(same status as ../spikes/): a one-off dev tool, run ONCE against the real
AMPS_DATA directory, with its output committed to
`src/icon4py/model/atmosphere/subgrid_scale_physics/amps/data/amps_luts.npz`
(see data/README.md for provenance).

Run (from the icon4py worktree root):
    uv run --frozen python \
        model/atmosphere/subgrid_scale_physics/amps/codegen/convert_luts.py
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


DEFAULT_AMPS_DATA = pathlib.Path(
    "/Users/jcanton/projects/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA"
)
DEFAULT_OUTPUT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "icon4py"
    / "model"
    / "atmosphere"
    / "subgrid_scale_physics"
    / "amps"
    / "data"
    / "amps_luts.npz"
)

# Collision-efficiency 2D LUTs read by RDCETB (F3 SS2.1, lines 79-158): 2
# comment lines + a "nr nc xs dx ys dy" header line + nr rows of nc floats
# each. Target dims from F3 SS4's table (com_amps.F90 common/ECTBL).
_COL_LUT_FILES: dict[str, tuple[str, int, int]] = {
    "drpdrp": ("drop_drop_Rey4.dat", 201, 201),
    "hexdrp": ("hex_drop_Nre_Ec.dat", 64, 71),
    "bbcdrp": ("bbc_drop_Nre_Ec.dat", 64, 71),
    "coldrp": ("col_drop_Nre_Ec.dat", 62, 71),
    "gp1drp": ("grp01_ratNre_Ec.dat", 37, 125),
    "gp4drp": ("grp04_ratNre_Ec.dat", 27, 125),
    "gp8drp": ("grp08_ratNre_Ec.dat", 21, 125),
}

# Habit-frequency tables (F3 SS2.1, lines 161-215): 1 comment line + "nrow
# ncol" header + nrow rows of ncol floats. pol/pla/col are clipped to >= 0
# and renormalized by their per-cell sum; ros/ppo are clipped only (F3 SS6,
# "Post-read transform to replicate", RDCETB lines 198-215).
_FRQ_FILES: dict[str, str] = {
    "pol_frq": "pol_frq.dat",
    "pla_frq": "pla_frq.dat",
    "col_frq": "col_frq.dat",
    "ros_frq": "ros_frq.dat",
    "ppo_frq": "ppo_frq.dat",
}

# a/c-axis (tmp_map_*) and density (tmd_map_*) diagnostic maps (F3 SS2.1,
# lines 217-259): same 1-comment/"nrow ncol" header convention, stacked into
# a trailing length-2 axis to mirror the Fortran mtac_map_col/pla(:,:,1:2)
# common-block target (F3 SS4 table: index 1 = tmp (a/c-axis), index 2 = tmd
# (density)).
_MAP_FILES: dict[str, tuple[str, str]] = {
    "mtac_map_col": ("tmp_map_col.dat", "tmd_map_col.dat"),
    "mtac_map_pla": ("tmp_map_pla.dat", "tmd_map_pla.dat"),
}

# lmt_mass_{col,pla}.dat: same header convention, 4 data columns per row;
# only column 2 (0-based index 1) is kept as lmt_mass_col/pla(i) (F3 SS3
# "lmt_mass_col.dat" sample + SS6 parser note: "4 columns, only col 2
# retained").
_LMT_MASS_FILES: dict[str, str] = {
    "lmt_mass_col": "lmt_mass_col.dat",
    "lmt_mass_pla": "lmt_mass_pla.dat",
}


def _read_col_lut(
    path: pathlib.Path, nr_expected: int, nc_expected: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a 2-comment-line collision-efficiency LUT file (F3 SS2.1, SS3).

    Returns (data[nr, nc], aux) where aux = [xs, dx, ys, dy, nr, nc], the
    `col_lut_aux` field order (class_Group.F90:179-183, F3 SS4).
    """
    with path.open() as f:
        f.readline()  # comment line 1
        f.readline()  # comment line 2
        nr, nc, xs, dx, ys, dy = (float(tok) for tok in f.readline().split())
        nr, nc = int(nr), int(nc)
        if (nr, nc) != (nr_expected, nc_expected):
            raise ValueError(
                f"{path}: header dims {(nr, nc)} != expected {(nr_expected, nc_expected)}"
            )
        data = np.loadtxt(f, max_rows=nr)
    if data.shape != (nr, nc):
        raise ValueError(f"{path}: parsed data shape {data.shape} != header dims {(nr, nc)}")
    aux = np.array([xs, dx, ys, dy, float(nr), float(nc)], dtype=np.float64)
    return data, aux


def _read_grid_2d(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a 1-comment-line 'nrow ncol' grid file (F3 SS2.1, SS3).

    Returns (data[nrow, ncol], aux) where aux = [nrow, ncol] (no
    xs/dx/ys/dy -- "grid is implicit" per F3 SS3).
    """
    with path.open() as f:
        f.readline()  # comment line
        nrow, ncol = (int(tok) for tok in f.readline().split())
        data = np.loadtxt(f, max_rows=nrow)
    if data.shape != (nrow, ncol):
        raise ValueError(f"{path}: parsed data shape {data.shape} != header dims {(nrow, ncol)}")
    aux = np.array([nrow, ncol], dtype=np.float64)
    return data, aux


def _read_stdnorm(path: pathlib.Path) -> np.ndarray:
    """Parse stdnorm.dat: headerless 451x6, keep columns 3-6 as znorm(451,
    4) (F3 SS2.3, SS3). Columns 1-2 (dum1, dum2 in RDSTTB) are discarded;
    the implicit x-grid is 0.01*(row-1) but is not stored (no aux)."""
    data = np.loadtxt(path)
    if data.shape != (451, 6):
        raise ValueError(f"{path}: shape {data.shape} != expected (451, 6)")
    return data[:, 2:6].copy()


def convert(amps_data_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Parse the real AMPS_DATA tree and return the flat {key: array}
    mapping to be written to `amps_luts.npz` (brief's naming convention:
    `<name>` + `<name>_aux`)."""
    collision_dir = amps_data_dir / "collision_data"
    statpack_dir = amps_data_dir / "statpack"

    out: dict[str, np.ndarray] = {}

    for name, (filename, nr, nc) in _COL_LUT_FILES.items():
        data, aux = _read_col_lut(collision_dir / filename, nr, nc)
        out[name] = data
        out[f"{name}_aux"] = aux

    raw_frq: dict[str, np.ndarray] = {}
    frq_auxes: dict[str, np.ndarray] = {}
    for name, filename in _FRQ_FILES.items():
        data, aux = _read_grid_2d(collision_dir / filename)
        raw_frq[name] = data
        frq_auxes[name] = aux
    reference_aux = frq_auxes["pol_frq"]
    for name, aux in frq_auxes.items():
        if not np.array_equal(aux, reference_aux):
            raise ValueError(f"{name}: header dims {aux} != pol_frq's {reference_aux}")

    # Post-read transform, RDCETB lines 198-215 (F3 SS6): clip all five to
    # >= 0; pol/pla/col are additionally renormalized by their per-cell sum.
    pol = np.clip(raw_frq["pol_frq"], 0.0, None)
    pla = np.clip(raw_frq["pla_frq"], 0.0, None)
    col = np.clip(raw_frq["col_frq"], 0.0, None)
    ros = np.clip(raw_frq["ros_frq"], 0.0, None)
    ppo = np.clip(raw_frq["ppo_frq"], 0.0, None)
    denom = pol + pla + col
    if np.any(denom == 0.0):
        raise ValueError(
            "pol_frq + pla_frq + col_frq is zero somewhere; RDCETB's normalization "
            "would divide by zero there"
        )
    out["pol_frq"] = pol / denom
    out["pla_frq"] = pla / denom
    out["col_frq"] = col / denom
    out["ros_frq"] = ros
    out["ppo_frq"] = ppo
    for name in _FRQ_FILES:
        out[f"{name}_aux"] = reference_aux

    for name, (tmp_file, tmd_file) in _MAP_FILES.items():
        tmp_data, tmp_aux = _read_grid_2d(collision_dir / tmp_file)
        tmd_data, tmd_aux = _read_grid_2d(collision_dir / tmd_file)
        if not np.array_equal(tmp_aux, tmd_aux):
            raise ValueError(f"{name}: tmp/tmd header dims disagree: {tmp_aux} vs {tmd_aux}")
        out[name] = np.stack([tmp_data, tmd_data], axis=-1)
        out[f"{name}_aux"] = tmp_aux

    for name, filename in _LMT_MASS_FILES.items():
        data, aux = _read_grid_2d(collision_dir / filename)
        out[name] = data[:, 1].copy()
        out[f"{name}_aux"] = aux

    out["znorm"] = _read_stdnorm(statpack_dir / "stdnorm.dat")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "amps_data_dir",
        type=pathlib.Path,
        nargs="?",
        default=DEFAULT_AMPS_DATA,
        help=f"AMPS_DATA directory (default: {DEFAULT_AMPS_DATA})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"output .npz path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    tables = convert(args.amps_data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # mypy false positive: savez_compressed's stub has a keyword-only
    # `allow_pickle: bool` between `*args` and `**kwds`, which makes mypy
    # (incorrectly) worry that **tables could supply `allow_pickle` with a
    # non-bool value -- it never does (see `convert()`'s key names).
    np.savez_compressed(args.output, **tables)  # type: ignore[arg-type]

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {len(tables)} arrays ({len(tables) // 2} tables) to {args.output}")
    print(f"Output size: {size_kb:.1f} KiB")


if __name__ == "__main__":
    main()
