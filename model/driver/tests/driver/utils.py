# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the driver tests."""

import pathlib

import netCDF4 as nc
import numpy as np

from icon4py.model.driver import driver_io


def read_qv_frames(output_dir: pathlib.Path) -> np.ndarray:
    """qv from the driver output as (time, cell, level)."""
    output_files = sorted(output_dir.rglob(f"{driver_io.DEFAULT_OUTPUT_FILENAME}_*.nc"))
    assert output_files, f"no output file under {output_dir}"
    frames = []
    for output_file in output_files:
        with nc.Dataset(output_file) as ds:
            assert "qv" in ds.variables, "qv missing from driver output"
            var = ds.variables["qv"]
            axes = [var.dimensions.index(name) for name in ("time", "cell", "level")]
            frames.append(np.transpose(np.asarray(var[:]), axes))
    return np.concatenate(frames, axis=0)
