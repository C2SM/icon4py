# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for single-node output from the standalone driver.

Runs the Jablonowski-Williamson testcase for the default (one) step with
``enable_output=True`` and asserts that valid CF/UGRID NetCDF files are produced.
It exercises the full ``store -> file`` path, not just the in-memory bridge.

Being a datatest, it requires the JW grid.
"""

import pathlib

import netCDF4 as nc
import pytest

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import driver_io, main
from icon4py.model.testing import definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend_like

from ..fixtures import *  # noqa: F403


def _find_one(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(directory.rglob(pattern))
    assert matches, f"no file matching {pattern!r} under {directory}"
    return matches[0]


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_standalone_driver_writes_output(
    experiment: test_defs.Experiment,
    *,
    backend_like: model_backends.BackendLike,
    tmp_path: pathlib.Path,
) -> None:
    grid_file_path = grid_utils._download_grid_file(experiment.grid)
    output_path = tmp_path / "io_driver_output"

    main.main(
        grid_file_path=grid_file_path,
        icon4py_backend=backend_like,
        output_path=output_path,
        log_level="notset",
        print_distributed_debug_msg=False,
        force_serial_run=False,
        enable_output=True,
    )

    io_dir = output_path / driver_io.OUTPUT_SUBDIR
    assert io_dir.is_dir(), f"expected output subfolder at {io_dir}"

    # the UGRID grid file is written on monitor init
    _find_one(io_dir, "*_ugrid.nc")

    # single data file: all prognostic AND diagnostic fields together, with the expected
    # shapes and a single time slice (output is on by default; no group split)
    output_file = _find_one(io_dir, f"{driver_io.DEFAULT_OUTPUT_FILENAME}_*.nc")
    with nc.Dataset(output_file) as ds:
        assert ds.Conventions == "CF-1.7"
        for name in driver_io.DEFAULT_OUTPUT_VARIABLES:
            assert name in ds.variables, f"{name} missing from output"
            var = ds.variables[name]
            assert var.dimensions[0] == "time"
            assert var.shape[0] == 1
        # vertical placement: w on interface levels, the rest on full levels
        assert "interface_level" in ds.variables["upward_air_velocity"].dimensions
        assert "edge" in ds.variables["normal_velocity"].dimensions
        # diagnostics live on cells/full levels
        assert "cell" in ds.variables["temperature"].dimensions
        assert len(ds.dimensions["time"]) == 1
