# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import pytest
from gt4py.next.program_processors.runners.gtfn import run_gtfn
from gt4py.next.program_processors.runners.roundtrip import executor


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"


def pytest_addoption(parser):
    """Add custom commandline options for pytest."""
    try:
        parser.addoption(
            "--datatest",
            action="store_true",
            help="Run tests that use serialized data, can be slow since data might be downloaded from online storage.",
            default=False,
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--backend",
            action="store",
            default="embedded",
            help="GT4Py backend to use when executing stencils. Defaults to embedded, other options include gtfn_cpu",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--grid",
            action="store",
            default="simple_grid",
            help="Grid to use. Defaults to simple_grid, other options include icon_grid",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--enable-mixed-precision",
            action="store_true",
            help="Switch unit tests from double to mixed-precision",
            default=False,
        )
    except ValueError:
        pass


def pytest_runtest_setup(item):
    for _ in item.iter_markers(name="datatest"):
        if not item.config.getoption("--datatest"):
            pytest.skip("need '--datatest' option to run")


def pytest_generate_tests(metafunc):
    # parametrise backend
    if "backend" in metafunc.fixturenames:
        backend_option = metafunc.config.getoption("backend")

        params = []
        ids = []

        if backend_option == "gtfn_cpu":
            params.append(run_gtfn)
            ids.append("backend=gtfn_cpu")
        elif backend_option == "embedded":
            params.append(executor)
            ids.append("backend=embedded")
        # TODO (skellerhals): add gpu support
        else:
            raise Exception(
                "Need to select a backend. Select from: ['embedded', 'gtfn_cpu'] and pass it as an argument to --backend when invoking pytest."
            )

        metafunc.parametrize("backend", params, ids=ids)

    # parametrise grid
    if "grid" in metafunc.fixturenames:
        selected_grid_type = metafunc.config.getoption("--grid")

        try:
            if selected_grid_type == "simple_grid":
                from icon4py.model.common.grid.simple import SimpleGrid

                grid_instance = SimpleGrid()
            elif selected_grid_type == "icon_grid":
                from icon4py.model.common.test_utils.grid_utils import get_icon_grid

                grid_instance = get_icon_grid()
            else:
                raise ValueError(f"Unknown grid type: {selected_grid_type}")
            metafunc.parametrize("grid", [grid_instance], ids=[f"grid={selected_grid_type}"])
        except ValueError as e:
            available_grids = ["simple_grid", "icon_grid"]
            raise Exception(f"{e}. Select from: {available_grids}")
