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
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_gpu
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line("markers", "slow_tests: this test takes a very long time")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )

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
        # TODO (samkellerhals): set embedded to default as soon as all tests run in embedded mode
        parser.addoption(
            "--backend",
            action="store",
            default="roundtrip",
            help="GT4Py backend to use when executing stencils. Defaults to rountrip backend, other options include gtfn_cpu, gtfn_gpu, and embedded",
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

        backends = {
            "embedded": None,
            "roundtrip": run_roundtrip,
            "gtfn_cpu": run_gtfn,
            "gtfn_gpu": run_gtfn_gpu,
        }
        gpu_backends = ["gtfn_gpu"]

        try:
            from gt4py.next.program_processors.runners.dace_iterator import (
                run_dace_cpu,
                run_dace_gpu,
            )

            backends.update(
                {
                    "dace_cpu": run_dace_cpu,
                    "dace_gpu": run_dace_gpu,
                }
            )
            gpu_backends.append("dace_gpu")

        except ImportError:
            # dace module not installed, ignore dace backends
            pass

        if backend_option not in backends:
            available_backends = ", ".join([f"'{k}'" for k in backends.keys()])
            raise Exception(
                "Need to select a backend. Select from: ["
                + available_backends
                + "] and pass it as an argument to --backend when invoking pytest."
            )

        metafunc.parametrize(
            "backend", [backends[backend_option]], ids=[f"backend={backend_option}"]
        )

    # parametrise grid
    if "grid" in metafunc.fixturenames:
        on_gpu = backend_option in gpu_backends
        selected_grid_type = metafunc.config.getoption("--grid")

        try:
            if selected_grid_type == "simple_grid":
                from icon4py.model.common.grid.simple import SimpleGrid

                grid_instance = SimpleGrid()
            elif selected_grid_type == "icon_grid":
                from icon4py.model.common.test_utils.grid_utils import get_icon_grid

                grid_instance = get_icon_grid(on_gpu)
            else:
                raise ValueError(f"Unknown grid type: {selected_grid_type}")
            metafunc.parametrize("grid", [grid_instance], ids=[f"grid={selected_grid_type}"])
        except ValueError as e:
            available_grids = ["simple_grid", "icon_grid"]
            raise Exception(f"{e}. Select from: {available_grids}")
