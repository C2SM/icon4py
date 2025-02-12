# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
from typing import Final

import pytest
from gt4py.next import backend as gtx_backend

from icon4py.model.common import model_backends
from icon4py.model.common.grid import base as base_grid, simple as simple_grid
from icon4py.model.testing.datatest_utils import (
    GLOBAL_EXPERIMENT,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.testing.helpers import apply_markers


DEFAULT_GRID: Final[str] = "simple_grid"
VALID_GRIDS: tuple[str, str, str] = (DEFAULT_GRID, "icon_grid", "icon_grid_global")


def _check_backend_validity(backend_name: str) -> None:
    if backend_name not in model_backends.BACKENDS:
        available_backends = ", ".join([f"'{k}'" for k in model_backends.BACKENDS.keys()])
        raise Exception(
            "Need to select a backend. Select from: ["
            + available_backends
            + "] and pass it as an argument to --backend when invoking pytest."
        )


def _check_grid_validity(grid_name: str) -> None:
    assert (
        grid_name in VALID_GRIDS
    ), f"Invalid value for '--grid' option - possible names are {VALID_GRIDS}"


@pytest.fixture(scope="session")
def backend(request):
    try:
        backend_option = request.config.getoption("backend")
    except ValueError:
        backend_option = model_backends.DEFAULT_BACKEND
    else:
        _check_backend_validity(backend_option)

    selected_backend = model_backends.BACKENDS[backend_option]
    return selected_backend


@pytest.fixture(scope="session")
def grid(request, backend):
    try:
        grid_option = request.config.getoption("grid")
    except ValueError:
        grid_option = DEFAULT_GRID
    else:
        _check_grid_validity(grid_option)
    grid = _get_grid(grid_option, backend)
    return grid


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"

    if config.getoption("--backend"):
        backend_option = config.getoption("--backend")
        _check_backend_validity(backend_option)


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
            default=model_backends.DEFAULT_BACKEND,
            help="GT4Py backend to use when executing stencils. Defaults to roundtrip backend, other options include gtfn_cpu, gtfn_gpu, and embedded",
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


def _get_grid(
    selected_grid_type: str, selected_backend: gtx_backend.Backend | None
) -> base_grid.BaseGrid:
    match selected_grid_type:
        case "icon_grid":
            from icon4py.model.testing.grid_utils import (
                get_grid_manager_for_experiment,
            )

            grid_instance = get_grid_manager_for_experiment(
                REGIONAL_EXPERIMENT, backend=selected_backend
            ).grid
            return grid_instance
        case "icon_grid_global":
            from icon4py.model.testing.grid_utils import (
                get_grid_manager_for_experiment,
            )

            grid_instance = get_grid_manager_for_experiment(
                GLOBAL_EXPERIMENT, backend=selected_backend
            ).grid
            return grid_instance
        case _:
            return simple_grid.SimpleGrid()


def pytest_runtest_setup(item):
    backend = model_backends.BACKENDS[item.config.getoption("--backend")]
    grid_option = item.config.getoption("--grid")
    has_skip_values = grid_option is not None and grid_option != DEFAULT_GRID
    apply_markers(
        item.own_markers,
        backend,
        has_skip_values,
        is_datatest=item.config.getoption("--datatest"),
    )
