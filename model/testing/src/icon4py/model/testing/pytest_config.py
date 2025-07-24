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


TEST_LEVELS = ("any", "unit", "integration")
DEFAULT_GRID: Final[str] = "simple_grid"
VALID_GRIDS: tuple[str, str, str] = ("simple_grid", "icon_grid", "icon_grid_global")


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

@pytest.fixture(scope="session")
def benchmark_only(request):
    try:
        benchmark_only = request.config.getoption("--benchmark-only")
    except ValueError:
        benchmark_only = False
    return benchmark_only


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )
    config.addinivalue_line(
        "markers",
        "level(name): marks test as unit or integration tests, mostly applicable where both are available",
    )

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"

    if config.getoption("--backend"):
        backend_option = config.getoption("--backend")
        _check_backend_validity(backend_option)

    # Handle datatest options: --datatest-only  and --datatest-skip
    if m_option := config.getoption("-m", []):
        m_option = [f"({m_option})"]  # add parenthesis around original k_option just in case
    if config.getoption("--datatest-only"):
        config.option.markexpr = " and ".join(["datatest", *m_option])

    if config.getoption("--datatest-skip"):
        config.option.markexpr = " and ".join(["not datatest", *m_option])


def pytest_addoption(parser):
    """Add custom commandline options for pytest."""
    try:
        datatest = parser.getgroup("datatest", "Options for data testing")
        datatest.addoption(
            "--datatest-skip",
            action="store_true",
            default=False,
            help="Skip all data tests",
        )
        datatest.addoption(
            "--datatest-only",
            action="store_true",
            default=False,
            help="Run only data tests",
        )
    except ValueError:
        pass
    try:
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

    try:
        parser.addoption(
            "--level",
            action="store",
            choices=TEST_LEVELS,
            help="Set level (unit, integration) of the tests to run. Defaults to 'any'.",
            default="any",
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
                REGIONAL_EXPERIMENT, keep_skip_values=False, backend=selected_backend
            ).grid
            return grid_instance
        case "icon_grid_global":
            from icon4py.model.testing.grid_utils import (
                get_grid_manager_for_experiment,
            )

            grid_instance = get_grid_manager_for_experiment(
                GLOBAL_EXPERIMENT, keep_skip_values=False, backend=selected_backend
            ).grid
            return grid_instance
        case _:
            return simple_grid.SimpleGrid(selected_backend)


def pytest_collection_modifyitems(config, items):
    test_level = config.getoption("--level")
    if test_level == "any":
        return
    for item in items:
        if (marker := item.get_closest_marker("level")) is not None:
            assert all(
                level in TEST_LEVELS for level in marker.args
            ), f"Invalid test level argument on function '{item.name}' - possible values are {TEST_LEVELS}"
            if test_level not in marker.args:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Selected level '{test_level}' does not match the configured '{marker.args}' level for this test."
                    )
                )


def pytest_runtest_setup(item):
    backend = model_backends.BACKENDS[item.config.getoption("--backend")]
    if "grid" in item.funcargs:
        grid = item.funcargs["grid"]
    else:
        # use the default grid
        grid = simple_grid.SimpleGrid(backend)
    apply_markers(
        item.own_markers,
        grid,
        backend,
    )


# pytest benchmark hook, see:
#     https://pytest-benchmark.readthedocs.io/en/latest/hooks.html#pytest_benchmark.hookspec.pytest_benchmark_update_json
def pytest_benchmark_update_json(output_json):
    "Replace 'fullname' of pytest benchmarks with a shorter name for better readability in bencher."
    for bench in output_json["benchmarks"]:
        # Replace fullname with name and filter unnecessary prefix and suffix
        bench["fullname"] = bench["name"].replace("test_", "")
