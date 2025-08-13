# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os

import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.testing.fixtures import (
    # Note: has to be used from pytest_hooks, otherwise StencilTest will not respect the session scope
    backend,  # noqa: F401
    grid,  # noqa: F401
)
from icon4py.model.testing.helpers import apply_markers


__all__ = [
    "pytest_addoption",
    "pytest_benchmark_update_json",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_setup",
]

_TEST_LEVELS = ("any", "unit", "integration")


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
            choices=_TEST_LEVELS,
            help="Set level (unit, integration) of the tests to run. Defaults to 'any'.",
            default="any",
        )
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    test_level = config.getoption("--level")
    if test_level == "any":
        return
    for item in items:
        if (marker := item.get_closest_marker("level")) is not None:
            assert all(
                level in _TEST_LEVELS for level in marker.args
            ), f"Invalid test level argument on function '{item.name}' - possible values are {_TEST_LEVELS}"
            if test_level not in marker.args:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Selected level '{test_level}' does not match the configured '{marker.args}' level for this test."
                    )
                )


def pytest_runtest_setup(item):
    selected_backend = model_backends.BACKENDS[item.config.getoption("--backend")]
    if "grid" in item.funcargs:
        grid = item.funcargs["grid"]
    else:
        # use the default grid
        grid = simple_grid.simple_grid(selected_backend)
    apply_markers(
        item.own_markers,
        grid,
        selected_backend,
    )


# pytest benchmark hook, see:
#     https://pytest-benchmark.readthedocs.io/en/latest/hooks.html#pytest_benchmark.hookspec.pytest_benchmark_update_json
def pytest_benchmark_update_json(output_json):
    "Replace 'fullname' of pytest benchmarks with a shorter name for better readability in bencher."
    for bench in output_json["benchmarks"]:
        # Replace fullname with name and filter unnecessary prefix and suffix
        bench["fullname"] = bench["name"].replace("test_", "")
