# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Callable
from typing import Final

import pytest
from gt4py.next import gtfn_cpu, gtfn_gpu, itir_python

import icon4py.model.common.settings as settings
from icon4py.model.testing.datatest_utils import (
    GLOBAL_EXPERIMENT,
    REGIONAL_EXPERIMENT,
)


DEFAULT_BACKEND: Final = "roundtrip"

BACKENDS: dict[str, Callable] = {
    "embedded": None,
    "roundtrip": itir_python,
    "gtfn_cpu": gtfn_cpu,
    "gtfn_gpu": gtfn_gpu,
}

GPU_BACKENDS: list[str] =["gtfn_gpu"]

try:
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu,
        run_dace_cpu_noopt,
        run_dace_gpu,
        run_dace_gpu_noopt,
    )

    BACKENDS.update(
        {
            "dace_cpu": run_dace_cpu,
            "dace_gpu": run_dace_gpu,
            "dace_cpu_noopt": run_dace_cpu_noopt,
            "dace_gpu_noopt": run_dace_gpu_noopt,
        }
    )
    GPU_BACKENDS.extend(["dace_gpu", "dace_gpu_noopt"])

except ImportError:
    # dace module not installed, ignore dace backends
    pass


def _check_backend_validity(backend_name: str) -> None:
    if backend_name not in BACKENDS:
        available_backends = ", ".join([f"'{k}'" for k in BACKENDS.keys()])
        raise Exception(
            "Need to select a backend. Select from: ["
            + available_backends
            + "] and pass it as an argument to --backend when invoking pytest."
        )


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line("markers", "slow_tests: this test takes a very long time")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"

    if config.getoption("--backend"):
        backend = config.getoption("--backend")
        _check_backend_validity(backend)
        settings.backend = BACKENDS[backend]

    if config.getoption("--dace-orchestration"):
        settings.dace_orchestration = True


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
            default=DEFAULT_BACKEND,
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
            "--dace-orchestration",
            action="store",
            default=None,
            help="Performs DaCe orchestration. Any value will enable it.",
        )
    except ValueError:
        pass


def pytest_runtest_setup(item):
    for _ in item.iter_markers(name="datatest"):
        if not item.config.getoption("--datatest"):
            pytest.skip("need '--datatest' option to run")


def pytest_generate_tests(metafunc):
    selected_backend = BACKENDS[DEFAULT_BACKEND]

    # parametrise backend
    if "backend" in metafunc.fixturenames:
        backend_option = metafunc.config.getoption("backend")
        _check_backend_validity(backend_option)

        selected_backend = BACKENDS[backend_option]
        metafunc.parametrize("backend", [selected_backend], ids=[f"backend={backend_option}"])

    # parametrise grid
    if "grid" in metafunc.fixturenames:
        selected_grid_type = metafunc.config.getoption("--grid")

        try:
            if selected_grid_type == "simple_grid":
                from icon4py.model.common.grid.simple import SimpleGrid

                grid_instance = SimpleGrid()
            elif selected_grid_type == "icon_grid":
                from icon4py.model.testing.grid_utils import (
                    get_grid_manager_for_experiment,
                )

                grid_instance = get_grid_manager_for_experiment(
                    REGIONAL_EXPERIMENT, backend=selected_backend
                ).grid
            elif selected_grid_type == "icon_grid_global":
                from icon4py.model.testing.grid_utils import (
                    get_grid_manager_for_experiment,
                )

                grid_instance = get_grid_manager_for_experiment(
                    GLOBAL_EXPERIMENT, backend=selected_backend
                ).grid
            else:
                raise ValueError(f"Unknown grid type: {selected_grid_type}")
            metafunc.parametrize("grid", [grid_instance], ids=[f"grid={selected_grid_type}"])
        except ValueError as err:
            available_grids = [
                "simple_grid",
                "icon_grid",
                "icon_grid_global",
            ]
            raise Exception(f"{err}. Select from: {available_grids}") from err
