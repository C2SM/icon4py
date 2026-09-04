# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.driver import config as driver_config, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils


BENCHMARK_EXPERIMENTS: list[test_defs.ExperimentDescription] = [test_defs.Experiments.JW]
BENCHMARK_STEPS: int = 100
BENCHMARK_ROUNDS: int = 5
BENCHMARK_WARMUP_ROUNDS: int = 2


def _make_config(
    experiment: test_defs.ExperimentDescription,
    grid: test_defs.GridDescription,
    process_props: decomp_defs.ProcessProperties,
) -> driver_config.ExperimentConfig:
    dt_utils.download_experiment(experiment, process_props)
    experiment_path = dt_utils.get_path_for_experiment(experiment, process_props)
    config = driver_config.read_experiment_config_from_fortran(experiment_path)
    return config.with_overrides(
        driver={
            "dtime": time.RelativeTime(seconds=50),
            "enable_output": False,
            "end_of_simulation": time.NumTimeSteps(BENCHMARK_STEPS),
        }
    )


def _make_grid_manager(
    config: driver_config.ExperimentConfig,
    grid: test_defs.GridDescription,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
) -> gm.GridManager:
    allocator = model_backends.get_allocator(backend)
    grid_file_path = grid_utils._download_grid_file(grid)
    return driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )


@pytest.fixture
def driver_benchmark_experiment(request: pytest.FixtureRequest) -> test_defs.ExperimentDescription:
    return request.param


@pytest.fixture
def driver_benchmark_grid(
    request: pytest.FixtureRequest,
    driver_benchmark_experiment: test_defs.ExperimentDescription,
) -> test_defs.GridDescription:
    grid_option = request.config.getoption("--grid")
    if grid_option is None:
        grid = driver_benchmark_experiment.grid
    else:
        grid = grid_utils.resolve_grid_description(grid_option)
    return grid


@pytest.fixture
def driver_benchmark_config(
    driver_benchmark_experiment: test_defs.ExperimentDescription,
    driver_benchmark_grid: test_defs.GridDescription,
    process_props: decomp_defs.ProcessProperties,
) -> driver_config.ExperimentConfig:
    return _make_config(driver_benchmark_experiment, driver_benchmark_grid, process_props)


@pytest.fixture
def driver_benchmark_grid_manager(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid: test_defs.GridDescription,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
) -> gm.GridManager:
    return _make_grid_manager(
        config=driver_benchmark_config,
        grid=driver_benchmark_grid,
        process_props=process_props,
        backend=backend,
    )
