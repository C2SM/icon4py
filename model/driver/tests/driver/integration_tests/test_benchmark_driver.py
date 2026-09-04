# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.driver import config as driver_config, driver, driver_states, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend, process_props


BENCHMARK_EXPERIMENTS: list[test_defs.ExperimentDescription] = [test_defs.Experiments.JW]
BENCHMARK_STEPS: int = 100
BENCHMARK_ROUNDS: int = 5
BENCHMARK_WARMUP_ROUNDS: int = 2

_GRID_PRESETS: dict[str, test_defs.GridDescription] = {
    "icon_global": test_defs.Grids.R02B04_GLOBAL,
    "icon_benchmark_global": test_defs.Grids.R02B06_GLOBAL,
}


def _resolve_grid(
    request: pytest.FixtureRequest,
    experiment: test_defs.ExperimentDescription,
) -> test_defs.GridDescription:
    spec = request.config.getoption("--grid")
    if spec is None:
        return experiment.grid

    name = spec.split(":")[0].strip()
    grid = _GRID_PRESETS.get(name)
    if grid is None and hasattr(test_defs.Grids, name):
        grid = getattr(test_defs.Grids, name)
    if grid is None:
        for maybe_grid in vars(test_defs.Grids).values():
            if isinstance(maybe_grid, test_defs.GridDescription) and maybe_grid.name == name:
                grid = maybe_grid
                break
    if grid is None:
        raise pytest.UsageError(
            f"Unknown grid '{name}' in '--grid' option. "
            f"Use a preset, a 'Grids' attribute name, or a grid description name."
        )
    return grid


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
    return _resolve_grid(request, driver_benchmark_experiment)


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


@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize(
    "driver_benchmark_experiment",
    BENCHMARK_EXPERIMENTS,
    indirect=True,
    ids=lambda e: e.name,
)
def test_benchmark_driver_init(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.driver.enable_output is False

    def _setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (
            driver_benchmark_config,
            driver_benchmark_grid_manager,
            process_props,
            backend,
        ), {}

    def _timed(
        config: driver_config.ExperimentConfig,
        grid_manager: gm.GridManager,
        props: decomp_defs.ProcessProperties,
        bench_backend: gtx_typing.Backend | None,
    ) -> driver.Icon4pyDriver:
        return driver.initialize_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=props,
            backend=bench_backend,
        )

    benchmark.pedantic(
        _timed,
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )


@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize(
    "driver_benchmark_experiment",
    BENCHMARK_EXPERIMENTS,
    indirect=True,
    ids=lambda e: e.name,
)
def test_benchmark_driver_timeloop(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.driver.enable_output is False

    def _setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        icon4py_driver = driver.initialize_driver(
            config=driver_benchmark_config,
            grid_manager=driver_benchmark_grid_manager,
            process_props=process_props,
            backend=backend,
        )
        allocator = model_backends.get_allocator(backend)
        ds = driver.initialize_driver_states(icon4py_driver=icon4py_driver, allocator=allocator)
        return (icon4py_driver, ds), {}

    def _timed(fresh_driver: driver.Icon4pyDriver, ds: driver_states.DriverStates) -> None:
        fresh_driver.time_integration(ds)

    benchmark.pedantic(
        _timed,
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )


@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize(
    "driver_benchmark_experiment",
    BENCHMARK_EXPERIMENTS,
    indirect=True,
    ids=lambda e: e.name,
)
def test_benchmark_driver_total(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.driver.enable_output is False

    def _timed() -> tuple[driver_states.DriverStates, driver.Icon4pyDriver]:
        return driver.run_driver(
            config=driver_benchmark_config,
            grid_manager=driver_benchmark_grid_manager,
            process_props=process_props,
            backend=backend,
        )

    benchmark.pedantic(
        _timed,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )
