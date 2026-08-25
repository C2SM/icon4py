# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import initial_condition, model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    nonhydro_states,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.driver import config as driver_config, driver, driver_states, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend, process_props


if mpi_decomposition.mpi4py is None:
    pytest.skip(
        "Skipping parallel driver benchmark tests on single-node installation",
        allow_module_level=True,
    )


_log = logging.getLogger(__file__)
_DRIVER_BENCHMARK_EXPERIMENTS: dict[str, test_defs.ExperimentDescription] = {
    "jw": test_defs.Experiments.JW,
}

_GRID_PRESETS: dict[str, test_defs.GridDescription] = {
    "icon_global": test_defs.Grids.R02B04_GLOBAL,
    "icon_benchmark_global": test_defs.Grids.R02B06_GLOBAL,
    "icon_regional": test_defs.Grids.MCH_CH_R04B09_DSL,
    "icon_benchmark_regional": test_defs.Grids.MCH_OPR_R19B08_DOMAIN01,
}


def _resolve_experiment(request: pytest.FixtureRequest) -> test_defs.ExperimentDescription:
    name = request.config.getoption("--driver-benchmark-experiment")
    assert isinstance(name, str)
    key = name.lower()
    if key not in _DRIVER_BENCHMARK_EXPERIMENTS:
        raise pytest.UsageError(
            f"Unknown driver benchmark experiment '{name}'. "
            f"Supported values: {list(_DRIVER_BENCHMARK_EXPERIMENTS)}."
        )
    return _DRIVER_BENCHMARK_EXPERIMENTS[key]


def _resolve_grid(
    request: pytest.FixtureRequest, experiment: test_defs.ExperimentDescription
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
    if grid.limited_area:
        pytest.xfail("Limited-area grids are not yet supported in distributed driver benchmarks")
    return grid


def _num_steps(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--driver-benchmark-steps")


def _rounds(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--driver-benchmark-rounds")


def _warmup_rounds(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--driver-benchmark-warmup-rounds")


def _make_config(
    request: pytest.FixtureRequest,
    experiment: test_defs.ExperimentDescription,
    process_props: decomp_defs.ProcessProperties,
) -> driver_config.ExperimentConfig:
    dt_utils.download_experiment(experiment, process_props)
    experiment_path = dt_utils.get_path_for_experiment(experiment, process_props)
    config = driver_config.read_experiment_config_from_fortran(experiment_path)
    steps = _num_steps(request)
    return config.with_overrides(
        driver={
            "enable_output": False,
            "end_of_simulation": time.NumTimeSteps(steps),
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


def _assemble_driver_states(
    icon4py_driver: driver.Icon4pyDriver,
) -> driver_states.DriverStates:
    backend = icon4py_driver.backend
    allocator = model_backends.get_allocator(backend)
    grid = icon4py_driver.grid

    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=grid,
        allocator=allocator,
    )
    tracer_state_now = tracer_states.initialize_tracer_state(
        grid=grid,
        allocator=allocator,
        tracer_config=icon4py_driver.config.tracer_config,
    )
    solve_nonhydro_diagnostic_state = (
        nonhydro_states.initialize_solve_nonhydro_diagnostic_state(grid=grid, allocator=allocator)
        if icon4py_driver.config.nonhydrostatic is not None
        else None
    )
    initial_condition.create(
        config=icon4py_driver.config.initial_condition,
        vertical_config=icon4py_driver.config.vertical_grid,
        grid=grid,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        tracer_state_now=tracer_state_now,
        solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
        backend=backend,
        exchange=icon4py_driver.exchange,
        global_reductions=icon4py_driver.global_reductions,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=grid, allocator=allocator)
    ds = driver_states.assemble_driver_states(
        grid=grid,
        allocator=allocator,
        backend=backend,
        exchange=icon4py_driver.exchange,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        tracer_state_now=tracer_state_now,
        diagnostic_state=diagnostic_state,
        experiment_config=icon4py_driver.config,
        solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
    )
    driver_utils.validate_granule_state_consistency(
        config=icon4py_driver.config,
        granules=icon4py_driver.granules,
        states=ds,
    )
    return ds


def _barrier(process_props: decomp_defs.ProcessProperties) -> None:
    if process_props.comm is not None:
        process_props.comm.Barrier()


def _with_barriers[T](
    process_props: decomp_defs.ProcessProperties, fn: Callable[..., T]
) -> Callable[..., T]:
    def _wrapped(*args: Any, **kwargs: Any) -> T:
        _barrier(process_props)
        result = fn(*args, **kwargs)
        _barrier(process_props)
        return result

    return _wrapped


@pytest.fixture(scope="module")
def driver_benchmark_experiment(request: pytest.FixtureRequest) -> test_defs.ExperimentDescription:
    return _resolve_experiment(request)


@pytest.fixture(scope="module")
def driver_benchmark_grid(
    request: pytest.FixtureRequest,
    driver_benchmark_experiment: test_defs.ExperimentDescription,
) -> test_defs.GridDescription:
    return _resolve_grid(request, driver_benchmark_experiment)


@pytest.fixture
def driver_benchmark_config(
    request: pytest.FixtureRequest,
    driver_benchmark_experiment: test_defs.ExperimentDescription,
    process_props: decomp_defs.ProcessProperties,
) -> driver_config.ExperimentConfig:
    return _make_config(request, driver_benchmark_experiment, process_props)


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


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_benchmark_driver_init(  # noqa: PLR0917 [too-many-positional-arguments]
    request: pytest.FixtureRequest,
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
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=_rounds(request),
        iterations=1,
        warmup_rounds=_warmup_rounds(request),
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_benchmark_driver_timeloop(  # noqa: PLR0917 [too-many-positional-arguments]
    request: pytest.FixtureRequest,
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
        ds = _assemble_driver_states(icon4py_driver)
        return (icon4py_driver, ds), {}

    def _timed(fresh_driver: driver.Icon4pyDriver, ds: driver_states.DriverStates) -> None:
        fresh_driver.time_integration(ds)

    benchmark.pedantic(
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=_rounds(request),
        iterations=1,
        warmup_rounds=_warmup_rounds(request),
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_benchmark_driver_total(  # noqa: PLR0917 [too-many-positional-arguments]
    request: pytest.FixtureRequest,
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
        _with_barriers(process_props, _timed),
        rounds=_rounds(request),
        iterations=1,
        warmup_rounds=_warmup_rounds(request),
    )
