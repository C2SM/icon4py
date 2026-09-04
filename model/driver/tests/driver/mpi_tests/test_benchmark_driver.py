# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.driver import config as driver_config, driver, driver_states
from icon4py.model.testing.fixtures.datatest import backend, process_props

from ..fixtures import *  # noqa: F403


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


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
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
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
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
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
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
        _with_barriers(process_props, _timed),
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )
