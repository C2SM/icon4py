# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
from scipy.stats import linregress

from icon4py.model.common import model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geometry_meta, gridfile
from icon4py.model.common.initial_condition.analytical import (
    linear_horizontal_tracer_advection,
    linear_vertical_tracer_advection,
)
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import config as test_config, definitions as test_defs, grid_utils

from ..fixtures import *  # noqa: F403


_ZERO_ORDER = 0.0
_DEGRADED_FIRST_ORDER = 0.5
_SECOND_ORDER = 2.0
_THIRD_ORDER = 3.0
_TOL = 0.1
_DEGRADED_TOL = 0.5
_STD_TOL = 0.4
_HORIZONTAL_CONVERGENCE_GRIDS: Final = (
    test_defs.Grids.TORUS_100X116_1000M,
    test_defs.Grids.TORUS_200X232_500M,
    test_defs.Grids.TORUS_400X462_250M,
)
_VERTICAL_CONVERGENCE_GRID: Final = test_defs.Grids.TORUS_1000X1000_250M


def _compute_relative_errors(
    simulated_values: data_alloc.NDArray,
    reference: data_alloc.NDArray,
) -> tuple[float, float]:
    # compute the errors relative to the reference
    # note: the following lines take the errors of all the levels, which is fine
    array_ns = data_alloc.array_namespace(simulated_values)
    error_l1 = array_ns.sum(array_ns.abs(simulated_values - reference)) / array_ns.sum(
        array_ns.abs(reference)
    )
    error_linf = array_ns.max(array_ns.abs(simulated_values - reference)) / array_ns.max(
        array_ns.abs(reference)
    )
    return error_l1, error_linf


def _check_convergence(
    *,
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    error_l1: list[float] | np.ndarray,
    error_linf: list[float] | np.ndarray,
    grid_spacing: list[float] | np.ndarray,
) -> None:
    linreg_l1 = linregress(np.log(grid_spacing), np.log(error_l1))
    p_l1 = linreg_l1.slope
    linreg_linf = linregress(np.log(grid_spacing), np.log(error_linf))
    p_linf = linreg_linf.slope
    # check that the measured convergence rates are within the acceptable ranges
    assert l1_acceptable_range[0] <= p_l1 <= l1_acceptable_range[1]
    assert linf_acceptable_range[0] <= p_linf <= linf_acceptable_range[1]
    # check that the standard errors are within the acceptable tolerance
    assert linreg_l1.stderr <= _STD_TOL
    assert linreg_linf.stderr <= _STD_TOL


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, l1_acceptable_range, linf_acceptable_range",
    [
        (
            "linear_horizontal_advection_gaussian_2d",
            [_SECOND_ORDER - _TOL, _SECOND_ORDER + _TOL],
            [_SECOND_ORDER - _TOL, _SECOND_ORDER + _TOL],
        ),
        (
            "linear_horizontal_advection_circle_2d",
            [_DEGRADED_FIRST_ORDER - _DEGRADED_TOL, _DEGRADED_FIRST_ORDER + _DEGRADED_TOL],
            [_ZERO_ORDER - _TOL, _ZERO_ORDER + _TOL],
        ),
    ],
)
def test_horizontal_tracer_advection_convergence(
    *,
    experiment_case: str,
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_paths = [grid_utils._download_grid_file(grid_description) for grid_description in _HORIZONTAL_CONVERGENCE_GRIDS]

    error_l1: list[float] = []
    error_linf: list[float] = []
    mean_edge_length: list[float] = []

    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    experiment_config = config_io.read_yaml_str(
        config_path.read_text(), driver_config.ExperimentConfig
    ).with_overrides(driver={"output_path": tmp_path / "ci_driver_output"})

    grid_managers = [
        driver_utils.create_grid_manager(
            grid_file_path=grid_path,
            vertical_grid_config=experiment_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )
        for _, grid_path in enumerate(grid_file_paths)
    ]

    # compute the time step based on the CFL condition and the maximum velocity on the finest grid
    domain_length = grid_managers[-1].grid.grid_params.domain_length
    domain_height = grid_managers[-1].grid.grid_params.domain_height
    assert (
        type(experiment_config.initial_condition.config)
        is linear_horizontal_tracer_advection.LinearHorizontalAdvectionConfig
    )
    assert domain_length is not None
    assert domain_height is not None
    vel_max = linear_horizontal_tracer_advection.compute_max_velocity(
        velocity_field=experiment_config.initial_condition.config.velocity_field,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    match experiment_config.driver.end_of_simulation:
        case time.RelativeTime() as relative:
            integration_time = relative.total_seconds()
        case time.AbsoluteTime() as absolute:
            integration_time = (
                absolute - experiment_config.driver.start_of_simulation
            ).total_seconds()
        case _:
            raise ValueError(
                f"end_of_simulation {experiment_config.driver.end_of_simulation} must be specified as a RelativeTime or AbsoluteTime for this test"
            )
    dtime = min(
        experiment_config.initial_condition.config.cfl_number
        * grid_managers[-1].geometry_fields[gridfile.GeometryName.EDGE_LENGTH].asnumpy().mean()
        / vel_max,
        integration_time,
    )
    # recompute the integration time to be a multiple of dtime, so that the model stops at a time that is consistent with the CFL condition
    num_steps = int(integration_time / dtime)
    experiment_config = experiment_config.with_overrides(
        driver={
            "dtime": time.RelativeTime(seconds=dtime),
            "end_of_simulation": time.NumTimeSteps(num_steps),
        },
    )

    for i in range(len(grid_file_paths)):
        ds, icon4py_driver = driver.run_driver(
            config=experiment_config,
            grid_manager=grid_managers[i],
            process_props=process_props,
            backend=backend,
        )
        simulated_tracer = ds.tracers.current.qv.ndarray

        assert (
            type(experiment_config.initial_condition.config)
            is linear_horizontal_tracer_advection.LinearHorizontalAdvectionConfig
        )
        reference_tracer = linear_horizontal_tracer_advection.construct_reference_tracer(
            config=experiment_config.initial_condition.config,
            grid=grid_managers[i].grid,
            static_fields=icon4py_driver.static_field_factories,
            integration_time=num_steps * experiment_config.driver.dtime.total_seconds(),
            num_levels=experiment_config.vertical_grid.num_levels,
        )

        current_error_l1, current_error_linf = _compute_relative_errors(
            simulated_tracer, reference_tracer
        )
        error_l1.append(current_error_l1)
        error_linf.append(current_error_linf)
        mean_edge_length.append(
            icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
            )
        )

    _check_convergence(
        l1_acceptable_range=l1_acceptable_range,
        linf_acceptable_range=linf_acceptable_range,
        error_l1=error_l1,
        error_linf=error_linf,
        grid_spacing=mean_edge_length,
    )


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, num_levels, l1_acceptable_range, linf_acceptable_range",
    [
        (
            "linear_vertical_advection_gaussian",
            [400 * 2**i for i in range(3)],
            [_THIRD_ORDER - _TOL, _THIRD_ORDER + _TOL],
            [_THIRD_ORDER - _TOL, _THIRD_ORDER + _TOL],
        ),
        (
            "linear_vertical_advection_box",
            [400 * 2**i for i in range(3)],
            [_DEGRADED_FIRST_ORDER - _DEGRADED_TOL, _DEGRADED_FIRST_ORDER + _DEGRADED_TOL],
            [_ZERO_ORDER - _TOL, _ZERO_ORDER + _TOL],
        ),
    ],
)
def test_vertical_tracer_advection_convergence(
    *,
    experiment_case: str,
    num_levels: tuple[int, ...],
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_path = grid_utils._download_grid_file(_VERTICAL_CONVERGENCE_GRID)
    error_l1: list[float] = []
    error_linf: list[float] = []
    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    experiment_config = config_io.read_yaml_str(
        config_path.read_text(), driver_config.ExperimentConfig
    ).with_overrides(driver={"output_path": tmp_path / "ci_driver_output"})

    assert (
        type(experiment_config.initial_condition.config)
        is linear_vertical_tracer_advection.LinearVerticalAdvectionConfig
    )
    w_max = linear_vertical_tracer_advection.compute_max_velocity(
        velocity_field=experiment_config.initial_condition.config.velocity_field,
        model_top_height=experiment_config.vertical_grid.model_top_height,
    )
    match experiment_config.driver.end_of_simulation:
        case time.RelativeTime() as relative:
            integration_time = relative.total_seconds()
        case time.AbsoluteTime() as absolute:
            integration_time = (
                absolute - experiment_config.driver.start_of_simulation
            ).total_seconds()
        case _:
            raise ValueError(
                f"end_of_simulation {experiment_config.driver.end_of_simulation} must be specified as a RelativeTime or AbsoluteTime for this test"
            )
    dtime = min(
        experiment_config.initial_condition.config.cfl_number
        * experiment_config.vertical_grid.model_top_height
        / num_levels[-1]
        / w_max,
        integration_time,
    )
    num_steps = int(integration_time / dtime)

    for num_lev in num_levels:
        experiment_config_local = experiment_config.with_overrides(
            driver={
                "dtime": time.RelativeTime(seconds=dtime),
                "end_of_simulation": time.NumTimeSteps(num_steps),
            },
            vertical_grid={"num_levels": num_lev},
        )

        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=grid_path,
            vertical_grid_config=experiment_config_local.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )

        ds, icon4py_driver = driver.run_driver(
            config=experiment_config_local,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=backend,
        )
        simulated_tracer = ds.tracers.current.qv.ndarray

        assert (
            type(experiment_config_local.initial_condition.config)
            is linear_vertical_tracer_advection.LinearVerticalAdvectionConfig
        )
        reference_tracer = linear_vertical_tracer_advection.construct_reference_tracer(
            config=experiment_config_local.initial_condition.config,
            metrics=icon4py_driver.static_field_factories.metrics,
            vertical_config=experiment_config_local.vertical_grid,
            integration_time=num_steps * experiment_config_local.driver.dtime.total_seconds(),
        )

        current_error_l1, current_error_linf = _compute_relative_errors(
            simulated_tracer, reference_tracer
        )
        error_l1.append(current_error_l1)
        error_linf.append(current_error_linf)

    mean_thickness = experiment_config.vertical_grid.model_top_height / np.array(
        num_levels, dtype=float
    )

    _check_convergence(
        l1_acceptable_range=l1_acceptable_range,
        linf_acceptable_range=linf_acceptable_range,
        error_l1=error_l1,
        error_linf=error_linf,
        grid_spacing=mean_thickness,
    )
