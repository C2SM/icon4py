# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
from scipy.stats import linregress

from icon4py.model.common import model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from icon4py.model.common.initial_condition.analytical import linear_advection
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import (
    config as test_config,
    definitions as test_defs,
    grid_utils,
    plot_utils,
)

from ..fixtures import *  # noqa: F403


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


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, grid_description, l1_acceptable_range, linf_acceptable_range, enable_plot",
    [
        (
            "linear_adv",
            (
                test_defs.Grids.TORUS_1000X1000_100M,
                test_defs.Grids.TORUS_1000X1000_50M,
                test_defs.Grids.TORUS_1000X1000_25M,
            ),
            [2 - 3e-2, 2 + 3e-2],
            [2 - 4e-2, 2 + 4e-2],
            False,
        ),
    ],
)
def horizontal_advection_test(
    experiment_case: str,
    grid_description: tuple[test_defs.GridDescription, ...],
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    enable_plot: bool,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = (grid_utils._download_grid_file(grid) for grid in grid_description)
    error_l1: list[float] = []
    error_linf: list[float] = []
    mean_edge_length: list[float] = []

    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    for grid in grid_file_path:
        experiment_config = config_io.read_yaml_str(
            config_path.read_text(), driver_config.ExperimentConfig
        ).with_overrides(
            driver={"output_path": tmp_path / "ci_driver_output"},
        )

        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=grid,
            vertical_grid_config=experiment_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )

        domain_length = grid_manager.grid.grid_params.domain_length
        domain_height = grid_manager.grid.grid_params.domain_height
        vel_max = linear_advection.compute_max_velocity(
            velocity_field=experiment_config.initial_condition.velocity_field,
            domain_length=domain_length if domain_length is not None else 0.0,
            domain_height=domain_height if domain_height is not None else 0.0,
        )
        match type(experiment_config.driver.end_of_simulation):
            case time.RelativeTime():
                end_time = experiment_config.driver.end_of_simulation.total_seconds()
            case time.AbsoluteTime():
                end_time = (
                    experiment_config.driver.end_of_simulation
                    - experiment_config.driver.start_of_simulation
                ).total_seconds()
            case _:
                raise ValueError(
                    f"end_of_simulation {experiment_config.driver.end_of_simulation} must be specified as a RelativeTime or AbsoluteTime for this test"
                )
        dtime = min(
            experiment_config.initial_condition.cfl_number
            * grid_manager.grid.grid_params.domain_length
            / vel_max
            / (4.0 * (3.0**0.5)),
            experiment_config.driver.end_of_simulation.total_seconds(),
        )
        print("debugging num of steps: ", end_time / dtime, int(end_time / dtime), dtime, end_time)
        experiment_config.with_overrides(
            driver={"dtime": dtime},
        )

        ds, icon4py_driver = standalone_driver.run_driver(
            config=experiment_config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=backend,
        )
        simulated_tracer = ds.tracers.current.qv.ndarray

        # get reference solution
        reference_tracer = linear_advection.construct_reference_tracer(
            velocity_field=experiment_config.initial_condition.velocity_field,
            tracer_profile=experiment_config.initial_condition.tracer_profile,
            grid=grid_manager.grid,
            static_fields=grid_manager.static_fields,
            integration_time=end_time,
        )

        if enable_plot:
            vertex_x = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.VERTEX_X
            ).asnumpy()
            vertex_y = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.VERTEX_Y
            ).asnumpy()
            assert experiment_config.tracer_advection is not None
            adv_type_name = (
                experiment_config.tracer_advection.horizontal_advection_type.name.lower()
            )
            plot_utils.plot_torus_plane(
                grid=grid_manager.grid,
                node_x=vertex_x,
                node_y=vertex_y,
                values=np.asarray(reference_tracer[:, 0]),
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"experiment_{adv_type_name}_reference.pdf",
            )
            plot_utils.plot_torus_plane(
                grid=grid_manager.grid,
                node_x=vertex_x,
                node_y=vertex_y,
                values=np.asarray(simulated_tracer[:, 0] - reference_tracer[:, 0]),
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"experiment_{adv_type_name}_diff.pdf",
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

    if enable_plot:
        theoretical_orders = [1.0, 2.0]
        linestyles = ["--", "-."]
        assert experiment_config.tracer_advection is not None
        adv_type_name = experiment_config.tracer_advection.horizontal_advection_type.name.lower()
        plot_utils.plot_convergence(
            x=mean_edge_length,
            y=error_l1,
            label_name=adv_type_name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"experiment_{adv_type_name}_l1.pdf",
        )
        plot_utils.plot_convergence(
            x=mean_edge_length,
            y=error_linf,
            label_name=adv_type_name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"experiment_{adv_type_name}_linf.pdf",
        )

    linreg_l1 = linregress(np.log(mean_edge_length), np.log(error_l1))
    p_l1 = linreg_l1.slope
    assert l1_acceptable_range[0] <= p_l1 <= l1_acceptable_range[1]
    linreg_linf = linregress(np.log(mean_edge_length), np.log(error_linf))
    p_linf = linreg_linf.slope
    assert linf_acceptable_range[0] <= p_linf <= linf_acceptable_range[1]
