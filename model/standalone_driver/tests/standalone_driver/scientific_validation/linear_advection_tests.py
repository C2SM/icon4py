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
from icon4py.model.common.grid import geometry_attributes as geometry_meta, gridfile
from icon4py.model.common.initial_condition.analytical import linear_advection
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import config as test_config, definitions as test_defs, plot_utils

from ..fixtures import *  # noqa: F403


_FIRST_ORDER = 1.0
_FIRST_ORDER_TOL = 0.5
_ZERO_ORDER = 0.0
_ZERO_ORDER_TOL = 0.5


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
    "experiment_case, tracer_profile, grid_description, l1_acceptable_range, linf_acceptable_range, enable_plot",
    [
        (
            "linear_adv",
            linear_advection.TracerProfile.GAUSSIAN_2D,
            (
                test_defs.Grids.TORUS_1000X1000_100M,
                test_defs.Grids.TORUS_1000X1000_50M,
                test_defs.Grids.TORUS_1000X1000_25M,
            ),
            [_FIRST_ORDER - _FIRST_ORDER_TOL, _FIRST_ORDER + _FIRST_ORDER_TOL],
            [_FIRST_ORDER - _FIRST_ORDER_TOL, _FIRST_ORDER + _FIRST_ORDER_TOL],
            True,
        ),
        (
            "linear_adv",
            linear_advection.TracerProfile.CIRCLE_2D,
            (
                test_defs.Grids.TORUS_1000X1000_100M,
                test_defs.Grids.TORUS_1000X1000_50M,
                test_defs.Grids.TORUS_1000X1000_25M,
            ),
            [_FIRST_ORDER - _FIRST_ORDER_TOL, _FIRST_ORDER + _FIRST_ORDER_TOL],
            [_ZERO_ORDER - _ZERO_ORDER_TOL, _ZERO_ORDER + _ZERO_ORDER_TOL],
            True,
        ),
    ],
)
def test_horizontal_advection_convergence(
    *,
    experiment_case: str,
    tracer_profile: linear_advection.TracerProfile,
    grid_description: tuple[test_defs.GridDescription, ...],
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    enable_plot: bool,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    import pathlib

    base_path = pathlib.Path("/capstor/scratch/cscs/cong/icon4py/testdata/grids/")
    grid_file_paths = []
    for i in range(len(grid_description)):
        grid_file_paths.append(
            base_path.joinpath(grid_description[i].name, f"{grid_description[i].name}.nc")
        )
        print(grid_file_paths[i])
    # grid_file_paths = (grid_utils._download_grid_file(grid) for grid in grid_description)
    error_l1: list[float] = []
    error_linf: list[float] = []
    mean_edge_length: list[float] = []

    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    for grid_path in grid_file_paths:
        experiment_config = config_io.read_yaml_str(
            config_path.read_text(), driver_config.ExperimentConfig
        ).with_overrides(
            driver={"output_path": tmp_path / "ci_driver_output"},
        )

        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=grid_path,
            vertical_grid_config=experiment_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )

        domain_length = grid_manager.grid.grid_params.domain_length
        domain_height = grid_manager.grid.grid_params.domain_height
        assert (
            type(experiment_config.initial_condition.config)
            is linear_advection.LinearAdvectionConfig
        )
        vel_max = linear_advection.compute_max_velocity(
            velocity_field=experiment_config.initial_condition.config.velocity_field,
            domain_length=domain_length if domain_length is not None else 0.0,
            domain_height=domain_height if domain_height is not None else 0.0,
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
            * grid_manager.geometry_fields[gridfile.GeometryName.EDGE_LENGTH].asnumpy().mean()
            / vel_max,
            integration_time,
        )
        print(
            "debugging num of steps: ",
            integration_time / dtime,
            int(integration_time / dtime),
            dtime,
            integration_time,
            int(integration_time / dtime) * dtime,
        )
        experiment_config = experiment_config.with_overrides(
            driver={"dtime": time.RelativeTime(seconds=dtime)},
            initial_condition={
                "config": {
                    "tracer_profile": tracer_profile,
                }
            },
        )

        ds, icon4py_driver = standalone_driver.run_driver(
            config=experiment_config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=backend,
        )
        simulated_tracer = ds.tracers.current.qv.ndarray

        assert (
            type(experiment_config.initial_condition.config)
            is linear_advection.LinearAdvectionConfig
        )
        reference_tracer = linear_advection.construct_reference_tracer(
            velocity_field=experiment_config.initial_condition.config.velocity_field,
            tracer_profile=experiment_config.initial_condition.config.tracer_profile,
            grid=grid_manager.grid,
            static_fields=icon4py_driver.static_field_factories,
            integration_time=integration_time,
            num_levels=experiment_config.vertical_grid.num_levels,
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
            grid_name = grid_path.stem
            assert (
                type(experiment_config.initial_condition.config)
                is linear_advection.LinearAdvectionConfig
            )
            initial_tracer = linear_advection.construct_reference_tracer(
                velocity_field=experiment_config.initial_condition.config.velocity_field,
                tracer_profile=experiment_config.initial_condition.config.tracer_profile,
                grid=grid_manager.grid,
                static_fields=icon4py_driver.static_field_factories,
                integration_time=0.0,
                num_levels=experiment_config.vertical_grid.num_levels,
            )
            plot_utils.plot_torus_plane(
                c2v_connectivity=grid_manager.grid.connectivities["C2V"].asnumpy(),
                node_x=vertex_x,
                node_y=vertex_y,
                values=initial_tracer[:, 0],
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"grid_{grid_name}_prof_{tracer_profile}_adv_{adv_type_name}_initial.pdf",
            )
            plot_utils.plot_torus_plane(
                c2v_connectivity=grid_manager.grid.connectivities["C2V"].asnumpy(),
                node_x=vertex_x,
                node_y=vertex_y,
                values=reference_tracer[:, 0],
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"grid_{grid_name}_prof_{tracer_profile}_adv_{adv_type_name}_reference.pdf",
            )
            plot_utils.plot_torus_plane(
                c2v_connectivity=grid_manager.grid.connectivities["C2V"].asnumpy(),
                node_x=vertex_x,
                node_y=vertex_y,
                values=simulated_tracer[:, 0] - reference_tracer[:, 0],
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"grid_{grid_name}_prof_{tracer_profile}_adv_{adv_type_name}_diff.pdf",
            )
            plot_utils.plot_torus_plane(
                c2v_connectivity=grid_manager.grid.connectivities["C2V"].asnumpy(),
                node_x=vertex_x,
                node_y=vertex_y,
                values=simulated_tracer[:, 0],
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"grid_{grid_name}_prof_{tracer_profile}_adv_{adv_type_name}_sim.pdf",
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
            out_file=f"convergence_prof_{tracer_profile}_adv_{adv_type_name}_l1.pdf",
        )
        plot_utils.plot_convergence(
            x=mean_edge_length,
            y=error_linf,
            label_name=adv_type_name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"convergence_prof_{tracer_profile}_adv_{adv_type_name}_linf.pdf",
        )

    linreg_l1 = linregress(np.log(mean_edge_length), np.log(error_l1))
    p_l1 = linreg_l1.slope
    assert l1_acceptable_range[0] <= p_l1 <= l1_acceptable_range[1]
    linreg_linf = linregress(np.log(mean_edge_length), np.log(error_linf))
    p_linf = linreg_linf.slope
    assert linf_acceptable_range[0] <= p_linf <= linf_acceptable_range[1]
