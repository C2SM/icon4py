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
    linear_horizontal_advection,
    linear_vertical_advection,
)
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import config as test_config, plot_utils, torus_grid_generator

from ..fixtures import *  # noqa: F403


_FIRST_ORDER: Final = 1.0
_SECOND_ORDER: Final = 2.0
_ORDER_TOL: Final = 0.5

# 12 rows by 10 columns of 100 m edges: domain_length = 1000 m and
# domain_height = 12 * 100 * sqrt(3)/2 = 1039.2304845413264 m. Refining multiplies the row and
# column counts and divides the edge length, so every level of the family discretises the same
# continuous problem. Only a power of two keeps both extents bit-identical, a factor of 3 or 5
# perturbs them in the last ulp. The downloaded TORUS_1000X1000_* grids did not have that
# property at all: their domain_height was fitted to the requested extent and therefore varied
# with the resolution.
_BASE_TORUS_ROWS: Final = 12
_BASE_TORUS_COLS: Final = 10
_BASE_TORUS_EDGE_LENGTH: Final = 100.0
_REFINEMENT_FACTORS: Final = tuple(2**exponent for exponent in range(4))


def _order_range(order: float) -> tuple[float, float]:
    return (order - _ORDER_TOL, order + _ORDER_TOL)


def _generate_torus_grid(*, refinement_factor: int, out_dir: pathlib.Path) -> pathlib.Path:
    """Write the base torus grid refined by 'refinement_factor', see '_REFINEMENT_FACTORS'."""
    n_rows = _BASE_TORUS_ROWS * refinement_factor
    n_cols = _BASE_TORUS_COLS * refinement_factor
    edge_length = _BASE_TORUS_EDGE_LENGTH / refinement_factor
    return torus_grid_generator.generate_torus_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        edge_length=edge_length,
        # the stem is the label of the per-grid plots
        out_file=out_dir / f"torus_{n_rows}x{n_cols}_res{edge_length:g}m.nc",
    )


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


# The scheme of 'experiment_configs/linear_horizontal_advection.yaml' is second order
# ('linear_2nd_order'), and the one-dimensional profile attains it in both norms on the generated
# grid family. The two-dimensional profile does not, and the reason is the initial condition
# rather than the scheme: '_construct_idealized_tracer' evaluates the Gaussian on the minimum
# image separation, so the profile is only C0 on the torus and has a slope kink on the half
# domain line. GAUSSIAN_2D is wide enough to still be at 6.6% of its peak there, against 1.5e-4
# for GAUSSIAN_1D_X, and the kink radiates a dispersive wake that stalls the maximum norm and
# drags L1 down to the measured 1.7. Summing the Gaussian over its periodic images instead
# restores second order in both norms; until that lands, the L1 band below is thin by ~0.2 on
# its lower side. The limiter is not involved: at CFL 0.2 its multiplicative factor is 1.
@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, tracer_profile, l1_acceptable_range, linf_acceptable_range, enable_plot",
    [
        (
            "linear_horizontal_advection",
            linear_horizontal_advection.TracerProfile.GAUSSIAN_2D,
            _order_range(_SECOND_ORDER),
            _order_range(_FIRST_ORDER),
            True,
        ),
        (
            "linear_horizontal_advection",
            linear_horizontal_advection.TracerProfile.GAUSSIAN_1D_X,
            _order_range(_SECOND_ORDER),
            _order_range(_SECOND_ORDER),
            True,
        ),
    ],
)
def test_horizontal_advection_convergence(
    *,
    experiment_case: str,
    tracer_profile: linear_horizontal_advection.TracerProfile,
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    enable_plot: bool,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    # Generate the grid family instead of downloading it: only a generated family is
    # guaranteed to keep the domain extents fixed while the resolution changes, which is what
    # the convergence rate is measured against.
    grid_file_paths = [
        _generate_torus_grid(refinement_factor=factor, out_dir=tmp_path)
        for factor in _REFINEMENT_FACTORS
    ]
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
            is linear_horizontal_advection.LinearHorizontalAdvectionConfig
        )
        vel_max = linear_horizontal_advection.compute_max_velocity(
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

        # the driver takes floor(integration_time / dtime) steps, so the model stops short of
        # the nominal integration time by up to one dtime. As dtime is proportional to the mesh
        # spacing, comparing against the reference at the nominal time would add an O(h) phase
        # error to the measured error and destroy the convergence rate.
        simulated_time = (
            icon4py_driver.model_time_variables.n_time_steps
            * icon4py_driver.model_time_variables.dtime_in_seconds
        )

        assert (
            type(experiment_config.initial_condition.config)
            is linear_horizontal_advection.LinearHorizontalAdvectionConfig
        )
        reference_tracer = linear_horizontal_advection.construct_reference_tracer(
            config=experiment_config.initial_condition.config,
            grid=grid_manager.grid,
            static_fields=icon4py_driver.static_field_factories,
            integration_time=simulated_time,
            num_levels=experiment_config.vertical_grid.num_levels,
        )

        if enable_plot:
            vertex_x = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.VERTEX_X
            ).asnumpy()
            vertex_y = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.VERTEX_Y
            ).asnumpy()
            edge_x = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.EDGE_CENTER_X
            ).asnumpy()
            edge_y = icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.EDGE_CENTER_Y
            ).asnumpy()
            assert experiment_config.tracer_advection is not None
            adv_type_name = (
                experiment_config.tracer_advection.horizontal_advection_type.name.lower()
            )
            grid_name = grid_path.stem
            assert (
                type(experiment_config.initial_condition.config)
                is linear_horizontal_advection.LinearHorizontalAdvectionConfig
            )
            initial_tracer = linear_horizontal_advection.construct_reference_tracer(
                config=experiment_config.initial_condition.config,
                grid=grid_manager.grid,
                static_fields=icon4py_driver.static_field_factories,
                integration_time=0.0,
                num_levels=experiment_config.vertical_grid.num_levels,
            )
            assert ds.prep_tracer_advection_prognostic is not None
            plot_utils.plot_torus_scatter(
                node_x=edge_x,
                node_y=edge_y,
                values=ds.prep_tracer_advection_prognostic.vn_traj.asnumpy()[:, 0],
                c2v_connectivity=grid_manager.grid.connectivities["C2V"].asnumpy(),
                vertex_x=vertex_x,
                vertex_y=vertex_y,
                length_max=2
                * icon4py_driver.static_field_factories.geometry.get(
                    geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
                ),
                out_file=f"grid_{grid_name}_prof_{tracer_profile}_adv_{adv_type_name}_vn.pdf",
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


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_case, tracer_profile, num_levels, l1_acceptable_range, linf_acceptable_range, enable_plot",
    [
        (
            "linear_vertical_advection",
            linear_vertical_advection.TracerProfile.GAUSSIAN_1D,
            (
                100,
                200,
                400,
            ),
            _order_range(_FIRST_ORDER),
            _order_range(_FIRST_ORDER),
            True,
        ),
    ],
)
def test_vertical_advection_convergence(
    *,
    experiment_case: str,
    tracer_profile: linear_vertical_advection.TracerProfile,
    num_levels: tuple[int, ...],
    l1_acceptable_range: tuple[float, float],
    linf_acceptable_range: tuple[float, float],
    enable_plot: bool,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    # the horizontal mesh only carries the vertical profile here, so the coarsest member of the
    # convergence family is enough
    grid_path = _generate_torus_grid(refinement_factor=1, out_dir=tmp_path)
    error_l1: list[float] = []
    error_linf: list[float] = []
    mean_edge_length: list[float] = []

    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"

    for num_lev in num_levels:
        experiment_config = config_io.read_yaml_str(
            config_path.read_text(), driver_config.ExperimentConfig
        ).with_overrides(
            driver={"output_path": tmp_path / "ci_driver_output"},
            vertical_grid={"num_levels": num_lev},
        )

        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=grid_path,
            vertical_grid_config=experiment_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )

        assert (
            type(experiment_config.initial_condition.config)
            is linear_vertical_advection.LinearVerticalAdvectionConfig
        )
        # the time step is needed before the driver exists, so take the half level heights from
        # the vertical grid rather than from the metric fields. The topography is flat, so the
        # two agree; only the velocity fields that vary with height read the values at all.
        vertical_grid = driver_utils.create_vertical_grid(
            vertical_grid_config=experiment_config.vertical_grid, allocator=allocator
        )
        w_max = linear_vertical_advection.compute_max_velocity(
            velocity_field=experiment_config.initial_condition.config.velocity_field,
            z_ifc=vertical_grid.interface_physical_height.ndarray,
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
            / num_lev
            / w_max,
            integration_time,
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

        # see the comment in test_horizontal_advection_convergence: the reference must be
        # evaluated at the time the model actually reached, not the nominal one
        simulated_time = (
            icon4py_driver.model_time_variables.n_time_steps
            * icon4py_driver.model_time_variables.dtime_in_seconds
        )

        assert (
            type(experiment_config.initial_condition.config)
            is linear_vertical_advection.LinearVerticalAdvectionConfig
        )
        reference_tracer = linear_vertical_advection.construct_reference_tracer(
            velocity_field=experiment_config.initial_condition.config.velocity_field,
            tracer_profile=experiment_config.initial_condition.config.tracer_profile,
            metrics=icon4py_driver.static_field_factories.metrics,
            center_z=experiment_config.vertical_grid.model_top_height / 2.0,
            model_top_height=experiment_config.vertical_grid.model_top_height,
            integration_time=simulated_time,
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
            initial_tracer = linear_vertical_advection.construct_reference_tracer(
                velocity_field=experiment_config.initial_condition.config.velocity_field,
                tracer_profile=experiment_config.initial_condition.config.tracer_profile,
                metrics=icon4py_driver.static_field_factories.metrics,
                center_z=experiment_config.vertical_grid.model_top_height / 2.0,
                model_top_height=experiment_config.vertical_grid.model_top_height,
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
