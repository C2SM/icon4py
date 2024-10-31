# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import pytest

from icon4py.model.atmosphere.advection import advection
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils, torus_helpers
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from scipy.stats import linregress

from .utils import (
    InitialConditions,
    VelocityField,
    compute_relative_errors,
    construct_config,
    construct_diagnostic_exit_state,
    construct_diagnostic_init_state,
    construct_idealized_diagnostic_state,
    construct_idealized_prep_adv,
    construct_idealized_tracer,
    construct_idealized_tracer_reference,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    construct_test_config,
    get_idealized_velocity_max,
    get_torus_dimensions,
    log_serialized,
    prepare_torus_quadrature,
    verify_advection_fields,
)


# flake8: noqa
log = logging.getLogger(__name__)

# note about ntracer: The first tracer is always dry air which is not advected. Thus, originally
# ntracer=2 is the first tracer in transport_nml, ntracer=3 the second, and so on.
# Here though, ntracer=1 corresponds to the first tracer in transport_nml.

# ntracer legend for the serialization data used here in test_advection:
# ------------------------------------
# ntracer          |  1, 2, 3, 4, 5 |
# ------------------------------------
# ivadv_tracer     |  3, 0, 0, 2, 3 |
# itype_hlimit     |  3, 3, 4, 0, 0 |
# itype_vlimit     |  1, 0, 0, 1, 2 |
# ihadv_tracer     | 52, 2, 2, 0, 0 |
# ------------------------------------


@pytest.mark.datatest
@pytest.mark.parametrize(
    "date, even_timestep, ntracer, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type, vertical_advection_limiter",
    [
        (
            "2021-06-20T12:00:10.000",
            False,
            2,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            2,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
    ],
)
def est_advection_run_single_step(
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    least_squares_savepoint,
    advection_init_savepoint,
    advection_exit_savepoint,
    data_provider,
    data_provider_advection,
    backend,
    even_timestep,
    ntracer,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
):
    config = construct_config(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    least_squares_state = construct_least_squares_state(least_squares_savepoint)
    metric_state = construct_metric_state(icon_grid)
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()

    advection_granule = advection.convert_config_to_advection(
        config=config,
        grid=icon_grid,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        even_timestep=even_timestep,
        backend=backend,
    )

    diagnostic_state = construct_diagnostic_init_state(icon_grid, advection_init_savepoint, ntracer)
    prep_adv = construct_prep_adv(icon_grid, advection_init_savepoint)
    p_tracer_now = advection_init_savepoint.tracer(ntracer)
    p_tracer_new = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid)
    dtime = advection_init_savepoint.get_metadata("dtime").get("dtime")

    log_serialized(diagnostic_state, prep_adv, p_tracer_now, dtime)

    advection_granule.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=dtime,
    )

    diagnostic_state_ref = construct_diagnostic_exit_state(
        icon_grid, advection_exit_savepoint, ntracer
    )
    p_tracer_new_ref = advection_exit_savepoint.tracer(ntracer)

    verify_advection_fields(
        grid=icon_grid,
        diagnostic_state=diagnostic_state,
        diagnostic_state_ref=diagnostic_state_ref,
        p_tracer_new=p_tracer_new,
        p_tracer_new_ref=p_tracer_new_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiments, initial_conditions, velocity_field, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type,"
    "vertical_advection_limiter, cfl_number, time_end, order_range, use_exact_solution, use_high_order_quadrature, enable_plots",
    [
        (
            [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
            InitialConditions.GAUSSIAN_2D,
            VelocityField.CONSTANT,
            advection.HorizontalAdvectionType.UPWIND_1ST_ORDER,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            1.0,
            1e-2,
            [0.9, 1.1],
            True,
        ),
        (
            [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
            InitialConditions.GAUSSIAN_2D,
            VelocityField.CONSTANT,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            1.0,
            1e-2,
            [1.8, 2.2],
            True,
        ),
        (
            [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
            InitialConditions.GAUSSIAN_2D,
            VelocityField.CONSTANT,
            advection.HorizontalAdvectionType.BROKEN_LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            1.0,
            1e-2,
            [0.9, 1.1],
            True,
        ),
        # the next two tests work but take 10 mins each
        # (
        #    [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
        #    InitialConditions.GAUSSIAN_2D_OFFCENTER,
        #    VelocityField.VORTEX_2D,
        #    advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
        #    advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        #    advection.VerticalAdvectionType.NO_ADVECTION,
        #    advection.VerticalAdvectionLimiter.NO_LIMITER,
        #    1.0,
        #    2e0,
        #    [2.0, 2.4],
        #    True,
        # ),
        # (
        #     [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
        #     InitialConditions.GAUSSIAN_2D_OFFCENTER,
        #     VelocityField.VORTEX_2D,
        #     advection.HorizontalAdvectionType.BROKEN_LINEAR_2ND_ORDER,
        #     advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        #     advection.VerticalAdvectionType.NO_ADVECTION,
        #     advection.VerticalAdvectionLimiter.NO_LIMITER,
        #     1.0,
        #     2e0,
        #     [0.3, 1.0],
        #     True,
        # ),
        # the next two tests work and take 90 secs each
        # (
        #    [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
        #    InitialConditions.GAUSSIAN_2D_OFFCENTER,
        #    VelocityField.VORTEX_2D,
        #    advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
        #    advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        #    advection.VerticalAdvectionType.NO_ADVECTION,
        #    advection.VerticalAdvectionLimiter.NO_LIMITER,
        #    1.0,
        #    2e-1,
        #    [2.0, 3.0],
        #    True,
        # ),
        # (
        #     [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2],
        #     InitialConditions.GAUSSIAN_2D_OFFCENTER,
        #     VelocityField.VORTEX_2D,
        #     advection.HorizontalAdvectionType.BROKEN_LINEAR_2ND_ORDER,
        #     advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
        #     advection.VerticalAdvectionType.NO_ADVECTION,
        #     advection.VerticalAdvectionLimiter.NO_LIMITER,
        #     1.0,
        #     2e-1,
        #     [0.8, 1.0],
        #     True,
        # ),
        (
            [
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1,
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2,
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_3,
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_4,
            ],
            InitialConditions.GAUSSIAN_2D,
            VelocityField.CONSTANT,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            1.0,
            1e-2,
            [1.8, 2.2],
            False,
            True,
            True,
        ),
    ]
    if 0
    else [
        (
            [
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1,
                dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2,
                # dt_utils.TORUS_CONVERGENCE_EXPERIMENT_3,
                # dt_utils.TORUS_CONVERGENCE_EXPERIMENT_4,
                # dt_utils.TORUS_CONVERGENCE_EXPERIMENT_5,
            ],
            InitialConditions.CIRCLE_2D,
            VelocityField.INCREASING_2D,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            1.0,
            2e-2,
            [1.8, 2.2],
            False,
            True,
            True,
        ),
    ],
)
def test_horizontal_advection_convergence(
    processor_props,
    ranked_data_path,
    backend,
    experiments,
    initial_conditions,
    velocity_field,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
    cfl_number,
    time_end,
    order_range,
    use_exact_solution,
    use_high_order_quadrature,
    enable_plots,
):
    test_config = construct_test_config(
        initial_conditions=initial_conditions, velocity_field=velocity_field
    )
    config = construct_config(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )

    errors_l1 = []
    errors_linf = []
    lengths_min = []

    if not use_exact_solution:
        # obtain reference solution first
        experiment_ref = dt_utils.TORUS_CONVERGENCE_EXPERIMENT_5
        experiments.insert(0, experiment_ref)
    else:
        tracer_reference_high = None
        cell_center_x_high = None
        cell_center_y_high = None

    # run experiments sequentially
    for experiment in experiments:
        data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
        data_provider = dt_utils.create_icon_serial_data_provider(data_path, processor_props)

        root, level = dt_utils.get_global_grid_params(experiment)
        grid_id = dt_utils.get_grid_id_for_experiment(experiment)
        grid_savepoint = data_provider.from_savepoint_grid(grid_id, root, level)
        icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)

        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        least_squares_savepoint = data_provider.from_least_squares_savepoint(
            size=data_provider.grid_size
        )

        interpolation_state = construct_interpolation_state(interpolation_savepoint)
        least_squares_state = construct_least_squares_state(least_squares_savepoint)
        metric_state = construct_metric_state(icon_grid)
        edge_geometry = grid_savepoint.construct_edge_geometry()
        cell_geometry = grid_savepoint.construct_cell_geometry()

        horizontal_advection, _ = advection.convert_config_to_horizontal_vertical_advection(
            config=config,
            grid=icon_grid,
            interpolation_state=interpolation_state,
            least_squares_state=least_squares_state,
            metric_state=metric_state,
            edge_params=edge_geometry,
            cell_params=cell_geometry,
            backend=backend,
        )

        node_x = grid_savepoint.verts_vertex_x().asnumpy()
        node_y = grid_savepoint.verts_vertex_y().asnumpy()
        edges_center_x = grid_savepoint.edges_center_x().asnumpy()
        edges_center_y = grid_savepoint.edges_center_y().asnumpy()
        cell_center_x = grid_savepoint.cell_center_x().asnumpy()
        cell_center_y = grid_savepoint.cell_center_y().asnumpy()
        length_min = edge_geometry.primal_edge_lengths.asnumpy().min()
        x_center, y_center, x_range, y_range = get_torus_dimensions(experiment)

        weights, nodes = prepare_torus_quadrature(
            icon_grid,
            node_x,
            node_y,
            cell_center_x,
            cell_center_y,
            length_min,
            use_high_order_quadrature,
        )

        diagnostic_state = construct_idealized_diagnostic_state(icon_grid)
        p_tracer_now = construct_idealized_tracer(
            test_config,
            icon_grid,
            x_center,
            y_center,
            x_range,
            y_range,
            weights,
            nodes,
        )
        p_tracer_new = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid)
        log.debug(f"length_min: {length_min}")

        if enable_plots:
            name = horizontal_advection_type.name
            k = 0
            torus_helpers.plot_torus_plane(
                icon_grid,
                node_x,
                node_y,
                p_tracer_now.asnumpy()[:, k],
                2 * length_min,
                out_file=f"{experiment}_{name}_start.pdf",
            )

        # time loop
        time = 0.0
        while time != time_end:
            # calculate time step based on desired CFL number for triangles
            vel_max = get_idealized_velocity_max(test_config, x_range, y_range, time_end)
            assert vel_max > 0.0
            dtime = min(cfl_number * length_min / vel_max / (4.0 * (3.0**0.5)), time_end - time)
            log.debug(f"dtime: {dtime}")

            prep_adv = construct_idealized_prep_adv(
                test_config,
                icon_grid,
                edge_geometry,
                edges_center_x,
                edges_center_y,
                x_range,
                y_range,
                time,
                dtime,
                time_end,
            )

            horizontal_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )
            time += dtime

            p_tracer_now, p_tracer_new = p_tracer_new, p_tracer_now
            # note: no need to swap airmass fields here because they are equal

        if enable_plots:
            tracer_end_k = p_tracer_now.asnumpy()[:, k]
            torus_helpers.plot_torus_plane(
                icon_grid,
                node_x,
                node_y,
                tracer_end_k,
                2 * length_min,
                out_file=f"{experiment}_{name}_end.pdf",
            )

        if not use_exact_solution and experiment == experiment_ref:
            tracer_reference_high = p_tracer_now
            cell_center_x_high = cell_center_x
            cell_center_y_high = cell_center_y
            continue

        # get reference solution
        tracer_reference = construct_idealized_tracer_reference(
            test_config,
            icon_grid,
            x_center,
            y_center,
            x_range,
            y_range,
            edges_center_x,
            edges_center_y,
            node_x,
            node_y,
            time,
            time_end,
            weights,
            nodes,
            tracer_reference_high,
            cell_center_x_high,
            cell_center_y_high,
        )

        if enable_plots:
            tracer_reference_k = tracer_reference.asnumpy()[:, k]
            torus_helpers.plot_torus_plane(
                icon_grid,
                node_x,
                node_y,
                tracer_reference_k,
                2 * length_min,
                out_file=f"{experiment}_{name}_reference.pdf",
            )
            torus_helpers.plot_torus_plane(
                icon_grid,
                node_x,
                node_y,
                tracer_end_k - tracer_reference_k,
                2 * length_min,
                out_file=f"{experiment}_{name}_diff.pdf",
            )

        # compute errors
        error_l1, error_linf = compute_relative_errors(p_tracer_now, tracer_reference)
        assert math.isfinite(error_l1) and math.isfinite(error_linf)
        assert error_l1 >= 0.0 and error_linf >= 0.0
        errors_l1.append(error_l1)
        errors_linf.append(error_linf)
        lengths_min.append(length_min)

    log.debug(f"errors_l1: {errors_l1}")
    log.debug(f"errors_linf: {errors_linf}")
    log.debug(f"lengths_min: {lengths_min}")
    n = len(lengths_min)
    assert n == len(errors_l1) == len(errors_linf)

    if enable_plots:
        theoretical_orders = [1.0, 2.0]
        linestyles = ["--", "-."]
        ref = "" if use_exact_solution else "_ref"
        torus_helpers.plot_convergence(
            lengths_min,
            errors_l1,
            name=name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"{experiment[:-2]}_{name}_l1{ref}.pdf",
        )
        torus_helpers.plot_convergence(
            lengths_min,
            errors_linf,
            name=name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"{experiment[:-2]}_{name}_linf{ref}.pdf",
        )

    # check observed rate of convergence
    if 1:
        linreg_l1 = linregress(xp.log(lengths_min), xp.log(errors_l1))
        linreg_linf = linregress(xp.log(lengths_min), xp.log(errors_linf))
        p_l1 = linreg_l1.slope
        p_linf = linreg_linf.slope
        log.debug(f"p_l1: {p_l1}")
        log.debug(f"p_linf: {p_linf}")
        assert order_range[0] <= p_l1 <= order_range[1]
        assert order_range[0] <= p_linf <= order_range[1]
    else:
        for i in range(n - 1):
            # grid refinement factor
            r = lengths_min[i] / lengths_min[i + 1]

            # observed rate of convergence
            p_l1 = math.log(errors_l1[i] / errors_l1[i + 1]) / math.log(r)
            p_linf = math.log(errors_linf[i] / errors_linf[i + 1]) / math.log(r)
            log.debug(f"p_l1: {p_l1}")
            log.debug(f"p_linf: {p_linf}")

            # tolerance check only for the highest resolution
            if i == n - 2:
                assert order_range[0] <= p_l1 <= order_range[1]
                assert order_range[0] <= p_linf <= order_range[1]


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiments, initial_conditions, use_high_order_quadrature, enable_plots",
    [
        (
            [dt_utils.TORUS_CONVERGENCE_EXPERIMENT_5, dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1],
            InitialConditions.GAUSSIAN_2D,
            True,
            True,
        ),
    ],
)
def est_torus_interpolation(
    processor_props,
    ranked_data_path,
    experiments,
    initial_conditions,
    use_high_order_quadrature,
    enable_plots,
):
    test_config = construct_test_config(initial_conditions=initial_conditions)

    fields = []
    grids = []
    nodes_x = []
    nodes_y = []
    cell_centers_x = []
    cell_centers_y = []
    lengths_min = []

    # run experiments sequentially
    for experiment in experiments:
        data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
        data_provider = dt_utils.create_icon_serial_data_provider(data_path, processor_props)

        root, level = dt_utils.get_global_grid_params(experiment)
        grid_id = dt_utils.get_grid_id_for_experiment(experiment)
        grid_savepoint = data_provider.from_savepoint_grid(grid_id, root, level)
        icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)

        edge_geometry = grid_savepoint.construct_edge_geometry()
        cell_geometry = grid_savepoint.construct_cell_geometry()

        node_x = grid_savepoint.verts_vertex_x().asnumpy()
        node_y = grid_savepoint.verts_vertex_y().asnumpy()
        cell_center_x = grid_savepoint.cell_center_x().asnumpy()
        cell_center_y = grid_savepoint.cell_center_y().asnumpy()
        length_min = edge_geometry.primal_edge_lengths.asnumpy().min()
        x_center, y_center, x_range, y_range = get_torus_dimensions(experiment)

        weights, nodes = prepare_torus_quadrature(
            icon_grid,
            node_x,
            node_y,
            cell_center_x,
            cell_center_y,
            length_min,
            use_high_order_quadrature,
        )

        p_tracer_now = construct_idealized_tracer(
            test_config,
            icon_grid,
            x_center,
            y_center,
            x_range,
            y_range,
            weights,
            nodes,
        )

        k = 0
        fields.append(p_tracer_now.asnumpy()[:, k])
        grids.append(icon_grid)
        nodes_x.append(node_x)
        nodes_y.append(node_y)
        cell_centers_x.append(cell_center_x)
        cell_centers_y.append(cell_center_y)
        lengths_min.append(length_min)

    if enable_plots:
        for i in range(len(fields)):
            torus_helpers.plot_torus_plane(
                grids[i],
                nodes_x[i],
                nodes_y[i],
                fields[i],
                2 * lengths_min[i],
                out_file=f"original_{i}.pdf",
            )

    field_low = torus_helpers.interpolate_torus_plane(
        cell_centers_x[0],
        cell_centers_y[0],
        fields[0],
        nodes_x[1],
        nodes_y[1],
        weights,
        nodes,
    )

    if enable_plots:
        torus_helpers.plot_torus_plane(
            grids[1],
            nodes_x[1],
            nodes_y[1],
            field_low,
            2 * lengths_min[1],
            out_file=f"interpolated.pdf",
        )
        torus_helpers.plot_torus_plane(
            grids[1],
            nodes_x[1],
            nodes_y[1],
            field_low - fields[1],
            2 * lengths_min[1],
            out_file=f"diff.pdf",
        )

    # compute errors
    error_l1, error_linf = compute_relative_errors(field_low, fields[1])
    assert math.isfinite(error_l1) and math.isfinite(error_linf)
    log.debug(f"error_l1: {error_l1}")
    log.debug(f"error_linf: {error_linf}")

    # tolerance check
    assert 0.0 <= error_l1 <= 1e-4
    assert 0.0 <= error_linf <= 1e-4


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, initial_conditions, enable_plots",
    [(dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1, InitialConditions.CIRCLE_2D, True)],
)
def est_torus_quadrature(
    processor_props,
    ranked_data_path,
    experiment,
    initial_conditions,
    enable_plots,
):
    test_config = construct_test_config(initial_conditions=initial_conditions)

    data_path = dt_utils.get_datapath_for_experiment(ranked_data_path, experiment)
    data_provider = dt_utils.create_icon_serial_data_provider(data_path, processor_props)

    root, level = dt_utils.get_global_grid_params(experiment)
    grid_id = dt_utils.get_grid_id_for_experiment(experiment)
    grid_savepoint = data_provider.from_savepoint_grid(grid_id, root, level)
    icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)

    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()

    node_x = grid_savepoint.verts_vertex_x().asnumpy()
    node_y = grid_savepoint.verts_vertex_y().asnumpy()
    cell_center_x = grid_savepoint.cell_center_x().asnumpy()
    cell_center_y = grid_savepoint.cell_center_y().asnumpy()
    length_min = edge_geometry.primal_edge_lengths.asnumpy().min()
    x_center, y_center, x_range, y_range = get_torus_dimensions(experiment)

    weights, nodes = prepare_torus_quadrature(
        icon_grid,
        node_x,
        node_y,
        cell_center_x,
        cell_center_y,
        length_min,
        use_high_order_quadrature=True,
    )

    # quadrature rules with negative weights should not be used
    assert xp.all(weights >= 0.0)

    ics = construct_idealized_tracer(
        test_config,
        icon_grid,
        x_center,
        y_center,
        x_range,
        y_range,
        weights,
        nodes,
    )

    k = 0
    ics_k = ics.asnumpy()[:, k]

    if enable_plots:
        torus_helpers.plot_torus_plane(
            icon_grid, node_x, node_y, ics_k, 2 * length_min, out_file="ics.pdf"
        )
        torus_helpers.plot_torus_plane_quad(
            icon_grid,
            node_x,
            node_y,
            ics_k,
            2 * length_min,
            weights,
            nodes,
            out_file="ics_quad.pdf",
        )

    tol = 1e-16
    assert xp.all(0.0 - tol <= ics_k)
    assert xp.all(ics_k <= 1.0 + tol)
