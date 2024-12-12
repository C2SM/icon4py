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

import numpy as np

from icon4py.model.atmosphere.advection import advection
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.test_utils import helpers, plot_utils
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
    construct_idealized_metric_state,
    construct_idealized_prep_adv,
    construct_idealized_tracer,
    construct_idealized_tracer_reference,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    construct_test_config,
    get_idealized_velocity_max,
    log_serialized,
    verify_advection_fields,
)

# flake8: noqa
log = logging.getLogger(__name__)


# ntracer legend for the serialization data used here in test_advection:
# ------------------------------------
# ntracer          |  1, 2, 3, 4, 5 |
# ------------------------------------
# ivadv_tracer     |  3, 0, 0, 2, 3 |
# itype_hlimit     |  3, 4, 3, 0, 0 |
# itype_vlimit     |  1, 0, 0, 2, 1 |
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
        (
            "2021-06-20T12:00:10.000",
            False,
            5,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            5,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
    ],
)
def test_advection_run_single_step(
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    least_squares_savepoint,
    metrics_savepoint,
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
    metric_state = construct_metric_state(icon_grid, metrics_savepoint)
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
        even_timestep=even_timestep,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "nums_levels, num_levels_ref, initial_conditions, velocity_field, horizontal_advection_type, horizontal_advection_limiter,"
    "vertical_advection_type, vertical_advection_limiter, cfl_number, time_end, l1_acceptable_range, linf_acceptable_range",
    [
        (  # test positive upwind
            [2**11, 2**12, 2**13, 2**14],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.UPWIND_1ST_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            0.5,
            1e-2,
            [1 - 2e-4, 1 + 2e-4],
            [1 - 4e-4, 1 + 4e-4],
        ),
        (  # test negative upwind
            [2**11, 2**12, 2**13, 2**14],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_NEGATIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.UPWIND_1ST_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            0.5,
            1e-2,
            [1 - 2e-4, 1 + 2e-4],
            [1 - 4e-4, 1 + 4e-4],
        ),
        (  # test third-order accuracy
            [2**8, 2**9, 2**10, 2**11],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            0.5,
            1e-2,
            [3 - 2e-2, 3 + 2e-2],
            [3 - 3e-2, 3 + 3e-2],
        ),
        (  # test using reference solution
            [2**8, 2**9, 2**10, 2**11],
            2**13,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            0.5,
            1e-2,
            [3 - 2e-2, 3 + 2e-2],
            [3 - 3e-2, 3 + 3e-2],
        ),
        (  # test positive large time steps
            [2**8, 2**9, 2**10, 2**11],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            4.5,
            1e-1,
            [3 - 2e-2, 3 + 2e-2],
            [3 - 2e-2, 3 + 2e-2],
        ),
        (  # test negative large time steps
            [2**8, 2**9, 2**10, 2**11],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_NEGATIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            4.5,
            1e-1,
            [3 - 2e-2, 3 + 2e-2],
            [3 - 2e-2, 3 + 2e-2],
        ),
        (  # test that the broken scheme is different
            [2**8, 2**9, 2**10, 2**11],
            None,
            InitialConditions.GAUSSIAN_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.BROKEN_PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
            0.5,
            1e-2,
            [1 - 5e-3, 1 + 5e-3],
            [1 - 3e-3, 1 + 3e-3],
        ),
        (  # test L1 convergence with discontinuous ICs
            [2**8, 2**9, 2**10, 2**11],
            None,
            InitialConditions.BOX_1D,
            VelocityField.CONSTANT_POSITIVE,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
            0.5,
            1e-2,
            [1 - 3e-1, 1 + 3e-1],
            None,
        ),
    ],
)
def test_vertical_advection_convergence(
    processor_props,
    ranked_data_path,
    backend,
    use_high_order_quadrature,
    enable_plots,
    nums_levels,
    num_levels_ref,
    initial_conditions,
    velocity_field,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
    cfl_number,
    time_end,
    l1_acceptable_range,
    linf_acceptable_range,
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

    use_exact_solution = num_levels_ref is None
    if use_exact_solution:
        tracer_reference_high = None
        z_mc_high = None
    else:
        # obtain reference solution first
        nums_levels.insert(0, num_levels_ref)
        ref_config = construct_config(
            horizontal_advection_type=advection.HorizontalAdvectionType.NO_ADVECTION,
            horizontal_advection_limiter=advection.HorizontalAdvectionLimiter.NO_LIMITER,
            vertical_advection_type=advection.VerticalAdvectionType.PPM_3RD_ORDER,
            vertical_advection_limiter=advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        )

    z_range = 1e5
    z_center = z_range / 2

    # run experiments sequentially
    for num_levels in nums_levels:
        reference_run = not use_exact_solution and num_levels == num_levels_ref

        grid = simple_grid.SimpleGrid()
        grid._configure(num_levels=num_levels)

        dz = z_range / num_levels
        z_mc = np.linspace(z_range - dz / 2, dz / 2, num_levels)
        z_ifc = np.linspace(z_range, 0.0, num_levels + 1)
        ddqz_z_full = helpers.constant_field(grid, dz, dims.CellDim, dims.KDim)

        metric_state = construct_idealized_metric_state(grid, ddqz_z_full)

        vertical_advection = advection.convert_config_to_vertical_advection(
            config=ref_config if reference_run else config,
            grid=grid,
            metric_state=metric_state,
            backend=backend,
        )

        diagnostic_state = construct_idealized_diagnostic_state(grid, ddqz_z_full)
        p_tracer_now = construct_idealized_tracer(
            test_config, grid, z_mc, z_ifc, z_center, z_range, use_high_order_quadrature
        )
        p_tracer_new = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid)

        exp_name = f"{vertical_advection_type.name}"
        jc = 0
        if enable_plots:
            name = f"{exp_name}_{num_levels}"
            plot_utils.plot_1D(z_mc, p_tracer_now.ndarray[jc, :], out_file=name + "_start.pdf")

        # time loop
        time = 0.0
        while time != time_end:
            vel_max = get_idealized_velocity_max(test_config, z_range, time_end)
            assert vel_max > 0.0
            dtime = min(cfl_number * dz / vel_max, time_end - time)
            log.debug(f"dtime: {dtime}")

            prep_adv = construct_idealized_prep_adv(
                test_config, grid, z_ifc, z_range, time, dtime, time_end
            )

            vertical_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                dtime=dtime,
            )
            time += dtime

            p_tracer_now, p_tracer_new = p_tracer_new, p_tracer_now
            # note: no need to swap airmass fields here because they are equal

        tracer_end_jc = p_tracer_now.ndarray[jc, :]
        if enable_plots:
            plot_utils.plot_1D(z_mc, tracer_end_jc, out_file=f"{name}_end.pdf")

        if reference_run:
            tracer_reference_high = p_tracer_now
            z_mc_high = z_mc
            continue

        # get reference solution
        tracer_reference = construct_idealized_tracer_reference(
            test_config,
            grid,
            z_mc,
            z_center,
            z_range,
            z_ifc,
            time,
            time_end,
            use_high_order_quadrature,
            tracer_reference_high,
            z_mc_high,
        )

        tracer_reference_jc = tracer_reference.ndarray[jc, :]
        if enable_plots:
            plot_utils.plot_1D(z_mc, tracer_reference_jc, out_file=f"{name}_reference.pdf")
            plot_utils.plot_1D(
                z_mc, tracer_end_jc - tracer_reference_jc, out_file=f"{name}_diff.pdf"
            )

        # compute errors
        error_l1, error_linf = compute_relative_errors(tracer_end_jc, tracer_reference_jc)
        assert math.isfinite(error_l1) and math.isfinite(error_linf)
        assert error_l1 >= 0.0 and error_linf >= 0.0
        errors_l1.append(error_l1)
        errors_linf.append(error_linf)
        lengths_min.append(dz)

    log.debug(f"errors_l1: {errors_l1}")
    log.debug(f"errors_linf: {errors_linf}")
    log.debug(f"lengths_min: {lengths_min}")
    n = len(lengths_min)
    assert n == len(errors_l1) == len(errors_linf)

    if enable_plots:
        theoretical_orders = [1.0, 2.0, 3.0]
        linestyles = ["--", ":", "-."]
        ref = "" if use_exact_solution else "_ref"
        plot_utils.plot_convergence(
            lengths_min,
            errors_l1,
            name=exp_name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"{exp_name}_l1{ref}.pdf",
        )
        plot_utils.plot_convergence(
            lengths_min,
            errors_linf,
            name=exp_name,
            theoretical_orders=theoretical_orders,
            linestyles=linestyles,
            out_file=f"{exp_name}_linf{ref}.pdf",
        )

    # check observed rate of convergence
    if l1_acceptable_range is not None:
        linreg_l1 = linregress(np.log(lengths_min), np.log(errors_l1))
        p_l1 = linreg_l1.slope
        log.debug(f"p_l1: {p_l1}")
        assert l1_acceptable_range[0] <= p_l1 <= l1_acceptable_range[1]
    if linf_acceptable_range is not None:
        linreg_linf = linregress(np.log(lengths_min), np.log(errors_linf))
        p_linf = linreg_linf.slope
        log.debug(f"p_linf: {p_linf}")
        assert linf_acceptable_range[0] <= p_linf <= linf_acceptable_range[1]
