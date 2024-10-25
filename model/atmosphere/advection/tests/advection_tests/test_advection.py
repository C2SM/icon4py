# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.advection import advection
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from .utils import (
    construct_config,
    construct_diagnostic_exit_state,
    construct_diagnostic_init_state,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    log_serialized,
    verify_advection_fields,
)


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
def test_advection_run_single_step(
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
