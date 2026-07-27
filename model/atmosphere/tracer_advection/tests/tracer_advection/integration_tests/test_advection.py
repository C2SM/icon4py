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

from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    base as base_grid,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
)
from icon4py.model.common.interpolation.interpolation_fields import compute_lsq_coeffs
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils as gridtest_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)

from ..fixtures import advection_exit_savepoint, advection_init_savepoint
from ..utils import (
    construct_diagnostic_exit_state,
    construct_diagnostic_init_state,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    log_serialized,
    verify_advection_fields,
)


# ntracer legend for the serialization data used here in test_advection:
# ------------------------------------
# ntracer          |  0, 1, 2, 3, 4 |
# ------------------------------------
# ivadv_tracer     |  3, 0, 0, 2, 3 |
# itype_hlimit     |  3, 4, 3, 0, 0 |
# itype_vlimit     |  1, 0, 0, 2, 1 |
# ihadv_tracer     | 52, 2, 2, 0, 0 |
# ------------------------------------


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize(
    "date, even_timestep, ntracer, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type, vertical_advection_limiter",
    [
        (
            "2021-06-20T12:00:10.000",
            False,
            1,
            tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            tracer_advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            tracer_advection.VerticalAdvectionType.NO_ADVECTION,
            tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            1,
            tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            tracer_advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            tracer_advection.VerticalAdvectionType.NO_ADVECTION,
            tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:10.000",
            False,
            4,
            tracer_advection.HorizontalAdvectionType.NO_ADVECTION,
            tracer_advection.HorizontalAdvectionLimiter.NO_LIMITER,
            tracer_advection.VerticalAdvectionType.PPM_3RD_ORDER,
            tracer_advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            4,
            tracer_advection.HorizontalAdvectionType.NO_ADVECTION,
            tracer_advection.HorizontalAdvectionLimiter.NO_LIMITER,
            tracer_advection.VerticalAdvectionType.PPM_3RD_ORDER,
            tracer_advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
    ],
)
def test_advection_run_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    date: str,
    even_timestep: bool,
    ntracer: int,
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
    horizontal_advection_limiter: tracer_advection.HorizontalAdvectionLimiter,
    vertical_advection_type: tracer_advection.VerticalAdvectionType,
    vertical_advection_limiter: tracer_advection.VerticalAdvectionLimiter,
    *,
    grid_savepoint: Any,
    icon_grid: base_grid.Grid,
    interpolation_savepoint: Any,
    metrics_savepoint: Any,
    # data_provider,
    backend: gtx_typing.Backend | None,
    advection_init_savepoint: Any,
    advection_exit_savepoint: Any,
    experiment: test_defs.Experiment,
) -> None:
    # TODO(OngChia): the last datatest fails on GPU (or even CPU) backend when there is no advection because the horizontal flux is not zero. Further check required.
    if (
        even_timestep
        and horizontal_advection_type == tracer_advection.HorizontalAdvectionType.NO_ADVECTION
    ):
        pytest.xfail(
            "This test is skipped until the cause of nonzero horizontal advection if revealed."
        )
    config = tracer_advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )

    interpolation_state = construct_interpolation_state(interpolation_savepoint, backend=backend)
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    least_squares_coeffs = compute_lsq_coeffs(
        cell_center_x=geometry.get(geometry_attrs.CELL_CENTER_X).asnumpy(),
        cell_center_y=geometry.get(geometry_attrs.CELL_CENTER_Y).asnumpy(),
        cell_lat=geometry.get(geometry_attrs.CELL_LAT).asnumpy(),
        cell_lon=geometry.get(geometry_attrs.CELL_LON).asnumpy(),
        c2e2c=icon_grid.connectivities["C2E2C"].asnumpy(),
        cell_owner_mask=grid_savepoint.c_owner_mask().asnumpy(),
        domain_length=geometry.grid.grid_params.domain_length,  # type: ignore[arg-type]  # float | None, compute_lsq_coeffs handles None for icosahedron grids
        domain_height=geometry.grid.grid_params.domain_height,  # type: ignore[arg-type]  # float | None, compute_lsq_coeffs handles None for icosahedron grids
        grid_sphere_radius=constants.EARTH_RADIUS,
        lsq_dim_unk=experiment.config.interpolation.lsq_dim_unk,
        lsq_dim_c=experiment.config.interpolation.lsq_dim_c,
        lsq_wgt_exp=experiment.config.interpolation.lsq_wgt_exp,
        start_idx=icon_grid.start_index(
            h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        min_rlcell_int=icon_grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LOCAL)),
        geometry_type=icon_grid.grid_params.geometry_type,  # type: ignore[attr-defined]  # icon_grid is base_grid.Grid at type level, but actually IconGrid
        exchange=decomposition.single_node_exchange,
    )

    least_squares_state = construct_least_squares_state(least_squares_coeffs, backend=backend)

    metric_state = construct_metric_state(icon_grid, metrics_savepoint, backend=backend)
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()

    advection_granule = tracer_advection.convert_config_to_advection(
        config=config,
        grid=icon_grid,  # type: ignore[arg-type]  # fixture returns base_grid.Grid but is actually IconGrid
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        even_timestep=even_timestep,
        backend=backend,
        exchange=decomposition.single_node_exchange,
    )

    diagnostic_state = construct_diagnostic_init_state(
        icon_grid, advection_init_savepoint, ntracer, backend=backend
    )
    prep_adv = construct_prep_adv(advection_init_savepoint)
    p_tracer_now = advection_init_savepoint.tracer(ntracer)
    p_tracer_new = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
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
        grid=icon_grid,
        savepoint=advection_exit_savepoint,
        ntracer=ntracer,
        backend=backend,
    )
    p_tracer_new_ref = advection_exit_savepoint.tracer(ntracer)

    verify_advection_fields(
        grid=icon_grid,  # type: ignore[arg-type]  # fixture returns base_grid.Grid but is actually IconGrid
        diagnostic_state=diagnostic_state,
        diagnostic_state_ref=diagnostic_state_ref,
        p_tracer_new=p_tracer_new,
        p_tracer_new_ref=p_tracer_new_ref,
        even_timestep=even_timestep,
    )
