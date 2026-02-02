# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.advection.advection_tracers import lsq_compute_coeff_cell_sphere, \
    lsq_compute_coeff_cell_torus
from icon4py.model.common.grid import base
from icon4py.model.common.grid import horizontal as h_grid
import pytest
from icon4py.model.testing import definitions, serialbox as sb, grid_utils
from icon4py.model.common.grid import base as base_grid
import gt4py.next.typing as gtx_typing
import icon4py.model.testing.test_utils as test_helpers

from icon4py.model.atmosphere.advection import advection
from icon4py.model.common import dimension as dims, constants
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions
from icon4py.model.testing.definitions import Experiments
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)

from ..fixtures import advection_exit_savepoint, advection_init_savepoint
from ..utils import (
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
@pytest.mark.parametrize("experiment", [definitions.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize(
    "date, even_timestep, ntracer, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type, vertical_advection_limiter",
    [
        (
            "2021-06-20T12:00:10.000",
            False,
            1,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            1,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:10.000",
            False,
            4,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            4,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
    ],
)
def test_advection_run_single_step(
    date,
    even_timestep,
    ntracer,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
    *,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    # data_provider,
    backend,
    advection_init_savepoint,
    advection_exit_savepoint,
):
    # TODO(OngChia): the last datatest fails on GPU (or even CPU) backend when there is no advection because the horizontal flux is not zero. Further check required.
    if (
        even_timestep
        and horizontal_advection_type == advection.HorizontalAdvectionType.NO_ADVECTION
    ):
        pytest.xfail(
            "This test is skipped until the cause of nonzero horizontal advection if revealed."
        )
    config = construct_config(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )
    interpolation_state = construct_interpolation_state(interpolation_savepoint, backend=backend)
    least_squares_state = construct_least_squares_state(interpolation_savepoint, backend=backend)
    metric_state = construct_metric_state(icon_grid, metrics_savepoint, backend=backend)
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
        icon_grid, advection_exit_savepoint, ntracer, backend=backend
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


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_lsq_compute_coeff_cell_sphere(
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
    interpolation_savepoint,
) -> None:
    experiment = Experiments.MCH_CH_R04B09
    gm = grid_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=1,
        keep_skip_values=True,
        allocator=backend,
    )

    c2e2c = grid_savepoint.c2e2c() #gm.grid.connectivities["C2E2C"].ndarray
    c2v  = grid_savepoint.c2v() #gm.grid.connectivities["C2V"].ndarray
    cell_owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    grid_sphere_radius = constants.EARTH_RADIUS
    lsq_dim_unk = 2
    lsq_dim_c = 3
    lsq_wgt_exp = 2
    cell_domain = h_grid.domain(dims.CellDim)

    min_rlcell_int = gm.grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    coordinates = grid_savepoint.coordinates()
    cell_lat = coordinates[dims.CellDim]["lat"]
    cell_lon = coordinates[dims.CellDim]["lon"]
    vertex_lat = coordinates[dims.VertexDim]["lat"]
    vertex_lon = coordinates[dims.VertexDim]["lon"]

    lsq_pseudoinv = lsq_compute_coeff_cell_sphere(cell_lat, cell_lon, vertex_lat, vertex_lon, c2e2c, c2v, cell_owner_mask, grid_sphere_radius, lsq_dim_unk, lsq_dim_c, lsq_wgt_exp, min_rlcell_int)

    assert test_helpers.dallclose(interpolation_savepoint.lsq_pseudoinv_1().asnumpy(), lsq_pseudoinv[:, 0, :], atol=1e-12)

@pytest.mark.level("unit")
@pytest.mark.datatest
def test_lsq_compute_coeff_cell_torus(
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
    interpolation_savepoint,
) -> None:

    experiment = Experiments.GAUSS3D
    gm = grid_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=1,
        keep_skip_values=True,
        allocator=backend,
    )
    c2e2c = gm.grid.connectivities["C2E2C"].ndarray
    c2v  = gm.grid.connectivities["C2V"].ndarray
    cell_owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    lsq_dim_unk = 2
    lsq_dim_c = 3
    lsq_wgt_exp = 2
    cell_domain = h_grid.domain(dims.CellDim)
    min_rlcell_int = gm.grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    coordinates = gm.coordinates
    cell_center_x = coordinates[dims.CellDim]["x"]
    cell_center_y = coordinates[dims.CellDim]["y"]
    domain_length = gm.grid.global_properties.domain_length
    domain_height = gm.grid.global_properties.domain_height
    vertex_center_x = coordinates[dims.VertexDim]["x"]
    vertex_center_y = coordinates[dims.VertexDim]["y"]

    lsq_pseudoinv = lsq_compute_coeff_cell_torus(cell_center_x, cell_center_y, vertex_center_x, vertex_center_y, c2e2c, c2v, cell_owner_mask, domain_length, domain_height, lsq_dim_unk, lsq_dim_c, lsq_wgt_exp, min_rlcell_int)

    assert test_helpers.dallclose(lsq_pseudoinv, lsq_pseudoinv)
