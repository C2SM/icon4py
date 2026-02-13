# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

import icon4py.model.testing.test_utils as test_helpers
from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.atmosphere.advection.advection_lsq_coeffs import lsq_compute_coeffs
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as nh
from icon4py.model.common import constants, dimension as dims, type_alias as ta, utils
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import (
    base as base_grid,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
    icon,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes as intp_attrs
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions as test_defs,
    grid_utils,
    grid_utils as gridtest_utils,
    parallel_helpers,
    serialbox,
    serialbox as sb,
    test_utils,
)
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
)

from .. import utils
from ..fixtures import *  # noqa: F403
from ..fixtures import advection_exit_savepoint, advection_init_savepoint, advection_lsq_state
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


try:
    import mpi4py

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [test_defs.Experiments.MCH_CH_R04B09])
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
@pytest.mark.mpi
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
    experiment: test_defs.Experiment,
    processor_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,  # : F811 fixture
    advection_lsq_state,
):
    if test_utils.is_embedded(backend):
        # https://github.com/GridTools/gt4py/issues/1583
        pytest.xfail("ValueError: axes don't match array")
    # TODO(OngChia): the last datatest fails on GPU (or even CPU) backend when there is no advection because the horizontal flux is not zero. Further check required.
    if (
        even_timestep
        and horizontal_advection_type == advection.HorizontalAdvectionType.NO_ADVECTION
    ):
        pytest.xfail(
            "This test is skipped until the cause of nonzero horizontal advection if revealed."
        )

    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    config = construct_config(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )

    interpolation_state = construct_interpolation_state(interpolation_savepoint, backend=backend)

    metric_state = construct_metric_state(icon_grid, metrics_savepoint, backend=backend)
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    advection_granule = advection.convert_config_to_advection(
        config=config,
        grid=icon_grid,
        interpolation_state=interpolation_state,
        least_squares_state=advection_lsq_state,
        metric_state=metric_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        even_timestep=even_timestep,
        backend=backend,
        exchange=exchange,
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

    assert test_helpers.dallclose(
        diagnostic_state.hfl_tracer.asnumpy(), diagnostic_state_ref.hfl_tracer.asnumpy(), atol=1e-8
    )
    assert test_utils.dallclose(
        diagnostic_state.vfl_tracer.asnumpy(),
        diagnostic_state_ref.vfl_tracer.asnumpy(),
        rtol=1e-10,
    )
    assert test_helpers.dallclose(p_tracer_new_ref.asnumpy(), p_tracer_new.asnumpy(), atol=1e-10)


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.mpi
def test_lsq_compute_coeffs(
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    experiment: test_defs.Experiment,
    processor_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,  # : F811 fixture
) -> None:
    gm = grid_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=1,
        keep_skip_values=True,
        allocator=backend,
    )

    c2e2c = gm.grid.connectivities["C2E2C"].asnumpy()
    cell_owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    grid_sphere_radius = constants.EARTH_RADIUS
    lsq_dim_unk = 2
    lsq_dim_c = 3
    lsq_wgt_exp = 2
    cell_domain = h_grid.domain(dims.CellDim)

    min_rlcell_int = gm.grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    start_idx = gm.grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    grid_geometry = grid_utils.get_grid_geometry(backend, experiment)
    cell_center_x = grid_geometry.get(geometry_attrs.CELL_CENTER_X).asnumpy()
    cell_center_y = grid_geometry.get(geometry_attrs.CELL_CENTER_Y).asnumpy()
    domain_length = gm.grid.global_properties.domain_length
    domain_height = gm.grid.global_properties.domain_height
    lsq_dim_stencil = 3
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    coordinates = gm.coordinates
    cell_lat = coordinates[dims.CellDim]["lat"].asnumpy()
    cell_lon = coordinates[dims.CellDim]["lon"].asnumpy()
    lsq_pseudoinv = lsq_compute_coeffs(
        cell_center_x,
        cell_center_y,
        cell_lat,
        cell_lon,
        c2e2c,
        cell_owner_mask,
        domain_length,
        domain_height,
        grid_sphere_radius,
        lsq_dim_unk,
        lsq_dim_c,
        lsq_wgt_exp,
        lsq_dim_stencil,
        start_idx,
        min_rlcell_int,
        icon_grid.geometry_type.value,
        exchange,
    )

    assert test_helpers.dallclose(
        interpolation_savepoint.lsq_pseudoinv_1().asnumpy(), lsq_pseudoinv[:, 0, :], atol=1e-15
    )
    assert test_helpers.dallclose(
        interpolation_savepoint.lsq_pseudoinv_2().asnumpy(), lsq_pseudoinv[:, 1, :], atol=1e-15
    )
