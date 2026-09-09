# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import simple, vertical as v_grid
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.states import diagnostic_state as diagnostics, tracer_states as tracers
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
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


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_edge_2_cell_vector_rbf_interpolation(
    data_provider: sb.IconSerialDataProvider,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    prognostics_init_savepoint = data_provider.from_savepoint_prognostics_initial()
    vn = prognostics_init_savepoint.vn_now()
    rbv_vec_coeff_c1 = interpolation_savepoint.rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = interpolation_savepoint.rbf_vec_coeff_c2()

    diagnostics_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    u_ref = diagnostics_reference_savepoint.zonal_wind().asnumpy()
    v_ref = diagnostics_reference_savepoint.meridional_wind().asnumpy()

    u = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    v = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)

    cell_domain = h_grid.domain(dims.CellDim)
    cell_end_lateral_boundary_level_2 = icon_grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.END))

    rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=vn,
        ptr_coeff_1=rbv_vec_coeff_c1,
        ptr_coeff_2=rbv_vec_coeff_c2,
        p_u_out=u,
        p_v_out=v,
        horizontal_start=cell_end_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_connectivity("C2E2C2E"),
        },
    )

    assert test_utils.dallclose(
        u.asnumpy(),
        u_ref,
    )

    assert test_utils.dallclose(
        v.asnumpy(),
        v_ref,
        atol=1.0e-13,
    )
