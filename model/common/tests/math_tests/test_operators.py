# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.math.stencils.compute_nabla2_on_cell import compute_nabla2_on_cell
from icon4py.model.common.math.stencils.compute_nabla2_on_cell_k import compute_nabla2_on_cell_k
from icon4py.model.common.test_utils import helpers as test_helpers, reference_funcs
from icon4py.model.common.test_utils.helpers import constant_field, zero_field


def test_nabla2_on_cell(
    grid,
    backend,
):
    psi_c = constant_field(
        grid,
        1.0,
        dims.CellDim,
    )
    geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
    nabla2_psi_c = zero_field(
        grid,
        dims.CellDim,
    )

    compute_nabla2_on_cell.with_backend(backend)(
        psi_c=psi_c,
        geofac_n2s=geofac_n2s,
        nabla2_psi_c=nabla2_psi_c,
        horizontal_start=0,
        horizontal_end=grid.num_cells,
        offset_provider={
            "C2E2CO": grid.get_offset_provider("C2E2CO"),
        },
    )

    nabla2_psi_c_np = reference_funcs.nabla2_on_cell_numpy(
        grid, psi_c.asnumpy(), geofac_n2s.asnumpy()
    )

    assert test_helpers.dallclose(nabla2_psi_c.asnumpy(), nabla2_psi_c_np)


def test_nabla2_on_cell_k(
    grid,
    backend,
):
    psi_c = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
    geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
    nabla2_psi_c = zero_field(grid, dims.CellDim, dims.KDim)

    compute_nabla2_on_cell_k.with_backend(backend)(
        psi_c=psi_c,
        geofac_n2s=geofac_n2s,
        nabla2_psi_c=nabla2_psi_c,
        horizontal_start=0,
        horizontal_end=grid.num_cells,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider={
            "C2E2CO": grid.get_offset_provider("C2E2CO"),
        },
    )

    nabla2_psi_c_np = reference_funcs.nabla2_on_cell_k_numpy(
        grid, psi_c.asnumpy(), geofac_n2s.asnumpy()
    )

    assert test_helpers.dallclose(nabla2_psi_c.asnumpy(), nabla2_psi_c_np)
