# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_horizontal_diffusion_and_update_scalar import (
    apply_horizontal_diffusion_and_update_scalar,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestApplyHorizontalDiffusionAndUpdateScalar(stencil_tests.StencilTest):
    """
    Outside the computed domain ``tend`` keeps its input values (the vertical
    diffusion tendency) and ``new_scalar`` stays zero.
    """

    PROGRAM = apply_horizontal_diffusion_and_update_scalar
    OUTPUTS = ("new_scalar", "tend")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar: np.ndarray,
        nabla2_flux: np.ndarray,
        geofac_div: np.ndarray,
        rho: np.ndarray,
        tend: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        c2e = connectivities[dims.C2E]  # (n_cells, 3)
        hori_tend = np.sum(geofac_div[:, :, np.newaxis] * nabla2_flux[c2e], axis=1) / rho

        hs, he = horizontal_start, horizontal_end
        tend_out = tend.copy()
        tend_out[hs:he] = tend[hs:he] + hori_tend[hs:he]
        new_scalar = np.zeros_like(scalar)
        new_scalar[hs:he] = scalar[hs:he] + tend_out[hs:he] * dtime
        return dict(new_scalar=new_scalar, tend=tend_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            scalar=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            nabla2_flux=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            geofac_div=data_alloc.random_field(
                dims.CellDim, dims.C2EDim, low=-1.0e-4, high=1.0e-4, dtype=wpfloat
            ),
            rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.4, dtype=wpfloat),
            new_scalar=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            tend=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            dtime=wpfloat(300.0),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
