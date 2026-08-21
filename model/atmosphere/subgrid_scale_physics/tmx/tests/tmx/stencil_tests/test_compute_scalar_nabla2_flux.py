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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_scalar_nabla2_flux import (
    compute_scalar_nabla2_flux,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestComputeScalarNabla2Flux(stencil_tests.StencilTest):
    PROGRAM = compute_scalar_nabla2_flux
    OUTPUTS = ("nabla2_flux",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar: np.ndarray,
        km_ie: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        nabla2_flux: np.ndarray,
        rturb_prandtl: float,
        prefac: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        e2c = connectivities[dims.E2C]  # (n_edges, 2)
        flux = (
            0.5
            * prefac
            * rturb_prandtl
            * (km_ie[:, :-1] + km_ie[:, 1:])
            * inv_dual_edge_length[:, np.newaxis]
            * (scalar[e2c[:, 1]] - scalar[e2c[:, 0]])
        )
        nabla2_flux_out = nabla2_flux.copy()
        nabla2_flux_out[horizontal_start:horizontal_end] = flux[horizontal_start:horizontal_end]
        return dict(nabla2_flux=nabla2_flux_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        # Fortran: edges rl_start = grf_bdywidth_e, rl_end = min_rledge_int - 1
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO))
        assert horizontal_start < horizontal_end

        return dict(
            scalar=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            km_ie=data_alloc.random_field(
                dims.EdgeDim,
                dims.KDim,
                low=0.0,
                high=10.0,
                extend={dims.KDim: 1},
                dtype=wpfloat,
            ),
            inv_dual_edge_length=data_alloc.random_field(
                dims.EdgeDim, low=1.0e-5, high=1.0e-3, dtype=wpfloat
            ),
            nabla2_flux=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            rturb_prandtl=wpfloat(3.0),
            prefac=wpfloat(0.9),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
