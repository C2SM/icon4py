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
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeScalarNabla2Flux(StencilTest):
    PROGRAM = compute_scalar_nabla2_flux
    OUTPUTS = ("nabla2_flux",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
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
        e2c = connectivities[dims.E2CDim]  # (n_edges, 2)
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

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        # Fortran: edges rl_start = grf_bdywidth_e, rl_end = min_rledge_int - 1
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO))
        assert horizontal_start < horizontal_end

        return dict(
            scalar=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            km_ie=data_alloc.random_field(
                grid,
                dims.EdgeDim,
                dims.KDim,
                low=0.0,
                high=10.0,
                extend={dims.KDim: 1},
                dtype=wpfloat,
            ),
            inv_dual_edge_length=data_alloc.random_field(
                grid, dims.EdgeDim, low=1.0e-5, high=1.0e-3, dtype=wpfloat
            ),
            nabla2_flux=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat),
            rturb_prandtl=wpfloat(3.0),
            prefac=wpfloat(0.9),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
