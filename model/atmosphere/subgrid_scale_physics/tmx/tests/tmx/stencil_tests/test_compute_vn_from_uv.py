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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_from_uv import (
    compute_vn_from_uv,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeVnFromUv(stencil_tests.StencilTest):
    PROGRAM = compute_vn_from_uv
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u: np.ndarray,
        v: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        c_lin_e: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]  # (n_edges, 2)

        # (n_edges, 2, nlev) gathers of the cell-center wind components
        u_e = u[e2c]
        v_e = v[e2c]

        # (n_edges, 2, 1) geometrical factors and weights per E2C neighbor
        pn_x = np.expand_dims(primal_normal_cell_x, axis=-1)
        pn_y = np.expand_dims(primal_normal_cell_y, axis=-1)
        w = np.expand_dims(c_lin_e, axis=-1)

        vn = np.sum(w * (u_e * pn_x + v_e * pn_y), axis=1)

        vn_out = np.zeros_like(vn)
        vn_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = vn[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(vn=vn_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        u = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        primal_normal_cell_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        primal_normal_cell_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat)
        vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: compute_normal_velocity_edge runs on
        # rl_start = grf_bdywidth_e + 1, rl_end = min_rledge_int.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            u=u,
            v=v,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            c_lin_e=c_lin_e,
            vn=vn,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
