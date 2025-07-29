# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.states.utils as state_utils
import icon4py.model.testing.helpers as test_helpers
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_velocity_advection import (
    compute_horizontal_advection_of_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc

from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)


class TestComputeHorizontalAdvectionOfW(test_helpers.StencilTest):
    PROGRAM = compute_horizontal_advection_of_w
    OUTPUTS = ("horizontal_advection_of_w_at_edges_on_half_levels",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        w: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        w_at_vertices = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            connectivities, w, c_intp
        )

        horizontal_advection_of_w_at_edges_on_half_levels = (
            compute_horizontal_advection_term_for_vertical_velocity_numpy(
                connectivities,
                vn_on_half_levels[:, :-1],
                inv_dual_edge_length,
                w,
                tangential_wind_on_half_levels,
                inv_primal_edge_length,
                tangent_orientation,
                w_at_vertices,
            )
        )

        return dict(
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        tangential_wind_on_half_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        c_intp = data_alloc.random_field(grid, dims.VertexDim, dims.V2CDim)

        nlev = grid.num_levels

        edge_domain = h_grid.domain(dims.EdgeDim)
        # For the ICON grid we use the proper domain bounds (otherwise we will run into non-protected skip values)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = nlev

        return dict(
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
