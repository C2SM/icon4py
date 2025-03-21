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
        edge: np.ndarray,
        vertex: np.ndarray,
        lateral_boundary_7: int,
        halo_1: int,
        start_vertex_lateral_boundary_level_2: int,
        end_vertex_halo: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        edge = edge[:, np.newaxis]
        vertex = vertex[:, np.newaxis]
        condition_mask1 = (start_vertex_lateral_boundary_level_2 <= vertex) & (
            vertex < end_vertex_halo
        )
        condition_mask2 = (lateral_boundary_7 <= edge) & (edge < halo_1)

        w_at_vertices = np.where(
            condition_mask1,
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(connectivities, w, c_intp),
            0.0,
        )

        horizontal_advection_of_w_at_edges_on_half_levels = np.where(
            condition_mask2,
            compute_horizontal_advection_term_for_vertical_velocity_numpy(
                connectivities,
                vn_on_half_levels[:, :-1],
                inv_dual_edge_length,
                w,
                tangential_wind_on_half_levels,
                inv_primal_edge_length,
                tangent_orientation,
                w_at_vertices,
            ),
            horizontal_advection_of_w_at_edges_on_half_levels,
        )

        return dict(
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
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

        edge = data_alloc.index_field(dim=dims.EdgeDim, grid=grid)
        vertex = data_alloc.index_field(dim=dims.VertexDim, grid=grid)

        nlev = grid.num_levels

        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)
        # For the ICON grid we use the proper domain bounds (otherwise we will run into non-protected skip values)
        lateral_boundary_7 = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
        halo_1 = grid.end_index(edge_domain(h_grid.Zone.HALO))
        start_vertex_lateral_boundary_level_2 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        end_vertex_halo = grid.end_index(vertex_domain(h_grid.Zone.HALO))
        horizontal_start = 0
        horizontal_end = grid.num_edges
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
            edge=edge,
            vertex=vertex,
            lateral_boundary_7=lateral_boundary_7,
            halo_1=halo_1,
            start_vertex_lateral_boundary_level_2=start_vertex_lateral_boundary_level_2,
            end_vertex_halo=end_vertex_halo,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
