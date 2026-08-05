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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_horizontal_stress_tendency import (
    compute_vn_horizontal_stress_tendency,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeVnHorizontalStressTendency(stencil_tests.StencilTest):
    PROGRAM = compute_vn_horizontal_stress_tendency
    OUTPUTS = ("tot_tend",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        vn: np.ndarray,
        km_c: np.ndarray,
        div_c: np.ndarray,
        km_iv: np.ndarray,
        inv_rhoe: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        e2c2v = connectivities[dims.E2C2VDim]  # (n_edges, 4)
        e2c = connectivities[dims.E2CDim]  # (n_edges, 2)
        z_2by3 = 2.0 / 3.0

        # (n_edges, 4, nlev) normal/tangential projections at the four vertices
        vn_vert = u_vert[e2c2v] * np.expand_dims(primal_normal_vert_x, axis=-1) + v_vert[
            e2c2v
        ] * np.expand_dims(primal_normal_vert_y, axis=-1)
        vt_vert = u_vert[e2c2v] * np.expand_dims(dual_normal_vert_x, axis=-1) + v_vert[
            e2c2v
        ] * np.expand_dims(dual_normal_vert_y, axis=-1)
        dvt = vt_vert[:, 3] - vt_vert[:, 2]

        # (n_edges, 1) edge geometry
        tang = np.expand_dims(tangent_orientation, axis=-1)
        inv_pel = np.expand_dims(inv_primal_edge_length, axis=-1)
        inv_vvl = np.expand_dims(inv_vert_vert_length, axis=-1)
        inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

        # Fortran cell/vertex neighbor indices: iecidx 1/2 -> E2C 0/1,
        # ividx 1..4 -> E2C2V 0..3.
        flux_up_c = km_c[e2c[:, 1]] * (
            4.0 * (vn_vert[:, 3] - vn) * inv_vvl - z_2by3 * div_c[e2c[:, 1]]
        )
        flux_dn_c = km_c[e2c[:, 0]] * (
            4.0 * (vn - vn_vert[:, 2]) * inv_vvl - z_2by3 * div_c[e2c[:, 0]]
        )

        # km_iv at the two edge endpoint vertices, summed over half levels k, k+1
        km_iv_e = km_iv[e2c2v[:, 0:2]]  # (n_edges, 2, nlev + 1)
        km_iv_sum = km_iv_e[:, :, :-1] + km_iv_e[:, :, 1:]  # (n_edges, 2, nlev)

        flux_up_v = km_iv_sum[:, 1] * (tang * (vn_vert[:, 1] - vn) * inv_pel + 0.5 * dvt * inv_vvl)
        flux_dn_v = km_iv_sum[:, 0] * (tang * (vn - vn_vert[:, 0]) * inv_pel + 0.5 * dvt * inv_vvl)

        tot_tend = (
            (flux_up_c - flux_dn_c) * inv_del + 2.0 * tang * (flux_up_v - flux_dn_v) * inv_pel
        ) * inv_rhoe

        tot_tend_out = np.zeros_like(tot_tend)
        tot_tend_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = tot_tend[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(tot_tend=tot_tend_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        km_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        div_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        km_iv = data_alloc.random_field(
            grid,
            dims.VertexDim,
            dims.KDim,
            low=0.0,
            high=1.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        inv_rhoe = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=0.5, high=2.0, dtype=ta.wpfloat
        )

        primal_normal_vert_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        dual_normal_vert_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        dual_normal_vert_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )

        tangent_orientation = data_alloc.random_sign(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)

        tot_tend = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: rl_start = grf_bdywidth_e + 1, rl_end = min_rledge_int.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            vn=vn,
            km_c=km_c,
            div_c=div_c,
            km_iv=km_iv,
            inv_rhoe=inv_rhoe,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_dual_edge_length=inv_dual_edge_length,
            tot_tend=tot_tend,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
