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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_shear_and_div_of_stress import (
    compute_shear_and_div_of_stress,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeShearAndDivOfStress(stencil_tests.StencilTest):
    PROGRAM = compute_shear_and_div_of_stress
    OUTPUTS = ("shear", "div_stress")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        w_vert: np.ndarray,
        w: np.ndarray,
        vn_ie: np.ndarray,
        vt_ie: np.ndarray,
        w_ie: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_ddqz_z_full_e: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        e2c2v = connectivities[dims.E2C2VDim]  # (n_edges, 4)
        e2c = connectivities[dims.E2CDim]  # (n_edges, 2)

        # (n_edges, 4, nlev) gathers of the vertex velocities
        u_vert_e = u_vert[e2c2v]
        v_vert_e = v_vert[e2c2v]

        # (n_edges, 4, 1) geometrical factors per E2C2V neighbor
        pn_x = np.expand_dims(primal_normal_vert_x, axis=-1)
        pn_y = np.expand_dims(primal_normal_vert_y, axis=-1)
        dn_x = np.expand_dims(dual_normal_vert_x, axis=-1)
        dn_y = np.expand_dims(dual_normal_vert_y, axis=-1)

        # (n_edges, 1) edge geometry
        tang = np.expand_dims(tangent_orientation, axis=-1)
        inv_pel = np.expand_dims(inv_primal_edge_length, axis=-1)
        inv_vvl = np.expand_dims(inv_vert_vert_length, axis=-1)
        inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

        # Normal/tangential velocity components at the four vertices, (n_edges, 4, nlev)
        vn_vert = u_vert_e * pn_x + v_vert_e * pn_y
        vt_vert = u_vert_e * dn_x + v_vert_e * dn_y

        # Vertical wind at full levels: cells (E2C) and edge endpoints (E2C2V 0, 1)
        w_c = w[e2c]  # (n_edges, 2, nlev + 1)
        w_full_c = 0.5 * (w_c[:, :, :-1] + w_c[:, :, 1:])  # (n_edges, 2, nlev)
        w_v = w_vert[e2c2v[:, 0:2]]  # (n_edges, 2, nlev + 1)
        w_full_v = 0.5 * (w_v[:, :, :-1] + w_v[:, :, 1:])  # (n_edges, 2, nlev)

        # Velocity gradient tensor at edge of full levels
        vgrad_11 = (vn_vert[:, 3] - vn_vert[:, 2]) * inv_vvl
        vgrad_12 = (vn_vert[:, 1] - vn_vert[:, 0]) * tang * inv_pel
        vgrad_13 = (vn_ie[:, :-1] - vn_ie[:, 1:]) * inv_ddqz_z_full_e

        vgrad_21 = (vt_vert[:, 3] - vt_vert[:, 2]) * inv_vvl
        vgrad_22 = (vt_vert[:, 1] - vt_vert[:, 0]) * tang * inv_pel
        vgrad_23 = (vt_ie[:, :-1] - vt_ie[:, 1:]) * inv_ddqz_z_full_e

        vgrad_31 = (w_full_c[:, 1] - w_full_c[:, 0]) * inv_del
        vgrad_32 = (w_full_v[:, 1] - w_full_v[:, 0]) * tang * inv_pel
        vgrad_33 = (w_ie[:, :-1] - w_ie[:, 1:]) * inv_ddqz_z_full_e

        # Strain rates at edge center
        d_12 = vgrad_12 + vgrad_21
        d_13 = vgrad_13 + vgrad_31
        d_23 = vgrad_23 + vgrad_32

        shear = 4.0 * (vgrad_11**2 + vgrad_22**2 + vgrad_33**2) + 2.0 * (
            d_12**2 + d_13**2 + d_23**2
        )
        div_stress = vgrad_11 + vgrad_22 + vgrad_33

        shear_out = np.zeros_like(shear)
        div_stress_out = np.zeros_like(div_stress)
        shear_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = shear[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        div_stress_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = div_stress[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]

        return dict(shear=shear_out, div_stress=div_stress_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        w_vert = data_alloc.random_field(
            grid, dims.VertexDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        w = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vn_ie = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vt_ie = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        w_ie = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
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
        inv_ddqz_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        shear = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        div_stress = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: compute_velocity_gradient_tensor / compute_shear run on
        # rl_start = 4, rl_end = min_rledge_int - 2.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        assert horizontal_start < horizontal_end

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            w_vert=w_vert,
            w=w,
            vn_ie=vn_ie,
            vt_ie=vt_ie,
            w_ie=w_ie,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_ddqz_z_full_e=inv_ddqz_z_full_e,
            shear=shear,
            div_stress=div_stress,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
