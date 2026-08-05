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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vn_vertical_diffusion_rhs import (
    compute_vn_vertical_diffusion_rhs,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeVnVerticalDiffusionRhs(stencil_tests.StencilTest):
    PROGRAM = compute_vn_vertical_diffusion_rhs
    OUTPUTS = ("rhs", "inv_maire")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        w: np.ndarray,
        km_ie: np.ndarray,
        inv_rhoe: np.ndarray,
        inv_ddqz_z_full_e: np.ndarray,
        u_stress: np.ndarray,
        v_stress: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        c_lin_e: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        nlev: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

        inv_maire = inv_ddqz_z_full_e * inv_rhoe  # (n_edges, nlev)

        # Vertical flux of the dw/dn stress at all half levels, (n_edges, nlev + 1)
        grad = km_ie * inv_del * (w[e2c[:, 1]] - w[e2c[:, 0]])

        # Interior rows: rhs(k) = (grad(k) - grad(k+1)) * inv_maire(k)
        rhs = (grad[:, :-1] - grad[:, 1:]) * inv_maire

        # Top row (jk = 1): the flux through the model top is dropped.
        rhs[:, 0] = -grad[:, 1] * inv_maire[:, 0]

        # Bottom row (jk = nlev): dwdn - net surface stress along the edge normal.
        stress_n = u_stress[e2c] * primal_normal_cell_x + v_stress[e2c] * primal_normal_cell_y
        flux_dn_e = np.sum(stress_n * c_lin_e, axis=1)  # (n_edges,)
        rhs[:, nlev - 1] = (
            grad[:, nlev - 1] * inv_maire[:, nlev - 1] - flux_dn_e * inv_maire[:, nlev - 1]
        )

        rhs_out = np.zeros_like(rhs)
        inv_maire_out = np.zeros_like(inv_maire)
        rhs_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = rhs[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        inv_maire_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = inv_maire[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(rhs=rhs_out, inv_maire=inv_maire_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        w = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        km_ie = data_alloc.random_field(
            grid,
            dims.EdgeDim,
            dims.KDim,
            low=0.0,
            high=1.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        inv_rhoe = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=0.5, high=2.0, dtype=ta.wpfloat
        )
        inv_ddqz_z_full_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=0.1, high=2.0, dtype=ta.wpfloat
        )
        u_stress = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        v_stress = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        primal_normal_cell_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        primal_normal_cell_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        c_lin_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=0.1, high=0.9, dtype=ta.wpfloat
        )
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)

        rhs = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        inv_maire = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: rl_start = grf_bdywidth_e + 1, rl_end = min_rledge_int.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            w=w,
            km_ie=km_ie,
            inv_rhoe=inv_rhoe,
            inv_ddqz_z_full_e=inv_ddqz_z_full_e,
            u_stress=u_stress,
            v_stress=v_stress,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            c_lin_e=c_lin_e,
            inv_dual_edge_length=inv_dual_edge_length,
            rhs=rhs,
            inv_maire=inv_maire,
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
