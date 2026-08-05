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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_w_horizontal_stress_tendency import (
    compute_w_horizontal_stress_tendency,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeWHorizontalStressTendency(stencil_tests.StencilTest):
    """
    Fortran (mo_vdf.f90, 'Compute_diffusion_vert_wind') computes hori_tend_e on
    half-level rows jk = 2..nlev (1-based) -> 0-based rows 1..nlev-1; rows 0 and
    nlev of the half-level output stay untouched.
    """

    PROGRAM = compute_w_horizontal_stress_tendency
    OUTPUTS = ("hori_tend_e",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u: np.ndarray,
        v: np.ndarray,
        km_ic: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        w_vert: np.ndarray,
        km_iv: np.ndarray,
        inv_ddqz_z_half_v: np.ndarray,
        w_ie: np.ndarray,
        vt_e: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        edge_cell_length: np.ndarray,
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
        e2c = connectivities[dims.E2CDim]  # (n_edges, 2)
        e2c2v = connectivities[dims.E2C2VDim]  # (n_edges, 4)
        nlev = u.shape[1]

        # Half-level rows k = 1..nlev-1: full-level slices [k-1] = [:, :-1] and
        # [k] = [:, 1:]; half-level fields sliced at [:, 1:nlev].
        km = slice(0, nlev - 1)  # full level k-1 for k = 1..nlev-1
        kk = slice(1, nlev)  # full level k / half level k

        tang = np.expand_dims(tangent_orientation, axis=-1)
        inv_pel = np.expand_dims(inv_primal_edge_length, axis=-1)
        inv_vvl = np.expand_dims(inv_vert_vert_length, axis=-1)
        inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

        # Normal direction: flux = visc_c * D_31 at the half-level cell centers.
        u_c2 = u[e2c[:, 1]]
        v_c2 = v[e2c[:, 1]]
        dvn2 = (u_c2[:, km] - u_c2[:, kk]) * primal_normal_cell_x[:, 1:2] + (
            v_c2[:, km] - v_c2[:, kk]
        ) * primal_normal_cell_y[:, 1:2]
        flux_up_c = km_ic[e2c[:, 1]][:, kk] * (
            dvn2 * inv_ddqz_z_half[e2c[:, 1]][:, kk]
            + (w_vert[e2c2v[:, 3]][:, kk] - w_ie[:, kk]) * 2.0 * inv_vvl
        )

        u_c1 = u[e2c[:, 0]]
        v_c1 = v[e2c[:, 0]]
        dvn1 = (u_c1[:, km] - u_c1[:, kk]) * primal_normal_cell_x[:, 0:1] + (
            v_c1[:, km] - v_c1[:, kk]
        ) * primal_normal_cell_y[:, 0:1]
        flux_dn_c = km_ic[e2c[:, 0]][:, kk] * (
            dvn1 * inv_ddqz_z_half[e2c[:, 0]][:, kk]
            + (w_ie[:, kk] - w_vert[e2c2v[:, 2]][:, kk]) * 2.0 * inv_vvl
        )

        # Tangential direction: flux = visc_v * D_32 between vertex and edge center.
        u_v2 = u_vert[e2c2v[:, 1]]
        v_v2 = v_vert[e2c2v[:, 1]]
        dvt2 = 0.5 * (
            u_v2[:, km] * dual_normal_vert_x[:, 1:2]
            + v_v2[:, km] * dual_normal_vert_y[:, 1:2]
            + vt_e[:, km]
        ) - 0.5 * (
            u_v2[:, kk] * dual_normal_vert_x[:, 1:2]
            + v_v2[:, kk] * dual_normal_vert_y[:, 1:2]
            + vt_e[:, kk]
        )
        flux_up_v = km_iv[e2c2v[:, 1]][:, kk] * (
            dvt2 * inv_ddqz_z_half_v[e2c2v[:, 1]][:, kk]
            + tang * (w_vert[e2c2v[:, 1]][:, kk] - w_ie[:, kk]) / edge_cell_length[:, 1:2]
        )

        u_v1 = u_vert[e2c2v[:, 0]]
        v_v1 = v_vert[e2c2v[:, 0]]
        dvt1 = 0.5 * (
            u_v1[:, km] * dual_normal_vert_x[:, 0:1]
            + v_v1[:, km] * dual_normal_vert_y[:, 0:1]
            + vt_e[:, km]
        ) - 0.5 * (
            u_v1[:, kk] * dual_normal_vert_x[:, 0:1]
            + v_v1[:, kk] * dual_normal_vert_y[:, 0:1]
            + vt_e[:, kk]
        )
        flux_dn_v = km_iv[e2c2v[:, 0]][:, kk] * (
            dvt1 * inv_ddqz_z_half_v[e2c2v[:, 0]][:, kk]
            + tang * (w_ie[:, kk] - w_vert[e2c2v[:, 0]][:, kk]) / edge_cell_length[:, 0:1]
        )

        hori_tend = (flux_up_c - flux_dn_c) * inv_del + (
            flux_up_v - flux_dn_v
        ) * tang * 2.0 * inv_pel  # (n_edges, nlev - 1), rows 1..nlev-1

        hori_tend_e = np.zeros_like(w_ie)
        hori_tend_e[:, 1:nlev] = hori_tend
        out = np.zeros_like(hori_tend_e)
        out[horizontal_start:horizontal_end, vertical_start:vertical_end] = hori_tend_e[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(hori_tend_e=out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        u = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        km_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.0,
            high=1.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        inv_ddqz_z_half = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.1,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        w_vert = data_alloc.random_field(
            grid, dims.VertexDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        km_iv = data_alloc.random_field(
            grid,
            dims.VertexDim,
            dims.KDim,
            low=0.0,
            high=1.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        inv_ddqz_z_half_v = data_alloc.random_field(
            grid,
            dims.VertexDim,
            dims.KDim,
            low=0.1,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        w_ie = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vt_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        primal_normal_cell_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        primal_normal_cell_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        dual_normal_vert_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        dual_normal_vert_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        edge_cell_length = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=0.1, high=2.0, dtype=ta.wpfloat
        )

        tangent_orientation = data_alloc.random_sign(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)

        hori_tend_e = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        # Fortran: rl_start = grf_bdywidth_e, rl_end = min_rledge_int - 1.
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO))
        assert horizontal_start < horizontal_end

        return dict(
            u=u,
            v=v,
            km_ic=km_ic,
            inv_ddqz_z_half=inv_ddqz_z_half,
            u_vert=u_vert,
            v_vert=v_vert,
            w_vert=w_vert,
            km_iv=km_iv,
            inv_ddqz_z_half_v=inv_ddqz_z_half_v,
            w_ie=w_ie,
            vt_e=vt_e,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            edge_cell_length=edge_cell_length,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_dual_edge_length=inv_dual_edge_length,
            hori_tend_e=hori_tend_e,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            # Fortran jk = 2..nlev (1-based half levels) -> rows 1..nlev-1
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
