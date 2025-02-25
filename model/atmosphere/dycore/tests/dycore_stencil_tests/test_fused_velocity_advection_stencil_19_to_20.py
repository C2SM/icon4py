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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.fused_velocity_advection_stencil_19_to_20 import (
    fused_velocity_advection_stencil_19_to_20,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing.helpers import StencilTest

from .test_add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy,
)
from .test_compute_advective_normal_wind_tendency import (
    compute_advective_normal_wind_tendency_numpy,
)
from .test_mo_math_divrot_rot_vertex_ri_dsl import mo_math_divrot_rot_vertex_ri_dsl_numpy


class TestFusedVelocityAdvectionStencil19To20(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_19_to_20
    OUTPUTS = ("ddt_vn_apc",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vn: np.ndarray,
        geofac_rot: np.ndarray,
        z_kin_hor_e: np.ndarray,
        coeff_gradekin: np.ndarray,
        z_ekinh: np.ndarray,
        vt: np.ndarray,
        f_e: np.ndarray,
        c_lin_e: np.ndarray,
        z_w_con_c_full: np.ndarray,
        vn_ie: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        levelmask: np.ndarray,
        area_edge: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        geofac_grdiv: np.ndarray,
        k: np.ndarray,
        cfl_w_limit: np.ndarray,
        scalfac_exdiff: np.ndarray,
        d_time: float,
        extra_diffu: bool,
        nlev: int,
        nrdmax: int,
        ddt_vn_apc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        ddt_vn_apc_cp = ddt_vn_apc.copy()
        zeta = mo_math_divrot_rot_vertex_ri_dsl_numpy(connectivities, vn, geofac_rot)

        ddt_vn_apc = compute_advective_normal_wind_tendency_numpy(
            connectivities,
            z_kin_hor_e,
            coeff_gradekin,
            z_ekinh,
            zeta,
            vt,
            f_e,
            c_lin_e,
            z_w_con_c_full,
            vn_ie,
            ddqz_z_full_e,
        )

        condition = (np.maximum(2, nrdmax - 2) <= k) & (k < nlev - 3)

        ddt_vn_apc_extra_diffu = add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
            connectivities,
            levelmask,
            c_lin_e,
            z_w_con_c_full,
            ddqz_z_full_e,
            area_edge,
            tangent_orientation,
            inv_primal_edge_length,
            zeta,
            geofac_grdiv,
            vn,
            ddt_vn_apc,
            cfl_w_limit,
            scalfac_exdiff,
            d_time,
        )

        ddt_vn_apc = np.where(condition & extra_diffu, ddt_vn_apc_extra_diffu, ddt_vn_apc)
        # restriction of execution domain
        ddt_vn_apc[0 : kwargs["horizontal_start"], :] = ddt_vn_apc_cp[
            0 : kwargs["horizontal_start"], :
        ]
        ddt_vn_apc[kwargs["horizontal_end"] :, :] = ddt_vn_apc_cp[kwargs["horizontal_end"] :, :]

        return dict(ddt_vn_apc=ddt_vn_apc)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict:
        z_kin_hor_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        coeff_gradekin = data_alloc.random_field(grid, dims.ECDim)
        z_ekinh = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        f_e = data_alloc.random_field(grid, dims.EdgeDim)
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        z_w_con_c_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        vn_ie = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        ddqz_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddt_vn_apc = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        levelmask = data_alloc.random_mask(grid, dims.KDim, extend={dims.KDim: 1})
        area_edge = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_rot = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim)
        cfl_w_limit = 4.0
        scalfac_exdiff = 6.0
        d_time = 2.0

        k = data_alloc.zero_field(grid, dims.KDim, dtype=gtx.int32)
        nlev = grid.num_levels

        for level in range(nlev):
            k[level] = level

        nrdmax = 5
        extra_diffu = True
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        return dict(
            vn=vn,
            geofac_rot=geofac_rot,
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=coeff_gradekin,
            z_ekinh=z_ekinh,
            vt=vt,
            f_e=f_e,
            c_lin_e=c_lin_e,
            z_w_con_c_full=z_w_con_c_full,
            vn_ie=vn_ie,
            ddqz_z_full_e=ddqz_z_full_e,
            levelmask=levelmask,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            k=k,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=d_time,
            extra_diffu=extra_diffu,
            nlev=nlev,
            nrdmax=nrdmax,
            ddt_vn_apc=ddt_vn_apc,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
