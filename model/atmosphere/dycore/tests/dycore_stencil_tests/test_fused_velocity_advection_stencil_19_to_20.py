# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_19_to_20 import (
    fused_velocity_advection_stencil_19_to_20,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    random_mask,
    zero_field,
)

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

    @staticmethod
    def reference(
        grid,
        vn,
        geofac_rot,
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        levelmask,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        geofac_grdiv,
        k,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        extra_diffu,
        nlev,
        nrdmax,
        ddt_vn_apc,
        **kwargs,
    ):
        ddt_vn_apc_cp = ddt_vn_apc.copy()
        zeta = mo_math_divrot_rot_vertex_ri_dsl_numpy(grid, vn, geofac_rot)

        coeff_gradekin = np.reshape(coeff_gradekin, (grid.num_edges, 2))

        ddt_vn_apc = compute_advective_normal_wind_tendency_numpy(
            grid,
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
            grid,
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
    def input_data(self, grid):
        z_kin_hor_e = random_field(grid, dims.EdgeDim, dims.KDim)
        coeff_gradekin = random_field(grid, dims.EdgeDim, dims.E2CDim)
        coeff_gradekin_new = as_1D_sparse_field(coeff_gradekin, dims.ECDim)
        z_ekinh = random_field(grid, dims.CellDim, dims.KDim)
        vt = random_field(grid, dims.EdgeDim, dims.KDim)
        f_e = random_field(grid, dims.EdgeDim)
        c_lin_e = random_field(grid, dims.EdgeDim, dims.E2CDim)
        z_w_con_c_full = random_field(grid, dims.CellDim, dims.KDim)
        vn_ie = random_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        ddqz_z_full_e = random_field(grid, dims.EdgeDim, dims.KDim)
        ddt_vn_apc = zero_field(grid, dims.EdgeDim, dims.KDim)
        levelmask = random_mask(grid, dims.KDim, extend={dims.KDim: 1})
        area_edge = random_field(grid, dims.EdgeDim)
        tangent_orientation = random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = random_field(grid, dims.EdgeDim)
        geofac_grdiv = random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        vn = random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_rot = random_field(grid, dims.VertexDim, dims.V2EDim)
        cfl_w_limit = 4.0
        scalfac_exdiff = 6.0
        d_time = 2.0

        k = zero_field(grid, dims.KDim, dtype=int32)
        nlev = grid.num_levels

        for level in range(nlev):
            k[level] = level

        nrdmax = 5
        extra_diffu = True
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = (
            grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
            if hasattr(grid, "start_index")
            else 0
        )

        return dict(
            vn=vn,
            geofac_rot=geofac_rot,
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=coeff_gradekin_new,
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
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
