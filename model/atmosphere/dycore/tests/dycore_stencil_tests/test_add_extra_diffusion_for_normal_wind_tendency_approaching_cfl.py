# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.common.dimension import (
    CellDim,
    E2C2EODim,
    E2CDim,
    E2VDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask
from icon4py.model.common.type_alias import vpfloat, wpfloat


def add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
    grid,
    levelmask: np.array,
    c_lin_e: np.array,
    z_w_con_c_full: np.array,
    ddqz_z_full_e: np.array,
    area_edge: np.array,
    tangent_orientation: np.array,
    inv_primal_edge_length: np.array,
    zeta: np.array,
    geofac_grdiv: np.array,
    vn: np.array,
    ddt_vn_apc: np.array,
    cfl_w_limit,
    scalfac_exdiff,
    dtime,
) -> np.array:
    w_con_e = np.zeros_like(vn)
    difcoef = np.zeros_like(vn)

    levelmask_offset_0 = levelmask[:-1]
    levelmask_offset_1 = levelmask[1:]

    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    area_edge = np.expand_dims(area_edge, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    w_con_e = np.where(
        (levelmask_offset_0) | (levelmask_offset_1),
        np.sum(
            np.where(
                (grid.connectivities[E2CDim] != -1)[:, :, np.newaxis],
                c_lin_e * z_w_con_c_full[grid.connectivities[E2CDim]],
                0,
            ),
            axis=1,
        ),
        w_con_e,
    )
    difcoef = np.where(
        ((levelmask_offset_0) | (levelmask_offset_1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(w_con_e) * dtime / ddqz_z_full_e - cfl_w_limit * dtime,
        ),
        difcoef,
    )
    ddt_vn_apc = np.where(
        ((levelmask_offset_0) | (levelmask_offset_1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        ddt_vn_apc
        + difcoef
        * area_edge
        * (
            np.sum(
                np.where(
                    (grid.connectivities[E2C2EODim] != -1)[:, :, np.newaxis],
                    geofac_grdiv * vn[grid.connectivities[E2C2EODim]],
                    0,
                ),
                axis=1,
            )
            + tangent_orientation
            * inv_primal_edge_length
            * (zeta[grid.connectivities[E2VDim]][:, 1] - zeta[grid.connectivities[E2VDim]][:, 0])
        ),
        ddt_vn_apc,
    )
    return ddt_vn_apc


class TestAddExtraDiffusionForWnApproachingCfl(StencilTest):
    PROGRAM = add_extra_diffusion_for_normal_wind_tendency_approaching_cfl
    OUTPUTS = ("ddt_vn_apc",)

    @pytest.fixture
    def input_data(self, grid):
        levelmask = random_mask(grid, KDim, extend={KDim: 1})
        c_lin_e = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        z_w_con_c_full = random_field(grid, CellDim, KDim, dtype=vpfloat)
        ddqz_z_full_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        area_edge = random_field(grid, EdgeDim)
        tangent_orientation = random_field(grid, EdgeDim)
        inv_primal_edge_length = random_field(grid, EdgeDim)
        zeta = random_field(grid, VertexDim, KDim, dtype=vpfloat)
        geofac_grdiv = random_field(grid, EdgeDim, E2C2EODim)
        vn = random_field(grid, EdgeDim, KDim)
        ddt_vn_apc = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        cfl_w_limit = vpfloat("4.0")
        scalfac_exdiff = 6.0
        dtime = 2.0
        return dict(
            levelmask=levelmask,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            zeta=zeta,
            geofac_grdiv=geofac_grdiv,
            vn=vn,
            ddt_vn_apc=ddt_vn_apc,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            c_lin_e=c_lin_e,
            z_w_con_c_full=z_w_con_c_full,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        grid,
        levelmask: np.array,
        c_lin_e: np.array,
        z_w_con_c_full: np.array,
        ddqz_z_full_e: np.array,
        area_edge: np.array,
        tangent_orientation: np.array,
        inv_primal_edge_length: np.array,
        zeta: np.array,
        geofac_grdiv: np.array,
        vn: np.array,
        ddt_vn_apc: np.array,
        cfl_w_limit,
        scalfac_exdiff,
        dtime,
        **kwargs,
    ) -> dict:
        ddt_vn_apc = add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
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
            dtime,
        )
        return dict(ddt_vn_apc=ddt_vn_apc)
