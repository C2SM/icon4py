# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.common import dimension as dims
import numpy as xp
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask
from icon4py.model.common.type_alias import vpfloat, wpfloat


def add_extra_diffusion_for_w_con_approaching_cfl_numpy(
    grid,
    levmask: xp.array,
    cfl_clipping: xp.array,
    owner_mask: xp.array,
    z_w_con_c: xp.array,
    ddqz_z_half: xp.array,
    area: xp.array,
    geofac_n2s: xp.array,
    w: xp.array,
    ddt_w_adv: xp.array,
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
) -> xp.array:
    levmask = xp.expand_dims(levmask, axis=0)
    owner_mask = xp.expand_dims(owner_mask, axis=-1)
    area = xp.expand_dims(area, axis=-1)
    geofac_n2s = xp.expand_dims(geofac_n2s, axis=-1)

    difcoef = xp.where(
        (levmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * xp.minimum(
            0.85 - cfl_w_limit * dtime,
            xp.abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
        ),
        0,
    )

    ddt_w_adv = xp.where(
        (levmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
        ddt_w_adv
        + difcoef
        * area
        * xp.sum(
            xp.where(
                (xp.asarray(grid.connectivities[dims.C2E2CODim]) != -1)[:, :, xp.newaxis],
                w[xp.asarray(grid.connectivities[dims.C2E2CODim])] * geofac_n2s,
                0,
            ),
            axis=1,
        ),
        ddt_w_adv,
    )
    return ddt_w_adv


class TestAddExtraDiffusionForWConApproachingCfl(StencilTest):
    PROGRAM = add_extra_diffusion_for_w_con_approaching_cfl
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid,
        levmask: xp.array,
        cfl_clipping: xp.array,
        owner_mask: xp.array,
        z_w_con_c: xp.array,
        ddqz_z_half: xp.array,
        area: xp.array,
        geofac_n2s: xp.array,
        w: xp.array,
        ddt_w_adv: xp.array,
        scalfac_exdiff: wpfloat,
        cfl_w_limit: wpfloat,
        dtime: wpfloat,
        **kwargs,
    ):
        ddt_w_adv = add_extra_diffusion_for_w_con_approaching_cfl_numpy(
            grid,
            levmask,
            cfl_clipping,
            owner_mask,
            z_w_con_c,
            ddqz_z_half,
            area,
            geofac_n2s,
            w,
            ddt_w_adv,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        levmask = random_mask(grid, dims.KDim)
        cfl_clipping = random_mask(grid, dims.CellDim, dims.KDim)
        owner_mask = random_mask(grid, dims.CellDim)
        z_w_con_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        area = random_field(grid, dims.CellDim, dtype=wpfloat)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_w_adv = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        scalfac_exdiff = wpfloat("10.0")
        cfl_w_limit = vpfloat("3.0")
        dtime = wpfloat("2.0")

        return dict(
            levmask=levmask,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            z_w_con_c=z_w_con_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            w=w,
            ddt_w_adv=ddt_w_adv,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
