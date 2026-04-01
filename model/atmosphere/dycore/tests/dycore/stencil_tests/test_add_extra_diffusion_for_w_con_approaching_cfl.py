# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def add_extra_diffusion_for_w_con_approaching_cfl_numpy(
    connectivities: Mapping[gtx.Dimension, np.ndarray],
    cfl_clipping: np.ndarray,
    owner_mask: np.ndarray,
    z_w_con_c: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    ddt_w_adv: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    owner_mask = np.expand_dims(owner_mask, axis=-1)
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
        ),
        0,
    )

    c2e2cO = connectivities[dims.C2E2CODim]
    ddt_w_adv = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        ddt_w_adv
        + difcoef
        * area
        * np.sum(
            np.where(
                (c2e2cO != -1)[:, :, np.newaxis],
                w[c2e2cO] * geofac_n2s,
                0,
            ),
            axis=1,
        ),
        ddt_w_adv,
    )
    return ddt_w_adv


@pytest.mark.embedded_remap_error
class TestAddExtraDiffusionForWConApproachingCfl(StencilTest):
    PROGRAM = add_extra_diffusion_for_w_con_approaching_cfl
    OUTPUTS = ("ddt_w_adv",)

    @static_reference
    def reference(
        grid: base.Grid,
        cfl_clipping: np.ndarray,
        owner_mask: np.ndarray,
        z_w_con_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        w: np.ndarray,
        ddt_w_adv: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        ddt_w_adv = add_extra_diffusion_for_w_con_approaching_cfl_numpy(
            connectivities,
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

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        cfl_clipping = self.data_alloc.random_mask(dims.CellDim, dims.KDim)
        owner_mask = self.data_alloc.random_mask(dims.CellDim)
        z_w_con_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        ddqz_z_half = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        area = self.data_alloc.random_field(dims.CellDim, dtype=wpfloat)
        geofac_n2s = self.data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        w = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_w_adv = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        scalfac_exdiff = wpfloat("10.0")
        cfl_w_limit = vpfloat("3.0")
        dtime = wpfloat("2.0")

        return dict(
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
