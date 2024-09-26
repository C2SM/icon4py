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

from icon4py.model.atmosphere.dycore.compute_advective_vertical_wind_tendency import (
    compute_advective_vertical_wind_tendency,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_advective_vertical_wind_tendency_numpy(
    z_w_con_c: np.array,
    w: np.array,
    coeff1_dwdz: np.array,
    coeff2_dwdz: np.array,
    **kwargs,
) -> np.array:
    ddt_w_adv = np.zeros_like(coeff1_dwdz)
    ddt_w_adv[:, 1:] = -z_w_con_c[:, 1:] * (
        w[:, :-2] * coeff1_dwdz[:, 1:]
        - w[:, 2:] * coeff2_dwdz[:, 1:]
        + w[:, 1:-1] * (coeff2_dwdz[:, 1:] - coeff1_dwdz[:, 1:])
    )
    return ddt_w_adv


class TestComputeAdvectiveVerticalWindTendency(StencilTest):
    PROGRAM = compute_advective_vertical_wind_tendency
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid,
        z_w_con_c: np.array,
        w: np.array,
        coeff1_dwdz: np.array,
        coeff2_dwdz: np.array,
        **kwargs,
    ) -> dict:
        ddt_w_adv = compute_advective_vertical_wind_tendency_numpy(
            z_w_con_c, w, coeff1_dwdz, coeff2_dwdz
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat)
        coeff1_dwdz = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        coeff2_dwdz = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddt_w_adv = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            z_w_con_c=z_w_con_c,
            w=w,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            ddt_w_adv=ddt_w_adv,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
