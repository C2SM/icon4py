# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_01 import face_val_ppm_stencil_01
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    Output,
    StencilTest,
    _shape,
    random_field,
    zero_field,
)


class TestFaceValPpmStencil01(StencilTest):
    PROGRAM = face_val_ppm_stencil_01
    OUTPUTS = (
        Output(
            "z_slope", refslice=(slice(None), slice(None, -1)), gtslice=(slice(None), slice(1, -1))
        ),
    )

    @staticmethod
    def reference(
        grid, p_cc: np.array, p_cellhgt_mc_now: np.array, k: np.array, elev: int32, **kwargs
    ):
        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 2:] - p_cc[:, 1:-1]) / (p_cellhgt_mc_now[:, 2:] + p_cellhgt_mc_now[:, 1:-1])
        z_slope_a = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 2:])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 2:]) * zfac_m1
        )

        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 1:-1] - p_cc[:, 1:-1]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1]
        )
        z_slope_b = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 1:-1]) * zfac_m1
        )

        z_slope = np.where(k[1:-1] < elev, z_slope_a, z_slope_b)
        return dict(z_slope=z_slope)

    @pytest.fixture
    def input_data(self, grid):
        z_slope = zero_field(grid, dims.CellDim, dims.KDim)
        p_cc = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        p_cellhgt_mc_now = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        k = as_field(
            (dims.KDim,),
            np.arange(0, _shape(grid, dims.KDim, extend={dims.KDim: 1})[0], dtype=int32),
        )
        elev = k[-2].as_scalar()
        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            k=k,
            elev=elev,
            z_slope=z_slope,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=1,
            vertical_end=grid.num_levels - 1,
        )
