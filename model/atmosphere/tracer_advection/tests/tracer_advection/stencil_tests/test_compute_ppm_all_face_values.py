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

from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm import (
    compute_ppm_all_face_values,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


@pytest.mark.uses_concat_where
class TestComputePpmAllFaceValues(stencil_tests.StencilTest):
    PROGRAM = compute_ppm_all_face_values
    OUTPUTS = ("p_face",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        p_cc: np.ndarray,
        p_cellhgt_mc_now: np.ndarray,
        p_face_in: np.ndarray,
        p_face: np.ndarray,
        start_lvl: gtx.int32,
        end_lvl: gtx.int32,
        start_lvl_plus1: gtx.int32,
        end_lvl_plus1: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        p_face_a = p_face_in.copy()
        p_face_a[:, 1:-1] = p_cc[:, 1:] * (
            1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])
        ) + (p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])) * (
            (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1]
        )

        p_face = p_face.copy()
        p_face[:, :-1] = p_face_in[:, :-1]
        p_face[:, start_lvl_plus1] = p_face_a[:, start_lvl_plus1]
        p_face[:, end_lvl] = p_face_a[:, end_lvl]
        p_face[:, start_lvl] = p_cc[:, start_lvl]
        p_face[:, end_lvl_plus1] = p_cc[:, end_lvl_plus1 - 1]
        return dict(p_face=p_face)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_cellhgt_mc_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_face_in = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        p_face = data_alloc.zero_field(dims.CellDim, dims.KHalfDim)
        start_lvl = gtx.int32(1)
        start_lvl_plus1 = gtx.int32(2)
        end_lvl = grid.num_levels - 2
        end_lvl_plus1 = gtx.int32(end_lvl + 1)

        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            p_face_in=p_face_in,
            p_face=p_face,
            start_lvl=start_lvl,
            end_lvl=end_lvl,
            start_lvl_plus1=start_lvl_plus1,
            end_lvl_plus1=end_lvl_plus1,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
