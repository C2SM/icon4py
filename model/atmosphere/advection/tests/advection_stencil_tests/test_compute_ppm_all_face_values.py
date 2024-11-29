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
from gt4py.next import as_field

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ppm_all_face_values import (
    compute_ppm_all_face_values,
)
from icon4py.model.common import dimension as dims


class TestComputePpmAllFaceValues(helpers.StencilTest):
    PROGRAM = compute_ppm_all_face_values
    OUTPUTS = ("p_face",)

    @staticmethod
    def reference(
        grid,
        p_cc: np.array,
        p_cellhgt_mc_now: np.array,
        p_face_in: np.array,
        k: np.array,
        slev: gtx.int32,
        elev: gtx.int32,
        slevp1: gtx.int32,
        elevp1: gtx.int32,
        **kwargs,
    ) -> dict:
        p_face_a = p_face_in
        p_face_a[:, 1:] = p_cc[:, 1:] * (
            1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])
        ) + (p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])) * (
            (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1]
        )

        p_face = np.where((k == slevp1) | (k == elev), p_face_a, p_face_in)
        p_face = np.where((k == slev), p_cc, p_face)
        p_face[:, 1:] = np.where((k[1:] == elevp1), p_cc[:, :-1], p_face[:, 1:])
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_cellhgt_mc_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_face_in = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_face = helpers.zero_field(grid, dims.CellDim, dims.KDim)

        k = as_field(
            (dims.KDim,), np.arange(0, helpers._shape(grid, dims.KDim)[0], dtype=gtx.int32)
        )
        slev = gtx.int32(1)
        slevp1 = gtx.int32(2)
        elev = gtx.int32(k[-3].as_scalar())
        elevp1 = elev + gtx.int32(1)

        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            p_face_in=p_face_in,
            p_face=p_face,
            k=k,
            slev=slev,
            elev=elev,
            slevp1=slevp1,
            elevp1=elevp1,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
