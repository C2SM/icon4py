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

import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ppm_all_face_values import (
    compute_ppm_all_face_values,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


class TestComputePpmAllFaceValues(helpers.StencilTest):
    PROGRAM = compute_ppm_all_face_values
    OUTPUTS = ("p_face",)
    MARKERS = (pytest.mark.requires_concat_where,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_cc: np.ndarray,
        p_cellhgt_mc_now: np.ndarray,
        p_face_in: np.ndarray,
        k: np.ndarray,
        slev: gtx.int32,
        elev: gtx.int32,
        slevp1: gtx.int32,
        elevp1: gtx.int32,
        **kwargs: Any,
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
    def input_data(self, grid: base.BaseGrid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_cellhgt_mc_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_face_in = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_face = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        k = data_alloc.index_field(grid, dims.KDim)
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
