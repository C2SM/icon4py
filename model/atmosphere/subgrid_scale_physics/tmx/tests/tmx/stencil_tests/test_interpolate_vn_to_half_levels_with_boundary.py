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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_vn_to_half_levels_with_boundary import (
    interpolate_vn_to_half_levels_with_boundary,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def interpolate_vn_to_half_levels_with_boundary_numpy(
    *,
    vn: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq1_e_1: np.ndarray,
    wgtfacq1_e_2: np.ndarray,
    wgtfacq1_e_3: np.ndarray,
    wgtfacq_e_1: np.ndarray,
    wgtfacq_e_2: np.ndarray,
    wgtfacq_e_3: np.ndarray,
) -> np.ndarray:
    nlev = vn.shape[1]
    vn_ie = np.zeros((vn.shape[0], nlev + 1), dtype=vn.dtype)
    vn_ie[:, 0] = wgtfacq1_e_1 * vn[:, 0] + wgtfacq1_e_2 * vn[:, 1] + wgtfacq1_e_3 * vn[:, 2]
    vn_ie[:, 1:nlev] = (
        wgtfac_e[:, 1:nlev] * vn[:, 1:nlev] + (1.0 - wgtfac_e[:, 1:nlev]) * vn[:, 0 : nlev - 1]
    )
    vn_ie[:, nlev] = (
        wgtfacq_e_1 * vn[:, nlev - 1]
        + wgtfacq_e_2 * vn[:, nlev - 2]
        + wgtfacq_e_3 * vn[:, nlev - 3]
    )
    return vn_ie


class TestInterpolateVnToHalfLevelsWithBoundary(StencilTest):
    PROGRAM = interpolate_vn_to_half_levels_with_boundary
    OUTPUTS = ("vn_ie",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq1_e_1: np.ndarray,
        wgtfacq1_e_2: np.ndarray,
        wgtfacq1_e_3: np.ndarray,
        wgtfacq_e_1: np.ndarray,
        wgtfacq_e_2: np.ndarray,
        wgtfacq_e_3: np.ndarray,
        **kwargs,
    ) -> dict:
        vn_ie = interpolate_vn_to_half_levels_with_boundary_numpy(
            vn=vn,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e_1=wgtfacq1_e_1,
            wgtfacq1_e_2=wgtfacq1_e_2,
            wgtfacq1_e_3=wgtfacq1_e_3,
            wgtfacq_e_1=wgtfacq_e_1,
            wgtfacq_e_2=wgtfacq_e_2,
            wgtfacq_e_3=wgtfacq_e_3,
        )
        return dict(vn_ie=vn_ie)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        wgtfac_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )
        wgtfacq1_e_1 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        wgtfacq1_e_2 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        wgtfacq1_e_3 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_1 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_2 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_3 = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        vn_ie = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            vn=vn,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e_1=wgtfacq1_e_1,
            wgtfacq1_e_2=wgtfacq1_e_2,
            wgtfacq1_e_3=wgtfacq1_e_3,
            wgtfacq_e_1=wgtfacq_e_1,
            wgtfacq_e_2=wgtfacq_e_2,
            wgtfacq_e_3=wgtfacq_e_3,
            vn_ie=vn_ie,
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
