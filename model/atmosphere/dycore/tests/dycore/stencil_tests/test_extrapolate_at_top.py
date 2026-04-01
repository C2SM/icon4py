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

from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import extrapolate_at_top
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def extrapolate_at_top_numpy(wgtfacq_e: np.ndarray, vn: np.ndarray) -> np.ndarray:
    vn_k_minus_1 = vn[:, -1]
    vn_k_minus_2 = vn[:, -2]
    vn_k_minus_3 = vn[:, -3]
    wgtfacq_e_k_minus_1 = wgtfacq_e[:, -1]
    wgtfacq_e_k_minus_2 = wgtfacq_e[:, -2]
    wgtfacq_e_k_minus_3 = wgtfacq_e[:, -3]
    shape = vn.shape
    vn_ie = np.zeros((shape[0], shape[1] + 1), dtype=vpfloat)
    vn_ie[:, -1] = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )
    return vn_ie


class TestExtrapolateAtTop(stencil_tests.StencilTest):
    PROGRAM = extrapolate_at_top
    OUTPUTS = ("vn_ie",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        wgtfacq_e: np.ndarray,
        vn: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vn_ie = extrapolate_at_top_numpy(wgtfacq_e, vn)
        return dict(vn_ie=vn_ie)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfacq_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vn_ie = self.data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, dtype=vpfloat, extend={dims.KDim: 1}
        )

        return dict(
            wgtfacq_e=wgtfacq_e,
            vn=vn,
            vn_ie=vn_ie,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(grid.num_levels),
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
