# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w import (
    compute_contravariant_correction_of_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def compute_contravariant_correction_of_w_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    e_bln_c_s: np.ndarray,
    z_w_concorr_me: np.ndarray,
    wgtfac_c: np.ndarray,
) -> np.ndarray:
    c2e = connectivities[dims.C2E]

    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_mc_m0 = np.sum(e_bln_c_s * z_w_concorr_me[c2e], axis=1)
    _nlev = z_w_concorr_mc_m0.shape[1]
    w_concorr_c = np.zeros((z_w_concorr_mc_m0.shape[0], _nlev + 1))
    _w = wgtfac_c[:, 1:_nlev]
    w_concorr_c[:, 1:_nlev] = (
        _w * z_w_concorr_mc_m0[:, 1:_nlev] + (1.0 - _w) * z_w_concorr_mc_m0[:, 0 : _nlev - 1]
    )
    w_concorr_c[:, 0] = 0
    return w_concorr_c


class TestComputeContravariantCorrectionOfW(stencil_tests.StencilTest):
    PROGRAM = compute_contravariant_correction_of_w
    OUTPUTS = ("w_concorr_c",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        e_bln_c_s: np.ndarray,
        z_w_concorr_me: np.ndarray,
        wgtfac_c: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        w_concorr_c = compute_contravariant_correction_of_w_numpy(
            connectivities, e_bln_c_s, z_w_concorr_me, wgtfac_c
        )
        return dict(w_concorr_c=w_concorr_c)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=wpfloat)
        z_w_concorr_me = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)
        w_concorr_c = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w_concorr_c=w_concorr_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
