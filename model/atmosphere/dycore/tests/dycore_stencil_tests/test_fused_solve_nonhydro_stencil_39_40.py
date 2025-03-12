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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.fused_solve_nonhydro_stencil_39_40 import (
    fused_solve_nonhydro_stencil_39_40,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing.helpers import StencilTest

from .test_compute_contravariant_correction_of_w import compute_contravariant_correction_of_w_numpy
from .test_compute_contravariant_correction_of_w_for_lower_boundary import (
    compute_contravariant_correction_of_w_for_lower_boundary_numpy,
)


def _fused_solve_nonhydro_stencil_39_40_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_bln_c_s,
    z_w_concorr_me,
    wgtfac_c,
    wgtfacq_c,
    vert_idx,
    nlev,
    nflatlev,
):
    w_concorr_c = np.where(
        (nflatlev < vert_idx) & (vert_idx < nlev),
        compute_contravariant_correction_of_w_numpy(
            connectivities, e_bln_c_s, z_w_concorr_me, wgtfac_c
        ),
        compute_contravariant_correction_of_w_for_lower_boundary_numpy(
            connectivities, e_bln_c_s, z_w_concorr_me, wgtfacq_c
        ),
    )

    w_concorr_c_res = np.zeros_like(w_concorr_c)
    w_concorr_c_res[:, -1] = w_concorr_c[:, -1]
    return w_concorr_c_res


class TestFusedSolveNonhydroStencil39To40(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_39_40
    OUTPUTS = ("w_concorr_c",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        e_bln_c_s: np.ndarray,
        z_w_concorr_me: np.ndarray,
        wgtfac_c: np.ndarray,
        wgtfacq_c: np.ndarray,
        vert_idx: np.ndarray,
        nlev: int,
        nflatlev: int,
        **kwargs: Any,
    ) -> dict:
        w_concorr_c_result = _fused_solve_nonhydro_stencil_39_40_numpy(
            connectivities, e_bln_c_s, z_w_concorr_me, wgtfac_c, wgtfacq_c, vert_idx, nlev, nflatlev
        )
        return dict(w_concorr_c=w_concorr_c_result)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict:
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim, dtype=ta.wpfloat)
        z_w_concorr_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        wgtfacq_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        w_concorr_c = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        vert_idx = data_alloc.index_field(grid, dims.KDim, dtype=gtx.int32)

        nlev = grid.num_levels
        nflatlev = 13

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            vert_idx=vert_idx,
            nlev=nlev,
            nflatlev=nflatlev,
            w_concorr_c=w_concorr_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )
