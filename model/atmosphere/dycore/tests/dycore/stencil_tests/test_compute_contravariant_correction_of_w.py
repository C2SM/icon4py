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

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w import (
    compute_contravariant_correction_of_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_contravariant_correction_of_w_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_bln_c_s: np.ndarray,
    z_w_concorr_me: np.ndarray,
    wgtfac_c: np.ndarray,
) -> np.ndarray:
    c2e = connectivities[dims.C2EDim]

    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_me_offset_1 = np.roll(z_w_concorr_me, shift=1, axis=1)
    z_w_concorr_mc_m0 = np.sum(e_bln_c_s * z_w_concorr_me[c2e], axis=1)
    z_w_concorr_mc_m1 = np.sum(e_bln_c_s * z_w_concorr_me_offset_1[c2e], axis=1)
    contravariant_correction_at_cells_on_half_levels = (
        wgtfac_c * z_w_concorr_mc_m0 + (1.0 - wgtfac_c) * z_w_concorr_mc_m1
    )
    contravariant_correction_at_cells_on_half_levels[:, 0] = 0
    return contravariant_correction_at_cells_on_half_levels


class TestComputeContravariantCorrectionOfW(StencilTest):
    PROGRAM = compute_contravariant_correction_of_w
    OUTPUTS = ("contravariant_correction_at_cells_on_half_levels",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        e_bln_c_s: np.ndarray,
        z_w_concorr_me: np.ndarray,
        wgtfac_c: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        contravariant_correction_at_cells_on_half_levels = (
            compute_contravariant_correction_of_w_numpy(
                connectivities, e_bln_c_s, z_w_concorr_me, wgtfac_c
            )
        )
        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_bln_c_s = random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat)
        z_w_concorr_me = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        contravariant_correction_at_cells_on_half_levels = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
