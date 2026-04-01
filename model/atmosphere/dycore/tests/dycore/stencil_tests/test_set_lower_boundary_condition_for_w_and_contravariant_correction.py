# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.set_lower_boundary_condition_for_w_and_contravariant_correction import (
    set_lower_boundary_condition_for_w_and_contravariant_correction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    w_concorr_c: np.ndarray,
    z_contr_w_fl_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w_nnew = w_concorr_c
    z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
    return (w_nnew, z_contr_w_fl_l)


class TestInitLowerBoundaryConditionForWAndContravariantCorrection(stencil_tests.StencilTest):
    PROGRAM = set_lower_boundary_condition_for_w_and_contravariant_correction
    OUTPUTS = ("w_nnew", "z_contr_w_fl_l")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        w_concorr_c: np.ndarray,
        z_contr_w_fl_l: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        (
            w_nnew,
            z_contr_w_fl_l,
        ) = set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
            connectivities, w_concorr_c, z_contr_w_fl_l
        )
        return dict(w_nnew=w_nnew, z_contr_w_fl_l=z_contr_w_fl_l)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_concorr_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_contr_w_fl_l = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        w_nnew = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            w_nnew=w_nnew,
            z_contr_w_fl_l=z_contr_w_fl_l,
            w_concorr_c=w_concorr_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
