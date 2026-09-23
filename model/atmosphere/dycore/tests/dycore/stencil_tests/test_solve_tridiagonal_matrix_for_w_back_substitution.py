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

from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def solve_tridiagonal_matrix_for_w_back_substitution_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    z_q: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    # the surface half level is set elsewhere; the backward sweep runs over [1, nlev)
    w_new = np.copy(w)
    nlev = w.shape[1] - 1

    w_new[:, nlev - 1] = w[:, nlev - 1]
    for k in reversed(range(1, nlev - 1)):
        w_new[:, k] = w[:, k] + w_new[:, k + 1] * z_q[:, k]
    w_new[:, 0] = w[:, 0]
    return w_new


class TestSolveTridiagonalMatrixForWBackSubstitution(stencil_tests.StencilTest):
    PROGRAM = solve_tridiagonal_matrix_for_w_back_substitution
    OUTPUTS = ("w",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        z_q: np.ndarray,
        w: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        w_new = solve_tridiagonal_matrix_for_w_back_substitution_numpy(connectivities, z_q=z_q, w=w)
        return dict(w=w_new)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_q = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)
        h_start = 0
        h_end = gtx.int32(grid.num_cells)
        v_start = 1
        v_end = gtx.int32(grid.num_levels)
        return dict(
            z_q=z_q,
            w=w,
            horizontal_start=h_start,
            horizontal_end=h_end,
            vertical_start=v_start,
            vertical_end=v_end,
        )
