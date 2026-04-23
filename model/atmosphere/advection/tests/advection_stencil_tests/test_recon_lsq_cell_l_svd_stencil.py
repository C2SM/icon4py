# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.recon_lsq_cell_l_svd_stencil import (
    recon_lsq_cell_l_svd_stencil,
)
from icon4py.model.common.dimension import C2E2CDim, CECDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    reshape,
    zero_field,
)


class TestReconLsqCellLSvdStencil(StencilTest):
    PROGRAM = recon_lsq_cell_l_svd_stencil
    OUTPUTS = (
        "p_coeff_1_dsl",
        "p_coeff_2_dsl",
        "p_coeff_3_dsl",
    )

    @staticmethod
    def reference(
        grid,
        p_cc: np.array,
        lsq_pseudoinv_1: np.array,
        lsq_pseudoinv_2: np.array,
        **kwargs,
    ):
        c2e2c = grid.connectivities[C2E2CDim]
        p_cc_e = np.expand_dims(p_cc, axis=1)
        n_diff = p_cc[c2e2c] - p_cc_e
        lsq_pseudoinv_1 = reshape(lsq_pseudoinv_1, c2e2c.shape)
        lsq_pseudoinv_1 = np.expand_dims(lsq_pseudoinv_1, axis=-1)
        lsq_pseudoinv_2 = reshape(lsq_pseudoinv_2, c2e2c.shape)
        lsq_pseudoinv_2 = np.expand_dims(lsq_pseudoinv_2, axis=-1)
        p_coeff_1_dsl = p_cc
        p_coeff_2_dsl = np.sum(lsq_pseudoinv_1 * n_diff, axis=1)
        p_coeff_3_dsl = np.sum(lsq_pseudoinv_2 * n_diff, axis=1)
        return dict(
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
        )

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, CellDim, KDim)
        lsq_pseudoinv_1_field = as_1D_sparse_field(random_field(grid, CellDim, C2E2CDim), CECDim)
        lsq_pseudoinv_2_field = as_1D_sparse_field(random_field(grid, CellDim, C2E2CDim), CECDim)
        p_coeff_1_dsl = zero_field(grid, CellDim, KDim)
        p_coeff_2_dsl = zero_field(grid, CellDim, KDim)
        p_coeff_3_dsl = zero_field(grid, CellDim, KDim)
        return dict(
            p_cc=p_cc,
            lsq_pseudoinv_1=lsq_pseudoinv_1_field,
            lsq_pseudoinv_2=lsq_pseudoinv_2_field,
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
        )
