# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.upwind_hflux_miura_stencil_01 import (
    upwind_hflux_miura_stencil_01,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    constant_field,
    random_field,
    zero_field,
)


class TestUpwindHfluxMiuraStencil01(StencilTest):
    PROGRAM = upwind_hflux_miura_stencil_01
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        z_lsq_coeff_1: np.array,
        z_lsq_coeff_2: np.array,
        z_lsq_coeff_3: np.array,
        distv_bary_1: np.array,
        distv_bary_2: np.array,
        p_mass_flx_e: np.array,
        cell_rel_idx_dsl: np.array,
        **kwargs,
    ):
        e2c = grid.connectivities[dims.E2CDim]
        z_lsq_coeff_1_e2c = z_lsq_coeff_1[e2c]
        z_lsq_coeff_2_e2c = z_lsq_coeff_2[e2c]
        z_lsq_coeff_3_e2c = z_lsq_coeff_3[e2c]

        p_out_e = (
            np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_1_e2c[:, 1],
                z_lsq_coeff_1_e2c[:, 0],
            )
            + distv_bary_1
            * np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_2_e2c[:, 1],
                z_lsq_coeff_2_e2c[:, 0],
            )
            + distv_bary_2
            * np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_3_e2c[:, 1],
                z_lsq_coeff_3_e2c[:, 0],
            )
        ) * p_mass_flx_e

        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid):
        z_lsq_coeff_1 = random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_2 = random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_3 = random_field(grid, dims.CellDim, dims.KDim)
        distv_bary_1 = random_field(grid, dims.EdgeDim, dims.KDim)
        distv_bary_2 = random_field(grid, dims.EdgeDim, dims.KDim)
        p_mass_flx_e = random_field(grid, dims.EdgeDim, dims.KDim)
        cell_rel_idx_dsl = constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=int32)
        p_out_e = zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_lsq_coeff_1=z_lsq_coeff_1,
            z_lsq_coeff_2=z_lsq_coeff_2,
            z_lsq_coeff_3=z_lsq_coeff_3,
            distv_bary_1=distv_bary_1,
            distv_bary_2=distv_bary_2,
            p_mass_flx_e=p_mass_flx_e,
            cell_rel_idx_dsl=cell_rel_idx_dsl,
            p_out_e=p_out_e,
        )
