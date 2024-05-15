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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_01 import (
    upwind_hflux_miura_cycl_stencil_01,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)


class TestUpwindHfluxMiuraCyclStencil01(StencilTest):
    PROGRAM = upwind_hflux_miura_cycl_stencil_01
    OUTPUTS = ("z_tracer_mflx_dsl",)

    @staticmethod
    def reference(
        grid,
        z_lsq_coeff_1_dsl: np.array,
        z_lsq_coeff_2_dsl: np.array,
        z_lsq_coeff_3_dsl: np.array,
        distv_bary_1: np.array,
        distv_bary_2: np.array,
        p_mass_flx_e: np.array,
        cell_rel_idx_dsl: np.array,
        **kwargs,
    ):
        e2c = grid.connectivities[E2CDim]
        z_lsq_coeff_1_dsl_e2c = z_lsq_coeff_1_dsl[e2c]
        z_lsq_coeff_2_dsl_e2c = z_lsq_coeff_2_dsl[e2c]
        z_lsq_coeff_3_dsl_e2c = z_lsq_coeff_3_dsl[e2c]

        z_tracer_mflx_dsl = (
            np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_1_dsl_e2c[:, 1],
                z_lsq_coeff_1_dsl_e2c[:, 0],
            )
            + distv_bary_1
            * np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_2_dsl_e2c[:, 1],
                z_lsq_coeff_2_dsl_e2c[:, 0],
            )
            + distv_bary_2
            * np.where(
                cell_rel_idx_dsl == int32(1),
                z_lsq_coeff_3_dsl_e2c[:, 1],
                z_lsq_coeff_3_dsl_e2c[:, 0],
            )
        ) * p_mass_flx_e

        return dict(z_tracer_mflx_dsl=z_tracer_mflx_dsl)

    @pytest.fixture
    def input_data(self, grid):
        z_lsq_coeff_1_dsl = random_field(grid, CellDim, KDim)
        z_lsq_coeff_2_dsl = random_field(grid, CellDim, KDim)
        z_lsq_coeff_3_dsl = random_field(grid, CellDim, KDim)
        distv_bary_1 = random_field(grid, EdgeDim, KDim)
        distv_bary_2 = random_field(grid, EdgeDim, KDim)
        p_mass_flx_e = random_field(grid, EdgeDim, KDim)
        cell_rel_idx_dsl = random_mask(grid, EdgeDim, KDim, dtype=int32)
        z_tracer_mflx_dsl = zero_field(grid, EdgeDim, KDim)
        return dict(
            z_lsq_coeff_1_dsl=z_lsq_coeff_1_dsl,
            z_lsq_coeff_2_dsl=z_lsq_coeff_2_dsl,
            z_lsq_coeff_3_dsl=z_lsq_coeff_3_dsl,
            distv_bary_1=distv_bary_1,
            distv_bary_2=distv_bary_2,
            p_mass_flx_e=p_mass_flx_e,
            cell_rel_idx_dsl=cell_rel_idx_dsl,
            z_tracer_mflx_dsl=z_tracer_mflx_dsl,
        )
