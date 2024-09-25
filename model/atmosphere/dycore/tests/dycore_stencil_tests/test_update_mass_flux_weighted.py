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

from icon4py.model.atmosphere.dycore.update_mass_flux_weighted import update_mass_flux_weighted
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestUpdateMassFluxWeighted(StencilTest):
    PROGRAM = update_mass_flux_weighted
    OUTPUTS = ("mass_flx_ic",)

    @staticmethod
    def reference(
        grid,
        rho_ic: np.array,
        vwind_expl_wgt: np.array,
        vwind_impl_wgt: np.array,
        w_now: np.array,
        w_new: np.array,
        w_concorr_c: np.array,
        mass_flx_ic: np.array,
        r_nsubsteps: float,
        **kwargs,
    ) -> dict:
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
        mass_flx_ic = mass_flx_ic + (
            r_nsubsteps * rho_ic * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c)
        )
        return dict(mass_flx_ic=mass_flx_ic)

    @pytest.fixture
    def input_data(self, grid):
        r_nsubsteps = wpfloat("10.0")
        rho_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        vwind_expl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        vwind_impl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        w_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_new = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        mass_flx_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            rho_ic=rho_ic,
            vwind_expl_wgt=vwind_expl_wgt,
            vwind_impl_wgt=vwind_impl_wgt,
            w_now=w_now,
            w_new=w_new,
            w_concorr_c=w_concorr_c,
            mass_flx_ic=mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
