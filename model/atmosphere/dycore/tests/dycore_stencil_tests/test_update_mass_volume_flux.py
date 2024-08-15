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

from icon4py.model.atmosphere.dycore.update_mass_volume_flux import update_mass_volume_flux
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import wpfloat


class TestUpdateMassVolumeFlux(StencilTest):
    PROGRAM = update_mass_volume_flux
    OUTPUTS = (
        "mass_flx_ic",
        "vol_flx_ic",
    )

    @staticmethod
    def reference(
        grid,
        z_contr_w_fl_l: np.array,
        rho_ic: np.array,
        vwind_impl_wgt: np.array,
        w: np.array,
        mass_flx_ic: np.array,
        vol_flx_ic: np.array,
        r_nsubsteps,
        **kwargs,
    ) -> dict:
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
        z_a = r_nsubsteps * (z_contr_w_fl_l + rho_ic * vwind_impl_wgt * w)
        mass_flx_ic = mass_flx_ic + z_a
        vol_flx_ic = vol_flx_ic + z_a / rho_ic
        return dict(mass_flx_ic=mass_flx_ic, vol_flx_ic=vol_flx_ic)

    @pytest.fixture
    def input_data(self, grid):
        z_contr_w_fl_l = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        vwind_impl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        mass_flx_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        vol_flx_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        r_nsubsteps = 7.0

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
