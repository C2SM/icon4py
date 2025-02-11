# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


class TestComputeExplicitVerticalWindFromAdvectionAndVerticalWindDensity(StencilTest):
    PROGRAM = compute_explicit_vertical_wind_from_advection_and_vertical_wind_density
    OUTPUTS = ("z_w_expl", "z_contr_w_fl_l")

    @staticmethod
    def reference(
        grid,
        w_nnow: np.array,
        ddt_w_adv_ntl1: np.array,
        ddt_w_adv_ntl2: np.array,
        z_th_ddz_exner_c: np.array,
        rho_ic: np.array,
        w_concorr_c: np.array,
        vwind_expl_wgt: np.array,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        **kwargs,
    ) -> tuple[np.array]:
        z_w_expl = w_nnow + dtime * (
            wgt_nnow_vel * ddt_w_adv_ntl1 + wgt_nnew_vel * ddt_w_adv_ntl2 - cpd * z_th_ddz_exner_c
        )
        vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
        z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        w_nnow = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_w_adv_ntl1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddt_w_adv_ntl2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_th_ddz_exner_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_w_expl = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        vwind_expl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        z_contr_w_fl_l = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        wgt_nnow_vel = wpfloat("8.0")
        wgt_nnew_vel = wpfloat("9.0")
        cpd = wpfloat("10.0")

        return dict(
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
