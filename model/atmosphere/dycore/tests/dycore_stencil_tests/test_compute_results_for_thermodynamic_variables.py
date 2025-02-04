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

from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


class TestComputeResultsForThermodynamicVariables(StencilTest):
    PROGRAM = compute_results_for_thermodynamic_variables
    OUTPUTS = ("rho_new", "exner_new", "theta_v_new")

    @staticmethod
    def reference(
        grid,
        z_rho_expl: np.array,
        vwind_impl_wgt: np.array,
        inv_ddqz_z_full: np.array,
        rho_ic: np.array,
        w: np.array,
        z_exner_expl: np.array,
        exner_ref_mc: np.array,
        z_alpha: np.array,
        z_beta: np.array,
        rho_now: np.array,
        theta_v_now: np.array,
        exner_now: np.array,
        dtime,
        cvd_o_rd,
        **kwargs,
    ) -> dict:
        rho_ic_offset_1 = rho_ic[:, 1:]
        w_offset_0 = w[:, :-1]
        w_offset_1 = w[:, 1:]
        z_alpha_offset_1 = z_alpha[:, 1:]
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=1)
        rho_new = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full * (
            rho_ic[:, :-1] * w_offset_0 - rho_ic_offset_1 * w_offset_1
        )
        exner_new = (
            z_exner_expl
            + exner_ref_mc
            - z_beta * (z_alpha[:, :-1] * w_offset_0 - z_alpha_offset_1 * w_offset_1)
        )
        theta_v_new = (
            rho_now * theta_v_now * ((exner_new / exner_now - 1.0) * cvd_o_rd + 1.0) / rho_new
        )
        return dict(rho_new=rho_new, exner_new=exner_new, theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, grid):
        z_rho_expl = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        vwind_impl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        inv_ddqz_z_full = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        rho_ic = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat)
        z_exner_expl = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_ref_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_alpha = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat)
        z_beta = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        rho_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        cvd_o_rd = wpfloat("9.0")

        return dict(
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_now=exner_now,
            rho_new=rho_new,
            exner_new=exner_new,
            theta_v_new=theta_v_new,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
