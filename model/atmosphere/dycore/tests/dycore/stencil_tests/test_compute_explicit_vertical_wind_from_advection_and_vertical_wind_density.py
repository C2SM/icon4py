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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
    connectivities: Mapping[gtx.Dimension, np.ndarray],
    w_nnow: np.ndarray,
    ddt_w_adv_ntl1: np.ndarray,
    ddt_w_adv_ntl2: np.ndarray,
    z_th_ddz_exner_c: np.ndarray,
    rho_ic: np.ndarray,
    w_concorr_c: np.ndarray,
    vwind_expl_wgt: np.ndarray,
    dtime: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    cpd: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_w_expl = w_nnow + dtime * (
        wgt_nnow_vel * ddt_w_adv_ntl1 + wgt_nnew_vel * ddt_w_adv_ntl2 - cpd * z_th_ddz_exner_c
    )
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
    z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
    return (z_w_expl, z_contr_w_fl_l)


class TestComputeExplicitVerticalWindFromAdvectionAndVerticalWindDensity(StencilTest):
    PROGRAM = compute_explicit_vertical_wind_from_advection_and_vertical_wind_density
    OUTPUTS = ("z_w_expl", "z_contr_w_fl_l")

    @static_reference
    def reference(
        grid: base.Grid,
        w_nnow: np.ndarray,
        ddt_w_adv_ntl1: np.ndarray,
        ddt_w_adv_ntl2: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        rho_ic: np.ndarray,
        w_concorr_c: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        dtime: float,
        wgt_nnow_vel: float,
        wgt_nnew_vel: float,
        cpd: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        (
            z_w_expl,
            z_contr_w_fl_l,
        ) = compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
            connectivities=connectivities,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
        )
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_nnow = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        ddt_w_adv_ntl1 = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        ddt_w_adv_ntl2 = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_th_ddz_exner_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_w_expl = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_ic = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        w_concorr_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        vwind_expl_wgt = self.data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat)
        z_contr_w_fl_l = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")
        wgt_nnow_vel = ta.wpfloat("8.0")
        wgt_nnew_vel = ta.wpfloat("9.0")
        cpd = ta.wpfloat("10.0")

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
