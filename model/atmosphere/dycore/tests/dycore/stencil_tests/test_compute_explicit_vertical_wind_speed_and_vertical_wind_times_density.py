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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    w_nnow: np.ndarray,
    ddt_w_adv_ntl1: np.ndarray,
    z_th_ddz_exner_c: np.ndarray,
    rho_ic: np.ndarray,
    w_concorr_c: np.ndarray,
    vwind_expl_wgt: np.ndarray,
    dtime: float,
    cpd: float,
) -> tuple[np.ndarray, np.ndarray]:
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, -1)
    z_w_expl = w_nnow + dtime * (ddt_w_adv_ntl1 - cpd * z_th_ddz_exner_c)
    z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)
    return (z_w_expl, z_contr_w_fl_l)


class TestComputeExplicitVerticalWindSpeedAndVerticalWindTimesDensity(stencil_tests.StencilTest):
    PROGRAM = compute_explicit_vertical_wind_speed_and_vertical_wind_times_density
    OUTPUTS = ("z_w_expl", "z_contr_w_fl_l")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        w_nnow: np.ndarray,
        ddt_w_adv_ntl1: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        rho_ic: np.ndarray,
        w_concorr_c: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        dtime: float,
        cpd: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        (
            z_w_expl,
            z_contr_w_fl_l,
        ) = compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
            connectivities,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            cpd=cpd,
        )
        return dict(z_w_expl=z_w_expl, z_contr_w_fl_l=z_contr_w_fl_l)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_nnow = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_w_adv_ntl1 = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_th_ddz_exner_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_w_expl = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_ic = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        w_concorr_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        vwind_expl_wgt = self.data_alloc.random_field(dims.CellDim, dtype=wpfloat)
        z_contr_w_fl_l = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        cpd = wpfloat("10.0")

        return dict(
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
