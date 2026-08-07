# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy(
    *,
    w: np.ndarray,
    w_concorr_c: np.ndarray,
    ddqz_z_half: np.ndarray,
    rho_now: np.ndarray,
    rho_var: np.ndarray,
    theta_now: np.ndarray,
    theta_var: np.ndarray,
    wgtfac_c: np.ndarray,
    theta_ref_mc: np.ndarray,
    vwind_expl_wgt: np.ndarray,
    exner_pr: np.ndarray,
    d_exner_dz_ref_ic: np.ndarray,
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
) -> tuple[np.ndarray, ...]:
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)
    nlev = rho_now.shape[1]
    lo, hi = slice(0, nlev - 1), slice(1, nlev)  # model levels k-1 and k, for half level k

    def half(a: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0], nlev + 1))
        out[:, 1:nlev] = a
        return out

    w_ = wgtfac_c[:, hi]
    z_w_backtraj = -(w[:, hi] - w_concorr_c[:, hi]) * dtime * 0.5 / ddqz_z_half[:, hi]
    z_rho_tavg_m1 = wgt_nnow_rth * rho_now[:, lo] + wgt_nnew_rth * rho_var[:, lo]
    z_theta_tavg_m1 = wgt_nnow_rth * theta_now[:, lo] + wgt_nnew_rth * theta_var[:, lo]
    z_rho_tavg = wgt_nnow_rth * rho_now[:, hi] + wgt_nnew_rth * rho_var[:, hi]
    z_theta_tavg = wgt_nnow_rth * theta_now[:, hi] + wgt_nnew_rth * theta_var[:, hi]

    rho_ic = half(
        w_ * z_rho_tavg + (1 - w_) * z_rho_tavg_m1 + z_w_backtraj * (z_rho_tavg_m1 - z_rho_tavg)
    )
    z_theta_v_pr_mc_m1 = z_theta_tavg_m1 - theta_ref_mc[:, lo]
    z_theta_v_pr_mc = z_theta_tavg - theta_ref_mc[:, hi]
    z_theta_v_pr_ic = half(w_ * z_theta_v_pr_mc + (1 - w_) * z_theta_v_pr_mc_m1)
    theta_v_ic = half(
        w_ * z_theta_tavg
        + (1 - w_) * z_theta_tavg_m1
        + z_w_backtraj * (z_theta_tavg_m1 - z_theta_tavg)
    )
    z_th_ddz_exner_c = half(
        vwind_expl_wgt
        * theta_v_ic[:, hi]
        * (exner_pr[:, lo] - exner_pr[:, hi])
        / ddqz_z_half[:, hi]
        + z_theta_v_pr_ic[:, hi] * d_exner_dz_ref_ic[:, hi]
    )
    return (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c)


class TestComputeRhoVirtualPotentialTemperaturesAndPressureGradient(StencilTest):
    PROGRAM = compute_rho_virtual_potential_temperatures_and_pressure_gradient
    OUTPUTS = ("rho_ic", "z_theta_v_pr_ic", "theta_v_ic", "z_th_ddz_exner_c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        w: np.ndarray,
        w_concorr_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        rho_now: np.ndarray,
        rho_var: np.ndarray,
        theta_now: np.ndarray,
        theta_var: np.ndarray,
        wgtfac_c: np.ndarray,
        theta_ref_mc: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        exner_pr: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        dtime: ta.wpfloat,
        wgt_nnow_rth: ta.wpfloat,
        wgt_nnew_rth: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        (
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
        ) = compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy(
            w=w,
            w_concorr_c=w_concorr_c,
            ddqz_z_half=ddqz_z_half,
            rho_now=rho_now,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            wgtfac_c=wgtfac_c,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
        )
        return dict(
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = ta.wpfloat("1.0")
        wgt_nnow_rth = ta.wpfloat("2.0")
        wgt_nnew_rth = ta.wpfloat("3.0")
        w = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        w_concorr_c = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        rho_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_var = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_var = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        theta_ref_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        vwind_expl_wgt = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        exner_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        d_exner_dz_ref_ic = data_alloc.random_field(
            grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat
        )
        rho_ic = data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        z_theta_v_pr_ic = data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        theta_v_ic = data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        z_th_ddz_exner_c = data_alloc.zero_field(
            grid, dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat
        )
        return dict(
            w=w,
            w_concorr_c=w_concorr_c,
            ddqz_z_half=ddqz_z_half,
            rho_now=rho_now,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            wgtfac_c=wgtfac_c,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            rho_ic=rho_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
