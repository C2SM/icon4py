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
    w: np.ndarray,
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    current_rho: np.ndarray,
    rho_var: np.ndarray,
    theta_now: np.ndarray,
    theta_var: np.ndarray,
    wgtfac_c: np.ndarray,
    reference_theta_at_cells_on_model_levels: np.ndarray,
    exner_w_explicit_weight_parameter: np.ndarray,
    perturbed_exner_at_cells_on_model_levels: np.ndarray,
    d_exner_dz_ref_ic: np.ndarray,
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
) -> tuple[np.ndarray, ...]:
    exner_w_explicit_weight_parameter = np.expand_dims(exner_w_explicit_weight_parameter, axis=-1)
    rho_now_offset = np.roll(current_rho, shift=1, axis=1)
    rho_var_offset = np.roll(rho_var, shift=1, axis=1)
    theta_now_offset = np.roll(theta_now, shift=1, axis=1)
    theta_var_offset = np.roll(theta_var, shift=1, axis=1)
    theta_ref_mc_offset = np.roll(reference_theta_at_cells_on_model_levels, shift=1, axis=1)
    exner_pr_offset = np.roll(perturbed_exner_at_cells_on_model_levels, shift=1, axis=1)

    z_w_backtraj = (
        -(w - contravariant_correction_at_cells_on_half_levels) * dtime * 0.5 / ddqz_z_half
    )
    z_rho_tavg_m1 = wgt_nnow_rth * rho_now_offset + wgt_nnew_rth * rho_var_offset
    z_theta_tavg_m1 = wgt_nnow_rth * theta_now_offset + wgt_nnew_rth * theta_var_offset
    z_rho_tavg = wgt_nnow_rth * current_rho + wgt_nnew_rth * rho_var
    z_theta_tavg = wgt_nnow_rth * theta_now + wgt_nnew_rth * theta_var
    rho_at_cells_on_half_levels = (
        wgtfac_c * z_rho_tavg
        + (1 - wgtfac_c) * z_rho_tavg_m1
        + z_w_backtraj * (z_rho_tavg_m1 - z_rho_tavg)
    )
    rho_at_cells_on_half_levels[:, 0] = 0
    z_theta_v_pr_mc_m1 = z_theta_tavg_m1 - theta_ref_mc_offset
    z_theta_v_pr_mc = z_theta_tavg - reference_theta_at_cells_on_model_levels
    perturbed_theta_v_at_cells_on_half_levels = (
        wgtfac_c * z_theta_v_pr_mc + (1 - wgtfac_c) * z_theta_v_pr_mc_m1
    )
    perturbed_theta_v_at_cells_on_half_levels[:, 0] = 0
    theta_v_at_cells_on_half_levels = (
        wgtfac_c * z_theta_tavg
        + (1 - wgtfac_c) * z_theta_tavg_m1
        + z_w_backtraj * (z_theta_tavg_m1 - z_theta_tavg)
    )
    theta_v_at_cells_on_half_levels[:, 0] = 0
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        exner_w_explicit_weight_parameter
        * theta_v_at_cells_on_half_levels
        * (exner_pr_offset - perturbed_exner_at_cells_on_model_levels)
        / ddqz_z_half
        + perturbed_theta_v_at_cells_on_half_levels * d_exner_dz_ref_ic
    )
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels[:, 0] = 0
    return (
        rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )


class TestComputeRhoVirtualPotentialTemperaturesAndPressureGradient(StencilTest):
    PROGRAM = compute_rho_virtual_potential_temperatures_and_pressure_gradient
    OUTPUTS = (
        "rho_at_cells_on_half_levels",
        "perturbed_theta_v_at_cells_on_half_levels",
        "theta_v_at_cells_on_half_levels",
        "ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        ddqz_z_half: np.ndarray,
        current_rho: np.ndarray,
        rho_var: np.ndarray,
        theta_now: np.ndarray,
        theta_var: np.ndarray,
        wgtfac_c: np.ndarray,
        reference_theta_at_cells_on_model_levels: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        dtime: ta.wpfloat,
        wgt_nnow_rth: ta.wpfloat,
        wgt_nnew_rth: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        (
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ) = compute_rho_virtual_potential_temperatures_and_pressure_gradient_numpy(
            w=w,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            current_rho=current_rho,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            wgtfac_c=wgtfac_c,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
        )
        return dict(
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = ta.wpfloat("1.0")
        wgt_nnow_rth = ta.wpfloat("2.0")
        wgt_nnew_rth = ta.wpfloat("3.0")
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_var = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_var = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        reference_theta_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        exner_w_explicit_weight_parameter = data_alloc.random_field(
            grid, dims.CellDim, dtype=ta.wpfloat
        )
        perturbed_exner_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        d_exner_dz_ref_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        rho_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        return dict(
            w=w,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            current_rho=current_rho,
            rho_var=rho_var,
            theta_now=theta_now,
            theta_var=theta_var,
            wgtfac_c=wgtfac_c,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            dtime=dtime,
            wgt_nnow_rth=wgt_nnow_rth,
            wgt_nnew_rth=wgt_nnew_rth,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
