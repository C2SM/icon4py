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

from icon4py.model.atmosphere.dycore.stencils.compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    wgtfac_c: np.ndarray,
    perturbed_theta_v_at_cells_on_model_levels_2: np.ndarray,
    theta_v: np.ndarray,
    exner_w_explicit_weight_parameter: np.ndarray,
    perturbed_exner_at_cells_on_model_levels: np.ndarray,
    d_exner_dz_ref_ic: np.ndarray,
    ddqz_z_half: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_rth_pr_2_offset = np.roll(perturbed_theta_v_at_cells_on_model_levels_2, axis=1, shift=1)
    theta_v_offset = np.roll(theta_v, axis=1, shift=1)
    exner_pr_offset = np.roll(perturbed_exner_at_cells_on_model_levels, axis=1, shift=1)
    exner_w_explicit_weight_parameter = np.expand_dims(exner_w_explicit_weight_parameter, axis=-1)

    perturbed_theta_v_at_cells_on_half_levels = (
        wgtfac_c * perturbed_theta_v_at_cells_on_model_levels_2
        + (1.0 - wgtfac_c) * z_rth_pr_2_offset
    )
    perturbed_theta_v_at_cells_on_half_levels[:, 0] = 0
    theta_v_at_cells_on_half_levels = wgtfac_c * theta_v + (1 - wgtfac_c) * theta_v_offset
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
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )


class TestComputeVirtualPotentialTemperaturesAndPressureGradient(StencilTest):
    PROGRAM = compute_virtual_potential_temperatures_and_pressure_gradient
    OUTPUTS = (
        "perturbed_theta_v_at_cells_on_half_levels",
        "theta_v_at_cells_on_half_levels",
        "ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        wgtfac_c: np.ndarray,
        perturbed_theta_v_at_cells_on_model_levels_2: np.ndarray,
        theta_v: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        ddqz_z_half: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ) = compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
            connectivities=connectivities,
            wgtfac_c=wgtfac_c,
            perturbed_theta_v_at_cells_on_model_levels_2=perturbed_theta_v_at_cells_on_model_levels_2,
            theta_v=theta_v,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
        )

        return dict(
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        perturbed_theta_v_at_cells_on_model_levels_2 = random_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_w_explicit_weight_parameter = random_field(grid, dims.CellDim, dtype=wpfloat)
        perturbed_exner_at_cells_on_model_levels = random_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat
        )
        d_exner_dz_ref_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        perturbed_theta_v_at_cells_on_half_levels = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        theta_v_at_cells_on_half_levels = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )

        return dict(
            wgtfac_c=wgtfac_c,
            perturbed_theta_v_at_cells_on_model_levels_2=perturbed_theta_v_at_cells_on_model_levels_2,
            theta_v=theta_v,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
