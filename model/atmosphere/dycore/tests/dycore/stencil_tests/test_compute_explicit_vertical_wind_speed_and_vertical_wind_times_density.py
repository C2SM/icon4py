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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    w_nnow: np.ndarray,
    ddt_w_adv_ntl1: np.ndarray,
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
    rho_at_cells_on_half_levels: np.ndarray,
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    exner_w_explicit_weight_parameter: np.ndarray,
    dtime: float,
    cpd: float,
) -> tuple[np.ndarray, np.ndarray]:
    exner_w_explicit_weight_parameter = np.expand_dims(exner_w_explicit_weight_parameter, -1)
    w_explicit_term = w_nnow + dtime * (
        ddt_w_adv_ntl1 - cpd * ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )
    vertical_mass_flux_at_cells_on_half_levels = rho_at_cells_on_half_levels * (
        -contravariant_correction_at_cells_on_half_levels
        + exner_w_explicit_weight_parameter * w_nnow
    )
    return (w_explicit_term, vertical_mass_flux_at_cells_on_half_levels)


class TestComputeExplicitVerticalWindSpeedAndVerticalWindTimesDensity(StencilTest):
    PROGRAM = compute_explicit_vertical_wind_speed_and_vertical_wind_times_density
    OUTPUTS = ("w_explicit_term", "vertical_mass_flux_at_cells_on_half_levels")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w_nnow: np.ndarray,
        ddt_w_adv_ntl1: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        dtime: float,
        cpd: float,
        **kwargs: Any,
    ) -> dict:
        (
            w_explicit_term,
            vertical_mass_flux_at_cells_on_half_levels,
        ) = compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
            connectivities,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            dtime=dtime,
            cpd=cpd,
        )
        return dict(
            w_explicit_term=w_explicit_term,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_nnow = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_w_adv_ntl1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = random_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        w_explicit_term = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_at_cells_on_half_levels = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        contravariant_correction_at_cells_on_half_levels = random_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        exner_w_explicit_weight_parameter = random_field(grid, dims.CellDim, dtype=wpfloat)
        vertical_mass_flux_at_cells_on_half_levels = zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat
        )
        dtime = wpfloat("5.0")
        cpd = wpfloat("10.0")

        return dict(
            w_explicit_term=w_explicit_term,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
