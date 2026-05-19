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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    w_nnow: np.ndarray,
    ddt_w_adv_ntl1: np.ndarray,
    ddt_w_adv_ntl2: np.ndarray,
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
    rho_at_cells_on_half_levels: np.ndarray,
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    exner_w_explicit_weight_parameter: np.ndarray,
    dtime: float,
    advection_explicit_weight_parameter: float,
    advection_implicit_weight_parameter: float,
    cpd: float,
) -> tuple[np.ndarray, np.ndarray]:
    w_explicit_term = w_nnow + dtime * (
        advection_explicit_weight_parameter * ddt_w_adv_ntl1
        + advection_implicit_weight_parameter * ddt_w_adv_ntl2
        - cpd * ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )
    exner_w_explicit_weight_parameter = np.expand_dims(exner_w_explicit_weight_parameter, axis=-1)
    vertical_mass_flux_at_cells_on_half_levels = rho_at_cells_on_half_levels * (
        -contravariant_correction_at_cells_on_half_levels
        + exner_w_explicit_weight_parameter * w_nnow
    )
    return (w_explicit_term, vertical_mass_flux_at_cells_on_half_levels)


class TestComputeExplicitVerticalWindFromAdvectionAndVerticalWindDensity(StencilTest):
    PROGRAM = compute_explicit_vertical_wind_from_advection_and_vertical_wind_density
    OUTPUTS = ("w_explicit_term", "vertical_mass_flux_at_cells_on_half_levels")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w_nnow: np.ndarray,
        ddt_w_adv_ntl1: np.ndarray,
        ddt_w_adv_ntl2: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        dtime: float,
        advection_explicit_weight_parameter: float,
        advection_implicit_weight_parameter: float,
        cpd: float,
        **kwargs: Any,
    ) -> dict:
        (
            w_explicit_term,
            vertical_mass_flux_at_cells_on_half_levels,
        ) = compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
            connectivities=connectivities,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            dtime=dtime,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
            cpd=cpd,
        )
        return dict(
            w_explicit_term=w_explicit_term,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        ddt_w_adv_ntl1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        ddt_w_adv_ntl2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        w_explicit_term = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        exner_w_explicit_weight_parameter = data_alloc.random_field(
            grid, dims.CellDim, dtype=ta.wpfloat
        )
        vertical_mass_flux_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        dtime = ta.wpfloat("5.0")
        advection_explicit_weight_parameter = ta.wpfloat("8.0")
        advection_implicit_weight_parameter = ta.wpfloat("9.0")
        cpd = ta.wpfloat("10.0")

        return dict(
            w_explicit_term=w_explicit_term,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            dtime=dtime,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
