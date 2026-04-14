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

from icon4py.model.atmosphere.dycore.stencils.update_mass_volume_flux import update_mass_volume_flux
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def update_mass_volume_flux_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
    rho_at_cells_on_half_levels: np.ndarray,
    exner_w_implicit_weight_parameter: np.ndarray,
    w: np.ndarray,
    dynamical_vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: np.ndarray,
    r_nsubsteps: float,
) -> tuple[np.ndarray, np.ndarray]:
    exner_w_implicit_weight_parameter = np.expand_dims(exner_w_implicit_weight_parameter, axis=1)
    z_a = r_nsubsteps * (vertical_mass_flux_at_cells_on_half_levels + rho_at_cells_on_half_levels * exner_w_implicit_weight_parameter * w)
    dynamical_vertical_mass_flux_at_cells_on_half_levels = dynamical_vertical_mass_flux_at_cells_on_half_levels + z_a
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels = dynamical_vertical_volumetric_flux_at_cells_on_half_levels + z_a / rho_at_cells_on_half_levels
    return (dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels)


class TestUpdateMassVolumeFlux(StencilTest):
    PROGRAM = update_mass_volume_flux
    OUTPUTS = (
        "dynamical_vertical_mass_flux_at_cells_on_half_levels",
        "dynamical_vertical_volumetric_flux_at_cells_on_half_levels",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        exner_w_implicit_weight_parameter: np.ndarray,
        w: np.ndarray,
        dynamical_vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels: np.ndarray,
        r_nsubsteps: float,
        **kwargs: Any,
    ) -> dict:
        (dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels) = update_mass_volume_flux_numpy(
            connectivities,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            w=w,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            r_nsubsteps=r_nsubsteps,
        )
        return dict(dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vertical_mass_flux_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_w_implicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dynamical_vertical_mass_flux_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        r_nsubsteps = 7.0

        return dict(
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            w=w,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
