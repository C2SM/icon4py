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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    inv_dual_edge_length: np.ndarray,
    temporal_extrapolation_of_perturbed_exner: np.ndarray,
    ddxn_z_full: np.ndarray,
    c_lin_e: np.ndarray,
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
) -> np.ndarray:
    e2c = connectivities[dims.E2CDim]
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    z_exner_ex_pr_e2c = temporal_extrapolation_of_perturbed_exner[e2c]
    z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

    horizontal_pressure_gradient = inv_dual_edge_length * z_exner_ex_weighted - ddxn_z_full * np.sum(
        c_lin_e * ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels[e2c], axis=1
    )
    return horizontal_pressure_gradient


@pytest.mark.skip_value_error
class TestComputeHorizontalGradientOfExnerPressureForNonflatCoordinates(StencilTest):
    PROGRAM = compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates
    OUTPUTS = ("horizontal_pressure_gradient",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        inv_dual_edge_length: np.ndarray,
        temporal_extrapolation_of_perturbed_exner: np.ndarray,
        ddxn_z_full: np.ndarray,
        c_lin_e: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        horizontal_pressure_gradient = compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_numpy(
            connectivities,
            inv_dual_edge_length,
            temporal_extrapolation_of_perturbed_exner,
            ddxn_z_full,
            c_lin_e,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )
        return dict(horizontal_pressure_gradient=horizontal_pressure_gradient)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        temporal_extrapolation_of_perturbed_exner = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddxn_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        c_lin_e = random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=wpfloat)
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        horizontal_pressure_gradient = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
