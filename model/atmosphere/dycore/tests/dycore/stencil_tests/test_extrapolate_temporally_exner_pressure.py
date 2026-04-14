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

from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    extrapolate_temporally_exner_pressure,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def extrapolate_temporally_exner_pressure_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    exner: np.ndarray,
    reference_exner_at_cells_on_model_levels: np.ndarray,
    perturbed_exner_at_cells_on_model_levels: np.ndarray,
    time_extrapolation_parameter_for_exner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temporal_extrapolation_of_perturbed_exner = (
        (1 + time_extrapolation_parameter_for_exner)
        * (exner - reference_exner_at_cells_on_model_levels)
        - time_extrapolation_parameter_for_exner * perturbed_exner_at_cells_on_model_levels
    )
    perturbed_exner_at_cells_on_model_levels = exner - reference_exner_at_cells_on_model_levels
    return (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels)


class TestExtrapolateTemporallyExnerPressure(StencilTest):
    PROGRAM = extrapolate_temporally_exner_pressure
    OUTPUTS = ("temporal_extrapolation_of_perturbed_exner", "perturbed_exner_at_cells_on_model_levels")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        exner: np.ndarray,
        reference_exner_at_cells_on_model_levels: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        time_extrapolation_parameter_for_exner: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels) = extrapolate_temporally_exner_pressure_numpy(
            connectivities,
            exner=exner,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
        )

        return dict(temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        exner = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        reference_exner_at_cells_on_model_levels = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        perturbed_exner_at_cells_on_model_levels = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        time_extrapolation_parameter_for_exner = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        temporal_extrapolation_of_perturbed_exner = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
            exner=exner,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
