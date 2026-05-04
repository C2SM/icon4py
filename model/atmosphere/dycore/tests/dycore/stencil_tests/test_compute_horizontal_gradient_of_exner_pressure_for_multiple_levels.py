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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    inv_dual_edge_length: np.ndarray,
    temporal_extrapolation_of_perturbed_exner: np.ndarray,
    zdiff_gradp: np.ndarray,
    ikoffset: np.ndarray,
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
) -> np.ndarray:
    def _apply_index_field(
        shape: tuple, to_index: np.ndarray, neighbor_table: np.ndarray, offset_field: np.ndarray
    ) -> np.ndarray:
        indexed = np.zeros(shape)
        for iprimary in range(shape[0]):
            for isparse in range(shape[1]):
                for ik in range(shape[2]):
                    indexed[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik],
                    ]
        return indexed

    e2c = connectivities[dims.E2CDim]
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    full_shape = ikoffset.shape
    z_exner_ex_pr_at_kidx = _apply_index_field(
        full_shape, temporal_extrapolation_of_perturbed_exner, e2c, ikoffset
    )
    z_dexner_dz_c_1_at_kidx = _apply_index_field(
        full_shape, ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, e2c, ikoffset
    )
    z_dexner_dz_c_2_at_kidx = _apply_index_field(
        full_shape,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        e2c,
        ikoffset,
    )

    def at_neighbor(i: int) -> np.ndarray:
        return z_exner_ex_pr_at_kidx[:, i, :] + zdiff_gradp[:, i, :] * (
            z_dexner_dz_c_1_at_kidx[:, i, :]
            + zdiff_gradp[:, i, :] * z_dexner_dz_c_2_at_kidx[:, i, :]
        )

    sum_expr = at_neighbor(1) - at_neighbor(0)

    horizontal_pressure_gradient = inv_dual_edge_length * sum_expr
    return horizontal_pressure_gradient


@pytest.mark.skip_value_error
@pytest.mark.uses_as_offset
class TestComputeHorizontalGradientOfExnerPressureForMultipleLevels(StencilTest):
    PROGRAM = compute_horizontal_gradient_of_exner_pressure_for_multiple_levels
    OUTPUTS = ("horizontal_pressure_gradient",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        inv_dual_edge_length: np.ndarray,
        temporal_extrapolation_of_perturbed_exner: np.ndarray,
        zdiff_gradp: np.ndarray,
        ikoffset: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        horizontal_pressure_gradient = (
            compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_numpy(
                connectivities,
                inv_dual_edge_length,
                temporal_extrapolation_of_perturbed_exner,
                zdiff_gradp,
                ikoffset,
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            )
        )
        return dict(horizontal_pressure_gradient=horizontal_pressure_gradient)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        temporal_extrapolation_of_perturbed_exner = random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        zdiff_gradp = random_field(grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=ta.vpfloat)
        ikoffset = zero_field(grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32)
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset.ndarray[:, :, k] = rng.integers(  # type: ignore[index]
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )

        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        horizontal_pressure_gradient = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            zdiff_gradp=zdiff_gradp,
            ikoffset=ikoffset,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
