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

import icon4py.model.common.utils.data_allocation as data_alloc
import icon4py.model.testing.helpers as test_helpers
from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels import (
    interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils

from .test_compute_contravariant_correction_of_w import compute_contravariant_correction_of_w_numpy
from .test_compute_contravariant_correction_of_w_for_lower_boundary import (
    compute_contravariant_correction_of_w_for_lower_boundary_numpy,
)


def _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_bln_c_s: np.ndarray,
    contravariant_correction_at_edges_on_model_levels: np.ndarray,
    wgtfac_c: np.ndarray,
    wgtfacq_c: np.ndarray,
    nlev: int,
    nflatlev: int,
) -> np.ndarray:
    vert_idx = np.arange(nlev)
    contravariant_correction_at_cells_on_half_levels = np.where(
        (nflatlev < vert_idx) & (vert_idx < nlev),
        compute_contravariant_correction_of_w_numpy(
            connectivities, e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfac_c
        ),
        compute_contravariant_correction_of_w_for_lower_boundary_numpy(
            connectivities, e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfacq_c
        ),
    )

    contravariant_correction_at_cells_on_half_levels_res = np.zeros_like(
        contravariant_correction_at_cells_on_half_levels
    )
    contravariant_correction_at_cells_on_half_levels_res[
        :, -1
    ] = contravariant_correction_at_cells_on_half_levels[:, -1]
    return contravariant_correction_at_cells_on_half_levels_res


class TestInterpolateContravariantCorrectionFromEdgesOnModelLevelsToCellsOnHalfLevels(
    test_helpers.StencilTest
):
    PROGRAM = (
        interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels
    )
    OUTPUTS = ("contravariant_correction_at_cells_on_half_levels",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        e_bln_c_s: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        wgtfac_c: np.ndarray,
        wgtfacq_c: np.ndarray,
        nlev: int,
        nflatlev: int,
        **kwargs: Any,
    ) -> dict:
        contravariant_correction_at_cells_on_half_levels_result = _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels_numpy(
            connectivities,
            e_bln_c_s,
            contravariant_correction_at_edges_on_model_levels,
            wgtfac_c,
            wgtfacq_c,
            nlev,
            nflatlev,
        )
        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_result
        )

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim, dtype=ta.wpfloat)
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat
        )
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        wgtfacq_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )

        nlev = grid.num_levels
        nflatlev = 13

        return dict(
            e_bln_c_s=e_bln_c_s,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            nlev=nlev,
            nflatlev=nflatlev,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )
