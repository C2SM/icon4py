# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_cell_to_half_levels import (
    interpolate_cell_to_half_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def interpolate_cell_to_half_levels_numpy(
    interpolant: np.ndarray,
    wgtfac_c: np.ndarray,
    *,
    wgtfacq1_c_1: np.ndarray,
    wgtfacq1_c_2: np.ndarray,
    wgtfacq1_c_3: np.ndarray,
    wgtfacq_c_1: np.ndarray,
    wgtfacq_c_2: np.ndarray,
    wgtfacq_c_3: np.ndarray,
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation = np.zeros((interpolant.shape[0], nlev + 1), dtype=interpolant.dtype)
    # Fortran jk = 1 (1-based) -> k = 0
    interpolation[:, 0] = (
        wgtfacq1_c_1 * interpolant[:, 0]
        + wgtfacq1_c_2 * interpolant[:, 1]
        + wgtfacq1_c_3 * interpolant[:, 2]
    )
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1
    interpolation[:, 1:nlev] = (
        wgtfac_c[:, 1:nlev] * interpolant[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * interpolant[:, 0 : nlev - 1]
    )
    # Fortran jk = nlevp1 (1-based) -> k = nlev
    interpolation[:, nlev] = (
        wgtfacq_c_1 * interpolant[:, nlev - 1]
        + wgtfacq_c_2 * interpolant[:, nlev - 2]
        + wgtfacq_c_3 * interpolant[:, nlev - 3]
    )
    return interpolation


class TestInterpolateCellToHalfLevels(StencilTest):
    PROGRAM = interpolate_cell_to_half_levels
    OUTPUTS = ("interpolation",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        interpolant: np.ndarray,
        wgtfac_c: np.ndarray,
        wgtfacq1_c_1: np.ndarray,
        wgtfacq1_c_2: np.ndarray,
        wgtfacq1_c_3: np.ndarray,
        wgtfacq_c_1: np.ndarray,
        wgtfacq_c_2: np.ndarray,
        wgtfacq_c_3: np.ndarray,
        **kwargs,
    ) -> dict:
        interpolation = interpolate_cell_to_half_levels_numpy(
            interpolant,
            wgtfac_c,
            wgtfacq1_c_1=wgtfacq1_c_1,
            wgtfacq1_c_2=wgtfacq1_c_2,
            wgtfacq1_c_3=wgtfacq1_c_3,
            wgtfacq_c_1=wgtfacq_c_1,
            wgtfacq_c_2=wgtfacq_c_2,
            wgtfacq_c_3=wgtfacq_c_3,
        )
        return dict(interpolation=interpolation)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolant = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        wgtfac_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )
        wgtfacq1_c_1 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        wgtfacq1_c_2 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        wgtfacq1_c_3 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        wgtfacq_c_1 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        wgtfacq_c_2 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        wgtfacq_c_3 = data_alloc.random_field(grid, dims.CellDim, dtype=wpfloat)
        interpolation = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            interpolant=interpolant,
            wgtfac_c=wgtfac_c,
            wgtfacq1_c_1=wgtfacq1_c_1,
            wgtfacq1_c_2=wgtfacq1_c_2,
            wgtfacq1_c_3=wgtfacq1_c_3,
            wgtfacq_c_1=wgtfacq_c_1,
            wgtfacq_c_2=wgtfacq_c_2,
            wgtfacq_c_3=wgtfacq_c_3,
            interpolation=interpolation,
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
