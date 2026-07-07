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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_km_to_vertices import (
    interpolate_km_to_vertices,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def interpolate_km_to_vertices_numpy(
    km_ic: np.ndarray, *, cells_aw_verts: np.ndarray, v2c: np.ndarray, km_min: float
) -> np.ndarray:
    return np.maximum(km_min, np.sum(cells_aw_verts[:, :, np.newaxis] * km_ic[v2c], axis=1))


@pytest.mark.skip_value_error
class TestInterpolateKmToVertices(StencilTest):
    PROGRAM = interpolate_km_to_vertices
    OUTPUTS = ("km_iv",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        km_ic: np.ndarray,
        cells_aw_verts: np.ndarray,
        km_min: float,
        **kwargs,
    ) -> dict:
        km_iv = interpolate_km_to_vertices_numpy(
            km_ic,
            cells_aw_verts=cells_aw_verts,
            v2c=connectivities[dims.V2CDim],
            km_min=km_min,
        )
        return dict(km_iv=km_iv)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        km_ic = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=wpfloat, extend={dims.KDim: 1}
        )
        cells_aw_verts = data_alloc.random_field(
            grid, dims.VertexDim, dims.V2CDim, low=0.0, high=1.0 / 6.0, dtype=wpfloat
        )
        km_iv = data_alloc.zero_field(
            grid, dims.VertexDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            km_ic=km_ic,
            cells_aw_verts=cells_aw_verts,
            km_iv=km_iv,
            # large enough that the floor is active for part of the field
            km_min=wpfloat(0.25),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
