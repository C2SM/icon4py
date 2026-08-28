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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import interpolate_km
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def interpolate_km_to_full_level_cells_numpy(km_ic: np.ndarray, *, km_min: float) -> np.ndarray:
    return np.maximum(km_min, 0.5 * (km_ic[:, :-1] + km_ic[:, 1:]))


def interpolate_km_to_vertices_numpy(
    km_ic: np.ndarray, *, cells_aw_verts: np.ndarray, v2c: np.ndarray, km_min: float
) -> np.ndarray:
    return np.maximum(km_min, np.sum(cells_aw_verts[:, :, np.newaxis] * km_ic[v2c], axis=1))


def interpolate_km_to_edges_numpy(
    km_ic: np.ndarray, *, c_lin_e: np.ndarray, e2c: np.ndarray, km_min: float
) -> np.ndarray:
    return np.maximum(km_min, np.sum(km_ic[e2c] * c_lin_e[:, :, np.newaxis], axis=1))


@pytest.mark.skip_value_error
class TestInterpolateKm(stencil_tests.StencilTest):
    PROGRAM = interpolate_km
    OUTPUTS = ("km_c", "km_iv", "km_ie")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        km_ic: np.ndarray,
        cells_aw_verts: np.ndarray,
        c_lin_e: np.ndarray,
        km_min: float,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            km_c=interpolate_km_to_full_level_cells_numpy(km_ic, km_min=km_min),
            km_iv=interpolate_km_to_vertices_numpy(
                km_ic,
                cells_aw_verts=cells_aw_verts,
                v2c=connectivities[dims.V2C],
                km_min=km_min,
            ),
            km_ie=interpolate_km_to_edges_numpy(
                km_ic,
                c_lin_e=c_lin_e,
                e2c=connectivities[dims.E2C],
                km_min=km_min,
            ),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        km_ic = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=wpfloat, extend={dims.KDim: 1}
        )
        cells_aw_verts = data_alloc.random_field(
            dims.VertexDim, dims.V2CDim, low=0.0, high=1.0 / 6.0, dtype=wpfloat
        )
        c_lin_e = data_alloc.random_field(
            dims.EdgeDim, dims.E2CDim, low=0.0, high=1.0, dtype=wpfloat
        )

        return dict(
            km_ic=km_ic,
            cells_aw_verts=cells_aw_verts,
            c_lin_e=c_lin_e,
            km_c=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            km_iv=data_alloc.zero_field(
                dims.VertexDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
            ),
            km_ie=data_alloc.zero_field(
                dims.EdgeDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
            ),
            # large enough that the floor is active for part of each field
            km_min=wpfloat(0.5),
            cell_start=0,
            cell_end=gtx.int32(grid.num_cells),
            vertex_start=0,
            vertex_end=gtx.int32(grid.num_vertices),
            edge_start=0,
            edge_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
            vertical_end_half=gtx.int32(grid.num_levels + 1),
        )
