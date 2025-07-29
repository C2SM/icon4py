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

from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn import spatially_average_flux_or_velocity
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


def spatially_average_flux_or_velocity_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_flx_avg: np.ndarray,
    vn: np.ndarray,
) -> np.ndarray:
    e2c2eO = connectivities[dims.E2C2EODim]
    e_flx_avg = np.expand_dims(e_flx_avg, axis=-1)
    z_vn_avg = np.sum(
        np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * e_flx_avg, 0), axis=1
    )

    return z_vn_avg


class TestSpatiallyAverageFluxOrVelocity(StencilTest):
    PROGRAM = spatially_average_flux_or_velocity
    OUTPUTS = ("z_vn_avg",)
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        e_flx_avg: np.ndarray,
        vn: np.ndarray,
    ) -> dict:
        z_vn_avg = spatially_average_flux_or_velocity_numpy(connectivities, e_flx_avg, vn)

        return dict(z_vn_avg=z_vn_avg)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_flx_avg = random_field(grid, dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_vn_avg = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            z_vn_avg=z_vn_avg,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
