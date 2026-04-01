# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.spatially_average_flux_or_velocity import (
    spatially_average_flux_or_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def spatially_average_flux_or_velocity_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    e_flx_avg: np.ndarray,
    flux_or_velocity: np.ndarray,
) -> np.ndarray:
    e2c2eO = connectivities[dims.E2C2EO]
    e_flx_avg = np.expand_dims(e_flx_avg, axis=-1)
    spatially_averaged_flux_or_velocity = np.sum(flux_or_velocity[e2c2eO] * e_flx_avg, axis=1)

    return spatially_averaged_flux_or_velocity


@pytest.mark.embedded_remap_error
class TestSpatiallyAverageFluxOrVelocity(stencil_tests.StencilTest):
    PROGRAM = spatially_average_flux_or_velocity
    OUTPUTS = ("spatially_averaged_flux_or_velocity",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        e_flx_avg: np.ndarray,
        flux_or_velocity: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        spatially_averaged_flux_or_velocity = spatially_average_flux_or_velocity_numpy(
            connectivities, e_flx_avg, flux_or_velocity
        )

        return dict(spatially_averaged_flux_or_velocity=spatially_averaged_flux_or_velocity)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        e_flx_avg = self.data_alloc.random_field(dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        flux_or_velocity = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        spatially_averaged_flux_or_velocity = self.data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, dtype=wpfloat
        )

        return dict(
            e_flx_avg=e_flx_avg,
            flux_or_velocity=flux_or_velocity,
            spatially_averaged_flux_or_velocity=spatially_averaged_flux_or_velocity,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
