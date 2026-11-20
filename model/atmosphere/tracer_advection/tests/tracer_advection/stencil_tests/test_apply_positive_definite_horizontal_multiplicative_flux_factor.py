# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.tracer_advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


def apply_positive_definite_horizontal_multiplicative_flux_factor_numpy(
    r_m: np.ndarray,
    p_mflx_tracer_h: np.ndarray,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
) -> np.ndarray:
    e2c = connectivities[dims.E2C]
    return np.where(
        p_mflx_tracer_h >= 0.0,
        p_mflx_tracer_h * r_m[e2c[:, 0]],
        p_mflx_tracer_h * r_m[e2c[:, 1]],
    )


class TestApplyPositiveDefiniteHorizontalMultiplicativeFluxFactor(stencil_tests.StencilTest):
    PROGRAM = apply_positive_definite_horizontal_multiplicative_flux_factor
    OUTPUTS = ("p_mflx_tracer_h",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        r_m: np.ndarray,
        p_mflx_tracer_h: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        result = p_mflx_tracer_h.copy()
        limited = apply_positive_definite_horizontal_multiplicative_flux_factor_numpy(
            r_m, p_mflx_tracer_h, connectivities
        )
        result[horizontal_start:horizontal_end] = limited[horizontal_start:horizontal_end]
        return dict(p_mflx_tracer_h=result)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        edge_domain = h_grid.domain(dims.EdgeDim)
        return dict(
            r_m=data_alloc.random_field(dims.CellDim, dims.KDim),
            p_mflx_tracer_h=data_alloc.random_field(dims.EdgeDim, dims.KDim),
            # boundary edges have only one neighbouring cell
            horizontal_start=grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)),
            horizontal_end=grid.end_index(edge_domain(h_grid.Zone.END)),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
