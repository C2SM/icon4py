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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_exchange_coefficient_diagnostics import (
    update_exchange_coefficient_diagnostics,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def exchange_coefficient_diagnostics_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    km_ic: np.ndarray,
    kh_ic: np.ndarray,
    km_const: float,
    rturb_prandtl: float,
    use_km_const: bool,
    **kwargs: Any,
) -> dict:
    km = km_ic[:, 1:-1].copy()
    kh = kh_ic[:, 1:-1].copy()
    km_bottom = (km_const, km_const * rturb_prandtl) if use_km_const else (0.0, 0.0)
    km = np.concatenate((km, np.full((km.shape[0], 1), km_bottom[0])), axis=1)
    kh = np.concatenate((kh, np.full((kh.shape[0], 1), km_bottom[1])), axis=1)
    return dict(km=km, kh=kh)


def exchange_coefficient_diagnostics_input_data(
    grid: base.Grid, use_km_const: bool
) -> dict[str, Any]:
    def half_level_field() -> gtx.Field:
        return data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=1.0e-3,
            high=10.0,
            extend={dims.KDim: 1},
            dtype=wpfloat,
        )

    return dict(
        km_ic=half_level_field(),
        kh_ic=half_level_field(),
        km=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        kh=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        km_const=wpfloat(1.0),
        rturb_prandtl=wpfloat(3.0),
        use_km_const=use_km_const,
        nlev=gtx.int32(grid.num_levels),
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels),
    )


# Static-params variants: prove that the config bool can be passed both as a regular
# runtime scalar ("none") and as a static (compile-time) argument selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_km_const",),
}


class TestUpdateExchangeCoefficientDiagnostics(StencilTest):
    PROGRAM = update_exchange_coefficient_diagnostics
    OUTPUTS = ("km", "kh")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(exchange_coefficient_diagnostics_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return exchange_coefficient_diagnostics_input_data(grid, use_km_const=False)


class TestUpdateExchangeCoefficientDiagnosticsKmConst(StencilTest):
    PROGRAM = update_exchange_coefficient_diagnostics
    OUTPUTS = ("km", "kh")
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(exchange_coefficient_diagnostics_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return exchange_coefficient_diagnostics_input_data(grid, use_km_const=True)
