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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_surface_flux_rhs import (
    compute_surface_flux_rhs,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestComputeSurfaceFluxRhs(stencil_tests.StencilTest):
    """
    The program runs on the single bottom K row (Fortran
    ``rhs(jc,nlev,jb) = - sfc_flx * zfactor * inv_mair(jc,nlev,jb)``); all
    other rows keep their input values (zero in the granule).
    """

    PROGRAM = compute_surface_flux_rhs
    OUTPUTS = ("rhs",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        sfc_flx: np.ndarray,
        inv_air_mass: np.ndarray,
        rhs: np.ndarray,
        prefac: float,
        vertical_start: int,
        **kwargs: Any,
    ) -> dict:
        rhs_out = rhs.copy()
        rhs_out[:, vertical_start] = -sfc_flx * prefac * inv_air_mass[:, vertical_start]
        return dict(rhs=rhs_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return dict(
            sfc_flx=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            inv_air_mass=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=1.0e-4, high=1.0e-1, dtype=wpfloat
            ),
            rhs=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            prefac=wpfloat(0.9),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            # single bottom row, Fortran jk = nlev (1-based) -> row nlev-1
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )
