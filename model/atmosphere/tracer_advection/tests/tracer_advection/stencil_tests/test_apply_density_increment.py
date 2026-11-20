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

from icon4py.model.atmosphere.tracer_advection.stencils.apply_density_increment import (
    apply_density_increment,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


def apply_density_increment_numpy(  # noqa: PLR0917
    rhodz_in: np.ndarray,
    p_mflx_contra_v: np.ndarray,
    deepatmo_divzl: np.ndarray,
    deepatmo_divzu: np.ndarray,
    p_dtime: float,
    even_timestep: bool,
) -> np.ndarray:
    rhodz_incr = p_dtime * (
        p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
    )
    if even_timestep:
        return rhodz_in + rhodz_incr
    return np.maximum(0.1 * rhodz_in, rhodz_in) - rhodz_incr


class TestApplyDensityIncrement(stencil_tests.StencilTest):
    PROGRAM = apply_density_increment
    OUTPUTS = ("rhodz_out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rhodz_in: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        deepatmo_divzl: np.ndarray,
        deepatmo_divzu: np.ndarray,
        p_dtime: float,
        even_timestep: bool,
        **kwargs: Any,
    ) -> dict:
        rhodz_out = apply_density_increment_numpy(
            rhodz_in, p_mflx_contra_v, deepatmo_divzl, deepatmo_divzu, p_dtime, even_timestep
        )
        return dict(rhodz_out=rhodz_out)

    @stencil_tests.input_data_fixture(params=[True, False])
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        even_timestep = request.param
        return dict(
            rhodz_in=data_alloc.random_field(dims.CellDim, dims.KDim),
            p_mflx_contra_v=data_alloc.random_field(dims.CellDim, dims.KHalfDim),
            deepatmo_divzl=data_alloc.random_field(dims.KDim),
            deepatmo_divzu=data_alloc.random_field(dims.KDim),
            rhodz_out=data_alloc.zero_field(dims.CellDim, dims.KDim),
            p_dtime=ta.wpfloat(0.1),
            even_timestep=even_timestep,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
