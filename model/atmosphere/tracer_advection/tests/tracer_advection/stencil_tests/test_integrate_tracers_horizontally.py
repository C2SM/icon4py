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

from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_horizontally import (
    integrate_tracers_horizontally,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


NTRACER = 3


class TestIntegrateTracersHorizontally(stencil_tests.StencilTest):
    PROGRAM = integrate_tracers_horizontally(NTRACER)
    OUTPUTS = ("tracers_new_hor",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        fluxes_and_tracers: tuple[tuple[gtx.Field, gtx.Field], ...],
        deepatmo_divh: np.ndarray,
        rhodz_now: np.ndarray,
        rhodz_new: np.ndarray,
        geofac_div: np.ndarray,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        c2e = stencil_tests.connectivities_asnumpy(grid)[dims.C2E]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        return dict(
            tracers_new_hor=tuple(
                (
                    tracer.asnumpy() * rhodz_now
                    - p_dtime * deepatmo_divh * np.sum(flux.asnumpy()[c2e] * geofac_div, axis=1)
                )
                / rhodz_new
                for flux, tracer in fluxes_and_tracers
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            fluxes_and_tracers=tuple(
                (
                    data_alloc.random_field(dims.EdgeDim, dims.KDim),
                    data_alloc.random_field(dims.CellDim, dims.KDim),
                )
                for _ in range(NTRACER)
            ),
            deepatmo_divh=data_alloc.random_field(dims.KDim),
            rhodz_now=data_alloc.random_field(dims.CellDim, dims.KDim),
            rhodz_new=data_alloc.random_field(dims.CellDim, dims.KDim),
            geofac_div=data_alloc.random_field(dims.CellDim, dims.C2EDim),
            p_dtime=np.float64(5.0),
            tracers_new_hor=tuple(
                data_alloc.zero_field(dims.CellDim, dims.KDim) for _ in range(NTRACER)
            ),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
