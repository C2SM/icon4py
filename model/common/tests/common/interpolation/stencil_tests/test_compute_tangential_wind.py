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
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_on_half_levels,
    compute_tangential_wind_vp,
    compute_tangential_wind_wp,
)
from icon4py.model.testing import reference_funcs, stencil_tests


def tangential_wind_reference(
    grid: base.Grid,
    *,
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    **kwargs: Any,
) -> dict:
    connectivities = stencil_tests.connectivities_asnumpy(grid)
    e2c2e = connectivities[dims.E2C2E]  # (n_edges, 4)

    # (n_edges, 4, nlev[+1]) gather of the normal velocity at the neighbor edges
    vn_e = vn[e2c2e]
    coeff = np.expand_dims(rbf_vec_coeff_e, axis=-1)  # (n_edges, 4, 1)
    vt = np.sum(coeff * vn_e, axis=1)

    vt_out = np.zeros_like(vt)
    vt_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = vt[
        horizontal_start:horizontal_end, vertical_start:vertical_end
    ]
    return dict(vt=vt_out)


def tangential_wind_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, on_half_levels: bool
) -> dict[str, Any]:
    vertical_dim = dims.KHalfDim if on_half_levels else dims.KDim
    vn = data_alloc.random_field(dims.EdgeDim, vertical_dim, dtype=ta.wpfloat)
    rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim, dtype=ta.wpfloat)
    vt = data_alloc.zero_field(dims.EdgeDim, vertical_dim, dtype=ta.wpfloat)

    edge_domain = h_grid.domain(dims.EdgeDim)
    horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
    horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    assert horizontal_start < horizontal_end

    num_levels = grid.num_levels + 1 if on_half_levels else grid.num_levels

    return dict(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        vt=vt,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=gtx.int32(num_levels),
    )


class TestComputeTangentialWindOnHalfLevels(stencil_tests.StencilTest):
    PROGRAM = _compute_tangential_wind_on_half_levels
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vn_ie: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        domain: dict[gtx.Dimension, tuple[int, int]],
        **kwargs: Any,
    ) -> dict:
        (horizontal_start, horizontal_end) = domain[dims.EdgeDim]
        (vertical_start, vertical_end) = domain[dims.KHalfDim]
        vt = tangential_wind_reference(
            grid,
            vn=vn_ie,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )["vt"]
        return dict(out=vt)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        data = tangential_wind_input_data(data_alloc, grid, on_half_levels=True)
        return dict(
            vn_ie=data["vn"],
            rbf_vec_coeff_e=data["rbf_vec_coeff_e"],
            domain={
                dims.EdgeDim: (data["horizontal_start"], data["horizontal_end"]),
                dims.KHalfDim: (data["vertical_start"], data["vertical_end"]),
            },
            out=data["vt"],
        )


class TestComputeTangentialWindWp(stencil_tests.StencilTest):
    PROGRAM = compute_tangential_wind_wp
    OUTPUTS = ("vt",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return tangential_wind_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tangential_wind_input_data(data_alloc, grid, on_half_levels=False)


@pytest.mark.embedded_remap_error
class TestComputeTangentialWind(stencil_tests.StencilTest):
    """Variable-precision variant used by the dycore velocity advection."""

    PROGRAM = compute_tangential_wind_vp
    OUTPUTS = ("vt",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vn: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        vt = reference_funcs.compute_tangential_wind_numpy(connectivities, vn, rbf_vec_coeff_e)
        return dict(vt=vt)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim, dtype=ta.wpfloat)
        vt = data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            vt=vt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
