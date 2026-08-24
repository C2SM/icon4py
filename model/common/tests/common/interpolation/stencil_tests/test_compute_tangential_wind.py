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
    compute_tangential_wind,
    compute_tangential_wind_wp,
)
from icon4py.model.testing import stencil_tests


def compute_tangential_wind_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
) -> np.ndarray:
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    e2c2e = connectivities[dims.E2C2E]
    vt = np.sum(np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1)
    return vt


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
    extend = {dims.KDim: 1} if on_half_levels else {}
    vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, extend=extend, dtype=ta.wpfloat)
    rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim, dtype=ta.wpfloat)
    vt = data_alloc.zero_field(dims.EdgeDim, dims.KDim, extend=extend, dtype=ta.wpfloat)

    # Fortran: rbf_vec_interpol_edge is called in tmx with
    # opt_rlstart = 3, opt_rlend = min_rledge_int - 2.
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


class TestComputeTangentialWindWpHalfLevels(stencil_tests.StencilTest):
    """Half-level input (nlev + 1 rows), e.g. vt_ie from vn_ie in tmx Stage A."""

    PROGRAM = compute_tangential_wind_wp
    OUTPUTS = ("vt",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return tangential_wind_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tangential_wind_input_data(data_alloc, grid, on_half_levels=True)


class TestComputeTangentialWindWpFullLevels(stencil_tests.StencilTest):
    """Full-level input (nlev rows), e.g. vt from vn in tmx Stage E1."""

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

    PROGRAM = compute_tangential_wind
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
        vt = compute_tangential_wind_numpy(connectivities, vn, rbf_vec_coeff_e)
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
