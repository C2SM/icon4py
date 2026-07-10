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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_tangential_wind_wp import (
    compute_tangential_wind_wp,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


def tangential_wind_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    **kwargs: Any,
) -> dict:
    e2c2e = connectivities[dims.E2C2EDim]  # (n_edges, 4)

    # (n_edges, 4, nlev[+1]) gather of the normal velocity at the neighbor edges
    vn_e = vn[e2c2e]
    coeff = np.expand_dims(rbf_vec_coeff_e, axis=-1)  # (n_edges, 4, 1)
    vt = np.sum(coeff * vn_e, axis=1)

    vt_out = np.zeros_like(vt)
    vt_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = vt[
        horizontal_start:horizontal_end, vertical_start:vertical_end
    ]
    return dict(vt=vt_out)


def tangential_wind_input_data(grid: base.Grid, on_half_levels: bool) -> dict[str, Any]:
    extend = {dims.KDim: 1} if on_half_levels else {}
    vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, extend=extend, dtype=ta.wpfloat)
    rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim, dtype=ta.wpfloat)
    vt = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, extend=extend, dtype=ta.wpfloat)

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
    """Stage A use: vt_ie from vn_ie on half levels (nlev + 1 rows)."""

    PROGRAM = compute_tangential_wind_wp
    OUTPUTS = ("vt",)

    reference = staticmethod(tangential_wind_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return tangential_wind_input_data(grid, on_half_levels=True)


class TestComputeTangentialWindWpFullLevels(stencil_tests.StencilTest):
    """Stage E1 use: vt from vn on full levels (nlev rows)."""

    PROGRAM = compute_tangential_wind_wp
    OUTPUTS = ("vt",)

    reference = staticmethod(tangential_wind_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return tangential_wind_input_data(grid, on_half_levels=False)
