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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    compute_vertex_wind_diagnostics,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_cell_2_vertex_interpolation_numpy(
    cell_in: np.ndarray, *, c_int: np.ndarray, v2c: np.ndarray
) -> np.ndarray:
    return np.sum(c_int[:, :, np.newaxis] * cell_in[v2c], axis=1)


def mo_intp_rbf_rbf_vec_interpol_vertex_numpy(
    p_e_in: np.ndarray, *, ptr_coeff_1: np.ndarray, ptr_coeff_2: np.ndarray, v2e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p_u_out = np.sum(ptr_coeff_1[:, :, np.newaxis] * p_e_in[v2e], axis=1)
    p_v_out = np.sum(ptr_coeff_2[:, :, np.newaxis] * p_e_in[v2e], axis=1)
    return p_u_out, p_v_out


@pytest.mark.skip_value_error
class TestComputeVertexWindDiagnostics(stencil_tests.StencilTest):
    """
    The two vertex gathers of ``Compute_diagnostics`` (``cells2verts_scalar`` for
    ``w`` and ``rbf_vec_interpol_vertex`` for ``vn``), fused into one program.

    The three outputs share one horizontal range but not the vertical one:
    ``w_vert`` lives on the nlev + 1 half levels, ``u_vert`` / ``v_vert`` on the
    nlev full levels. The horizontal range is deliberately narrower than the
    field, so an output written on the whole field is caught.
    """

    PROGRAM = compute_vertex_wind_diagnostics
    OUTPUTS = ("w_vert", "u_vert", "v_vert")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        w: np.ndarray,
        vn: np.ndarray,
        cells_aw_verts: np.ndarray,
        rbf_coeff_v1: np.ndarray,
        rbf_coeff_v2: np.ndarray,
        w_vert: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        vertical_end: int,
        vertex_start_lateral_boundary_level_2: int,
        vertex_end_local: int,
        **kwargs: Any,
    ) -> dict:
        nlev = vertical_end
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        w_vert_full = compute_cell_2_vertex_interpolation_numpy(
            w, c_int=cells_aw_verts, v2c=connectivities[dims.V2C]
        )
        u_vert_full, v_vert_full = mo_intp_rbf_rbf_vec_interpol_vertex_numpy(
            vn,
            ptr_coeff_1=rbf_coeff_v1,
            ptr_coeff_2=rbf_coeff_v2,
            v2e=connectivities[dims.V2E],
        )

        # Each output keeps its initial value outside its own domain.
        horizontal = slice(vertex_start_lateral_boundary_level_2, vertex_end_local)
        w_vert_out = w_vert.copy()
        w_vert_out[horizontal, 0 : nlev + 1] = w_vert_full[horizontal, 0 : nlev + 1]
        u_vert_out = u_vert.copy()
        u_vert_out[horizontal, 0:nlev] = u_vert_full[horizontal, 0:nlev]
        v_vert_out = v_vert.copy()
        v_vert_out[horizontal, 0:nlev] = v_vert_full[horizontal, 0:nlev]

        return dict(w_vert=w_vert_out, u_vert=u_vert_out, v_vert=v_vert_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        # Non-trivial bounds: the vertex zones of the simple grid all collapse to
        # (0, num_vertices), which would hide a program ignoring its domain.
        vertex_start_lateral_boundary_level_2 = 1
        vertex_end_local = grid.num_vertices - 1
        assert vertex_start_lateral_boundary_level_2 < vertex_end_local

        return dict(
            w=data_alloc.random_field(
                dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat
            ),
            vn=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            cells_aw_verts=data_alloc.random_field(
                dims.VertexDim, dims.V2CDim, low=0.0, high=1.0 / 6.0, dtype=wpfloat
            ),
            rbf_coeff_v1=data_alloc.random_field(dims.VertexDim, dims.V2EDim, dtype=wpfloat),
            rbf_coeff_v2=data_alloc.random_field(dims.VertexDim, dims.V2EDim, dtype=wpfloat),
            w_vert=data_alloc.zero_field(
                dims.VertexDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat
            ),
            u_vert=data_alloc.zero_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            v_vert=data_alloc.zero_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
            vertical_end_half=gtx.int32(grid.num_levels + 1),
            vertex_start_lateral_boundary_level_2=gtx.int32(vertex_start_lateral_boundary_level_2),
            vertex_end_local=gtx.int32(vertex_end_local),
        )
