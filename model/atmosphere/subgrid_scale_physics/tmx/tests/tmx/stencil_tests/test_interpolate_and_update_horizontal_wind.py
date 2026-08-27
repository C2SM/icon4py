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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.wind_diffusion import (
    interpolate_and_update_horizontal_wind,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


def edge_2_cell_vector_rbf_interpolation_numpy(
    p_e_in: np.ndarray,
    *,
    ptr_coeff_1: np.ndarray,
    ptr_coeff_2: np.ndarray,
    c2e2c2e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference for 'rbf_vec_interpol_cell' (mo_intp_rbf.f90)."""
    p_u_out = np.sum(ptr_coeff_1[:, :, np.newaxis] * p_e_in[c2e2c2e], axis=1)
    p_v_out = np.sum(ptr_coeff_2[:, :, np.newaxis] * p_e_in[c2e2c2e], axis=1)
    return p_u_out, p_v_out


def update_two_fields_with_tendency_numpy(
    *,
    field_1: np.ndarray,
    field_2: np.ndarray,
    tendency_1: np.ndarray,
    tendency_2: np.ndarray,
    dtime: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference for the final update loop 'new = old + tend * dtime'."""
    return field_1 + tendency_1 * dtime, field_2 + tendency_2 * dtime


class TestInterpolateAndUpdateHorizontalWind(stencil_tests.StencilTest):
    """
    Composition of 'rbf_vec_interpol_cell' (tot_tend -> tend_u, tend_v) with the
    final update loop of 'Compute_diffusion_hor_wind' (new_u/v = u/v + tend * dt).

    The four outputs live on two different horizontal ranges: the tendencies on
    the Fortran default 'opt_rlstart = 2' cells of the interpolation, the updated
    winds on the narrower tmx t_domain cells. The two ranges are deliberately
    distinct so a program writing an output on the wrong one is caught.
    """

    PROGRAM = interpolate_and_update_horizontal_wind
    OUTPUTS = ("tend_u", "tend_v", "new_u", "new_v")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        tot_tend: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        rbf_coeff_c1: np.ndarray,
        rbf_coeff_c2: np.ndarray,
        tend_u: np.ndarray,
        tend_v: np.ndarray,
        new_u: np.ndarray,
        new_v: np.ndarray,
        dtime: float,
        vertical_end: int,
        cell_start_nudging: int,
        cell_start_lateral_boundary_level_2: int,
        cell_end_local: int,
        **kwargs: Any,
    ) -> dict:
        nlev = vertical_end
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        tend_u_full, tend_v_full = edge_2_cell_vector_rbf_interpolation_numpy(
            tot_tend,
            ptr_coeff_1=rbf_coeff_c1,
            ptr_coeff_2=rbf_coeff_c2,
            c2e2c2e=connectivities[dims.C2E2C2E],
        )
        new_u_full, new_v_full = update_two_fields_with_tendency_numpy(
            field_1=u,
            field_2=v,
            tendency_1=tend_u_full,
            tendency_2=tend_v_full,
            dtime=dtime,
        )

        # Each output keeps its initial value outside its own domain.
        tendency_rows = slice(cell_start_lateral_boundary_level_2, cell_end_local)
        wind_rows = slice(cell_start_nudging, cell_end_local)
        tend_u_out = tend_u.copy()
        tend_u_out[tendency_rows, 0:nlev] = tend_u_full[tendency_rows, 0:nlev]
        tend_v_out = tend_v.copy()
        tend_v_out[tendency_rows, 0:nlev] = tend_v_full[tendency_rows, 0:nlev]
        new_u_out = new_u.copy()
        new_u_out[wind_rows, 0:nlev] = new_u_full[wind_rows, 0:nlev]
        new_v_out = new_v.copy()
        new_v_out[wind_rows, 0:nlev] = new_v_full[wind_rows, 0:nlev]

        return dict(tend_u=tend_u_out, tend_v=tend_v_out, new_u=new_u_out, new_v=new_v_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        # The cell zones of the simple grid all collapse to (0, num_cells), which
        # would make the two output ranges indistinguishable. Explicit bounds
        # keep them distinct and ordered as on a real grid
        # (lateral_boundary_level_2 < nudging < end_local).
        cell_start_lateral_boundary_level_2 = 1
        cell_start_nudging = 3
        cell_end_local = grid.num_cells - 1
        assert cell_start_lateral_boundary_level_2 < cell_start_nudging < cell_end_local

        return dict(
            tot_tend=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
            u=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            v=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            rbf_coeff_c1=data_alloc.random_field(dims.CellDim, dims.C2E2C2EDim, dtype=ta.wpfloat),
            rbf_coeff_c2=data_alloc.random_field(dims.CellDim, dims.C2E2C2EDim, dtype=ta.wpfloat),
            tend_u=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            tend_v=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            new_u=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            new_v=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
            dtime=ta.wpfloat(2.0),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
            cell_start_nudging=gtx.int32(cell_start_nudging),
            cell_start_lateral_boundary_level_2=gtx.int32(cell_start_lateral_boundary_level_2),
            cell_end_local=gtx.int32(cell_end_local),
        )
