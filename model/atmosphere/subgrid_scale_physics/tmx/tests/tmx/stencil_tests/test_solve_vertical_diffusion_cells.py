# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.solve_vertical_diffusion_cells import (
    solve_vertical_diffusion_cells,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def tdma_solver_numpy(
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    minlvl: int,
    maxlvl: int,
) -> np.ndarray:
    """Reference for 'tdma_solver_vec' (iconmath mo_math_utilities.F90)."""
    c_p = np.zeros_like(a)
    d_p = np.zeros_like(a)
    varout = np.zeros_like(a)
    c_p[:, minlvl] = c[:, minlvl] / b[:, minlvl]
    d_p[:, minlvl] = d[:, minlvl] / b[:, minlvl]
    for k in range(minlvl + 1, maxlvl + 1):
        m = 1.0 / (b[:, k] - c_p[:, k - 1] * a[:, k])
        c_p[:, k] = c[:, k] * m
        d_p[:, k] = (d[:, k] - d_p[:, k - 1] * a[:, k]) * m
    varout[:, maxlvl] = d_p[:, maxlvl]
    for k in range(maxlvl - 1, minlvl - 1, -1):
        varout[:, k] = d_p[:, k] - c_p[:, k] * varout[:, k + 1]
    return varout


def diffuse_vertical_implicit_numpy(
    *,
    a: np.ndarray,
    bb: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray,
    var: np.ndarray,
    tend: np.ndarray,
    dtime: float,
    minlvl: int,
    maxlvl: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference for 'diffuse_vertical_implicit' (mo_tmx_numerics.f90).

    Rows outside [minlvl, maxlvl] are left untouched: new_var stays zero
    (zero-initialized output field) and tend keeps its input values.
    """
    rdtime = 1.0 / dtime
    b = rdtime + bb
    d = var * rdtime + rhs
    x = tdma_solver_numpy(a=a, b=b, c=c, d=d, minlvl=minlvl, maxlvl=maxlvl)
    new_var = np.zeros_like(var)
    new_var[:, minlvl : maxlvl + 1] = x[:, minlvl : maxlvl + 1]
    tend_out = tend.copy()
    tend_out[:, minlvl : maxlvl + 1] = (
        tend[:, minlvl : maxlvl + 1]
        + (x[:, minlvl : maxlvl + 1] - var[:, minlvl : maxlvl + 1]) * rdtime
    )
    return new_var, tend_out


def _solver_input_data(
    grid: base.Grid, horizontal_dim: gtx.Dimension, vertical_start: int
) -> dict[str, gtx.Field | state_utils.ScalarType]:
    num_horizontal = grid.num_cells if horizontal_dim == dims.CellDim else grid.num_edges
    return dict(
        # a, c < 0 and bb >= |a| + |c| as produced by prepare_tridiagonal_matrix_*:
        # the system is diagonally dominant, hence the Thomas algorithm is stable.
        a=data_alloc.random_field(
            grid, horizontal_dim, dims.KDim, low=-1.0, high=-0.1, dtype=wpfloat
        ),
        b=data_alloc.random_field(
            grid, horizontal_dim, dims.KDim, low=2.5, high=4.0, dtype=wpfloat
        ),
        c=data_alloc.random_field(
            grid, horizontal_dim, dims.KDim, low=-1.0, high=-0.1, dtype=wpfloat
        ),
        rhs=data_alloc.random_field(grid, horizontal_dim, dims.KDim, dtype=wpfloat),
        var=data_alloc.random_field(grid, horizontal_dim, dims.KDim, dtype=wpfloat),
        new_var=data_alloc.zero_field(grid, horizontal_dim, dims.KDim, dtype=wpfloat),
        tend=data_alloc.random_field(grid, horizontal_dim, dims.KDim, dtype=wpfloat),
        dtime=wpfloat(2.0),
        horizontal_start=0,
        horizontal_end=gtx.int32(num_horizontal),
        vertical_start=vertical_start,
        vertical_end=gtx.int32(grid.num_levels),
    )


def solve_vertical_diffusion_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray,
    var: np.ndarray,
    tend: np.ndarray,
    dtime: float,
    vertical_start: int,
    **kwargs,
) -> dict:
    new_var, tend_out = diffuse_vertical_implicit_numpy(
        a=a,
        bb=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=var.shape[1] - 1,
    )
    return dict(new_var=new_var, tend=tend_out)


class TestSolveVerticalDiffusionCells(StencilTest):
    PROGRAM = solve_vertical_diffusion_cells
    OUTPUTS = ("new_var", "tend")
    reference = staticmethod(solve_vertical_diffusion_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return _solver_input_data(grid, dims.CellDim, vertical_start=0)


class TestSolveVerticalDiffusionCellsFromSecondLevel(StencilTest):
    """
    Half-level (w) solve case: the solve starts at vertical_start=1 (Fortran minlvl=2).

    This verifies that the scan init is applied at the start of the restricted
    KDim domain (k=1), not at k=0, and that row 0 is left untouched.
    """

    PROGRAM = solve_vertical_diffusion_cells
    OUTPUTS = ("new_var", "tend")
    reference = staticmethod(solve_vertical_diffusion_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return _solver_input_data(grid, dims.CellDim, vertical_start=1)
