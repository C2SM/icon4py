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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.scalar_diffusion import (
    apply_explicit_scalar_vertical_diffusion,
    solve_scalar_vertical_diffusion,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_prepare_scalar_diffusion_matrix import on_domain


def surface_flux_rhs_numpy(
    *,
    sfc_flx: np.ndarray,
    inv_air_mass: np.ndarray,
    prefac: float,
    maxlvl: int,
) -> np.ndarray:
    """
    Reference for the right-hand-side rows of 'Compute_diffusion_hydrometeors'
    and 'Compute_diffusion_temperature' (mo_vdf.f90): the surface flux enters
    the bottom row only, every other row is zero.
    """
    rhs = np.zeros_like(inv_air_mass)
    rhs[:, maxlvl] = -sfc_flx * prefac * inv_air_mass[:, maxlvl]
    return rhs


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
) -> np.ndarray:
    """
    Reference for 'diffuse_vertical_implicit' (mo_tmx_numerics.f90).

    Rows outside [minlvl, maxlvl] keep their input tendency. The tridiagonal
    solution only enters through the tendency and is not returned.
    """
    rdtime = 1.0 / dtime
    b = rdtime + bb
    d = var * rdtime + rhs
    x = tdma_solver_numpy(a=a, b=b, c=c, d=d, minlvl=minlvl, maxlvl=maxlvl)
    tend_out = tend.copy()
    tend_out[:, minlvl : maxlvl + 1] = (
        tend[:, minlvl : maxlvl + 1]
        + (x[:, minlvl : maxlvl + 1] - var[:, minlvl : maxlvl + 1]) * rdtime
    )
    return tend_out


def diffuse_vertical_explicit_numpy(
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray,
    var: np.ndarray,
    tend: np.ndarray,
    minlvl: int,
    maxlvl: int,
) -> np.ndarray:
    """Reference for 'diffuse_vertical_explicit' (mo_tmx_numerics.f90)."""
    tend_out = tend.copy()
    # interior rows
    for jk in range(minlvl + 1, maxlvl):
        tend_out[:, jk] = (
            tend[:, jk]
            - a[:, jk] * var[:, jk - 1]
            - b[:, jk] * var[:, jk]
            - c[:, jk] * var[:, jk + 1]
            + rhs[:, jk]
        )
    # upper boundary row
    tend_out[:, minlvl] = (
        tend[:, minlvl]
        - b[:, minlvl] * var[:, minlvl]
        - c[:, minlvl] * var[:, minlvl + 1]
        + rhs[:, minlvl]
    )
    # lower boundary row
    tend_out[:, maxlvl] = (
        tend[:, maxlvl]
        - a[:, maxlvl] * var[:, maxlvl - 1]
        - b[:, maxlvl] * var[:, maxlvl]
        + rhs[:, maxlvl]
    )
    return tend_out


def scalar_solver_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper,
    grid: base.Grid,
    *,
    vertical_start: int,
    dtime: float | None = None,
) -> dict[str, Any]:
    """
    Arguments shared by the implicit and the explicit scalar solve. Only the
    implicit scheme takes a time step, hence the optional ``dtime``.
    """
    # Fortran: cells rl_start = grf_bdywidth_c + 1, rl_end = min_rlcell_int
    cell_domain = h_grid.domain(dims.CellDim)
    horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    assert horizontal_start < horizontal_end

    time_step = {} if dtime is None else {"dtime": wpfloat(dtime)}
    return dict(
        # a, c < 0 and b >= |a| + |c| as produced by prepare_scalar_diffusion_matrix:
        # the system is diagonally dominant, hence the Thomas algorithm is stable.
        a=data_alloc.random_field(dims.CellDim, dims.KDim, low=-1.0, high=-0.1, dtype=wpfloat),
        b=data_alloc.random_field(dims.CellDim, dims.KDim, low=2.5, high=4.0, dtype=wpfloat),
        c=data_alloc.random_field(dims.CellDim, dims.KDim, low=-1.0, high=-0.1, dtype=wpfloat),
        sfc_flx=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
        inv_air_mass=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.0e-4, high=1.0e-1, dtype=wpfloat
        ),
        var=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        tend=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        prefac=wpfloat(0.9),
        **time_step,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(vertical_start),
        vertical_end=gtx.int32(grid.num_levels),
    )


def solve_scalar_vertical_diffusion_reference(
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    sfc_flx: np.ndarray,
    inv_air_mass: np.ndarray,
    var: np.ndarray,
    tend: np.ndarray,
    prefac: float,
    dtime: float,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    **kwargs: Any,
) -> dict:
    maxlvl = vertical_end - 1
    rhs = surface_flux_rhs_numpy(
        sfc_flx=sfc_flx, inv_air_mass=inv_air_mass, prefac=prefac, maxlvl=maxlvl
    )
    tend_computed = diffuse_vertical_implicit_numpy(
        a=a,
        bb=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=maxlvl,
    )
    return dict(
        tend=on_domain(
            tend,
            tend_computed,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
    )


class TestSolveScalarVerticalDiffusion(stencil_tests.StencilTest):
    """
    Composition of the surface-flux right-hand side with the implicit vertical
    solve; only the accumulated tendency is returned.
    """

    PROGRAM = solve_scalar_vertical_diffusion
    OUTPUTS = ("tend",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return solve_scalar_vertical_diffusion_reference(**kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return scalar_solver_input_data(data_alloc, grid, vertical_start=0, dtime=2.0)


class TestSolveScalarVerticalDiffusionFromSecondLevel(stencil_tests.StencilTest):
    """
    The same solve restricted to vertical_start=1: the scan init must be applied
    at the start of the restricted KDim domain (k=1), not at k=0, and row 0 must
    be left untouched.
    """

    PROGRAM = solve_scalar_vertical_diffusion
    OUTPUTS = ("tend",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return solve_scalar_vertical_diffusion_reference(**kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return scalar_solver_input_data(data_alloc, grid, vertical_start=1, dtime=2.0)


class TestApplyExplicitScalarVerticalDiffusion(stencil_tests.StencilTest):
    """
    Composition of the surface-flux right-hand side with the explicit vertical
    diffusion tendency.
    """

    PROGRAM = apply_explicit_scalar_vertical_diffusion
    OUTPUTS = ("tend",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        sfc_flx: np.ndarray,
        inv_air_mass: np.ndarray,
        var: np.ndarray,
        tend: np.ndarray,
        prefac: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        maxlvl = vertical_end - 1
        rhs = surface_flux_rhs_numpy(
            sfc_flx=sfc_flx, inv_air_mass=inv_air_mass, prefac=prefac, maxlvl=maxlvl
        )
        tend_computed = diffuse_vertical_explicit_numpy(
            a=a,
            b=b,
            c=c,
            rhs=rhs,
            var=var,
            tend=tend,
            minlvl=vertical_start,
            maxlvl=maxlvl,
        )
        return dict(
            tend=on_domain(
                tend,
                tend_computed,
                horizontal_start=horizontal_start,
                horizontal_end=horizontal_end,
                vertical_start=vertical_start,
                vertical_end=vertical_end,
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return scalar_solver_input_data(data_alloc, grid, vertical_start=0)
