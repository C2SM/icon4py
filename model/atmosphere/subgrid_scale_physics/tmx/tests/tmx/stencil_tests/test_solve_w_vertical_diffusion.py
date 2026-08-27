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
    solve_w_vertical_diffusion,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_prepare_scalar_diffusion_matrix import on_domain, prepare_diffusion_matrix_numpy
from .test_solve_scalar_vertical_diffusion import diffuse_vertical_implicit_numpy


def w_vertical_diffusion_rhs_numpy(
    *,
    rho_ic: np.ndarray,
    inv_ddqz_z_half: np.ndarray,
    km_c: np.ndarray,
    div_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference for the first loop of 'Compute_diffusion_vert_wind' (mo_vdf.f90).

    rho_ic and inv_ddqz_z_half are half-level fields (nlev + 1 rows), km_c and
    div_c are full-level fields read at the full levels below (k) and above
    (k-1) the half level k. Only the half-level rows 1..nlev-1 of the
    right-hand side are defined (Fortran jk = 2..nlev).
    """
    z_1by3 = 1.0 / 3.0
    inv_rho_ic = 1.0 / rho_ic
    inv_mair_ic = inv_rho_ic * inv_ddqz_z_half
    rhs = np.zeros_like(rho_ic)
    rhs[:, 1:-1] = (
        2.0
        * inv_mair_ic[:, 1:-1]
        * (km_c[:, 1:] * z_1by3 * div_c[:, 1:] - km_c[:, :-1] * z_1by3 * div_c[:, :-1])
    )
    return rhs, inv_rho_ic, inv_mair_ic


def modify_w_diffusion_matrix_boundary_numpy(
    *,
    b: np.ndarray,
    km_c: np.ndarray,
    inv_dz: np.ndarray,
    inv_mair_ic: np.ndarray,
    minlvl: int,
    maxlvl: int,
) -> np.ndarray:
    """
    Reference for the w = 0 boundary-condition terms of the main diagonal
    (mo_vdf.f90, 'Compute_diffusion_vert_wind'):
        b(2)    += 2 * km_c(1)    * inv_dzf(1)    * inv_mair_ic(2)
        b(nlev) += 2 * km_c(nlev) * inv_dzf(nlev) * inv_mair_ic(nlev)
    (1-based rows) -> 0-based rows minlvl and maxlvl.
    """
    b_out = b.copy()
    b_out[:, minlvl] += 2.0 * km_c[:, minlvl - 1] * inv_dz[:, minlvl - 1] * inv_mair_ic[:, minlvl]
    b_out[:, maxlvl] += 2.0 * km_c[:, maxlvl] * inv_dz[:, maxlvl] * inv_mair_ic[:, maxlvl]
    return b_out


class TestSolveWVerticalDiffusion(stencil_tests.StencilTest):
    """
    Composition of the w right-hand side, the half-level matrix rows
    ('prepare_diffusion_matrix_wp' with lhalflvl=.TRUE., zprefac=2), the w = 0
    boundary terms of the main diagonal and the implicit solve.

    The Fortran (mo_vdf.f90, 'Compute_diffusion_vert_wind') runs on half-level
    rows jk = 2..nlev (1-based) -> 0-based rows 1..nlev-1, hence the vertical
    bounds (1, nlev) on the (nlev + 1)-row half-level fields. Rows 0 and nlev
    stay untouched.
    """

    PROGRAM = solve_w_vertical_diffusion
    OUTPUTS = ("inv_rho_ic", "tend")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        w: np.ndarray,
        rho_ic: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        km_c: np.ndarray,
        div_c: np.ndarray,
        inv_rho_ic: np.ndarray,
        tend: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        minlvl = vertical_start
        maxlvl = vertical_end - 1

        rhs, inv_rho_ic_computed, inv_mair_ic = w_vertical_diffusion_rhs_numpy(
            rho_ic=rho_ic, inv_ddqz_z_half=inv_ddqz_z_half, km_c=km_c, div_c=div_c
        )
        # half-level variant: lhalflvl=.TRUE. => lvlcorr_a=-1, lvlcorr_c=0
        a, b, c = prepare_diffusion_matrix_numpy(
            inv_mair=inv_mair_ic,
            inv_dz=inv_ddqz_z_full,
            zk=km_c,
            zprefac=2.0,
            lvlcorr_a=-1,
            lvlcorr_c=0,
            minlvl=minlvl,
            maxlvl=maxlvl,
        )
        b = modify_w_diffusion_matrix_boundary_numpy(
            b=b,
            km_c=km_c,
            inv_dz=inv_ddqz_z_full,
            inv_mair_ic=inv_mair_ic,
            minlvl=minlvl,
            maxlvl=maxlvl,
        )
        tend_computed = diffuse_vertical_implicit_numpy(
            a=a,
            bb=b,
            c=c,
            rhs=rhs,
            var=w,
            tend=tend,
            dtime=dtime,
            minlvl=minlvl,
            maxlvl=maxlvl,
        )

        domain = dict(
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
        return dict(
            inv_rho_ic=on_domain(inv_rho_ic, inv_rho_ic_computed, **domain),
            tend=on_domain(tend, tend_computed, **domain),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        half_level = {dims.KDim: 1}
        return dict(
            w=data_alloc.random_field(dims.CellDim, dims.KDim, extend=half_level, dtype=wpfloat),
            rho_ic=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.5, high=2.0, extend=half_level, dtype=wpfloat
            ),
            inv_ddqz_z_half=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.1, high=2.0, extend=half_level, dtype=wpfloat
            ),
            inv_ddqz_z_full=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            km_c=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=wpfloat),
            div_c=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            inv_rho_ic=data_alloc.zero_field(
                dims.CellDim, dims.KDim, extend=half_level, dtype=wpfloat
            ),
            tend=data_alloc.random_field(dims.CellDim, dims.KDim, extend=half_level, dtype=wpfloat),
            dtime=wpfloat(2.0),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            # Fortran jk = 2..nlev (1-based half levels) -> rows 1..nlev-1
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(grid.num_levels),
        )
