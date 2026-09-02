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
    solve_vn_vertical_diffusion,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests

from .test_prepare_scalar_diffusion_matrix import on_domain, prepare_diffusion_matrix_numpy
from .test_solve_scalar_vertical_diffusion import diffuse_vertical_implicit_numpy


def vn_vertical_diffusion_rhs_numpy(
    *,
    w: np.ndarray,
    km_ie: np.ndarray,
    inv_rhoe: np.ndarray,
    inv_ddqz_z_full_e: np.ndarray,
    u_stress: np.ndarray,
    v_stress: np.ndarray,
    primal_normal_cell_x: np.ndarray,
    primal_normal_cell_y: np.ndarray,
    c_lin_e: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    nlev: int,
    e2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference for the '2) Vertical tendency' loops of
    'Compute_diffusion_hor_wind' (mo_vdf.f90).
    """
    inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

    inv_maire = inv_ddqz_z_full_e * inv_rhoe  # (n_edges, nlev)

    # Vertical flux of the dw/dn stress at all half levels, (n_edges, nlev + 1)
    grad = km_ie * inv_del * (w[e2c[:, 1]] - w[e2c[:, 0]])

    # Interior rows: rhs(k) = (grad(k) - grad(k+1)) * inv_maire(k)
    rhs = (grad[:, :-1] - grad[:, 1:]) * inv_maire

    # Top row (jk = 1): the flux through the model top is dropped.
    rhs[:, 0] = -grad[:, 1] * inv_maire[:, 0]

    # Bottom row (jk = nlev): dwdn - net surface stress along the edge normal.
    stress_n = u_stress[e2c] * primal_normal_cell_x + v_stress[e2c] * primal_normal_cell_y
    flux_dn_e = np.sum(stress_n * c_lin_e, axis=1)  # (n_edges,)
    rhs[:, nlev - 1] = (
        grad[:, nlev - 1] * inv_maire[:, nlev - 1] - flux_dn_e * inv_maire[:, nlev - 1]
    )

    return rhs, inv_maire


class TestSolveVnVerticalDiffusion(stencil_tests.StencilTest):
    """
    Composition of the vn right-hand side, the full-level edge matrix rows
    ('prepare_diffusion_matrix_wp' with lhalflvl=.FALSE., zprefac=1) and the
    implicit solve, accumulating onto the tendency that already holds the
    horizontal stress contribution.
    """

    PROGRAM = solve_vn_vertical_diffusion
    OUTPUTS = ("tot_tend",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        w: np.ndarray,
        vn: np.ndarray,
        km_ie: np.ndarray,
        inv_rhoe: np.ndarray,
        inv_ddqz_z_full_e: np.ndarray,
        inv_ddqz_z_half_e: np.ndarray,
        u_stress: np.ndarray,
        v_stress: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        c_lin_e: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        tot_tend: np.ndarray,
        dtime: float,
        nlev: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        e2c = connectivities[dims.E2C]
        maxlvl = vertical_end - 1

        rhs, inv_maire = vn_vertical_diffusion_rhs_numpy(
            w=w,
            km_ie=km_ie,
            inv_rhoe=inv_rhoe,
            inv_ddqz_z_full_e=inv_ddqz_z_full_e,
            u_stress=u_stress,
            v_stress=v_stress,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            c_lin_e=c_lin_e,
            inv_dual_edge_length=inv_dual_edge_length,
            nlev=nlev,
            e2c=e2c,
        )
        # full-level variant: lhalflvl=.FALSE. => lvlcorr_a=0, lvlcorr_c=1
        a, b, c = prepare_diffusion_matrix_numpy(
            inv_mair=inv_maire,
            inv_dz=inv_ddqz_z_half_e,
            zk=km_ie,
            zprefac=1.0,
            lvlcorr_a=0,
            lvlcorr_c=1,
            minlvl=vertical_start,
            maxlvl=maxlvl,
        )
        tot_tend_computed = diffuse_vertical_implicit_numpy(
            a=a,
            bb=b,
            c=c,
            rhs=rhs,
            var=vn,
            tend=tot_tend,
            dtime=dtime,
            minlvl=vertical_start,
            maxlvl=maxlvl,
        )
        return dict(
            tot_tend=on_domain(
                tot_tend,
                tot_tend_computed,
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
        # Fortran: edges rl_start = grf_bdywidth_e + 1, rl_end = min_rledge_int.
        # Those zones all collapse to (0, num_edges) on the simple grid, which
        # would hide a program ignoring its domain, hence the explicit bounds.
        horizontal_start = 1
        horizontal_end = grid.num_edges - 1
        assert horizontal_start < horizontal_end

        return dict(
            # w and km_ie are half-level fields; inv_ddqz_z_half_e is the
            # half-level edge layer thickness read by the matrix rows.
            w=data_alloc.random_field(
                dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
            ),
            vn=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
            km_ie=data_alloc.random_field(
                dims.EdgeDim,
                dims.KDim,
                low=0.1,
                high=1.0,
                extend={dims.KDim: 1},
                dtype=ta.wpfloat,
            ),
            inv_rhoe=data_alloc.random_field(
                dims.EdgeDim, dims.KDim, low=0.5, high=2.0, dtype=ta.wpfloat
            ),
            inv_ddqz_z_full_e=data_alloc.random_field(
                dims.EdgeDim, dims.KDim, low=0.1, high=2.0, dtype=ta.wpfloat
            ),
            inv_ddqz_z_half_e=data_alloc.random_field(
                dims.EdgeDim,
                dims.KDim,
                low=0.1,
                high=2.0,
                extend={dims.KDim: 1},
                dtype=ta.wpfloat,
            ),
            u_stress=data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat),
            v_stress=data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat),
            primal_normal_cell_x=data_alloc.random_field(
                dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
            ),
            primal_normal_cell_y=data_alloc.random_field(
                dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
            ),
            c_lin_e=data_alloc.random_field(
                dims.EdgeDim, dims.E2CDim, low=0.1, high=0.9, dtype=ta.wpfloat
            ),
            inv_dual_edge_length=data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat),
            # the horizontal stress tendency the vertical solve accumulates onto
            tot_tend=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
            dtime=ta.wpfloat(2.0),
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=gtx.int32(horizontal_start),
            horizontal_end=gtx.int32(horizontal_end),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
