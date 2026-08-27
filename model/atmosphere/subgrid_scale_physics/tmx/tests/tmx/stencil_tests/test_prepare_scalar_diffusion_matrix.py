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
    prepare_scalar_diffusion_matrix,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def prepare_diffusion_matrix_numpy(
    *,
    inv_mair: np.ndarray,
    inv_dz: np.ndarray,
    zk: np.ndarray,
    zprefac: float,
    lvlcorr_a: int,
    lvlcorr_c: int,
    minlvl: int,
    maxlvl: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference for 'prepare_diffusion_matrix_wp' (mo_vdf_atmo.f90)."""
    a = np.zeros_like(inv_mair)
    b = np.zeros_like(inv_mair)
    c = np.zeros_like(inv_mair)
    # interior rows
    for jk in range(minlvl + 1, maxlvl):
        jk_corr_a = jk + lvlcorr_a
        jk_corr_c = jk + lvlcorr_c
        a[:, jk] = -zprefac * zk[:, jk_corr_a] * inv_dz[:, jk_corr_a] * inv_mair[:, jk]
        c[:, jk] = -zprefac * zk[:, jk_corr_c] * inv_dz[:, jk_corr_c] * inv_mair[:, jk]
        b[:, jk] = -a[:, jk] - c[:, jk]
    # upper boundary row
    jk_corr_c = minlvl + lvlcorr_c
    a[:, minlvl] = 0.0
    c[:, minlvl] = -zprefac * zk[:, jk_corr_c] * inv_dz[:, jk_corr_c] * inv_mair[:, minlvl]
    b[:, minlvl] = -c[:, minlvl]
    # lower boundary row
    jk_corr_a = maxlvl + lvlcorr_a
    a[:, maxlvl] = -zprefac * zk[:, jk_corr_a] * inv_dz[:, jk_corr_a] * inv_mair[:, maxlvl]
    c[:, maxlvl] = 0.0
    b[:, maxlvl] = -a[:, maxlvl]
    return a, b, c


def on_domain(
    out: np.ndarray,
    computed: np.ndarray,
    *,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> np.ndarray:
    """Value of an output field written by a program only on its domain."""
    result = out.copy()
    result[horizontal_start:horizontal_end, vertical_start:vertical_end] = computed[
        horizontal_start:horizontal_end, vertical_start:vertical_end
    ]
    return result


class TestPrepareScalarDiffusionMatrix(stencil_tests.StencilTest):
    """
    Composition of 1/air_mass with the full-level cell matrix rows
    ('prepare_diffusion_matrix_wp' with lhalflvl=.FALSE.).
    """

    PROGRAM = prepare_scalar_diffusion_matrix
    OUTPUTS = ("inv_air_mass", "a", "b", "c")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        air_mass: np.ndarray,
        inv_dz: np.ndarray,
        zk: np.ndarray,
        inv_air_mass: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        zprefac: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        inv_mair = 1.0 / air_mass
        # full-level variant: lhalflvl=.FALSE. => lvlcorr_a=0, lvlcorr_c=1
        a_computed, b_computed, c_computed = prepare_diffusion_matrix_numpy(
            inv_mair=inv_mair,
            inv_dz=inv_dz,
            zk=zk,
            zprefac=zprefac,
            lvlcorr_a=0,
            lvlcorr_c=1,
            minlvl=vertical_start,
            maxlvl=vertical_end - 1,
        )
        domain = dict(
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
        return dict(
            inv_air_mass=on_domain(inv_air_mass, inv_mair, **domain),
            a=on_domain(a, a_computed, **domain),
            b=on_domain(b, b_computed, **domain),
            c=on_domain(c, c_computed, **domain),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        # Fortran: cells rl_start = grf_bdywidth_c + 1, rl_end = min_rlcell_int
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            air_mass=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.5, high=10.0, dtype=wpfloat
            ),
            # zk (kh_ic) and inv_dz (inv_ddqz_z_half) are half-level fields
            inv_dz=data_alloc.random_field(
                dims.CellDim,
                dims.KDim,
                low=0.1,
                high=2.0,
                extend={dims.KDim: 1},
                dtype=wpfloat,
            ),
            zk=data_alloc.random_field(
                dims.CellDim,
                dims.KDim,
                low=0.1,
                high=2.0,
                extend={dims.KDim: 1},
                dtype=wpfloat,
            ),
            inv_air_mass=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            a=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            b=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            c=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            zprefac=wpfloat(0.5),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
