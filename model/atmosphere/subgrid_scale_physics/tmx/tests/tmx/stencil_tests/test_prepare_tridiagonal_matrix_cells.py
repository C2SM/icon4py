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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_cells import (
    prepare_tridiagonal_matrix_cells,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


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


class TestPrepareTridiagonalMatrixCells(StencilTest):
    PROGRAM = prepare_tridiagonal_matrix_cells
    OUTPUTS = ("a", "b", "c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        inv_mair: np.ndarray,
        inv_dz: np.ndarray,
        zk: np.ndarray,
        zprefac: float,
        **kwargs,
    ) -> dict:
        # full-level variant: lhalflvl=.FALSE. => lvlcorr_a=0, lvlcorr_c=1
        a, b, c = prepare_diffusion_matrix_numpy(
            inv_mair=inv_mair,
            inv_dz=inv_dz,
            zk=zk,
            zprefac=zprefac,
            lvlcorr_a=0,
            lvlcorr_c=1,
            minlvl=0,
            maxlvl=inv_mair.shape[1] - 1,
        )
        return dict(a=a, b=b, c=c)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            inv_mair=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            inv_dz=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            zk=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            a=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            b=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            c=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            zprefac=wpfloat(0.5),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
