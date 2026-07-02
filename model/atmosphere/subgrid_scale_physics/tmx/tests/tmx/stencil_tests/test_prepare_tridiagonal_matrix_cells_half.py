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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_cells_half import (
    prepare_tridiagonal_matrix_cells_half,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest

from .test_prepare_tridiagonal_matrix_cells import prepare_diffusion_matrix_numpy


class TestPrepareTridiagonalMatrixCellsHalf(StencilTest):
    """
    Half-level variant (lhalflvl=.TRUE.) used for the w solve, which in the Fortran
    runs over half levels minlvl=2..maxlvl=nlev (mo_vdf.f90, 'Compute_diffusion_vert_wind'),
    i.e. rows 1..nlev-1 in 0-based indexing. Row 0 must remain untouched.
    """

    PROGRAM = prepare_tridiagonal_matrix_cells_half
    OUTPUTS = ("a", "b", "c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        inv_mair: np.ndarray,
        inv_dz: np.ndarray,
        zk: np.ndarray,
        zprefac: float,
        vertical_start: int,
        **kwargs,
    ) -> dict:
        # half-level variant: lhalflvl=.TRUE. => lvlcorr_a=-1, lvlcorr_c=0
        # rows outside [vertical_start, nlev-1] stay zero (untouched zero-initialized output)
        a, b, c = prepare_diffusion_matrix_numpy(
            inv_mair=inv_mair,
            inv_dz=inv_dz,
            zk=zk,
            zprefac=zprefac,
            lvlcorr_a=-1,
            lvlcorr_c=0,
            minlvl=vertical_start,
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
            # Fortran w solve: minlvl=2 (1-based half levels) => vertical_start=1 (0-based)
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
