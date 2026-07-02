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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_explicit_vertical_diffusion_cells import (
    apply_explicit_vertical_diffusion_cells,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


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


class TestApplyExplicitVerticalDiffusionCells(StencilTest):
    PROGRAM = apply_explicit_vertical_diffusion_cells
    OUTPUTS = ("tend",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        rhs: np.ndarray,
        var: np.ndarray,
        tend: np.ndarray,
        **kwargs,
    ) -> dict:
        tend_out = diffuse_vertical_explicit_numpy(
            a=a,
            b=b,
            c=c,
            rhs=rhs,
            var=var,
            tend=tend,
            minlvl=0,
            maxlvl=var.shape[1] - 1,
        )
        return dict(tend=tend_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            a=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            b=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            c=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            rhs=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            var=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            tend=data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
