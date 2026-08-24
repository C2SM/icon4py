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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.prepare_tridiagonal_matrix_edges import (
    prepare_tridiagonal_matrix_edges,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_prepare_tridiagonal_matrix_cells import prepare_diffusion_matrix_numpy


class TestPrepareTridiagonalMatrixEdges(stencil_tests.StencilTest):
    PROGRAM = prepare_tridiagonal_matrix_edges
    OUTPUTS = ("a", "b", "c")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
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

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            inv_mair=data_alloc.random_field(
                dims.EdgeDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            inv_dz=data_alloc.random_field(
                dims.EdgeDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat
            ),
            zk=data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.1, high=2.0, dtype=wpfloat),
            a=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            b=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            c=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            zprefac=wpfloat(1.0),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
