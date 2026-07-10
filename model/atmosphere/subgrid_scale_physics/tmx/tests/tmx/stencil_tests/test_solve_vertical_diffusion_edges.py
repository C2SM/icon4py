# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.solve_vertical_diffusion_edges import (
    solve_vertical_diffusion_edges,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing.stencil_tests import StencilTest

from .test_solve_vertical_diffusion_cells import (
    _solver_input_data,
    solve_vertical_diffusion_reference,
)


class TestSolveVerticalDiffusionEdges(StencilTest):
    """Implicit vertical diffusion solve with EdgeDim as horizontal dimension (vn diffusion)."""

    PROGRAM = solve_vertical_diffusion_edges
    OUTPUTS = ("new_var", "tend")
    reference = staticmethod(solve_vertical_diffusion_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return _solver_input_data(grid, dims.EdgeDim, vertical_start=0)
