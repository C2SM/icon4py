# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from gt4py.next import Field
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common.dimension import E2V, EdgeDim, VertexDim

from icon4pytools.common import ICON4PY_MODEL_QUALIFIED_NAME


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"{ICON4PY_MODEL_QUALIFIED_NAME}.{stencil_module}.{stencil_name}:{stencil_name}"


@pytest.fixture
def testee_prog():
    @field_operator
    def testee_op(a: Field[[VertexDim], float]) -> Field[[EdgeDim], float]:
        amul = a * 2.0
        return amul(E2V[0]) + amul(E2V[1])

    @program
    def testee_prog(
        a: Field[[VertexDim], float],
        out: Field[[EdgeDim], float],
    ) -> Field[[EdgeDim], float]:
        testee_op(a, out=out)

    yield testee_prog
