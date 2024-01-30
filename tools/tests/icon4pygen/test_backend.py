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
import re

import pytest
from gt4py.next import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.iterator import ir as itir
from icon4py.model.common.dimension import E2V, EdgeDim, VertexDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid

from icon4pytools.icon4pygen import backend
from icon4pytools.icon4pygen.backend import generate_gtheader, get_missing_domain_params


@pytest.mark.parametrize(
    "input_params, expected_complement",
    [
        ([backend.H_START], [backend.H_END, backend.V_START, backend.V_END]),
        ([backend.H_START, backend.H_END], [backend.V_END, backend.V_START]),
        (backend.DOMAIN_ARGS, []),
        ([], backend.DOMAIN_ARGS),
    ],
)
def test_missing_domain_args(input_params, expected_complement):
    params = [itir.Sym(id=p) for p in input_params]
    domain_boundaries = set(map(lambda s: str(s.id), get_missing_domain_params(params)))
    assert len(domain_boundaries) == len(expected_complement)
    assert domain_boundaries == set(expected_complement)


def search_for_grid_sizes(code: str) -> bool:
    patterns = [r"num_cells", r"num_edges", r"num_vertices"]
    return all(re.search(pattern, code) for pattern in patterns)


@pytest.mark.parametrize(
    "temporaries, imperative", [(True, True), (True, False), (False, True), (False, False)]
)
def test_grid_size_param_generation(temporaries, imperative):
    @field_operator
    def testee_op(a: Field[[VertexDim, KDim], float]) -> Field[[EdgeDim, KDim], float]:
        amul = a * 2.0
        return amul(E2V[0]) + amul(E2V[1])

    @program
    def testee_prog(
        a: Field[[VertexDim, KDim], float],
        out: Field[[EdgeDim, KDim], float],
    ) -> Field[[EdgeDim, KDim], float]:
        testee_op(a, out=out)

    grid = SimpleGrid()
    offset_provider = {"E2V": grid.get_offset_provider("E2V")}
    fencil = testee_prog.itir

    # validate the grid sizes appear in the generated code
    gtheader = generate_gtheader(fencil, offset_provider, None, temporaries, imperative)
    assert search_for_grid_sizes(gtheader)
