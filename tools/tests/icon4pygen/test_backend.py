# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

import gt4py.next as gtx
import pytest
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.iterator import ir as itir

from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import E2V
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.tools.icon4pygen import backend
from icon4py.tools.icon4pygen.backend import generate_gtheader, get_missing_domain_params


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


# FIXME[#1582](tehrengruber): implement new temporary pass, then add (False, True), (True, True) cases
@pytest.mark.parametrize("temporaries, imperative", [(True, False), (False, False)])
def test_grid_size_param_generation(temporaries, imperative):
    @field_operator
    def testee_op(
        a: gtx.Field[gtx.Dims[dims.VertexDim, dims.KDim], float],
    ) -> gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], float]:
        amul = a * 2.0
        return amul(E2V[0]) + amul(E2V[1])

    @program
    def testee_prog(
        a: gtx.Field[gtx.Dims[dims.VertexDim, dims.KDim], float],
        out: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], float],
    ) -> gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], float]:
        testee_op(a, out=out)

    grid = SimpleGrid()
    offset_provider = {"E2V": grid.get_offset_provider("E2V")}
    fencil = testee_prog.itir

    # validate the grid sizes appear in the generated code
    gtheader = generate_gtheader(fencil, offset_provider, temporaries, imperative)
    assert search_for_grid_sizes(gtheader)
