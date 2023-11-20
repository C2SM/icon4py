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
import copy

import numpy as np
import pytest

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common.dimension import CellDim, EdgeDim, E2V, E2VDim, KDim, VertexDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field


from icon4pytools.icon4pygen.metadata import _get_field_infos, provide_neighbor_table, add_grid_element_sizes

chain_false_skipvalues = [
    "C2E",
    "C2V",
    "E2C",
    "E2V",
    "E2C2E",
    "E2C2EO",
    "E2C2V",
    "C2E2C",
    "C2E2CO",
    "C2E2C2E",
    "C2E2C2E2C",
    "C2E2C2E2CO",
    "C2E2C2E2C2E",
]

chain_true_skipvalues = [
    "V2C",
    "V2E",
    "E2C2V2C",
    "C2V2C",
    "C2V2CO",
    "C2V2C2E",
    "E2V2E",
    "E2V2EO",
    "E2V2E2C",
    "V2E2C",
    "V2E2C2V",
    "V2E2C2VO",
    "V2E2C2V2E",
    "V2E2C2V2E2C",
]


@pytest.mark.parametrize(
    "chain",
    chain_false_skipvalues + chain_true_skipvalues,
)
def test_provide_neighbor_table_local(chain):
    expected = False
    actual = provide_neighbor_table(chain, is_global=False)
    assert actual.has_skip_values == expected


@pytest.mark.parametrize(
    "chain",
    chain_false_skipvalues,
)
def test_provide_neighbor_table_global_false_skipvalues(chain):
    expected = False
    actual = provide_neighbor_table(chain, is_global=True)
    assert actual.has_skip_values == expected


@pytest.mark.parametrize(
    "chain",
    chain_true_skipvalues,
)
def test_provide_neighbor_table_global_true_skipvalues(chain):
    expected = True
    actual = provide_neighbor_table(chain, is_global=True)
    assert actual.has_skip_values == expected


@field_operator
def _add(
    field1: Field[[CellDim, KDim], float], field2: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    return field1 + field2


@program
def with_domain(
    a: Field[[CellDim, KDim], float],
    b: Field[[CellDim, KDim], float],
    result: Field[[CellDim, KDim], float],
    vertical_start: int,
    vertical_end: int,
    k_start: int,
    k_end: int,
):
    _add(
        a,
        b,
        out=result,
        domain={CellDim: (k_start, k_end), KDim: (vertical_start, vertical_end)},
    )


@program
def without_domain(
    a: Field[[CellDim, KDim], float],
    b: Field[[CellDim, KDim], float],
    result: Field[[CellDim, KDim], float],
):
    _add(a, b, out=result)


@program
def with_constant_domain(
    a: Field[[CellDim, KDim], float],
    b: Field[[CellDim, KDim], float],
    result: Field[[CellDim, KDim], float],
):
    _add(a, b, out=result, domain={CellDim: (0, 3), KDim: (1, 8)})


@pytest.mark.parametrize("program", [with_domain, without_domain, with_constant_domain])
def test_get_field_infos_does_not_contain_domain_args(program):
    field_info = _get_field_infos(program)
    assert len(field_info) == 3
    assert not field_info["a"].out
    assert field_info["a"].inp
    assert not field_info["b"].out
    assert field_info["b"].inp
    assert field_info["result"].out
    assert not field_info["result"].inp


@field_operator
def testee_op(a: Field[[VertexDim], float]) -> Field[[EdgeDim], float]:
    amul = a * 2.0
    return amul(E2V[0]) + amul(E2V[1])


@program
def prog(
    a: Field[[VertexDim], float],
    out: Field[[EdgeDim], float],
) -> Field[[EdgeDim], float]:
    testee_op(a, out=out)


def reference(grid, a):
    amul = a * 2.0
    return amul[grid.connectivities[E2VDim][:, 0]] + amul[grid.connectivities[E2VDim][:, 1]]


def test_add_grid_sizes():
    original_prog = copy.deepcopy(prog.past_node)
    result = add_grid_element_sizes(prog.past_node)

    new_symbols = {"num_cells", "num_edges", "num_vertices"}
    result_symbols = {param.id for param in result.params}
    for symbol in new_symbols:
        assert symbol in result_symbols, f"Symbol {symbol} is not in program params"

    type_symbols = set(result.type.definition.pos_or_kw_args.keys())
    for symbol in new_symbols:
        assert symbol in type_symbols, f"Symbol {symbol} is not in new_program_type definition"

    assert result.id == original_prog.id, "Program ID has changed"
    assert result.body == original_prog.body, "Program body has changed"
    assert result.closure_vars == original_prog.closure_vars, "Program closure_vars have changed"


def test_stencil():
    grid = SimpleGrid()
    a = random_field(grid, VertexDim)
    out = random_field(grid, EdgeDim)
    offset_provider = {"E2V": grid.get_offset_provider("E2V")}

    ref = reference(grid, a.ndarray)

    # without addition of size args
    prog(a, out, offset_provider=offset_provider)

    # todo: add size args and test for equality with original program without size args
    #   use from gt4py.next.program_processors.runners.gtfn import run_gtfn_with_temporaries_and_sizes

    assert np.allclose(ref, out)
