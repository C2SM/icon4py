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
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common.dimension import CellDim, KDim

from icon4pytools.icon4pygen.metadata import _get_field_infos, provide_neighbor_table


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
    actual = provide_neighbor_table(chain, is_global=False, force_skip_values=False)
    assert actual.has_skip_values == expected


@pytest.mark.parametrize(
    "chain",
    chain_false_skipvalues,
)
def test_provide_neighbor_table_global_false_skipvalues(chain):
    expected = False
    actual = provide_neighbor_table(chain, is_global=True, force_skip_values=False)
    assert actual.has_skip_values == expected


@pytest.mark.parametrize(
    "chain",
    chain_true_skipvalues,
)
def test_provide_neighbor_table_global_true_skipvalues(chain):
    expected = True
    actual = provide_neighbor_table(chain, is_global=True, force_skip_values=False)
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
