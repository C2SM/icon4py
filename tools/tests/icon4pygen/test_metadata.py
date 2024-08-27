# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import dimension as dims

from icon4pytools.icon4pygen.metadata import _get_field_infos, provide_neighbor_table


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim

chain_false_skipvalues = [
    "C2E",
    "C2V",
    "E2C",
    "E2V",
    "E2ECV",
    "E2C2E",
    "E2C2EO",
    "E2C2V",
    "C2E2C",
    "C2CEC",
    "C2E2CO",
    "C2E2C2E",
    "C2CECEC",
]

chain_true_skipvalues = [
    "V2C",
    "V2E",
    "V2E2V",
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
    field1: Field[[dims.CellDim, dims.KDim], float], field2: Field[[dims.CellDim, dims.KDim], float]
) -> Field[[dims.CellDim, dims.KDim], float]:
    return field1 + field2


@program
def with_domain(
    a: Field[[dims.CellDim, dims.KDim], float],
    b: Field[[dims.CellDim, dims.KDim], float],
    result: Field[[dims.CellDim, dims.KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _add(
        a,
        b,
        out=result,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@program
def without_domain(
    a: Field[[dims.CellDim, dims.KDim], float],
    b: Field[[dims.CellDim, dims.KDim], float],
    result: Field[[dims.CellDim, dims.KDim], float],
):
    _add(a, b, out=result)


@program
def with_constant_domain(
    a: Field[[dims.CellDim, dims.KDim], float],
    b: Field[[dims.CellDim, dims.KDim], float],
    result: Field[[dims.CellDim, dims.KDim], float],
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
