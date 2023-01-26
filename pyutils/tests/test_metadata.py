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
from functional.common import Field
from functional.ffront.decorator import field_operator, program

from icon4py.common.dimension import CellDim, KDim
from icon4py.pyutils.metadata import get_field_infos, provide_neighbor_table


@pytest.mark.parametrize(
    ("chain", "is_global", "expected"),
    [
        ("C2E", False, False),
        ("C2E", True, False),
        ("C2V", False, False),
        ("C2V", True, False),
        ("E2C", False, False),
        ("E2C", True, False),
        ("E2V", False, False),
        ("E2V", True, False),
        ("V2C", False, False),
        ("V2C", True, True),
        ("V2E", False, False),
        ("V2E", True, True),
        ("E2C2E", False, False),
        ("E2C2E", True, False),
        ("E2C2EO", False, False),
        ("E2C2EO", True, False),
        ("E2C2V", False, False),
        ("E2C2V", True, False),
        ("E2C2V2C", False, False),
        ("E2C2V2C", True, True),
        ("C2V2C", False, False),
        ("C2V2C", True, True),
        ("C2V2CO", False, False),
        ("C2V2CO", True, True),
        ("C2V2C2E", False, False),
        ("C2V2C2E", True, True),
        ("E2V2E", False, False),
        ("E2V2E", True, True),
        ("E2V2EO", False, False),
        ("E2V2EO", True, True),
        ("E2V2E2C", False, False),
        ("E2V2E2C", True, True),
        ("V2E2C", False, False),
        ("V2E2C", True, True),
        ("V2E2C2V", False, False),
        ("V2E2C2V", True, True),
        ("V2E2C2VO", False, False),
        ("V2E2C2VO", True, True),
        ("V2E2C2V2E", False, False),
        ("V2E2C2V2E", True, True),
        ("V2E2C2V2E2C", False, False),
        ("V2E2C2V2E2C", True, True),
        ("C2E2C", False, False),
        ("C2E2C", True, False),
        ("C2E2CO", False, False),
        ("C2E2CO", True, False),
        ("C2E2C2E", False, False),
        ("C2E2C2E", True, False),
        ("C2E2C2E2C", False, False),
        ("C2E2C2E2C", True, False),
        ("C2E2C2E2CO", False, False),
        ("C2E2C2E2CO", True, False),
        ("C2E2C2E2C2E", False, False),
        ("C2E2C2E2C2E", True, False),
    ],
)
def test_provide_neighbor_table(chain, is_global, expected):
    actual = provide_neighbor_table(chain, is_global)
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
    field_info = get_field_infos(program)
    assert len(field_info) == 3
    assert not field_info["a"].out
    assert field_info["a"].inp
    assert not field_info["b"].out
    assert field_info["b"].inp
    assert field_info["result"].out
    assert not field_info["result"].inp
