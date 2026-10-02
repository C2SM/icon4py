# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Constant fields shaped like another field.

``value_of_size_*(value, field)`` is ``value`` on the domain of ``field``. Use it for the
scalar branch of a ``concat_where`` whose mask leaves that branch on an open range: the
embedded backend (gt4py 1.2.2) cannot bound a scalar branch there and fails with
"Cannot compute length of open 'UnitRange'". ``field`` must be a field that exists on the
whole output domain (an input, not a shifted expression). The result does not depend on
the values of ``field``, NaN and inf included.

One copy per (dimensions, dtype) because GT4Py field operators have no generics.
"""

from gt4py import next as gtx
from gt4py.next import where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def value_of_size_on_cells_on_half_levels_wp(
    value: wpfloat, field: fa.CellKHalfField[wpfloat]
) -> fa.CellKHalfField[wpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_cells_on_half_levels_vp(
    value: vpfloat, field: fa.CellKHalfField[vpfloat]
) -> fa.CellKHalfField[vpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_cells_on_half_levels_bool(
    value: bool, field: fa.CellKHalfField[vpfloat]
) -> fa.CellKHalfField[bool]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_cells_on_model_levels_wp(
    value: wpfloat, field: fa.CellKField[wpfloat]
) -> fa.CellKField[wpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_edges_on_model_levels_wp(
    value: wpfloat, field: fa.EdgeKField[wpfloat]
) -> fa.EdgeKField[wpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_edges_wp(
    value: wpfloat, field: fa.EdgeField[wpfloat]
) -> fa.EdgeField[wpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]


@gtx.field_operator
def value_of_size_on_half_levels_wp(
    value: wpfloat, field: fa.KHalfField[wpfloat]
) -> fa.KHalfField[wpfloat]:
    return where(field == field, value, value)  # noqa: PLR0124 [comparison-with-itself]
