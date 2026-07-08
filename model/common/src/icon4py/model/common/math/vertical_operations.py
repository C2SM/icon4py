# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Vertical level operations on unstructured grid fields.

Contains averaging and difference operations between adjacent vertical levels
on cell and edge fields.
"""

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=wpfloat("0.0"))
def _accumulate_from_top(state: wpfloat, integrand: wpfloat) -> wpfloat:
    return state + integrand


@gtx.field_operator
def _compute_vertical_integral(
    integrand: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Compute the running vertical sum of ``integrand`` from the top of the column downwards.

    The value at level k is sum_{j<=k} integrand(j), so the value at the last full
    level is the column integral. Callers pre-multiply the integrand with the
    appropriate weights (e.g. ``rho * dz`` for mass-weighted vertical integrals as in
    the ``*_vi`` diagnostics of ``Update_diagnostics`` in ICON's ``mo_vdf_atmo.f90``).

    Args:
        integrand: integrand on full levels

    Returns:
        running vertical sum of the integrand
    """
    return _accumulate_from_top(integrand)


@gtx.field_operator
def average_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(KDim + 1))


@gtx.field_operator
def average_level_plus1_on_edges(
    half_level_field: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over an edge field for storage
    in the corresponding full levels.
    Args:
        half_level_field: fa.EdgeKField[wpfloat]

    Returns: fa.EdgeKField[wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(KDim + 1))


@gtx.field_operator
def difference_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field - half_level_field(KDim + 1)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_integral(  # noqa: PLR0917 [too-many-positional-arguments]
    integrand: fa.CellKField[wpfloat],
    vertical_integral: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_integral(
        integrand=integrand,
        out=vertical_integral,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    input_field: fa.EdgeKField[wpfloat],
    average: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    average_level_plus1_on_edges(
        half_level_field=input_field,
        out=average,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    input_field: fa.CellKField[wpfloat],
    average: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    average_level_plus1_on_cells(
        half_level_field=input_field,
        out=average,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
