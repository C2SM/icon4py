# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _aggregate_surface_tiles(
    field_ocean: fa.CellField[wpfloat],
    field_ice: fa.CellField[wpfloat],
    field_land: fa.CellField[wpfloat],
    fraction_ocean: fa.CellField[wpfloat],
    fraction_ice: fa.CellField[wpfloat],
    fraction_land: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    Aggregate a per-tile field to its grid mean.

    Port of 'average_tiles' (mo_vdf_sfc.f90:1146-1157): the fraction-weighted sum
    ``X = frac_oce * X_oce + frac_ice * X_ice + frac_lnd * X_lnd``. Each term is
    guarded by ``where(frac > 0, ...)`` so an inactive tile (fraction 0) whose
    per-tile field is non-finite (the masked full-field computation runs every
    tile everywhere) contributes exactly zero instead of poisoning the sum.

    Args:
        field_ocean: ocean-tile field
        field_ice: sea-ice-tile field
        field_land: land-tile field
        fraction_ocean: ocean tile area fraction [-]
        fraction_ice: sea-ice tile area fraction [-]
        fraction_land: land tile area fraction [-]

    Returns:
        grid-mean field
    """
    return (
        where(fraction_ocean > wpfloat("0.0"), fraction_ocean * field_ocean, wpfloat("0.0"))
        + where(fraction_ice > wpfloat("0.0"), fraction_ice * field_ice, wpfloat("0.0"))
        + where(fraction_land > wpfloat("0.0"), fraction_land * field_land, wpfloat("0.0"))
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def aggregate_surface_tiles(
    field_ocean: fa.CellField[wpfloat],
    field_ice: fa.CellField[wpfloat],
    field_land: fa.CellField[wpfloat],
    fraction_ocean: fa.CellField[wpfloat],
    fraction_ice: fa.CellField[wpfloat],
    fraction_land: fa.CellField[wpfloat],
    grid_mean: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _aggregate_surface_tiles(
        field_ocean=field_ocean,
        field_ice=field_ice,
        field_land=field_land,
        fraction_ocean=fraction_ocean,
        fraction_ice=fraction_ice,
        fraction_land=fraction_land,
        out=grid_mean,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
