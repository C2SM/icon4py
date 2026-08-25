# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import minimum, power

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _init_smagorinsky_mixing_length(
    dz_ic: fa.CellKField[wpfloat],
    geopot_agl_ic: fa.CellKField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the square of the subgrid-scale mixing length for the Smagorinsky model.

    Port of ``compute_mixing_length`` in ICON's ``mo_tmx_smagorinsky.f90``:

        lambda^2 = (Cs * Delta)^2 * (kappa * x_3)^2 / ((Cs * Delta)^2 + (kappa * x_3)^2)
                 = (Cs * Delta * x_3)^2 / ((Cs * Delta / kappa)^2 + x_3^2)

    with Cs the Smagorinsky constant, Delta the filter/grid width (capped at
    ``max_turb_scale``), x_3 the height above ground, and kappa = 0.4 the
    von Karman constant. Reference: Dipankar et al. (2015).

    Args:
        dz_ic: layer thickness centered at half levels (nlev + 1 levels)
        geopot_agl_ic: geopotential above ground at half levels (nlev + 1 levels)
        cell_area: cell area
        smag_constant: Smagorinsky constant Cs
        max_turb_scale: maximum turbulence length scale
        grav: gravitational acceleration

    Returns:
        square of the Smagorinsky mixing length at half levels
    """
    kappa = wpfloat("0.4")  # von Karman constant

    z_agl = geopot_agl_ic * (wpfloat("1.0") / grav)
    les_filter = smag_constant * minimum(
        max_turb_scale, power(dz_ic * cell_area, wpfloat("0.33333"))
    )
    return (
        (les_filter * z_agl)
        * (les_filter * z_agl)
        / ((les_filter / kappa) * (les_filter / kappa) + z_agl * z_agl)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_smagorinsky_mixing_length(
    dz_ic: fa.CellKField[wpfloat],
    geopot_agl_ic: fa.CellKField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    mixing_length_sq: fa.CellKField[wpfloat],
    smag_constant: wpfloat,
    max_turb_scale: wpfloat,
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _init_smagorinsky_mixing_length(
        dz_ic=dz_ic,
        geopot_agl_ic=geopot_agl_ic,
        cell_area=cell_area,
        smag_constant=smag_constant,
        max_turb_scale=max_turb_scale,
        grav=grav,
        out=mixing_length_sq,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
