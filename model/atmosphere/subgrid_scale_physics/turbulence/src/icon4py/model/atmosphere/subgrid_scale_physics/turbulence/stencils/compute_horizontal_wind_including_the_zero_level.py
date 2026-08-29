# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The horizontal wind on the levels of the conserved-variable array, zero level included.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section 0),
lines 1043-1063 at icon commit 26d6b98cce, the commit that produced the reference capture: the
two wind assignments of the loop headed "Berechnung der horizontalen Windgeschwindigkeiten und
Schichtdicken" ("Calculation of the horizontal wind speeds and the layer thicknesses") and the
one-row loop that follows it. The scientific commentary in that file is by Matthias Raschendorfer
(DWD).

    DO k=1,ke
      zvari(i,k,u_m)=u(i,k)
      zvari(i,k,v_m)=v(i,k)
    ...
    zvari(i,ke1,u_m)=zvari(i,ke,u_m)*(z1-tfm(i))
    zvari(i,ke1,v_m)=zvari(i,ke,v_m)*(z1-tfm(i))

ONE PROGRAM, NOT TWO. The Fortran writes the surface row in a loop of its own because the second
statement reads the storage the first one just filled; the port reads 'u' and 'v' directly and
has no such constraint. The two rows are the same output field taking a different expression,
which is the case the port's boundary-row rule (spec D14, package README "Boundary rows") says
to merge with 'concat_where'. The embedded backend cannot execute 'concat_where' in gt4py 1.1.10,
so the datatest carries 'uses_concat_where' and xfails there.

WHAT THE SURFACE ROW MEANS. 'tfm' is the fraction of the momentum transfer resistance that the
laminar sub-layer of the Prandtl layer accounts for, so '1 - tfm' reduces the wind of the lowest
main level to the wind at the top of the roughness layer -- the "zero level" the scheme takes as
its lower boundary. The other three conserved variables get their zero-level value from
'adjust_satur_equil' instead; only the wind has this closed form, because momentum has no
thermodynamics.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _wind_including_the_zero_level(
    wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """One wind component on the main levels, with its Prandtl-layer boundary value below."""
    return concat_where(
        dims.KDim == nlev,
        wind(Koff[-1]) * (wpfloat("1.0") - laminar_reduction_factor_for_momentum),
        wind,
    )


@gtx.field_operator
def _compute_horizontal_wind_including_the_zero_level(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Both wind components on the levels of the conserved-variable array.

    Run the program with 'vertical_start = 0', 'vertical_end = nlev + 1'; 'nlev' must be that
    same last row, since it is what selects the reduced value.

    Args:
        zonal_wind: 'u' at the mass centre, main levels [m/s]
        meridional_wind: 'v' at the mass centre, main levels [m/s]
        laminar_reduction_factor_for_momentum: 'tfm' [-]
        nlev: 'ke', the row of the zero level

    Returns:
        'zvari(:,:,u_m)' and 'zvari(:,:,v_m)' [m/s]
    """
    return (
        _wind_including_the_zero_level(zonal_wind, laminar_reduction_factor_for_momentum, nlev),
        _wind_including_the_zero_level(
            meridional_wind, laminar_reduction_factor_for_momentum, nlev
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_wind_including_the_zero_level(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
    zonal_wind_on_conserved_variable_levels: fa.CellKField[wpfloat],
    meridional_wind_on_conserved_variable_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_wind_including_the_zero_level(
        zonal_wind=zonal_wind,
        meridional_wind=meridional_wind,
        laminar_reduction_factor_for_momentum=laminar_reduction_factor_for_momentum,
        nlev=nlev,
        out=(
            zonal_wind_on_conserved_variable_levels,
            meridional_wind_on_conserved_variable_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
