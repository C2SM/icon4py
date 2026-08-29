# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Current vertical profile of the potential temperature.

Translated from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE 'vertdiff'
(:646-659 at icon commit 26d6b98cce), the 'n.EQ.tem' branch of the block headed "Providing the
profiles of concentrations and their current tendencies". The scientific commentary in that
file is by Matthias Raschendorfer (DWD).

    cur_prof(i,k) = dvar_av(i,k)/epr(i,k)   !potential temperature
                                            !Note: here 'dvar_av' points to ordinary temperature

Temperature is the one first-order variable that is not diffused in its own units. The turbulent
flux of heat is a flux of potential temperature, so the profile is divided by the Exner pressure
on the way in and the resulting tendency is multiplied by it on the way out -- see
'compute_and_apply_potential_temperature_diffusion_tendency'.

The surface row is not written here: under 'lsfluse' the lower boundary condition for
temperature is a heat-flux density, and
'compute_surface_profile_value_from_flux_gradient' provides that row. The dead assignments the
Fortran makes first ('cur_prof(ke1) = shfl_s', then divided by 'eprs') are not translated.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_current_potential_temperature_profile(
    temperature: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'t(k)/epr(k)' [K]."""
    return temperature / exner_factor


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_current_potential_temperature_profile(
    temperature: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Lay the temperature out as a potential-temperature profile on the main levels.

    Args:
        temperature: 't' [K].
        exner_factor: 'epr', the Exner pressure on the main levels [-].
        current_profile: Output, 'cur_prof' [K]; the surface row is not written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First diffused main level; 0.
        vertical_end: 'nlev'.
    """
    _compute_current_potential_temperature_profile(
        temperature=temperature,
        exner_factor=exner_factor,
        out=current_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
