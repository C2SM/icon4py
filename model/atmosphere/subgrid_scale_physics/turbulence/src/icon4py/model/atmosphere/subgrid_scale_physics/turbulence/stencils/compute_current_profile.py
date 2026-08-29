# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Current vertical profile of a diffused variable, with a zero surface value.

Translated from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE 'vertdiff'
(:646-694 at icon commit 26d6b98cce), the block headed "Providing the profiles of
concentrations and their current tendencies" together with the "Surface concentrations" block
below it. The scientific commentary in that file is by Matthias Raschendorfer (DWD).

    DO k=k_st_up,ke:  cur_prof(i,k) = dvar_av(i,k)
    cur_prof(i,ke1) = z0          !no-slip / zero-concentration condition

'cur_prof' is the array the solve reads and the tendency is measured against; it carries the
model variable on the 'ke' main levels and its lower boundary value on the surface row, which
is what makes it 'ke1' rows long for a variable that has 'ke'.

THE SURFACE ROW IS ZERO FOR THREE OF THE FIVE VARIABLES. The Fortran picks between four
possibilities (:661-694):

    ASSOCIATED(dvar(n)%sv)                  cur_prof(ke1) = the surface value
    n <= nvel .OR. tdc%ilow_def_cond == 2   cur_prof(ke1) = 0
    otherwise (ilow_def_cond == 1)          cur_prof(ke1) = dvar_av(ke)   ! zero-flux

The two wind components take the second branch by "no-slip-condition for momentum"; cloud water
takes it too, because it has no surface value and 'ilow_def_cond' defaults to 2
(mo_turbdiff_config.f90:322). Water vapour and temperature DO have a surface value, but under
'lsfluse' it is a flux density rather than a concentration and
'compute_surface_profile_value_from_flux_gradient' overwrites this row before the solve reads
it -- so running this program for water vapour and then overwriting is exactly what the Fortran
does. The 'ilow_def_cond == 1' zero-flux branch is not ported; nothing in 'icon/run/' sets it.

'itndcon = 0' at the ported call site (mo_nwp_turbdiff_interface.f90), so the "Current
tendencies" block at :697-717 -- which would fill 'dif_tend' with the explicit tendency -- is
dead, and 'dif_tend' enters the solve as pure output.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _copy_current_profile(variable: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """'cur_prof(k) = dvar_av(k)' on the main levels."""
    return variable


@gtx.field_operator
def _zero_surface_profile_value() -> fa.CellKField[wpfloat]:
    """'cur_prof(ke1) = z0'."""
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_current_profile(
    variable: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Lay a variable out as a profile with a zero lower boundary value.

    The two statements write disjoint rows and neither reads what the other writes, so their
    order is immaterial; they are one program because the Fortran boundary block exists only to
    give the array its 'ke1'-th row.

    Args:
        variable: The model variable on the main levels, in its own units.
        current_profile: Output, 'cur_prof'. Rows 'vertical_start' to 'vertical_end - 1' get the
            variable, row 'vertical_end' the zero boundary value.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First diffused main level; 0 for every variable here, Fortran
            'k_st_up = dvar(n)%kstart'. It is 'kstart_cloud' for cloud water, which this capture
            has at 1 (Fortran), i.e. 0.
        vertical_end: 'nlev'; also the surface row, which is written by the second statement.
    """
    _copy_current_profile(
        variable=variable,
        out=current_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _zero_surface_profile_value(
        out=current_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
