# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery of the true TKE profile after diffusing the virtual one.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
"9) Aufdatieren des TKE-Profils durch die (erweiterte) Diffusions-Tendenz" ("Updating the TKE
profile by the (extended) diffusion tendency"), the block under 'IF (lcircterm)' (:2451-2462
at icon commit 26d6b98cce). The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

    IF (lcircterm) THEN !es wurden virtuelle Effektiv-Profile benutzt
       DO k = 2, ke
          upd_prof(i,k) = sav_prof(i,k) + upd_prof(i,k) - cur_prof(i,k) !aufdatierte echte Profile

"virtual effective profiles were used" / "updated true profiles". When the raw circulation term
is active, section 8) does not diffuse the TKE profile itself: it builds a virtual profile
whose extra curvature carries the circulation flux (turb_diffusion.f90:2353-2385), and the
diffusion of that profile is what produces the circulation tendency implicitly. 'upd_prof' and
'cur_prof' are then both virtual, their difference is the increment the diffusion produced, and
adding it to the true profile 'sav_prof' that section 6) saved gives the updated true profile.

The Fortran association is 'sav_prof + upd_prof' first and the subtraction second, and it is
kept: the three terms are of the same magnitude and the two orders round differently.

WHEN THE CALLER SHOULD RUN THIS. Only under 'lcircterm', which is 'pat_len > 0' and 'ltkenst'
(turb_diffusion.f90:945, :959). Without it section 8) sets 'cur_prof' to 'sav_prof' itself
(:2392), the increment is what the solve already produced, and this program must be skipped
rather than run with equal fields: 'sav_prof + upd_prof - sav_prof' is not 'upd_prof' in
floating point.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _add_virtual_diffusion_increment_to_tke_profile(
    saved_tke_profile: fa.CellKField[wpfloat],
    updated_virtual_profile: fa.CellKField[wpfloat],
    current_virtual_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Add the diffusion increment of the virtual profile to the true one."""
    return saved_tke_profile + updated_virtual_profile - current_virtual_profile


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_virtual_diffusion_increment_to_tke_profile(
    saved_tke_profile: fa.CellKField[wpfloat],
    updated_virtual_profile: fa.CellKField[wpfloat],
    current_virtual_profile: fa.CellKField[wpfloat],
    updated_tke_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Turn the diffused virtual TKE profile into the updated true profile, on half levels.

    Every row is a function of its own row alone, so the caller may pass the same field as
    'updated_virtual_profile' and 'updated_tke_profile', which is what the Fortran does.

    The vertical range is the Fortran 'DO k = 2, ke', which is 'vertical_start = 1,
    vertical_end = nlev' here: exactly the rows 'solve_tke_diffusion_equation' solved.

    Args:
        saved_tke_profile: 'sav_prof', the true profile before the diffusion, saved by
            section 6). A TKE [m2/s2] at 'imode_tkediff = 2', a turbulent velocity 'q' [m/s]
            at 1.
        updated_virtual_profile: 'upd_prof', the diffused virtual profile from
            'solve_tke_diffusion_equation'.
        current_virtual_profile: 'cur_prof', the undiffused virtual profile from section 8).
        updated_tke_profile: Output, 'upd_prof' on exit: the updated true profile; may alias
            'updated_virtual_profile'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the diffused half levels; 'nlev', mirroring Fortran 'k = ..,ke'.
    """
    _add_virtual_diffusion_increment_to_tke_profile(
        saved_tke_profile=saved_tke_profile,
        updated_virtual_profile=updated_virtual_profile,
        current_virtual_profile=current_virtual_profile,
        out=updated_tke_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
