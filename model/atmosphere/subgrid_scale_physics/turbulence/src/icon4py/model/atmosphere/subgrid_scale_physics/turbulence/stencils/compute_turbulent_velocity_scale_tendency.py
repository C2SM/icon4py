# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tendency of 'q = SQRT(2*TKE)' that the vertical TKE diffusion produces.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
"10) Speichern der zugehoerigen q-Tendenzen" ("Storing the associated q tendencies",
:2490-2554 at icon commit 26d6b98cce). The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

Two Fortran loops, one output array:

    IF (tdc%imode_tkediff == 2) THEN     !'upd_prof' ist ein TKE-Profil
      DO k = 2, ke
        tketens(i,k) = ( SQRT( 2*MAX( upd_prof(i,k), z0 ) ) - tke(i,k,ntur) )*fr_tke
    ...
    !Am Unterrand gibt es keine q-Tendenz durch Diffusionsterme:
    DO i
      tketens(i,ke1) = z0

    -- "at the lower boundary there is no q tendency from diffusion terms".

Section 9) leaves 'upd_prof' holding the TKE updated by the diffusion tendency; this converts
it back to the turbulent velocity the scheme is prognostic in, and divides the increment by the
TKE time step. So the output OVERWRITES 'tketens' -- it does not accumulate into it. What it
overwrites is the advection tendency 'tvt' that 'turbdiff' received as INTENT(INOUT) and passed
to 'solve_turb_budgets' in section 3); from here on the array is the scheme's own output.

THE CLAMP IS NOT COSMETIC. The tridiagonal solve is unconstrained and does return negative TKE:
measured on 'exp.mch_icon-ch2_small', 47 to 57 of the 8276 x 79 written values per timestep are
below zero, so 'MAX(upd_prof, 0)' is what keeps 'SQRT' real. Raschendorfer's alternative form,
'(MAX(upd_prof,0) - sav_prof)*fr_tke/tke', is commented out in the Fortran with the note
"Attention: this way appears to be numerically unstable!!"; the form above carries his
"This way appears to be numerically stable!!".

WHAT IS NOT PORTED. Two branches of the Fortran are unreachable in the operational
configuration and are deliberately absent here rather than translated untested:

  * 'imode_tkediff == 1' (:2516-2529), where the diffusion runs on 'q' itself and the
    conversion is a bare 'MAX(upd_prof,0) - tke'. 'TurbulenceConfig' freezes 'imode_tkediff'
    at 2 (turbulence.py:97), so this program implements the TKE branch only.
  * The reset 'tketens(:,2:ke1) = 0' (:2541-2552) taken when neither the TKE diffusion nor the
    circulation term runs. That is the same 'IF (ldotkedif .OR. lcircterm)' whose ELSE branch
    would also suppress the 'turbdiff-9-exit' savepoint, so a capture that exercises it cannot
    be one that validates this section; see the note at turb_diffusion.f90:2482-2486.

THE SURFACE ROW IS MERGED IN with 'concat_where' rather than being a program of its own: same
output field, a different expression on one boundary row, which is the rule in the package
README ("Boundary rows"). It costs the embedded backend, which cannot run 'concat_where' in
gt4py 1.1.10.

Bit-exactness needs nothing special of this expression. 'SQRT' is correctly rounded in
IEEE-754, so unlike 'EXP' or 'LOG' it cannot differ between nvhpc's libm and the backends';
'2*x' is exact; and there is no multiply-add to contract.
"""

import gt4py.next as gtx
from gt4py.next import broadcast, maximum, sqrt
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_turbulent_velocity_scale_tendency(
    updated_tke_profile: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """The diffusion tendency of 'q', zero on the surface half level.

    The association is the Fortran's: the clamped TKE is doubled and square-rooted, the old
    velocity scale is subtracted from THAT, and only the difference is scaled by '1/dt_tke'.
    Multiplying either term out first is mathematically the same and rounds differently.
    """
    tendency = (
        sqrt(wpfloat("2.0") * maximum(updated_tke_profile, wpfloat("0.0")))
        - turbulent_velocity_scale
    ) * inverse_tke_time_step
    return concat_where(
        dims.KDim == nlev, broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)), tendency
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_velocity_scale_tendency(
    updated_tke_profile: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    inverse_tke_time_step: wpfloat,
    nlev: gtx.int32,
    turbulent_velocity_scale_tendency: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Store the q tendency of the TKE diffusion on half levels, closed at the surface.

    The vertical range spans both Fortran loops: 'DO k = 2, ke' and the single row 'ke1', which
    is 'vertical_start = 1, vertical_end = nlev + 1' here. 'nlev' must be that same last row,
    since it is what selects the vanishing surface tendency. The model top (row 0) is outside
    the range and keeps whatever 'tketens' held on entry, exactly as in the Fortran.

    Args:
        updated_tke_profile: 'upd_prof', the TKE updated by the diffusion tendency [m2/s2], as
            section 9) leaves 'zaux(:,:,1)'. Read on rows 1..nlev-1; the surface row is not read
            because the boundary branch does not name it.
        turbulent_velocity_scale: 'tke(:,:,ntur)' [m/s], the profile the diffusion started from.
        inverse_tke_time_step: 'fr_tke', '1/dt_tke' [1/s].
        nlev: 'ke1' as a zero-based row, the surface half level -- the one row whose tendency
            vanishes.
        turbulent_velocity_scale_tendency: Output, 'tketens' [m/s2]. Overwritten, not
            accumulated into.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost half level with a diffusion tendency; 1, mirroring Fortran
            'k = 2'.
        vertical_end: End of the half levels; 'nlev + 1', mirroring Fortran 'k = ..,ke1'.
    """
    _compute_turbulent_velocity_scale_tendency(
        updated_tke_profile=updated_tke_profile,
        turbulent_velocity_scale=turbulent_velocity_scale,
        inverse_tke_time_step=inverse_tke_time_step,
        nlev=nlev,
        out=turbulent_velocity_scale_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
