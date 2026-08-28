# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The virtual TKE profile that smuggles the circulation term into the TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', the whole
of section

    "8) Bestimmung des Zirkulationstermes als zusaetzliche TKE-Flussdichte"
    -- "Determination of the circulation term as an additional TKE flux density"

(:2343-2406 at icon commit 26d6b98cce, the commit that produced the reference capture), from the
block Raschendorfer heads "Quasi-implizite Berechnung der Zirkulationstendenz" -- "quasi-implicit
calculation of the circulation tendency". The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

    cur_prof(i,2) = sav_prof(i,2)
    fakt = tdc%c_diff/c_diff_llim
    DO k = 3, ke1
      wert = frm(i,k)/expl_mom(i,k)
      cur_prof(i,k) = (cur_prof(i,k-1)-sav_prof(i,k-1))*fakt + wert + sav_prof(i,k)

WHAT THE TRICK IS. The raw "circulation term" is a TKE flux density driven by the non-turbulent
circulations that near-surface thermal inhomogeneity generates; section 6) built it on the flux
levels as 'frm'. Rather than adding its convergence to the TKE equation as an explicit source --
which is what the dead 'lcirflx' branch of section 6) does -- Raschendorfer folds it into the
profile that section 9) is about to diffuse. Dividing a flux density by the diffusion momentum
'expl_mom' gives the profile INCREMENT across a flux level that would have produced that flux by
ordinary down-gradient diffusion, and accumulating those increments downward gives a profile
whose diffusive flux is the turbulent one plus the circulation one. So the circulation tendency
comes out of the same implicit solve as the diffusion tendency, and is therefore stable at any
time step. Section 9) subtracts the fiction again ('add_virtual_diffusion_increment_to_tke_profile'):
'upd_prof - cur_prof' is the increment the solve produced, and it is added to the true profile.

Raschendorfer's own note, at :2382-2385:

    "'cur_prof' contains a virtual TKE-profile (or q-profile at 'imode_tkediff=1'), and its
     diffusion tendency includes the particular tendency by the 'circulation term'. Hence,
     'cur_prof' is also used for the explicit (non-gradient) part of vertical diffusion. This
     'circulation tendency', however, is independent on 'c_diff' (except numerical effects)."

WHY THIS IS A SCAN AND NOT A STENCIL. It is one of the few 'k'-loops in this scheme that really
is a recurrence: 'cur_prof(k)' reads 'cur_prof(k-1)', the value the previous iteration of the
same loop wrote, so no fixed-depth vertical stencil expresses it (package README, "Vertical
recurrences"). Note that the port spec places this statement in section 6) -- it cites ':2328',
the same statement in the uninstrumented upstream file -- and section 6) contains no recurrence
at all; the statement is here.

WHAT IS NOT TRANSLATED. The rest of the Fortran section is two pointer assignments and a host
scalar: under 'lcircterm' it aliases 'cur_prof' to the 'hlp' scratch, in the 'ELSEIF (ldotkedif)'
branch it aliases 'cur_prof' to 'sav_prof' itself -- no virtual profile, hence no call to this
program -- and 'itndcon = 0' in both, because the other TKE sources are treated by
'solve_turb_budgets' in section 3). Those are caller-level facts, recorded in the program
docstring below rather than in a stencil.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), True))
def _accumulate_virtual_tke_profile(
    state: tuple[wpfloat, bool],
    saved_tke_profile: wpfloat,
    saved_tke_profile_above: wpfloat,
    cke_flux_at_main_levels: wpfloat,
    explicit_diffusion_momentum: wpfloat,
    tke_diffusion_limit_correction: wpfloat,
) -> tuple[wpfloat, bool]:
    """One step of the downward accumulation, carrying the virtual profile of the level above.

    The carried flag is what distinguishes the uppermost row, where the Fortran has a separate
    statement outside the loop: there the virtual profile simply IS the saved one, since no flux
    level above it carries a circulation flux. Writing it as a flag in the carry rather than as a
    'concat_where' on an absolute level is what keeps this program runnable on the embedded
    backend, and it mirrors '_invert_diffusion_momentum' of section 9). Returning 'False'
    unconditionally makes the flag true on the first row only.

    The two Fortran statements are one scan and not two programs because the first is the scan's
    initial step, not a different quantity: it seeds the recurrence the second one continues.

    THE OPERAND ORDER IS THE FORTRAN'S. '(cur - sav)*fakt + wert + sav' associates left to right
    in both languages, and the three terms are of comparable magnitude, so regrouping them would
    round differently. The quotient is formed once, as the Fortran's 'wert' is.
    """
    virtual_profile_above, at_the_top = state
    circulation_increment = cke_flux_at_main_levels / explicit_diffusion_momentum
    virtual_profile = (
        saved_tke_profile
        if at_the_top
        else (virtual_profile_above - saved_tke_profile_above) * tke_diffusion_limit_correction
        + circulation_increment
        + saved_tke_profile
    )
    return virtual_profile, False


@gtx.field_operator
def _compute_virtual_tke_profile(
    saved_tke_profile: fa.CellKField[wpfloat],
    cke_flux_at_main_levels: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    tke_diffusion_limit_correction: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Run the accumulation downward, reading the saved profile one half level up.

    'sav_prof(k-1)' is a plain shifted input rather than a second carry: it is written by
    section 6) and read here, so nothing about it is sequential. Only 'cur_prof' is.
    """
    virtual_profile, _ = _accumulate_virtual_tke_profile(
        saved_tke_profile,
        saved_tke_profile(Koff[-1]),
        cke_flux_at_main_levels,
        explicit_diffusion_momentum,
        tke_diffusion_limit_correction,
    )
    return virtual_profile


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_virtual_tke_profile(
    saved_tke_profile: fa.CellKField[wpfloat],
    cke_flux_at_main_levels: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    tke_diffusion_limit_correction: wpfloat,
    virtual_tke_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'cur_prof' on half levels (turb_diffusion.f90:2346-2380).

    WHEN THE CALLER SHOULD RUN THIS. Only under 'lcircterm' -- 'pat_len > 0' and 'ltkenst'
    (turb_diffusion.f90:945, :959). Without it 'cur_prof' is 'sav_prof' itself (:2390) and
    section 9)'s 'add_virtual_diffusion_increment_to_tke_profile' must be skipped as well; with
    it, the two programs are a pair and running one without the other changes the physics.

    THE CORRECTION FACTOR UNDOES A LIMIT IMPOSED ELSEWHERE. 'expl_mom' was built by section 6)
    with 'c_diff_llim = MAX(epsi, c_diff)' rather than with 'c_diff', because this program
    divides by it and 'c_diff = 0' with an active circulation term would be a division by zero
    (icon commit 571b0c9e42, "Enabling c_diff=0 in case of lcircterm=T"). Multiplying the carried
    increment by 'fakt = c_diff/c_diff_llim' restores the unlimited coefficient for the pure TKE
    diffusion, which is Raschendorfer's note at :2376-2379:

        "Although 'c_diff_llim' (and with it 'expl_mom') applies a lower limit in order to
         prevent a division by zero, the correction factor 'fakt' makes pure TKE-diffusion
         acting always with the unlimited value 'c_diff'."

    In the reference capture the limit does not bind, so the factor is exactly 1.0 and this
    argument is untested against anything but the identity; the datatest
    'test_the_c_diff_limit_correction_is_unity_in_this_capture' recovers it from the data and
    says so.

    Args:
        saved_tke_profile: 'sav_prof' = 'zaux(:,:,2)', the true pre-diffusion profile saved by
            section 6). TKE [m2/s2] at 'imode_tkediff = 2', 'q' [m/s] at 1. Read at the row and
            at the row above.
        cke_flux_at_main_levels: 'frm' [kg/s3], the circulation-kinetic-energy flux density
            interpolated onto the flux levels by section 6). Read from
            'vertical_start + 1' down.
        explicit_diffusion_momentum: 'expl_mom' = 'zaux(:,:,3)' [kg/m2/s] on the flux levels,
            from section 6). Read from 'vertical_start + 1' down; its row at 'vertical_start'
            is not written by section 6) at all and is not read here.
        tke_diffusion_limit_correction: 'fakt' = 'c_diff / c_diff_llim' [-], one scalar per run;
            see above.
        virtual_tke_profile: Output, 'cur_prof', aliased onto the 'hlp' scratch by the Fortran
            and read by section 9) as 'current_virtual_profile'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost half level; 1, mirroring the Fortran's separate 'cur_prof(:,2)'
            statement. The model top is not written -- the diffusion has no level there.
        vertical_end: End of the half levels; 'ke1', mirroring Fortran 'DO k = 3,ke1'. The
            surface half level IS written, as it is in section 6).
    """
    _compute_virtual_tke_profile(
        saved_tke_profile=saved_tke_profile,
        cke_flux_at_main_levels=cke_flux_at_main_levels,
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        tke_diffusion_limit_correction=tke_diffusion_limit_correction,
        out=virtual_tke_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
