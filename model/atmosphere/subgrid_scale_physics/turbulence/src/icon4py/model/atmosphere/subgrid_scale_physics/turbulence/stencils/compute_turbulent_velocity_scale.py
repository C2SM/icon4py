# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Prognostic update of the turbulent velocity scale 'q = SQRT(2*TKE)'.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'solve_turb_budgets' (:1052-1861 at icon commit 26d6b98cce, the commit that produced the
reference capture), from the two blocks headed "Total TKE-forcing in [m/s2]" (:1420-1467) and
"Calculating updated turbulent velocity scale, which is SQRT(2*TKE)" (:1469-1546). The routine
is called from 'turbdiff' section 3) (turb_diffusion.f90:1795-1914). The scientific commentary
in both files is by Matthias Raschendorfer (DWD).

WHY THIS IS NOT A SCAN. The Fortran k-loop that contains these blocks
(turb_utilities.f90:1384) is declared '!$ACC LOOP SEQ', which reads like an irreducible
recurrence. It is not one: verified over the whole subroutine, no array reference anywhere in
it subscripts the vertical axis with anything but the loop index 'k' itself -- there is no
'k-1', no 'k+1', and no expression of 'k' other than 'k_tvs', which is set to 'k' on the branch
this configuration takes. The loop is sequential only because its scratch arrays ('dd', 'l_dis',
'l_frc', 'frc', 'tvs_u') are one-dimensional and are overwritten each level, and because 'dd'
carries the roughness-layer parameters forward -- machinery that is dead here ('lporous' is a
'.FALSE.' PARAMETER). Every level is an independent instance of the same expression, so this is
an ordinary wide field operator; lowering it to a 'scan_operator' would serialise fully parallel
work. See the port spec, section 3.2.
"""

import gt4py.next as gtx
from gt4py.next import maximum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _effective_tke_forcing(
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    frcsecu: wpfloat,
    rim: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Total forcing of the TKE equation, 'frc' [m/s2]: shear production less buoyancy loss.

    turb_utilities.f90:1423-1432. Both terms are a stability length times a squared forcing
    frequency, so their difference is an acceleration of the turbulent motion. Raschendorfer's
    note on the floor (:1434-1439), which is what 'frcsecu' controls:

        "Beachte: Bei 'frcsecu=1' wird 'frc' so nach unten beschraenkt, dass die krit. Rf-Zahl
         (Rf=1-'dd(i,7)') im TKE-Quellgleichgewicht (ohne Transport und Zeittendenz) nicht
         ueberschritten wird. Bei 'frcsecu<1' wird 'frc' entsprechend weniger eingeschraenkt.
         Bei 'frcsecu=0' wird Rf nicht > 1 und die Summe der TKE-Quellterme wird nicht negativ.
         Bei 'frcsecu<0' kann 'Rf>1' und die Summe der TKE-Quellterme auch negativ werden."

        -- "Note: at 'frcsecu = 1', 'frc' is bounded from below such that the critical flux
           Richardson number (Rf = 1 - dd(i,7)) is not exceeded in the equilibrium of the TKE
           sources (that is, without transport and time tendency). At 'frcsecu < 1', 'frc' is
           correspondingly less restricted. At 'frcsecu = 0', Rf does not exceed 1 and the sum
           of the TKE source terms does not become negative. At 'frcsecu < 0', 'Rf > 1' becomes
           possible and the sum of the TKE source terms may become negative as well."

    'dd(i,7)' is the free-atmosphere value 'rim = 1/(1 + (d_m - d_4)/d_5)', which 'turb_setup'
    (turb_utilities.f90:418) computes and labels "1-Rf_c"; the roughness-layer override of it
    is dead here.

    The Fortran's effective mechanical forcing is 'fm2_e', a pointer that aims at the pure
    turbulent shear 'ft2' under 'lssintact' and at the total mechanical shear 'frm' otherwise
    (:1275-1279). 'lssintact' is 'imode_adshear == 1' and this port freezes that switch at 2,
    so 'fm2_e' is always 'frm' and the pointer disappears.

    Shared with 'compute_stability_lengths', which needs the same 'frc' for the deviation from
    TKE equilibrium. Two programs recompute it rather than one program returning three fields,
    because Raschendorfer separates the velocity scale from the stability functions and each
    deserves its own numerical gate; the two could be fused later at the cost of that.

    Args:
        stability_length_for_momentum: 'lsm', turbulent master length scale times the stability
            function for momentum [m], as section 2c) leaves it.
        stability_length_for_scalars: 'lsh', the same for scalars (heat) [m].
        mechanical_forcing: 'fm2' ('frm'), squared frequency of the mechanical forcing [1/s2].
        thermal_forcing: 'fh2' ('frh'), squared frequency of the thermal forcing [1/s2].
        frcsecu: Security factor for the TKE forcing [-].
        rim: One minus the critical flux Richardson number [-].

    Returns:
        The effective TKE forcing 'frc' [m/s2].
    """
    shear_production = stability_length_for_momentum * mechanical_forcing
    buoyancy_loss = stability_length_for_scalars * thermal_forcing
    return maximum(shear_production - buoyancy_loss, frcsecu * rim * shear_production)


@gtx.field_operator
def _compute_turbulent_velocity_scale(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Integrate the TKE equation one step and apply the floors and the time smoothing.

    turb_utilities.f90:1494-1546, the branch 'imode_stke == 1', "1-st (former) type of
    prognostic solution", which is what 'imode_turb = 1' selects. Raschendorfer's heading is
    "Integration von SQRT(2TKE)" -- "integration of SQRT(2*TKE)".

    The step solves the quadratic that the implicit dissipation term produces. Writing
    'q1 = l_dis/dt' and 'q2' for the explicitly transported and forced value, the discretised
    equation 'q_new + dt*q_new**2/l_dis = q2' has the positive root

        q_new = q1/2 * (SQRT(1 + 4*q2/q1) - 1),

    which is 'tvs_u'. The dissipation length scale is 'l_dis = tls*d_m'; the forcing length
    scale 'l_frc = tls*d_4/b_m' sets the floor below.

    THE SQUARE IS WRITTEN AS A PRODUCT. There is no 'x**2' anywhere in this module, and the
    reason is measured rather than stylistic: GT4Py lowers Python's '**' to 'math.pow' and
    CUDA's 'pow' carries up to 2 ulp, so on the GPU backends 'x**2' is not 'x*x'. See the
    docstring of '_compute_mechanical_forcing', where it cost a day.

    Two floors and a smoothing are applied on top of the root (:1531-1535):

        tke = MAX( tvs_m, tkesecu*q2, w1*tvs_0 + w2*tvs_u )

    'tvs_m = tkesecu*vel_min' is the absolute minimum velocity scale -- constant here, because
    the 'turbdiff' call site passes 'imode_vel_min = 1' and no 'velmin' field
    (turb_diffusion.f90:1817). 'tkesecu*q2' is the equilibrium floor, on which Raschendorfer
    notes (:1538-1541):

        "'q2' ist ein Minimalwert fuer 'tke', mit dem die Abweichung vom TKE-Gleichgewicht den
         Wert besitzt, der mit dem gegebenen 'frc' bei neutraler Schichtung nicht ueberschritten
         werden kann. 'tke(:,:,ntur)' ist der Wert, der im naechsten Prognoseschritt benutzt
         wird!"

        -- "'q2' is a minimum value for 'tke' at which the deviation from TKE equilibrium takes
           the value that, for the given 'frc', cannot be exceeded under neutral stratification.
           'tke(:,:,ntur)' is the value used in the next prognostic step!"

    The third argument is the time smoothing with weight 'tkesmot' towards the previous value.

    WHAT IS NOT HERE, and why it is legitimately absent rather than forgotten:

    - The upper limit on 'frc' (:1445-1450) is guarded by 'lupfrclim', which the 'turbdiff'
      call site passes as a literal '.FALSE.' (turb_diffusion.f90:1811). Only 'turbtran' asks
      for it.
    - The addition of the scale-interaction shear (:1458-1467) needs 'lssintact', frozen off.
    - The advection increments (:1478-1486, 'lpres_avt') need 'tketadv', which the ICON
      interface never passes, so 'tvs_0' is the previous TKE time level itself.
    - The three other 'imode_stke' branches (:1505-1526) are alternative TKE solutions;
      'imode_turb' is frozen at 1.
    - The whole 'ltkeinp' path, under which TKE is an input and none of this runs.

    Args:
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        stability_length_for_momentum: 'lsm' [m], for the forcing.
        stability_length_for_scalars: 'lsh' [m], for the forcing.
        mechanical_forcing: 'fm2' ('frm') [1/s2].
        thermal_forcing: 'fh2' ('frh') [1/s2].
        previous_velocity_scale: 'tke(:,:,nvor)' [m/s], the previous time level.
        transport_tendency: 'tvt' ('tketens'), turbulent transport of the turbulent velocity
            scale [m/s2].
        d_m: Length-scale factor of the momentum dissipation [-].
        d_4: Closure constant '6*a_mom' [-].
        b_m: Momentum closure constant of the stability functions [-].
        rim: One minus the critical flux Richardson number [-].
        frcsecu: Security factor for the TKE forcing [-].
        tkesecu: Security factor in the TKE equation [-].
        tkesmot: Time smoothing factor towards the previous velocity scale [-].
        vel_min: Minimal velocity scale [m/s].
        tke_time_step: 'dt_tke' [s].
        inverse_tke_time_step: 'fr_tke', '1/dt_tke' [1/s]. Passed rather than derived because
            'turb_setup' derives it once (turb_utilities.f90:317) and the scheme multiplies by
            it, which is not the same rounding as dividing by 'dt_tke'.

    Returns:
        'tke(:,:,ntur)' [m/s], the updated turbulent velocity scale 'q = SQRT(2*TKE)'.
    """
    dissipation_length = master_length_scale * d_m
    forcing_length = master_length_scale * d_4 * (wpfloat("1.0") / b_m)
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    q1 = dissipation_length * inverse_tke_time_step
    q2 = (
        maximum(wpfloat("0.0"), previous_velocity_scale + transport_tendency * tke_time_step)
        + forcing * tke_time_step
    )
    updated = q1 * (sqrt(wpfloat("1.0") + wpfloat("4.0") * q2 / q1) - wpfloat("1.0")) * wpfloat(
        "0.5"
    )

    equilibrium_floor = sqrt(forcing_length * maximum(forcing, wpfloat("0.0")))
    smoothed = tkesmot * previous_velocity_scale + (wpfloat("1.0") - tkesmot) * updated
    return maximum(maximum(tkesecu * vel_min, tkesecu * equilibrium_floor), smoothed)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_velocity_scale(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Advance 'tke' one TKE time step on the half levels the turbulence model covers.

    The vertical domain is the Fortran 'DO k=k_st,k_en' with 'k_st = 2' and 'k_en = kem = ke'
    (turb_diffusion.f90:1814): half levels 1 to 'ke' - 1 zero-based. The model top is not
    written here -- 'set_turbulent_velocity_scale_at_model_top' copies it from the level below
    afterwards -- and the surface half level 'ke1' keeps the value 'turbtran' put there.

    'previous_velocity_scale' and 'turbulent_velocity_scale' are the same Fortran storage:
    'ntim = 1' for the NWP interface, so 'nvor' and 'ntur' are both 1 and 'tke(:,:,nvor)' and
    'tke(:,:,ntur)' alias. That is safe there because each level reads only its own level, and
    it is why the two appear as separate fields here.

    Args:
        master_length_scale: 'tls' [m].
        stability_length_for_momentum: 'lsm' [m].
        stability_length_for_scalars: 'lsh' [m].
        mechanical_forcing: 'fm2' [1/s2].
        thermal_forcing: 'fh2' [1/s2].
        previous_velocity_scale: 'tke(:,:,nvor)' [m/s].
        transport_tendency: 'tvt' [m/s2].
        d_m: Length-scale factor of the momentum dissipation [-].
        d_4: Closure constant '6*a_mom' [-].
        b_m: Momentum closure constant [-].
        rim: One minus the critical flux Richardson number [-].
        frcsecu: Security factor for the TKE forcing [-].
        tkesecu: Security factor in the TKE equation [-].
        tkesmot: Time smoothing factor [-].
        vel_min: Minimal velocity scale [m/s].
        tke_time_step: 'dt_tke' [s].
        inverse_tke_time_step: 'fr_tke' [1/s].
        turbulent_velocity_scale: Output, 'tke(:,:,ntur)' [m/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k_st = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'k_en = kem = ke'.
    """
    _compute_turbulent_velocity_scale(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        previous_velocity_scale=previous_velocity_scale,
        transport_tendency=transport_tendency,
        d_m=d_m,
        d_4=d_4,
        b_m=b_m,
        rim=rim,
        frcsecu=frcsecu,
        tkesecu=tkesecu,
        tkesmot=tkesmot,
        vel_min=vel_min,
        tke_time_step=tke_time_step,
        inverse_tke_time_step=inverse_tke_time_step,
        out=turbulent_velocity_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
