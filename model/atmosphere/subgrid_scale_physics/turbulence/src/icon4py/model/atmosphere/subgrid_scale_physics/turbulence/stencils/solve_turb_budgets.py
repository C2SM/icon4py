# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Section 3) of 'turbdiff': the turbulent budgets, under ICON's own name for them.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'solve_turb_budgets'
(:1052-1861 at icon commit 26d6b98cce, the commit that produced the reference capture), together
with the four short blocks that follow its call back in 'turbdiff' section 3)
(turb_diffusion.f90:1795-1914). The scientific commentary in both files is by Matthias
Raschendorfer (DWD).

ONE PROGRAM, FIVE STATEMENTS, TWO DOMAINS. The statements are, in source order,

  1. the prognostic step of the turbulent velocity scale 'tke', half levels 2..kem;
  2. the stability lengths 'lsm' and 'lsh', half levels 2..kem;
  3. the vertical acceleration of the raw circulation term, half levels 2..ke1 -- one row
     deeper than everything else in the section;
  4. the standard deviation of the local super-saturation, half levels 2..kem;
  5. the diffusion coefficients 'tkvm' and 'tkvh', half levels 2..kem.

THE MODEL-TOP ROW IS THE SIXTH PROGRAM OF THE SECTION AND STAYS ONE, and the reason is now
measured rather than stylistic. 'set_turbulent_velocity_scale_at_model_top' copies 'tke(:,2)'
onto 'tke(:,1)', so as a statement of this program it would read, through 'Koff[1]', the very
parameter it writes. THAT SHAPE IS SILENTLY MISCOMPILED BY DACE. Measured 2026-09-01 on
'dace_cpu' and 'dace_gpu' with a standalone reproducer -- variants A1 and A2 of
'docs/superpowers/notes/2026-09-01-gt4py-dace-intra-program-aliasing.py' in the workspace:
a statement whose 'out=' names the same program parameter as a SHIFTED input is dropped -- the destination row keeps
whatever it held -- while 'gtfn_cpu', 'gtfn_gpu' and 'embedded' execute it correctly. The
pointwise in-place form is not affected. It first showed up here as four failures of
'test_set_turbulent_velocity_scale_at_model_top_agrees_with_icon_within_its_gate' on 'dace_gpu'
against a 'gtfn_cpu' run that was green, which is the worst way to find it.

Two parameters bound to the same field BY THE CALLER are a different thing and do work, which is
what the separate program does and has always done. So the section is two programs, not one, and
this one must run first.

THE SOURCE ORDER IS THE SCHEME'S ORDER, and two of the edges it expresses are real:

  * statements 4 and 5 read the stability lengths statement 2 writes, not the ones section 2c)
    supplied.
  * statement 3 must precede statement 4 BECAUSE THE CALLER ALIASES THEM. 'rcld' enters as the
    cloud cover and leaves as the standard deviation of the local super-saturation, one Fortran
    storage with two meanings, and the granule reproduces that: it passes the same field as
    'cloud_cover' and as 'supersaturation_standard_deviation'. The circulation term is the last
    reader of the cloud cover and the SDSS is the next writer of it, so the order of these two
    statements is load-bearing for that caller and for no other reason. GT4Py cannot see the
    aliasing -- the two are separate program parameters -- so nothing but the source order
    enforces it. 'test_the_sdss_is_written_after_the_circulation_term_reads_the_cloud_cover'
    asserts that order off the program's own body, and the datatest asserts the result.

'tkvm'/'tkvh' are aliased the same way in the Fortran -- they arrive as stability lengths and
leave as diffusion coefficients -- but the port keeps those two roles in FOUR fields, so
statement 6 reads 'updated_stability_length_*' and writes 'diffusion_coefficient_*' and no
ordering rests on it.

WHY NONE OF THIS IS A SCAN. The Fortran k-loop that contains statements 1 and 3
(turb_utilities.f90:1384) is declared '!$ACC LOOP SEQ', which reads like an irreducible
recurrence. It is not one: verified over the whole subroutine, no array reference anywhere in
it subscripts the vertical axis with anything but the loop index 'k' itself -- there is no
'k-1', no 'k+1', and no expression of 'k' other than 'k_tvs', which is set to 'k' on the branch
this configuration takes. The loop is sequential only because its scratch arrays ('dd', 'l_dis',
'l_frc', 'frc', 'tvs_u') are one-dimensional and are overwritten each level, and because 'dd'
carries the roughness-layer parameters forward -- machinery that is dead here ('lporous' is a
'.FALSE.' PARAMETER). Every level is an independent instance of the same expression, so these
are ordinary wide field operators; lowering them to 'scan_operator's would serialise fully
parallel work. 'test_the_k_loop_of_solve_turb_budgets_has_no_vertical_neighbour_access'
re-measures that against the Fortran source on every run. See the port spec, section 3.2.

THE SQUARES ARE PRODUCTS, never 'x**2', in every operator below -- GT4Py lowers '**' to
'math.pow' and CUDA's 'pow' carries up to 2 ulp. See '_compute_mechanical_forcing' in
'compute_tke_forcing_functions', where that cost a day.
"""

import gt4py.next as gtx
from gt4py.next import abs, maximum, minimum, sqrt, where  # noqa: A004 [builtin-shadowing]

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

    Shared with '_compute_stability_lengths', which needs the same 'frc' for the deviation from
    TKE equilibrium. Both call it rather than one of them returning it, so it is computed twice;
    that was the price of two programs with two gates before the merge, and it is a candidate for
    GT4Py's own common-subexpression elimination now that the two are statements of one program.
    It has not been measured, so it is not claimed.

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
    updated = (
        q1 * (sqrt(wpfloat("1.0") + wpfloat("4.0") * q2 / q1) - wpfloat("1.0")) * wpfloat("0.5")
    )

    equilibrium_floor = sqrt(forcing_length * maximum(forcing, wpfloat("0.0")))
    smoothed = tkesmot * previous_velocity_scale + (wpfloat("1.0") - tkesmot) * updated
    return maximum(maximum(tkesecu * vel_min, tkesecu * equilibrium_floor), smoothed)


@gtx.field_operator
def _compute_stability_lengths(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Both admissible solutions for 'S_m*l' and 'S_h*l', and the choice between them.

    THE STANDARD SOLUTION (turb_utilities.f90:1583-1612). With the square of the turbulent time
    scale 'tim2 = (tls/tke)**2' and the dimensionless forcings 'gh = fh2*tim2' ("durch
    thermischen Auftrieb" -- by thermal buoyancy) and 'gm = fm2*tim2' ("durch mechanische
    Scherung" -- by mechanical shear), the closure is

        [ a11 a12 ] [ sh ]   [ b_h ]
        [ a21 a22 ] [ sm ] = [ b_m ]

    solved by Cramer's rule. The Fortran forms the determinant, then the two numerators, then
    inverts the determinant once and multiplies -- '1/det' followed by two products, not two
    divisions -- and this port does the same, because it is a different rounding.

    Only the un-preconditioned coefficient assignment is ported. The alternative
    (:1596-1610, "Koeffizientenbelegung mit Praeconditionierung" -- coefficient assignment with
    preconditioning) is selected by 'lstbsecu = (imode_stbcalc < 0)', and 'imode_stbcalc' is
    frozen at 1.

    THE MODIFIED SOLUTION (:1627-1667). 'gama' is the prescribed deviation from TKE equilibrium,

        gama = MIN( gam0, frc*tim2/tls )     for 'imode_stbcalc = 1' ("alt_gama"),

    capped by 'gam0 = stbsecu/d_m + (1-stbsecu)*b_m/d_4' (:1320), Raschendorfer's "obere
    Schranke fuer die Abweichung 'gama' vom TKE-Gleichgewicht" -- "upper bound for the deviation
    'gama' from TKE equilibrium". Substituting the equilibrium form of the TKE equation turns
    the 2x2 system into a scalar quadratic in the flux Richardson number, whose larger root
    'val2' gives 'fakt = fh2/(val2 - fh2)' and thence both stability functions. It is realizable
    for any stratification, which is the point of it.

    WHICH ONE APPLIES. The Fortran runs the standard block under
    'imode_stbcorr == 2 .OR. fh2 >= 0' and then sets 'lcorr = .FALSE.' only inside it, only when
    the solution came out positive, and -- through a MERGE on 'imode_stbcorr == 2' -- only for
    'imode_stbcorr == 2'. With 'imode_stbcalc = 1' frozen, 'imode_stbcorr = 1', so the
    modified solution is taken exactly when

        NOT (fh2 >= 0 AND det > 0 AND sh > 0 AND sm > 0),

    which is what 'solvable' is below. Raschendorfer's comment at :1607 asserts the last three
    conjuncts always hold at 'fh2 >= 0' ("solution possible, which always holds at fh2>=0"), and
    in the reference capture they do -- 'test_the_standard_stability_solution_never_fails_where_
    it_is_attempted' measures that -- but the condition is translated as written and not reduced
    to 'fh2 < 0'.

    Both branches are evaluated everywhere and one is selected. The unselected branch may be
    NaN: the modified solution takes 'SQRT(val1**2 - ...)' of a quantity that is negative for
    some stably stratified points, and the standard solution divides by a determinant that may
    be zero. Neither reaches the output, and IEEE-754 does not trap.

    THE SQUARES ARE PRODUCTS, never 'x**2' -- see '_compute_mechanical_forcing'. This operator
    has two of them, 'tim2' and 'val1**2', and they are the reason 'turbulent_time_scale' is
    named at all rather than inlined.

    Args:
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        stability_length_for_momentum: 'lsm' [m] as section 2c) leaves it, for the forcing.
        stability_length_for_scalars: 'lsh' [m] as section 2c) leaves it, for the forcing.
        mechanical_forcing: 'fm2' ('frm') [1/s2].
        thermal_forcing: 'fh2' ('frh') [1/s2].
        turbulent_velocity_scale: 'tke(:,:,ntur)' [m/s], the value just computed by
            'compute_turbulent_velocity_scale'.
        a_h: Length-scale factor of the turbulent heat transport [-].
        a_m: Length-scale factor of the turbulent momentum transport [-].
        b_h: Scalar closure constant of the stability functions [-].
        b_m: Momentum closure constant of the stability functions [-].
        d_m: Length-scale factor of the momentum dissipation [-].
        d_1: Closure constant '1/a_heat' [-].
        d_2: Closure constant '1/a_mom' [-].
        d_3: Closure constant '9*a_heat' [-].
        d_4: Closure constant '6*a_mom' [-].
        d_5: Closure constant '3*(d_heat + d_4)' [-].
        d_6: Closure constant 'd_3 + 3*d_4' [-].
        rim: One minus the critical flux Richardson number [-].
        frcsecu: Security factor for the TKE forcing [-].
        stbsecu: Security factor in the stability function [-].

    Returns:
        'lsm' and 'lsh' [m], the master length scale times the stability function for momentum
        and for scalars.
    """
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    turbulent_time_scale = master_length_scale / turbulent_velocity_scale
    tim2 = turbulent_time_scale * turbulent_time_scale
    gh = thermal_forcing * tim2
    gm = mechanical_forcing * tim2

    a11 = d_1 + (d_5 - d_4) * gh
    a12 = d_4 * gm
    a21 = (d_6 - d_4) * gh
    a22 = d_2 + d_3 * gh + d_4 * gm

    determinant = a11 * a22 - a12 * a21
    numerator_h = b_h * a22 - b_m * a12
    numerator_m = b_m * a11 - b_h * a21
    inverse_determinant = wpfloat("1.0") / determinant

    gam0 = stbsecu / d_m + (wpfloat("1.0") - stbsecu) * b_m / d_4
    gama = minimum(gam0, forcing * tim2 / master_length_scale)
    wert = d_4 * gama
    bb1 = (b_h - wert) * a_h
    bb2 = (b_m - wert) * a_m
    a3 = d_3 * gama * a_m
    a5 = d_5 * gama * a_h
    a6 = d_6 * gama * a_m

    val1 = (mechanical_forcing * bb2 + (a5 - a3 + bb1) * thermal_forcing) / (wpfloat("2.0") * bb1)
    val2 = val1 + sqrt(val1 * val1 - (a6 + bb2) * thermal_forcing * mechanical_forcing / bb1)
    fakt = thermal_forcing / (val2 - thermal_forcing)
    corrected_h = bb1 - a5 * fakt
    corrected_m = corrected_h * (bb2 - a6 * fakt) / (bb1 - (a5 - a3) * fakt)

    solvable = (
        (thermal_forcing >= wpfloat("0.0"))
        & (determinant > wpfloat("0.0"))
        & (numerator_h > wpfloat("0.0"))
        & (numerator_m > wpfloat("0.0"))
    )
    sh = where(solvable, numerator_h * inverse_determinant, corrected_h)
    sm = where(solvable, numerator_m * inverse_determinant, corrected_m)
    return master_length_scale * sm, master_length_scale * sh


@gtx.field_operator
def _compute_circulation_acceleration(
    cloud_cover: fa.CellKField[wpfloat],
    master_length_scale: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    half_level_pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    horizontal_grid_scale: fa.CellField[wpfloat],
    gravitational_acceleration: wpfloat,
) -> fa.CellKField[wpfloat]:
    """The CKE gradient 'l_coh*fh2' [m/s2].

    The Fortran is four statements (turb_utilities.f90:1738-1745), the middle two reusing the
    name 'fakt' for two unrelated quantities:

        fakt  = z1-z2*ABS(rcld(i,k)-z1d2)                  ! coherence factor by cloud cover
        l_coh = MAX( l_pat(i), SQRT(fakt*tls(i,k)*l_hori(i)) )
        fakt  = fh2(i,k)*grd(i,k,0)/(dens(i,k)*grav2)      ! = Rd/g*exnr*grad(tet_v)
        l_coh = l_coh*SIGN(z1,fakt)*MIN( ABS(fakt), z1 )
        grd(i,k,0) = l_coh*fh2(i,k)

    with 'grd(:,:,0)' the half-level pressure on input and the acceleration on output, and
    'grav2 = grav**2' precomputed in the caller (:1310).

    THE SIGNED CLIP. 'SIGN(1,x)*MIN(|x|,1)' is the clamp of 'x' to [-1, 1], and that is how it
    is written here: 'maximum(-1, minimum(x, 1))'. The two forms agree bit for bit, signed zeros
    included -- multiplying by an exact +-1 is exact, so the rewrite only reorders exact
    operations -- and the clamp additionally avoids the one place where a naive translation
    would NOT agree. Fortran's 'SIGN(1.0, -0.0)' is -1.0, while a 'where(x < 0, -1, +1)' gives
    +1.0, because '-0.0 < 0' is false; the two then differ in the sign of a zero result. No
    point of the reference capture has 'x == 0' at all (asserted by
    'test_the_coherence_scaling_is_never_handed_an_exact_zero'), so this is a gap the data
    cannot close and the clamp closes structurally.

    Args:
        cloud_cover: 'rcld' as 'adjust_satur_equil' and 'bound_level_interp' leave it, the
            saturation fraction on half levels [-].
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        thermal_forcing: 'fh2' ('frh'), squared frequency of the thermal forcing [1/s2].
        half_level_pressure: 'grd(:,:,0)' ('zvari(:,:,0)'), the half-level pressure [Pa].
        air_density: 'dens' ('rhon'), air density on half levels [kg/m3].
        pattern_length_scale: 'l_pat', effective length scale of the near-surface circulation
            patterns [m], one value per column.
        horizontal_grid_scale: 'l_hori', horizontal grid spacing [m], one value per column.
        gravitational_acceleration: 'grav' [m/s2]; squared here, as 'grav2' is in the Fortran.

    Returns:
        The circulation acceleration 'grd(:,:,0)' [m/s2].
    """
    coherence_factor = wpfloat("1.0") - wpfloat("2.0") * abs(cloud_cover - wpfloat("0.5"))
    coherence_length = maximum(
        pattern_length_scale,
        sqrt(coherence_factor * master_length_scale * horizontal_grid_scale),
    )
    virtual_temperature_gradient = (
        thermal_forcing
        * half_level_pressure
        / (air_density * (gravitational_acceleration * gravitational_acceleration))
    )
    coherence_length = coherence_length * maximum(
        -wpfloat("1.0"), minimum(virtual_temperature_gradient, wpfloat("1.0"))
    )
    return coherence_length * thermal_forcing


@gtx.field_operator
def _compute_supersaturation_standard_deviation(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    saturation_humidity_derivative: fa.CellKField[wpfloat],
    gradient_of_liquid_water_potential_temperature: fa.CellKField[wpfloat],
    gradient_of_total_water: fa.CellKField[wpfloat],
    d_h: wpfloat,
) -> fa.CellKField[wpfloat]:
    """A length scale times the effective gradient of the super-saturating water.

    Args:
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        stability_length_for_scalars: 'lsh' [m], as 'compute_stability_lengths' leaves it --
            the updated value, not the one section 2c) supplied.
        exner_factor: 'exner' ('zaux(:,:,1)'), the Exner factor on half levels [-].
        saturation_humidity_derivative: 'qst_t' ('zaux(:,:,3)'), 'dq_sat/dT' [1/K].
        gradient_of_liquid_water_potential_temperature: 'grd(:,:,tet_l)' [K/m].
        gradient_of_total_water: 'grd(:,:,h2o_g)' [1/m].
        d_h: Length-scale factor of the scalar (temperature) variance, 'd_heat' [-].

    Returns:
        The standard deviation of the local super-saturation [-].
    """
    length_scale = sqrt(master_length_scale * stability_length_for_scalars * d_h)
    supersaturation_gradient = (
        exner_factor
        * saturation_humidity_derivative
        * gradient_of_liquid_water_potential_temperature
        - gradient_of_total_water
    )
    return length_scale * abs(supersaturation_gradient)


@gtx.field_operator
def _compute_diffusion_coefficients_from_stability_lengths(
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    molecular_diffusivity_for_scalars: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """A stability length times the turbulent velocity scale is a diffusion coefficient.

    Args:
        stability_length_for_momentum: 'lsm' ('tkvm'), 'S_m*l' [m].
        stability_length_for_scalars: 'lsh' ('tkvh'), 'S_h*l' [m].
        turbulent_velocity_scale: 'tke(:,:,ntur)', 'q = SQRT(2*TKE)' [m/s].
        molecular_diffusivity_for_scalars: 'con_h', the scalar conductivity of dry air
            (mo_physical_constants.f90:116) [m2/s].

    Returns:
        'tkvm' and 'tkvh' [m2/s].
    """
    return (
        stability_length_for_momentum * turbulent_velocity_scale,
        maximum(
            stability_length_for_scalars * turbulent_velocity_scale,
            molecular_diffusivity_for_scalars,
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_turb_budgets(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    cloud_cover: fa.CellKField[wpfloat],
    half_level_pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    saturation_humidity_derivative: fa.CellKField[wpfloat],
    gradient_of_liquid_water_potential_temperature: fa.CellKField[wpfloat],
    gradient_of_total_water: fa.CellKField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    horizontal_grid_scale: fa.CellField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_h: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
    gravitational_acceleration: wpfloat,
    molecular_diffusivity_for_scalars: wpfloat,
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    circulation_acceleration: fa.CellKField[wpfloat],
    supersaturation_standard_deviation: fa.CellKField[wpfloat],
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Solve the turbulent budgets on the half levels the turbulence model covers.

    THE TWO VERTICAL RANGES, both expressed as arithmetic on the one pair the granule binds.
    '(vertical_start, vertical_end)' is the Fortran 'DO k=k_st,k_en' with 'k_st = 2' and
    'k_en = kem = ke' (turb_diffusion.f90:1814) -- half levels 1 to 'ke' - 1 zero-based -- and
    four of the five statements use it unchanged. The fifth is the circulation acceleration,
    '(vertical_start, vertical_end + 1)', the Fortran 'DO k=k_st,k_sf' with 'k_sf = ke1', "for
    all boundary levels including the extra lowermost boundary". The surface row matters there
    -- it is where the near-surface circulations are anchored -- and it matters nowhere else in
    the section. The model-top row belongs to 'set_turbulent_velocity_scale_at_model_top', which
    runs after this program; see the module docstring for why it is not a statement here.

    Half level 'ke1' therefore keeps, for every output but the circulation acceleration, the
    value 'turbtran' put there. For the SDSS that is not an accident of the range but a
    requirement the Fortran states (turb_utilities.f90:1774-1780): the surface value comes from
    the call of 'solve_turb_budgets' inside 'turbtran', possibly as a tile aggregation, and
    recomputing it from aggregated inputs would be wrong.

    'previous_velocity_scale' and 'turbulent_velocity_scale' are the same Fortran storage:
    'ntim = 1' for the NWP interface, so 'nvor' and 'ntur' are both 1 and 'tke(:,:,nvor)' and
    'tke(:,:,ntur)' alias. That is safe there because each level reads only its own level, and
    it is why the two appear as separate fields here.

    Args:
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        stability_length_for_momentum: 'lsm' [m] as section 2c) leaves it, for the TKE forcing.
        stability_length_for_scalars: 'lsh' [m] as section 2c) leaves it, for the TKE forcing.
        mechanical_forcing: 'fm2' ('frm'), squared frequency of the mechanical forcing [1/s2].
        thermal_forcing: 'fh2' ('frh'), squared frequency of the thermal forcing [1/s2].
        previous_velocity_scale: 'tke(:,:,nvor)' [m/s], the previous time level.
        transport_tendency: 'tvt' ('tketens') [m/s2].
        cloud_cover: 'rcld' as section 0) leaves it, the saturation fraction on half levels [-].
            The caller may pass the same field as 'supersaturation_standard_deviation'; see the
            module docstring for why the statement order then matters.
        half_level_pressure: 'grd(:,:,0)' ('zvari(:,:,0)') on entry, the half-level pressure
            [Pa].
        air_density: 'dens' ('rhon'), air density on half levels [kg/m3].
        exner_factor: 'exner' ('zaux(:,:,1)'), the Exner factor on half levels [-].
        saturation_humidity_derivative: 'qst_t' ('zaux(:,:,3)'), 'dq_sat/dT' [1/K].
        gradient_of_liquid_water_potential_temperature: 'grd(:,:,tet_l)' [K/m].
        gradient_of_total_water: 'grd(:,:,h2o_g)' [1/m].
        pattern_length_scale: 'l_pat' [m], one value per column.
        horizontal_grid_scale: 'l_hori' [m], one value per column.
        a_h: Length-scale factor of the turbulent heat transport [-].
        a_m: Length-scale factor of the turbulent momentum transport [-].
        b_h: Scalar closure constant of the stability functions [-].
        b_m: Momentum closure constant of the stability functions [-].
        d_h: Length-scale factor of the scalar (temperature) variance, 'd_heat' [-].
        d_m: Length-scale factor of the momentum dissipation [-].
        d_1: Closure constant '1/a_heat' [-].
        d_2: Closure constant '1/a_mom' [-].
        d_3: Closure constant '9*a_heat' [-].
        d_4: Closure constant '6*a_mom' [-].
        d_5: Closure constant '3*(d_heat + d_4)' [-].
        d_6: Closure constant 'd_3 + 3*d_4' [-].
        rim: One minus the critical flux Richardson number [-].
        frcsecu: Security factor for the TKE forcing [-].
        stbsecu: Security factor in the stability function [-].
        tkesecu: Security factor in the TKE equation [-].
        tkesmot: Time smoothing factor towards the previous velocity scale [-].
        vel_min: Minimal velocity scale [m/s].
        tke_time_step: 'dt_tke' [s].
        inverse_tke_time_step: 'fr_tke', '1/dt_tke' [1/s].
        gravitational_acceleration: 'grav' [m/s2]; squared inside, as 'grav2' is in the Fortran.
        molecular_diffusivity_for_scalars: 'con_h' [m2/s].
        turbulent_velocity_scale: Output, 'tke(:,:,ntur)' [m/s], rows 1..'ke' - 1. The model
            top is not written here.
        updated_stability_length_for_momentum: Output, 'lsm' [m].
        updated_stability_length_for_scalars: Output, 'lsh' [m].
        circulation_acceleration: Output, 'zvari(:,:,0)' [m/s2].
        supersaturation_standard_deviation: Output, 'rcld' [-].
        diffusion_coefficient_for_momentum: Output, 'tkvm' [m2/s].
        diffusion_coefficient_for_scalars: Output, 'tkvh' [m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k_st = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'k_en = kem = ke'. The circulation
            acceleration runs one row deeper.
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
    _compute_stability_lengths(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        turbulent_velocity_scale=turbulent_velocity_scale,
        a_h=a_h,
        a_m=a_m,
        b_h=b_h,
        b_m=b_m,
        d_m=d_m,
        d_1=d_1,
        d_2=d_2,
        d_3=d_3,
        d_4=d_4,
        d_5=d_5,
        d_6=d_6,
        rim=rim,
        frcsecu=frcsecu,
        stbsecu=stbsecu,
        out=(updated_stability_length_for_momentum, updated_stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_circulation_acceleration(
        cloud_cover=cloud_cover,
        master_length_scale=master_length_scale,
        thermal_forcing=thermal_forcing,
        half_level_pressure=half_level_pressure,
        air_density=air_density,
        pattern_length_scale=pattern_length_scale,
        horizontal_grid_scale=horizontal_grid_scale,
        gravitational_acceleration=gravitational_acceleration,
        out=circulation_acceleration,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end + 1),
        },
    )
    _compute_supersaturation_standard_deviation(
        master_length_scale=master_length_scale,
        stability_length_for_scalars=updated_stability_length_for_scalars,
        exner_factor=exner_factor,
        saturation_humidity_derivative=saturation_humidity_derivative,
        gradient_of_liquid_water_potential_temperature=gradient_of_liquid_water_potential_temperature,
        gradient_of_total_water=gradient_of_total_water,
        d_h=d_h,
        out=supersaturation_standard_deviation,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_diffusion_coefficients_from_stability_lengths(
        stability_length_for_momentum=updated_stability_length_for_momentum,
        stability_length_for_scalars=updated_stability_length_for_scalars,
        turbulent_velocity_scale=turbulent_velocity_scale,
        molecular_diffusivity_for_scalars=molecular_diffusivity_for_scalars,
        out=(diffusion_coefficient_for_momentum, diffusion_coefficient_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
