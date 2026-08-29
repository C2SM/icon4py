# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stability functions of the level-2.5 closure, as lengths 'S*l' rather than as 'S'.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'solve_turb_budgets', from the block headed "Calculating updated value for stability lenght
(stability-factor times turbulent master length-scale)" (:1548-1697 at icon commit 26d6b98cce).
Called from 'turbdiff' section 3) (turb_diffusion.f90:1795-1914). The scientific commentary is
by Matthias Raschendorfer (DWD).

Raschendorfer never stores the dimensionless 'S_m' and 'S_h': the last two statements of the
block multiply them by the master length scale, and every consumer wants the product. So do we,
and the outputs are metres.

TWO SOLUTIONS FOR THE SAME PAIR. The closure gives a 2x2 linear system for 'S_h' and 'S_m' once
'tke' is known, and that is the "standard solution". It is not always realizable: at strongly
unstable stratification the deviation from TKE equilibrium, 'gama := S_m*gm - S_h*gh', runs to
infinity. Raschendorfer's own note (:1614-1625, abridged):

    "The pure solution for the stability functions 'sh' and 'sm' through the above linear
     system, inserting given 'tke'-values from the just before solved TKE-equation, may become
     non-realizable at strongly unstable stratification ... This problem is circumvented by
     employing a modified solution based on a predescribed deviation 'gama', which is expressed
     by 'frc*tim2/tls' and an upper limit 'gam0'."

The modified solution is the one this port takes wherever the standard one is not taken; which
of the two applies is decided per point, and both are evaluated. See '_compute_stability_lengths'
for the exact condition.

WHY THIS IS NOT A SCAN: see the module docstring of 'compute_turbulent_velocity_scale'. The
enclosing k-loop has no vertical neighbour access at all.
"""

import gt4py.next as gtx
from gt4py.next import minimum, sqrt, where

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_velocity_scale import (
    _effective_tke_forcing,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths(
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
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Update 'lsm' and 'lsh' on the half levels the turbulence model covers.

    The vertical domain is the Fortran 'DO k=k_st,k_en' with 'k_st = 2' and 'k_en = kem = ke'
    (turb_diffusion.f90:1814): half levels 1 to 'ke' - 1 zero-based.

    Input and output stability lengths are the same Fortran storage, 'lsm' and 'lsh' being
    INTENT(INOUT). They are separate fields here because the input value enters the TKE forcing
    and the output value replaces it; each level reads only its own level, so the Fortran
    in-place update is safe there and means nothing here.

    Args:
        master_length_scale: 'tls' [m].
        stability_length_for_momentum: 'lsm' [m] on entry.
        stability_length_for_scalars: 'lsh' [m] on entry.
        mechanical_forcing: 'fm2' [1/s2].
        thermal_forcing: 'fh2' [1/s2].
        turbulent_velocity_scale: 'tke(:,:,ntur)' [m/s].
        a_h: Length-scale factor of the turbulent heat transport [-].
        a_m: Length-scale factor of the turbulent momentum transport [-].
        b_h: Scalar closure constant [-].
        b_m: Momentum closure constant [-].
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
        updated_stability_length_for_momentum: Output, 'lsm' [m].
        updated_stability_length_for_scalars: Output, 'lsh' [m].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k_st = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'k_en = kem = ke'.
    """
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
