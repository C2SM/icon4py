# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""TKE production by the separated, non-turbulent horizontal shear mode.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), :1466-1525 at icon commit 26d6b98cce, the commit that produced the reference capture. The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

Three consecutive loops, all on the scratch array 'hlp', and a fourth that copies it out:

    :1482  wert = fakt*hdiv(i,k)                                   with fakt = 1/(2*sm_0)**2
    :1483  hlp(i,k) = hor_scale(i,k)*(SQRT(wert**2+hdef2(i,k))-wert)
    :1509  hlp(i,k) = (hlp(i,k))**3/hor_scale(i,k)
    :1521  tket_hshr(i,k) = hlp(i,k)

They are fused here because the first two intermediates never reach a savepoint: 'hlp' is
overwritten in the same section by 'compute_sso_wake_energy_production', so what the exit
savepoint holds in it is the SSO energy conversion and not this.

WHAT THE THREE STEPS ARE. 'hlp' after :1483 is a strain velocity: a length scale times the
trace of the two-dimensional strain-rate tensor of the separated mode. Raschendorfer's comment
calls that out -- ":1472 not equal to trace of 2D-strain tensor" on the branch this port does
not take, ":1483 equal to trace of 2D-strain tensor" on the one it does -- and the difference is
exactly the '-wert' subtraction, which removes the divergent part that incompressibility already
assigns to the vertical. A velocity cubed over a length is a TKE production rate [m2/s3], which
is :1509, and it enters the budget as such.

'imode_shshear' SELECTS THIS FORM as well. At 'imode_shshear = 0' :1472 takes the former
variant, 'hor_scale*SQRT(hdef2 + hdiv**2)', which is a different number by orders of magnitude
here; the reference capture ran 2. See 'compute_effective_horizontal_shear_length_scale'.

'loutshshr' GATES ONLY THE COPY at :1521, not the computation: with the output switch off the
production is still formed and still added to 'frm' at :1534, just not published. The port
carries it in the output field either way, because 'compute_total_mechanical_forcing' is its
other consumer and needs it as an input.
"""

import gt4py.next as gtx
from gt4py.next import sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_separated_horizontal_shear_tke_source(
    effective_horizontal_shear_length_scale: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    neutral_momentum_stability_function: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'(hor_scale*(SQRT((fakt*hdiv)**2 + hdef2) - fakt*hdiv))**3 / hor_scale'.

    BOTH POWERS ARE WRITTEN AS PRODUCTS. Fortran's 'x**2' and 'x**3' with integer literal
    exponents are multiplications; GT4Py's '**' is 'math.pow', and CUDA's 'pow' carries up to
    2 ulp. Measured in '_compute_mechanical_forcing'. Cancelling the cube against the division
    algebraically -- 'hor_scale**2 * (...)**3' -- would be a third different rounding again, so
    the Fortran's two steps are kept as two steps.

    'fakt' is 'z1/(z2*sm_0)**2' (:1421), formed once per call from the neutral momentum
    stability function; it converts the divergence into the same units as the deformation.
    """
    scaled_divergence = (
        wpfloat("1.0")
        / (
            (wpfloat("2.0") * neutral_momentum_stability_function)
            * (wpfloat("2.0") * neutral_momentum_stability_function)
        )
    ) * horizontal_divergence
    strain_velocity = effective_horizontal_shear_length_scale * (
        sqrt(scaled_divergence * scaled_divergence + horizontal_deformation_square)
        - scaled_divergence
    )
    return (
        strain_velocity * strain_velocity * strain_velocity
    ) / effective_horizontal_shear_length_scale


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_separated_horizontal_shear_tke_source(
    effective_horizontal_shear_length_scale: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    neutral_momentum_stability_function: wpfloat,
    separated_horizontal_shear_tke_source: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'tket_hshr' [m2/s3], on half levels.

    Args:
        effective_horizontal_shear_length_scale: 'hor_scale' [m], from
            'compute_effective_horizontal_shear_length_scale'. It is stored with 'ke' rows.
        horizontal_divergence: 'hdiv' [1/s], half levels.
        horizontal_deformation_square: 'hdef2' [1/s2], half levels.
        neutral_momentum_stability_function: 'sm_0' [-], a derived closure constant of
            'TurbulenceParams'.
        separated_horizontal_shear_tke_source: Output, 'tket_hshr' [m2/s3], half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the half levels; 'ke', mirroring Fortran 'k=...,kem'.
    """
    _compute_separated_horizontal_shear_tke_source(
        effective_horizontal_shear_length_scale=effective_horizontal_shear_length_scale,
        horizontal_divergence=horizontal_divergence,
        horizontal_deformation_square=horizontal_deformation_square,
        neutral_momentum_stability_function=neutral_momentum_stability_function,
        out=separated_horizontal_shear_tke_source,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
