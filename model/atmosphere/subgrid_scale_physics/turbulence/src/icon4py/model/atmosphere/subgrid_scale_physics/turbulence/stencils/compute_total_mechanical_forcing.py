# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The total mechanical forcing of the TKE equation: mean shear plus the non-turbulent modes.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), the two loops that finish 'frm' at :1531-1536 and :1572-1596 of icon commit 26d6b98cce,
the commit that produced the reference capture. The scientific commentary in that file is by
Matthias Raschendorfer (DWD).

    :1534  frm(i,k) = frm(i,k) + hlp(i,k)/tkvm(i,k)                    !extended shear
    :1575  wert = MAX( z0, -zbnd_val(hlp(i,k), hlp(i,k-1), dp0(i,k), dp0(i,k-1)) )
    :1589  frm(i,k) = frm(i,k) + wert/tkvm(i,k)*MIN(1.0_wp,MAX(0.01_wp,xri(i,k)))

Both add an energy production rate divided by the momentum diffusion coefficient, which is what
turns [m2/s3] into the [1/s2] of a shear forcing: 'frm' is a squared shear, and dividing a TKE
source by 'tkvm' is the shear that would produce it. Raschendorfer's heading for the whole group
is "Erweiterung der Vertikalscherung mit verallgemeinerten Scher-Beitraegen durch die
nicht-turbulente subskalige Stroemung (STIC-Terme)" -- extension of the vertical shear by
generalised shear contributions of the non-turbulent sub-grid flow.

THE 'MAX' AND THE SIGN. 'hlp' is the rate at which the SSO drag drains kinetic energy from the
mean flow, which is negative, so it is negated to become a source. Raschendorfer's own note at
:1577-1579: "Although the SSO-tendencies 'ut_sso' and 'vt_sso' should never become positive, the
'MAX'-function is used for security."

THE INTERPOLATION. 'zbnd_val' (turb_utilities.f90:3372) is the mass-weighted interpolation of a
main-level profile onto the half level between two main levels,

    zbnd_val(val1, val2, dep1, dep2) = (val1*dep2 + val2*dep1)/(dep1 + dep2)

called here with the two main levels straddling half level 'k' and their pressure thicknesses.
It is the same interpolation section 0) performs with a precomputed weight
('compute_half_level_interpolation_weight'); this call site takes the unweighted branch, and the
two are not the same number to the last bit, so the branch is part of the translation.

WHAT IS NOT PORTED, AND WHY IT HAS NO ORACLE HERE.

  * 'imode_tkesso = 1', the SSO source without the Richardson-number reduction (:1587). The
    capture ran 2 or 3, established from the data; mode 1 is off by a factor of 90 and is
    excluded.
  * The distinction between 'imode_tkesso = 2' and '= 3' (:1589 vs :1591). Mode 3 multiplies by
    'MIN(1, l_hori/2000)', an additional reduction for meshes finer than 2 km. The capture's
    'l_hori' is 9863.8 m on every column, so that factor is exactly 1 and THE TWO MODES ARE
    INDISTINGUISHABLE IN THIS DATA. What is implemented is their common value; a mesh finer than
    2 km would need mode 3's factor and a capture that can tell it apart.
  * The convective circulation term at :1602-1614, 'frm += MAX(0, tket_conv/tkvm)'. It needs
    'ltkecon', which 'TurbulenceConfig' freezes '.FALSE.'; adding it here breaks the bit-exact
    agreement, which is how 'test_the_capture_adds_no_convective_circulation_shear' shows the
    branch is dead.

WHY THE MEAN SHEAR ARRIVES AS AN INPUT AND IS NOT ACCUMULATED IN PLACE. The Fortran adds to
'frm' three times in this section, and the intermediate after the first addition is what 'xri'
is formed from. The port keeps the mean-shear part in a field of its own -- see
'compute_three_dimensional_shear_forcing', which explains that it is the Fortran's own 'ftm' --
so that no program of this section both reads and writes the same buffer.
"""

import gt4py.next as gtx
from gt4py.next import maximum, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_total_mechanical_forcing(
    mean_shear_forcing: fa.CellKField[wpfloat],
    separated_horizontal_shear_tke_source: fa.CellKField[wpfloat],
    sso_wake_energy_production: fa.CellKField[wpfloat],
    layer_pressure_thickness: fa.CellKField[wpfloat],
    momentum_diffusion_coefficient: fa.CellKField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The mean shear plus the separated horizontal shear plus the SSO wake production.

    The three terms are summed in the Fortran's order and with its parenthesisation. That is
    not pedantry: '(a + b) + c' and 'a + (b + c)' round differently, and the section is
    bit-exact against ICON only in the first form.
    """
    sso_production_at_half_level = maximum(
        wpfloat("0.0"),
        -(
            sso_wake_energy_production * layer_pressure_thickness(Koff[-1])
            + sso_wake_energy_production(Koff[-1]) * layer_pressure_thickness
        )
        / (layer_pressure_thickness + layer_pressure_thickness(Koff[-1])),
    )
    with_separated_horizontal_shear = (
        mean_shear_forcing + separated_horizontal_shear_tke_source / momentum_diffusion_coefficient
    )
    stability_reduction = minimum(
        wpfloat("1.0"), maximum(wpfloat("0.01"), inverse_richardson_number_factor)
    )
    return (
        with_separated_horizontal_shear
        + sso_production_at_half_level / momentum_diffusion_coefficient * stability_reduction
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_total_mechanical_forcing(
    mean_shear_forcing: fa.CellKField[wpfloat],
    separated_horizontal_shear_tke_source: fa.CellKField[wpfloat],
    sso_wake_energy_production: fa.CellKField[wpfloat],
    layer_pressure_thickness: fa.CellKField[wpfloat],
    momentum_diffusion_coefficient: fa.CellKField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'frm' as section 2a) leaves it [1/s2], on half levels.

    NOT A RECURRENCE. The only neighbouring-level reads are of 'hlp' and 'dp0', neither of which
    this program writes; the Fortran loop is a plain '!$ACC LOOP GANG VECTOR COLLAPSE(2)'.

    Args:
        mean_shear_forcing: The mechanical forcing by the three-dimensional mean flow [1/s2],
            half levels, from 'compute_three_dimensional_shear_forcing'.
        separated_horizontal_shear_tke_source: 'tket_hshr' [m2/s3], half levels.
        sso_wake_energy_production: 'hlp' [m2/s3], MAIN levels, from
            'compute_sso_wake_energy_production'.
        layer_pressure_thickness: 'dp0' [Pa], main levels; the interpolation weight of the two
            main levels that meet at a half level.
        momentum_diffusion_coefficient: 'tkvm' [m2/s], half levels, as section 1c) left it.
        inverse_richardson_number_factor: 'xri' [-], from
            'compute_inverse_richardson_number_factor'.
        mechanical_forcing: Output, 'frm' [1/s2], half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the half levels; 'ke', mirroring Fortran 'k=...,kem' with
            'kem = ke'. The surface half level 'ke1' is never written by this section.
    """
    _compute_total_mechanical_forcing(
        mean_shear_forcing=mean_shear_forcing,
        separated_horizontal_shear_tke_source=separated_horizontal_shear_tke_source,
        sso_wake_energy_production=sso_wake_energy_production,
        layer_pressure_thickness=layer_pressure_thickness,
        momentum_diffusion_coefficient=momentum_diffusion_coefficient,
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        out=mechanical_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
