# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""'1/Ri**(2/3)', the stability factor that damps the non-turbulent TKE sources.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), the block Guenther Zaengl marks '!GZ: For tuning.' at :1396-1412 of icon commit
26d6b98cce, which is the commit that produced the reference capture:

    xri(i,k) = EXP( z2d3*LOG( MAX( 1.e-6_wp, frm(i,k) ) /   !1/Ri**(2/3)
                              MAX( 1.e-5_wp, frh(i,k) ) ) )

The scientific commentary in that file is by Matthias Raschendorfer (DWD); this block and its
banner are Zaengl's.

The gradient Richardson number is 'Ri = N**2 / S**2', the thermal forcing over the mechanical
one, so 'frm/frh' is '1/Ri' and the two-thirds power is the empirical exponent the tuning uses.
The two floors are what make the quotient defined: 'frh' is a buoyancy forcing and is negative
throughout the unstable boundary layer, and 'LOG' of a negative number is not a number. They are
not symmetric -- 1e-6 on the shear, 1e-5 on the buoyancy -- so the neutral limit this produces
is 'xri = (0.1)**(2/3)', a small number rather than one.

Its two consumers are both in the same family: 'compute_effective_horizontal_shear_length_scale'
scales the separated horizontal shear mode with it here in section 2a), and section 4) uses it
for the Richardson-number-dependent minimum diffusion coefficients.

THIS IS THE ONLY TRANSCENDENTAL IN SECTION 2a), AND THE ONLY THING IN IT THAT IS NOT BIT-EXACT.
'EXP(LOG())' is evaluated by the target's libm, and the reference was produced by nvhpc's. Every
other quantity of the section reproduces ICON's bits exactly when it is handed ICON's own 'xri';
the section's tolerant gates are all downstream of this one. Writing it as a 'power' instead
would not help -- the same libm decides -- and would additionally change which routine rounds.
"""

import gt4py.next as gtx
from gt4py.next import exp, log, maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_inverse_richardson_number_factor(
    mean_shear_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'EXP(2/3 * LOG(frm/frh))' with both arguments floored away from zero.

    'z2d3' is 'z2/z3' (turb_diffusion.f90:270), i.e. the double-precision quotient of the two
    integers and not a decimal literal, which is what '2.0/3.0' reproduces exactly.
    """
    return exp(
        (wpfloat("2.0") / wpfloat("3.0"))
        * log(
            maximum(wpfloat("1.0e-6"), mean_shear_forcing)
            / maximum(wpfloat("1.0e-5"), thermal_forcing)
        )
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_richardson_number_factor(
    mean_shear_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'xri', on the MAIN-level storage but at half-level positions.

    'xri' is declared 'xri(nvec,ke)' (turb_diffusion.f90:802) -- one row shorter than the half
    levels -- because the surface row is never needed. Its values are half-level quantities all
    the same, formed from half-level 'frm' and 'frh' at the same row index. The port keeps the
    Fortran's shape, so the field passed here has 'ke' rows and 'vertical_end' is 'ke'.

    Args:
        mean_shear_forcing: The mechanical forcing by the mean flow, 'frm' as
            'compute_three_dimensional_shear_forcing' leaves it [1/s2]. NOT the 'frm' of the
            section's exit savepoint, which already carries the non-turbulent contributions.
        thermal_forcing: The buoyancy forcing 'frh' [1/s2], written by section 1b) and not
            touched by this section.
        inverse_richardson_number_factor: Output, 'xri' [-].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First row; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the rows; 'ke', mirroring Fortran 'k=...,ke'.
    """
    _compute_inverse_richardson_number_factor(
        mean_shear_forcing=mean_shear_forcing,
        thermal_forcing=thermal_forcing,
        out=inverse_richardson_number_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
