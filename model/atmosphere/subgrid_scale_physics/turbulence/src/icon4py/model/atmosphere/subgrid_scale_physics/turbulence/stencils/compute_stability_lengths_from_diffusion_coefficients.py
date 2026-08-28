# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The stability lengths the turbulence model wants, from the diffusion coefficients it left.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2c), :1741-1752 at icon commit 26d6b98cce, under Raschendorfer's heading

    "Belegung von tkvh und tkvm mit den stabilitaetsabhaengigen Laengenmassen:"
    -- "Filling 'tkvh' and 'tkvm' with the stability-dependent length measures"

The scientific commentary in that file is by Matthias Raschendorfer (DWD).

    DO k=2,kem
      DO i=ivstart, ivend
        wert=z1/tke(i,k,nvor)
        tkvh(i,k)=tkvh(i,k)*wert
        tkvm(i,k)=tkvm(i,k)*wert
      END DO
    END DO

'turbdiff' receives 'tkvm' and 'tkvh' as the diffusion coefficients of the previous time step
and 'solve_turb_budgets' wants them as the stability lengths 'ls[m|h] = S[m|h]*len_scale' [m],
which is what this division by the turbulent velocity scale produces. Section 3) multiplies
them back, in 'compute_diffusion_coefficients_from_stability_lengths'; 'turbdiff-2c-exit' is the
one savepoint at which the two storages hold a length, which is why the serialized-data reader
names them 'stab_len_m()' and 'stab_len_h()' there and refuses 'tkvm()'/'tkvh()'.

Raschendorfer's note at :1755-1765 says what the pair now means and, in its last four lines,
that the value is not simply the stability length 'solve_turb_budgets' produced at the previous
time step: the artificial lower limits of section 4) have been applied to it since, and how much
of that impact survives depends on 'imode_tkemini' (frozen at 1 in this port).

IT IS A RECIPROCAL AND A MULTIPLICATION, NOT A DIVISION. The Fortran forms 'wert = z1/q' once
and multiplies both coefficients by it, and 'x * (1/y)' is not 'x / y' in floating point: the
reciprocal rounds first and the product rounds again, so the two agree only where the reciprocal
happens to be exact. Writing 'tkvm / q' here would therefore not be a simplification but a
different computation, and the reference data can tell them apart at a quarter of its values.
Measured over the 653 804 values this section writes at the first serialized date: the
reciprocal form reproduces ICON exactly, the division differs at 156 546 of them for 'ls_m' and
at 160 398 for 'ls_h', by up to 1 ulp. The same holds at all four dates, and
'test_the_shared_reciprocal_is_not_a_division' keeps measuring it.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_stability_lengths_from_diffusion_coefficients(
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """A diffusion coefficient divided by the turbulent velocity scale is a stability length.

    The shared reciprocal is a local of the Fortran ('wert', PRIVATE to the collapsed loop) and
    is one here too, for the reason the module docstring gives: it is the operation the
    reference performed, and reassociating it would cost bit-exactness.

    Args:
        diffusion_coefficient_for_momentum: 'tkvm' [m2/s], as 'turbtran' and the previous time
            step left it.
        diffusion_coefficient_for_scalars: 'tkvh' [m2/s].
        turbulent_velocity_scale: 'tke(:,:,nvor)', 'q = SQRT(2*TKE)' [m/s], of the time level the
            security iteration started from.

    Returns:
        'ls_m' and 'ls_h', the stability lengths 'S[m|h]*len_scale' [m], in that order.
    """
    reciprocal_velocity_scale = wpfloat("1.0") / turbulent_velocity_scale
    return (
        diffusion_coefficient_for_momentum * reciprocal_velocity_scale,
        diffusion_coefficient_for_scalars * reciprocal_velocity_scale,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths_from_diffusion_coefficients(
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Turn the diffusion coefficients into stability lengths on half levels 2..kem.

    The vertical domain is the Fortran 'DO k=2,kem' with 'kem = ke' (turb_diffusion.f90:843), so
    zero-based half levels 1 to 'ke' - 1: the same domain as the inverse operation in section 3),
    and for the same reason. Raschendorfer's own comment on the loop says it --

        "stability-dependent length-scales in 'tkvh/m' only for the here included atmospheric
         levels"

    -- the surface half level 'ke1' keeps the coefficients 'turbtran' produced, because the
    turbulence model is not applied there and section 3) would have nothing to multiply back.
    Row 0 is never written by 'turbdiff' at all.

    In the Fortran this is an in-place update of the two arrays; here the inputs and the outputs
    are separate fields, which is what lets the datatest assert that the rows outside the domain
    come out as the section found them.

    Args:
        diffusion_coefficient_for_momentum: 'tkvm' [m2/s] on entry to section 2c).
        diffusion_coefficient_for_scalars: 'tkvh' [m2/s] on entry to section 2c).
        turbulent_velocity_scale: 'tke(:,:,nvor)' [m/s].
        stability_length_for_momentum: Output, 'ls_m' [m]; the 'tkvm' storage as
            'turbdiff-2c-exit' holds it.
        stability_length_for_scalars: Output, 'ls_h' [m].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'kem = ke'.
    """
    _compute_stability_lengths_from_diffusion_coefficients(
        diffusion_coefficient_for_momentum=diffusion_coefficient_for_momentum,
        diffusion_coefficient_for_scalars=diffusion_coefficient_for_scalars,
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=(stability_length_for_momentum, stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
