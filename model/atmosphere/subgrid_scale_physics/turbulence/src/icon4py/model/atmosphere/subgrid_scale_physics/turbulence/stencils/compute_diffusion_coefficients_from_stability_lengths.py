# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Turbulent diffusion coefficients from the stability lengths and the velocity scale.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
3), :1863-1885 at icon commit 26d6b98cce. The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

    tkvh(i,k)=MAX(tkvh(i,k)*tke(i,k,ntur),con_h)
    tkvm(i,k)=tkvm(i,k)*tke(i,k,ntur)

with Raschendorfer's note on what the two arrays now mean (:1878-1882):

    "'tke' is always the turbulent velocity scale in [m/s] and NOT 'TKE = 1/2*q^2'!
     'tkv[m|h]' are no longer the stability length-scales 'S[m|h]*len_scale' now, but the turb.
     diff. coeffs. in [m^2/s] again."

Section 2c) divided the same storage by 'tke' to hand the turbulence model a length; this
multiplies it back by the freshly computed 'tke' and closes that round trip. The port keeps the
two roles in separate fields, so nothing here is in place.

Only the scalar coefficient gets the molecular floor 'con_h'. That asymmetry is in the Fortran
and is not an oversight: the momentum coefficient receives its own lower limits in section 4).
"""

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


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
def compute_diffusion_coefficients_from_stability_lengths(
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    molecular_diffusivity_for_scalars: wpfloat,
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Turn the stability lengths back into diffusion coefficients on half levels 2..kem.

    The vertical domain is the Fortran 'DO k=2, kem' with 'kem = ke': half levels 1 to 'ke' - 1
    zero-based. Half level 'ke1' holds the surface coefficients 'turbtran' produced and is not
    written; row 0 is never written by 'turbdiff' at all.

    Args:
        stability_length_for_momentum: 'lsm' [m].
        stability_length_for_scalars: 'lsh' [m].
        turbulent_velocity_scale: 'tke(:,:,ntur)' [m/s].
        molecular_diffusivity_for_scalars: 'con_h' [m2/s].
        diffusion_coefficient_for_momentum: Output, 'tkvm' [m2/s].
        diffusion_coefficient_for_scalars: Output, 'tkvh' [m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'kem = ke'.
    """
    _compute_diffusion_coefficients_from_stability_lengths(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        turbulent_velocity_scale=turbulent_velocity_scale,
        molecular_diffusivity_for_scalars=molecular_diffusivity_for_scalars,
        out=(diffusion_coefficient_for_momentum, diffusion_coefficient_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
