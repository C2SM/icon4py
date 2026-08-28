# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Standard deviation of the local super-saturation (SDSS).

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'solve_turb_budgets', from the block headed "Calculating Standard Deviation of local
Super-Saturation (SDSS)" (:1772-1793 at icon commit 26d6b98cce). Called from 'turbdiff' section
3) (turb_diffusion.f90:1795-1914). The scientific commentary is by Matthias Raschendorfer (DWD).

    rcld(i,k)=SQRT( tls(i,k)*lsh(i,k)*d_h ) * &
              ABS( exner(i,k)*qst_t(i,k)*grd(i,k,tet_l) - grd(i,k,h2o_g) )

An effective length scale, 'SQRT(tls*lsh*d_h)', times the effective vertical gradient of the
super-saturating water. The second factor converts the liquid-water potential temperature
gradient into a saturation humidity gradient with 'exner*dq_sat/dT' and subtracts the total
water gradient; where the two balance, the super-saturation does not vary along the vertical and
the sub-grid cloud is sharp.

IT OVERWRITES THE CLOUD COVER. 'rcld' arrives as the saturation fraction that
'adjust_satur_equil' produced in section 0) and leaves as the SDSS, in the same Fortran storage.
The raw circulation term is the other reader of the cloud cover, so in the Fortran that block
must run first; here they are separate fields and the ordering is free. Section 11) later
interpolates the SDSS back to main levels.

The Fortran writes only levels 'k_st..k_en', not down to 'k_sf', and says why (:1774-1780): with
'k_en = ke < k_sf', the surface value has already been produced by the call of
'solve_turb_budgets' from 'turbtran', possibly as a tile aggregation, and recomputing it from
aggregated inputs would be wrong.
"""

import gt4py.next as gtx
from gt4py.next import abs, sqrt  # noqa: A004 [builtin-shadowing]

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_supersaturation_standard_deviation(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    saturation_humidity_derivative: fa.CellKField[wpfloat],
    gradient_of_liquid_water_potential_temperature: fa.CellKField[wpfloat],
    gradient_of_total_water: fa.CellKField[wpfloat],
    d_h: wpfloat,
    supersaturation_standard_deviation: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the SDSS on the half levels the turbulence model covers, and no deeper.

    The vertical domain is the Fortran 'DO k=k_st,k_en' with 'k_st = 2' and 'k_en = kem = ke'
    (turb_diffusion.f90:1814): half levels 1 to 'ke' - 1 zero-based. Half level 'ke1' keeps the
    value 'turbtran' produced; see the module docstring for why it must.

    Args:
        master_length_scale: 'tls' [m].
        stability_length_for_scalars: 'lsh' [m], the updated value.
        exner_factor: 'exner' [-].
        saturation_humidity_derivative: 'qst_t' [1/K].
        gradient_of_liquid_water_potential_temperature: 'grd(:,:,tet_l)' [K/m].
        gradient_of_total_water: 'grd(:,:,h2o_g)' [1/m].
        d_h: 'd_heat' [-].
        supersaturation_standard_deviation: Output, 'rcld' [-].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k_st = 2'.
        vertical_end: End of the half levels; 'ke', mirroring 'k_en = kem = ke'.
    """
    _compute_supersaturation_standard_deviation(
        master_length_scale=master_length_scale,
        stability_length_for_scalars=stability_length_for_scalars,
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
