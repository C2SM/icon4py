# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion tendency of the potential temperature, applied to the temperature tendency.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2661-2670 at icon commit 26d6b98cce) and from 'icon/src/atm_phy_schemes/turb_vertdiff.f90',
SUBROUTINE 'vertdiff' (:773-785, the 'n.EQ.tem' branch of "Sichern der Tendenzen"). The
scientific commentary in those files is by Matthias Raschendorfer (DWD).

    dif_tend(i,k) = ( dif_tend(i,k) - cur_prof(i,k) )*fr_var
    dvar_at(i,k)  = dvar_at(i,k) + epr(i,k)*dicke(i,k)

The temperature is diffused as a potential temperature, so its tendency comes back in those
units and the Exner pressure converts it. This is 'compute_and_apply_diffusion_tendency' with
that one factor; the two are separate programs rather than one with a unit multiplier, because
a caller supplying a field of ones to a program that does not need it is worse than a second
program.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_potential_temperature_diffusion_tendency(
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'(upd_prof - cur_prof)*fr_var' [K/s], in potential-temperature units."""
    return (updated_profile - current_profile) * reciprocal_time_step


@gtx.field_operator
def _apply_potential_temperature_diffusion_tendency(
    temperature_tendency: fa.CellKField[wpfloat],
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'t_tens + epr*dicke', with the product grouped as the Fortran writes it."""
    return temperature_tendency + exner_factor * _compute_potential_temperature_diffusion_tendency(
        updated_profile, current_profile, reciprocal_time_step
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_and_apply_potential_temperature_diffusion_tendency(
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    temperature_tendency_before: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
    diffusion_tendency: fa.CellKField[wpfloat],
    temperature_tendency: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Turn the solved potential-temperature profile into a temperature tendency.

    Args:
        updated_profile: 'upd_prof' after the solve, a potential temperature [K].
        current_profile: 'cur_prof' before the solve, a potential temperature [K].
        exner_factor: 'epr' on the main levels [-].
        temperature_tendency_before: 't_tens' as the scheme found it [K/s]. ICON accumulates in
            place and a caller may pass the same field here and as the output.
        reciprocal_time_step: 'fr_var = 1/dt_var' [1/s].
        diffusion_tendency: Output, 'dif_tend' in potential-temperature units [K/s].
        temperature_tendency: Output, 't_tens' including the diffusion increment [K/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First diffused main level; 0.
        vertical_end: 'nlev'.
    """
    _compute_potential_temperature_diffusion_tendency(
        updated_profile=updated_profile,
        current_profile=current_profile,
        reciprocal_time_step=reciprocal_time_step,
        out=diffusion_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _apply_potential_temperature_diffusion_tendency(
        temperature_tendency=temperature_tendency_before,
        updated_profile=updated_profile,
        current_profile=current_profile,
        exner_factor=exner_factor,
        reciprocal_time_step=reciprocal_time_step,
        out=temperature_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
