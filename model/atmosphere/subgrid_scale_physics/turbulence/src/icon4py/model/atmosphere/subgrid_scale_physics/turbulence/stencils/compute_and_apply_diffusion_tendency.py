# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion tendency of a model variable, and its addition to the variable's tendency.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2661-2670 at icon commit 26d6b98cce, the block "Calculation of time tendencies for pure
vertical diffusion") and from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE
'vertdiff' (:786-799, the 'ELSE' branch of "Sichern der Tendenzen" -- "saving the tendencies").
The scientific commentary in those files is by Matthias Raschendorfer (DWD).

    dif_tend(i,k) = ( dif_tend(i,k) - cur_prof(i,k) )*fr_var    ! dif_tend held upd_prof
    dvar_at(i,k)  = dvar_at(i,k) + dicke(i,k)

Two Fortran statements in two subroutines, one program here, because the second is nothing but
the first added to an accumulator and because keeping the diffusion tendency as its own output
is what makes the pair checkable: it is the 'dicke' storage the exit savepoint holds.

The two outputs are computed from the same three inputs and neither reads the other, so the
addition re-forms '(upd_prof - cur_prof)*fr_var' instead of reading it back. That is bit-exact
-- the same expression rounds to the same number -- and it removes a cross-statement dependency
that would otherwise make the order of the two writes matter.

'vert_grad_diff' would follow this with a volume correction inside the roughness layer
('IF (PRESENT(r_air))', :2672-2684). Its loop runs 'DO k = kcm, k_lw' and ICON-NWP leaves 'kcm'
at 'ke+1', so it is empty; it is not ported. The same holds for 'vert_smooth', which
'vertdiff' does not call at all.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_tendency(
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'(upd_prof - cur_prof)*fr_var', the tendency of pure vertical diffusion."""
    return (updated_profile - current_profile) * reciprocal_time_step


@gtx.field_operator
def _apply_diffusion_tendency(
    variable_tendency: fa.CellKField[wpfloat],
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'dvar_at + dicke', with 'dicke' re-formed rather than read back."""
    return variable_tendency + _compute_diffusion_tendency(
        updated_profile, current_profile, reciprocal_time_step
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_and_apply_diffusion_tendency(
    updated_profile: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    variable_tendency_before: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
    diffusion_tendency: fa.CellKField[wpfloat],
    variable_tendency: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Turn the solved profile into a tendency and add it to the variable's tendency.

    The vertical range is the Fortran 'DO k = k_hi, k_lw' with 'k_hi = k_tp+1' and
    'k_lw = k_sf-1 = ke', i.e. 'vertical_start = 0, vertical_end = nlev'.

    Args:
        updated_profile: 'upd_prof' after the solve, in the variable's own units.
        current_profile: 'cur_prof' before the solve, same units.
        variable_tendency_before: 'dvar_at' as the scheme found it. ICON accumulates in place
            and a caller may pass the same field here and as the output; the statement is
            pointwise.
        reciprocal_time_step: 'fr_var = 1/dt_var' [1/s].
        diffusion_tendency: Output, 'dif_tend' -- the 'dicke' storage -- the tendency of pure
            vertical diffusion [variable units per s].
        variable_tendency: Output, 'dvar_at' including the diffusion increment.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First diffused main level; 0, Fortran 'k_st_pp'.
        vertical_end: 'nlev'.
    """
    _compute_diffusion_tendency(
        updated_profile=updated_profile,
        current_profile=current_profile,
        reciprocal_time_step=reciprocal_time_step,
        out=diffusion_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _apply_diffusion_tendency(
        variable_tendency=variable_tendency_before,
        updated_profile=updated_profile,
        current_profile=current_profile,
        reciprocal_time_step=reciprocal_time_step,
        out=variable_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
