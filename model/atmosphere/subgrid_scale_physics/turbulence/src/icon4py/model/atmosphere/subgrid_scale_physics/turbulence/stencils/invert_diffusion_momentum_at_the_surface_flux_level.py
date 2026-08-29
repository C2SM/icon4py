# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Last row of the LU factorisation under a surface-flux lower boundary condition.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2850-2858 at icon commit 26d6b98cce), the third and last elimination
loop of the branch "without preconditioning". 'vertdiff' reaches it through 'vert_grad_diff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    DO k = k_sf+1-m, k_sf-1   !only for a surface-flux condition and at level k_sf-1
       invs_fac(i,k) = invs_mom(i,k-1)*impl_mom(i,k)
       invs_mom(i,k) = 1/( disc_mom(i,k) + impl_mom(i,k)*(1 - invs_fac(i,k)) )

THIS IS THE COMPLEMENT OF 'compute_inverted_diffusion_momentum', not a replacement for it. That
program is the second Fortran loop, 'DO k = k_tp+2, k_sf-m', and its own docstring says the
third loop "is empty at 'm = 1' and is not translated; at 'm = 2' it would carry the
surface-flux condition". 'vertdiff' is where 'm = 2' actually occurs: the scalar type runs with
'tdc%lsflcnd = .TRUE.', so its second loop stops one row early and this row finishes it.

The formula differs from the second loop's in exactly one term: 'impl_mom(i,k+1)' is absent.
Under a flux condition the surface concentration is not an unknown of the system, so the last
diffused level has no sub-diagonal below it. Writing 'impl_mom(:,k_sf) = 0' and reusing the
scan would give the same number, but it would also destroy the row the momentum type left
there, which the exit savepoint still holds; this keeps the two ranges as ICON has them.

Only 'invs_mom' is computed here. 'invs_fac' is the same product on this row as on every other
and 'compute_diffusion_inversion_factor' covers the whole range in one go.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _invert_diffusion_momentum_at_the_surface_flux_level(
    discretisation_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The reciprocal pivot of the last diffused level, without a sub-diagonal below it."""
    inversion_factor = inverted_diffusion_momentum(Koff[-1]) * implicit_diffusion_momentum
    return wpfloat("1.0") / (
        discretisation_momentum + implicit_diffusion_momentum * (wpfloat("1.0") - inversion_factor)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def invert_diffusion_momentum_at_the_surface_flux_level(
    discretisation_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum_above: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Finish 'invs_mom' on the lowest diffused main level, for a surface-flux condition.

    Run it after 'compute_inverted_diffusion_momentum', which must have been given
    'vertical_end = nlev - 1' so that this row is the one it stopped short of. The vertical
    range here is that single row: 'vertical_start = nlev - 1, vertical_end = nlev'.

    Args:
        discretisation_momentum: 'disc_mom' [kg/m2/s].
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on the flux level above this row.
        inverted_diffusion_momentum_above: 'invs_mom' [m2 s/kg], read one level up: the last
            row 'compute_inverted_diffusion_momentum' wrote. ICON has one array and a caller
            passes the same field here and below; the row read is never the row written.
        inverted_diffusion_momentum: Output, 'invs_mom' [m2 s/kg] on this row.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The lowest diffused main level, 'nlev - 1' (Fortran 'k_sf-1 = ke').
        vertical_end: 'nlev'.
    """
    _invert_diffusion_momentum_at_the_surface_flux_level(
        discretisation_momentum=discretisation_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        inverted_diffusion_momentum=inverted_diffusion_momentum_above,
        out=inverted_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
