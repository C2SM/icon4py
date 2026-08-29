# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Right-hand side of the semi-implicit vertical diffusion equation.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff' (:2975-2991 at icon commit 26d6b98cce), the block headed "Resultant
right-hand side flux", reached from 'vertdiff' through 'vert_grad_diff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    k = k_tp+1
    eff_flux(i,k) = disc_mom(i,k)*old_prof(i,k) + eff_flux(i,k+1)
    DO k = k_tp+2, k_sf-1
       eff_flux(i,k) = disc_mom(i,k)*old_prof(i,k) + eff_flux(i,k+1) - eff_flux(i,k)

The layer's mass per time step times its current value, plus the net explicit flux convergence:
the flux in from below minus the flux out at the top, both positive upward.

ONE EXPRESSION, NOT TWO. The Fortran's first row omits the outgoing term because there is no
flux level above the top main level; here 'compute_explicit_flux_density' has already written
that level as an explicit zero, so subtracting it is a no-op and the interior expression covers
the whole range. The package README's boundary-row convention would otherwise put a
'concat_where' here; making the boundary condition a value instead of a row selection is
bit-exact ('x - 0.0' is 'x'), keeps this program runnable on the embedded backend, and states
the upper boundary condition once, where the flux lives, instead of once per consumer.

IT IS NOT A RECURRENCE, DESPITE THE '!$ACC LOOP SEQ'. The loop runs downward from 'k_tp+2' and
reads 'eff_flux(i,k+1)', a row it has not reached yet, so every read is of the explicit flux as
it was before the loop began. The Fortran is sequential only because it overwrites the array it
reads; the port keeps the two apart, which is why 'explicit_flux_density' and 'right_hand_side'
are different arguments. Writing them to one field would make the answer depend on the order in
which GT4Py visits the rows.

The row range stops one short of the surface, so the surface row of whatever field receives this
keeps the explicit surface flux -- which is what the exit savepoint's 'zvari(:,ke1,m)' holds.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_right_hand_side(
    discretisation_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    explicit_flux_density: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'disc_mom(k)*cur_prof(k) + flux(k+1) - flux(k)', left to right as the Fortran writes it."""
    return (
        discretisation_momentum * current_profile
        + explicit_flux_density(Koff[1])
        - explicit_flux_density
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diffusion_right_hand_side(
    discretisation_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    explicit_flux_density: fa.CellKField[wpfloat],
    right_hand_side: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Assemble the right-hand side on the diffused main levels.

    The vertical range is the Fortran 'k = k_tp+1 .. k_sf-1' with 'k_tp = 0' and 'k_sf = ke1',
    i.e. 'vertical_start = 0, vertical_end = nlev'.

    Args:
        discretisation_momentum: 'disc_mom' [kg/m2/s].
        current_profile: 'cur_prof', the variable profile the system is built around.
        explicit_flux_density: The explicit flux density, positive upward, from
            'compute_explicit_flux_density', including its zero at the model top; read one level
            down as well, so it must extend to row 'nlev'.
        right_hand_side: Output. Must not be the same field as 'explicit_flux_density'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused main level; 0.
        vertical_end: End of the diffused main levels; 'nlev'.
    """
    _compute_diffusion_right_hand_side(
        discretisation_momentum=discretisation_momentum,
        current_profile=current_profile,
        explicit_flux_density=explicit_flux_density,
        out=right_hand_side,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
