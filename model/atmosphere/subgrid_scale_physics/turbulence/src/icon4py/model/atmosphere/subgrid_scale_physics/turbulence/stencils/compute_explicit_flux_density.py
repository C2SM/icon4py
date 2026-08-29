# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit part of the vertical flux density of a diffused variable.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff' (:2951-2959 at icon commit 26d6b98cce), reached from 'vertdiff' through
'vert_grad_diff' (turb_vertdiff.f90:747-763). The scientific commentary in those files is by
Matthias Raschendorfer (DWD).

    eff_flux(i,k) = expl_mom(i,k)*( rhs_prof(i,k) - rhs_prof(i,k-1) )

The flux across a flux level, positive UPWARD, carried by the explicit part of the diffusion
momentum. It is the first of three roles the 'eff_flux' storage plays in one call -- explicit
flux, then right-hand side, then (under 'leff_flux') the effective flux density -- which is why
it is a distinct field here.

'rhs_prof' is 'cur_prof' at 'itndcon = 0', which is what the NWP interface passes; the
'old_prof'/'rhs_prof' selection at :2925-2939 collapses to that.

THE SURFACE ROW STILL USES THE FULL DIFFUSION MOMENTUM, as the Fortran notes right below the
loop: 'subtract_implicit_diffusion_momentum' stops one row short of it. That is what makes this
row the explicit surface flux and not merely a term of it.

THE MODEL-TOP ROW IS WRITTEN AS AN EXPLICIT ZERO, which the Fortran does not do. Its loop
starts at 'k_tp+2' and the right-hand side above it then simply omits the outgoing-flux term,
with the comment "Note: Zero flux condition just below top level". Stating the boundary
condition as a value rather than as a missing term makes 'compute_diffusion_right_hand_side'
one expression instead of two, so it needs no row selection, keeps the embedded backend, and
has no 'concat_where' for the DaCe passes to fold. It is bit-exact and not merely close: 'x -
0.0' is 'x' for every double, negative zero included.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_explicit_flux_density(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k)*(cur_prof(k) - cur_prof(k-1))'."""
    return explicit_diffusion_momentum * (current_profile - current_profile(Koff[-1]))


@gtx.field_operator
def _no_flux_through_the_model_top() -> fa.CellKField[wpfloat]:
    """The upper boundary condition of the diffusion equation: no flux above the top level."""
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_flux_density(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    model_top_level: gtx.int32,
    explicit_flux_density: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the explicit flux density on every flux level, both boundaries included.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf' with 'k_tp = 0' and 'k_sf = ke1',
    i.e. 'vertical_start = 1, vertical_end = nlev + 1', plus the zero at 'model_top_level'.

    Args:
        explicit_diffusion_momentum: 'expl_mom' [kg/m2/s], reduced by the implicit part on the
            interior levels and full on the surface row.
        current_profile: 'cur_prof', the variable profile including its lower boundary value;
            read one level up as well.
        model_top_level: The row above the uppermost flux level, Fortran 'k_tp+1' as a
            zero-based index, which is 0. It is a separate argument rather than
            'vertical_start - 1' so that the domain carries no arithmetic.
        explicit_flux_density: Output, the explicit flux density, positive upward.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost flux level; 1, mirroring Fortran 'k = 2'.
        vertical_end: 'nlev + 1', mirroring Fortran 'k_sf = ke1'.
    """
    _no_flux_through_the_model_top(
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (model_top_level, vertical_start),
        },
    )
    _compute_explicit_flux_density(
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        current_profile=current_profile,
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
