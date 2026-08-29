# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit part of the diffusion momentum of the vertical diffusion of the model variables.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2778-2786 at icon commit 26d6b98cce), reached from 'vertdiff' through
'vert_grad_diff' (turb_vertdiff.f90:747-763). The scientific commentary in those files is by
Matthias Raschendorfer (DWD).

    expl_mom(i,k) = expl_mom(i,k) - impl_mom(i,k)

An in-place update, and the Fortran's closing note is the whole point of the row range:
"Notice that 'expl_mom' still contains the whole diffusion momentum at level 'k_sf'!". The
surface flux level keeps the full momentum because 'calc_impl_vert_diff' uses it to form the
explicit surface flux, which is the lower boundary condition of the system.

'subtract_implicit_part_of_tke_diffusion_momentum' is the same statement for the TKE; see the
note in 'compute_implicit_diffusion_momentum' on why the two are not one program.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _subtract_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k) - impl_mom(k)' [kg/m2/s]."""
    return diffusion_momentum - implicit_diffusion_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def subtract_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Reduce the diffusion momentum to its explicit part on the interior flux levels.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf-1' with 'k_tp = 0' and 'k_sf = ke1',
    i.e. 'vertical_start = 1, vertical_end = nlev' -- one row short of the implicit part's
    range for the momentum type, and level with it for the scalar type.

    Args:
        diffusion_momentum: The full 'expl_mom' [kg/m2/s].
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s], from
            'compute_implicit_diffusion_momentum'.
        explicit_diffusion_momentum: Output, the reduced 'expl_mom' [kg/m2/s]. ICON updates in
            place and a caller may pass the same field for this and 'diffusion_momentum'; the
            statement is pointwise, so that is safe.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost interior flux level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the interior flux levels; 'nlev', mirroring Fortran 'k_sf-1 = ke'.
    """
    _subtract_implicit_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=explicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
