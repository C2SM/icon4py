# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion momentum of the vertical diffusion of one variable type.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2461-2469 at icon commit 26d6b98cce), reached from SUBROUTINE 'vertdiff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    expl_mom(i,k) = tkv(i,k)*rhon(i,k)/diff_dep(i,k)

'rho*K/dz' [kg/m2/s]: the off-diagonal of the tridiagonal system, the mass flux per unit
concentration difference across a flux level. Unlike the discretisation momentum and the
interior diffusion depth, this depends on the variable TYPE through 'tkv' -- 'tkvm' for the two
wind components and 'tkvh' for every scalar -- so 'vert_grad_diff' recomputes it under
'lnewvtype', once per type and not once per variable.

The commented-out 'MAX(tkmin, tkv(i,k))' at :2464 is not ported; ICON has it disabled with the
note "Eventuell tkmin-Beschraenkung nur bei VDiff" ("possibly restrict by 'tkmin' only in the
vertical diffusion").
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_momentum(
    diffusion_coefficient: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'tkv*rhon/diff_dep' [kg/m2/s], the product formed before the quotient as in the Fortran."""
    return diffusion_coefficient * air_density / diffusion_depth


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diffusion_momentum(
    diffusion_coefficient: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'expl_mom' on the interior flux levels, before it is split.

    The vertical range is the Fortran 'DO k = k_hi+1, k_lw', i.e.
    'vertical_start = 1, vertical_end = nlev'. The surface flux level has its own expression
    and its own program.

    Args:
        diffusion_coefficient: 'tkv' [m2/s] on the flux levels: 'tkvm' for the momentum type,
            'tkvh' for the scalar type.
        air_density: 'rhon' on half levels [kg/m3], with its surface row already rescaled by
            'compute_surface_air_density_and_exner_factor'.
        diffusion_depth: 'diff_dep' [m], from 'compute_diffusion_depth'.
        diffusion_momentum: Output, 'expl_mom' before the implicit part is subtracted [kg/m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost interior flux level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the interior flux levels; 'nlev', mirroring Fortran 'k_lw = ke'.
    """
    _compute_diffusion_momentum(
        diffusion_coefficient=diffusion_coefficient,
        air_density=air_density,
        diffusion_depth=diffusion_depth,
        out=diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
