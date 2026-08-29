# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion depth between adjacent main levels.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2449-2453 at icon commit 26d6b98cce), reached from SUBROUTINE 'vertdiff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    diff_dep(i,k) = 0.5*(expl_mom(i,k-1) + expl_mom(i,k))

with 'expl_mom' holding the layer depth 'hhl(k) - hhl(k+1)' at that point. So this is the
distance between the centres of the two layers a flux level separates: the denominator of the
finite-difference gradient the diffusion momentum is built from.

Like the discretisation momentum, it depends on the grid alone and 'vert_grad_diff' computes
it once, under 'linisetup'. Its surface row is NOT part of that: 'diff_dep(:,k_sf)' is
'tkv/tsv', a per-variable-type transfer depth, and belongs to
'compute_surface_diffusion_momentum_and_depth'.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_depth(half_level_height: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """'0.5*(dz(k-1) + dz(k))' [m], with 'dz(k) = hhl(k) - hhl(k+1)'.

    The two depths are formed as differences of half-level heights and added before the halving,
    which is the Fortran's grouping; '0.5*dz(k-1) + 0.5*dz(k)' would round twice.
    """
    depth_above = half_level_height(Koff[-1]) - half_level_height
    depth_here = half_level_height - half_level_height(Koff[1])
    return wpfloat("0.5") * (depth_above + depth_here)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diffusion_depth(
    half_level_height: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'diff_dep' on the interior flux levels.

    The vertical range is the Fortran 'DO k = k_hi+1, k_lw' with 'k_hi = 1' and 'k_lw = ke',
    i.e. 'vertical_start = 1, vertical_end = nlev'. Row 0 has no level above it and is not a
    flux level; the surface row is written by another program.

    Args:
        half_level_height: 'hhl' [m]; read one level up and one level down.
        diffusion_depth: Output, 'diff_dep' [m].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost interior flux level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the interior flux levels; 'nlev', mirroring Fortran 'k_lw = ke'.
    """
    _compute_diffusion_depth(
        half_level_height=half_level_height,
        out=diffusion_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
