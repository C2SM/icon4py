# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Discretisation momentum of the semi-implicit vertical diffusion of the model variables.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2438-2455 at icon commit 26d6b98cce), reached from SUBROUTINE 'vertdiff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    expl_mom(i,k) = hhl(i,k) - hhl(i,k+1)          ! the layer depth, in scratch storage
    disc_mom(i,k) = rho(i,k)*expl_mom(i,k)*fr_var

'disc_mom' is the mass per unit area of a model layer divided by the time step [kg/m2/s]: the
diagonal of the tridiagonal system before any diffusion momentum is added to it. It is a
function of the grid and of the density alone, so 'vert_grad_diff' computes it once -- under
'linisetup', for the first variable of the first type -- and every later variable and type
reuses it.

TWO ROUNDINGS THAT ARE NOT NEGOTIABLE. 'fr_var' is '1/dt_var', formed once and MULTIPLIED by;
'rho*dz/dt_var' is a different number. And the product associates to the left,
'(rho*dz)*fr_var'. Both were measured: getting either wrong puts 'disc_mom' one ulp off on
250707 of 662080 values, and the error survives into every tendency the scheme produces.

The layer depth itself is not a field here. The Fortran parks it in the 'expl_mom' storage,
which is why 'expl_mom(:,1)' still holds the top layer's depth at the exit savepoint -- the
one row the diffusion momentum never overwrites. That is an artefact of storage reuse and this
port does not reproduce it.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_discretisation_momentum(
    air_density: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'disc_mom(k) = rho(k)*(hhl(k) - hhl(k+1))*fr_var' [kg/m2/s]."""
    layer_depth = half_level_height - half_level_height(Koff[1])
    return air_density * layer_depth * reciprocal_time_step


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_discretisation_momentum(
    air_density: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    reciprocal_time_step: wpfloat,
    discretisation_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'disc_mom' on the main levels.

    The vertical range is the Fortran 'disc_mom(:,k_hi)' statement followed by
    'DO k = k_hi+1, k_lw' with 'k_hi = k_tp+1 = 1' and 'k_lw = k_sf-1 = ke': one range, split
    in the Fortran only because the loop above it starts one level lower. Here it is
    'vertical_start = 0, vertical_end = nlev'.

    Args:
        air_density: 'rho' = 'rhoh', the air density on the main levels [kg/m3].
        half_level_height: 'hhl' [m]; read one level down, so it must extend to row 'nlev'.
        reciprocal_time_step: 'fr_var = 1/dt_var' [1/s]. A scalar, and formed by the caller so
            that the division happens exactly once for the whole scheme.
        discretisation_momentum: Output, 'disc_mom' [kg/m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost main level; 0, mirroring Fortran 'k_hi = 1'.
        vertical_end: End of the main levels; 'nlev', mirroring Fortran 'k_lw = ke'.
    """
    _compute_discretisation_momentum(
        air_density=air_density,
        half_level_height=half_level_height,
        reciprocal_time_step=reciprocal_time_step,
        out=discretisation_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
