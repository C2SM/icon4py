# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit part of the diffusion momentum of the vertical diffusion of the model variables.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2764-2776 at icon commit 26d6b98cce), the branch under "use
precalculated implicit weights" -- the branch ICON takes, since 'ldynimp' defaults to
'.FALSE.' (mo_turbdiff_config.f90:101) and no configuration under 'icon/run/' sets it.
'vertdiff' reaches it through 'vert_grad_diff' (turb_vertdiff.f90:747-763). The scientific
commentary in those files is by Matthias Raschendorfer (DWD).

    impl_mom(i,k) = expl_mom(i,k)*tdc%impl_weight(k)

'compute_implicit_part_of_tke_diffusion_momentum' is the same statement, reached from
'turbdiff' section 9). The two are separate programs because that one is named, documented and
range-checked for the TKE diffusion, whose flux levels are the main levels and whose top level
is one row lower. Merging them under a scheme-neutral name would be an improvement to both;
it is not made here because it would edit a stencil section 9)'s datatest imports.

THE ROW RANGE DEPENDS ON THE LOWER BOUNDARY CONDITION. The Fortran runs
'DO k = k_tp+2, k_sf+1-m' with 'm = 1' for a surface-CONCENTRATION condition and 'm = 2' for a
surface-FLUX condition. 'vertdiff' uses the concentration condition for the two wind components
("no lower flux condition for momentum!", turb_vertdiff.f90:586) and 'tdc%lsflcnd' for the
scalars, which defaults to '.TRUE.'. So the momentum type gets an implicit momentum on the
surface flux level and the scalar type does not: with a flux condition the surface concentration
is not an unknown, so there is no sub-diagonal to it. The row is left holding whatever the
previous type put there, and nothing reads it.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k)*impl_weight(k)' [kg/m2/s]."""
    return diffusion_momentum * implicit_weight


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'impl_mom' on the flux levels.

    Args:
        diffusion_momentum: 'expl_mom' on entry to 'prep_impl_vert_diff', the FULL diffusion
            momentum [kg/m2/s], from 'compute_diffusion_momentum' and
            'compute_surface_diffusion_momentum_and_depth'.
        implicit_weight: 'tdc%impl_weight' [-], the fixed implicit weight of each flux level.
            Not a field of this scheme: ICON builds it once at model initialisation
            (mo_nwp_phy_init.f90:1541-1547) and never changes it.
        implicit_diffusion_momentum: Output, 'impl_mom' [kg/m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost interior flux level; 1, mirroring Fortran 'k = k_tp+2 = 2'.
        vertical_end: Fortran 'k_sf+1-m', i.e. 'nlev + 1' for the momentum type and 'nlev' for
            the scalar type. See the module docstring.
    """
    _compute_implicit_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_weight=implicit_weight,
        out=implicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
