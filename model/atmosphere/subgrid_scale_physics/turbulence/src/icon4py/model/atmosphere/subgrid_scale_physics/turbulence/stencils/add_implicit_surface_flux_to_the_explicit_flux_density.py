# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit closure of a surface-CONCENTRATION lower boundary condition.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff' (:2961-2973 at icon commit 26d6b98cce), the block under
'IF (.NOT.lsflucond)', reached from 'vertdiff' through 'vert_grad_diff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    eff_flux(i,k_sf) = eff_flux(i,k_sf) + impl_mom(i,k_sf)*rhs_prof(i,k_sf-1)

WHO RUNS THIS. Only the momentum type. 'vertdiff' sets 'lsflucond = .FALSE.' for the two wind
components with the comment "Attention: no lower flux condition for momentum!"
(turb_vertdiff.f90:585-586) and 'lsflucond = tdc%lsflcnd' for the scalars, which defaults to
'.TRUE.' (mo_turbdiff_config.f90:280). With a concentration condition the lowest main level is
coupled to a prescribed surface value, and this term is that coupling moved to the right-hand
side; the Fortran's note puts it as "in all, level 'k_sf-1' is treated (semi-)implicitly by
this".

At the surface row 'impl_mom' still carries no preconditioning scaling, and under a flux
condition it is not written for the type at all -- which is the other half of the reason this
program exists rather than a branch inside 'compute_explicit_flux_density'.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _add_implicit_surface_flux_to_the_explicit_flux_density(
    explicit_flux_density: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'eff_flux(ke1) + impl_mom(ke1)*cur_prof(ke)'."""
    return explicit_flux_density + implicit_diffusion_momentum * current_profile(Koff[-1])


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_implicit_surface_flux_to_the_explicit_flux_density(
    explicit_flux_density_at_the_surface: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    explicit_flux_density: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Add the implicit surface coupling to the surface row of the flux density.

    Args:
        explicit_flux_density_at_the_surface: The surface row as
            'compute_explicit_flux_density' left it. ICON updates in place and a caller may
            pass the same field here and below; the statement is pointwise on this row.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on the surface flux level, which only
            the momentum type has.
        current_profile: 'cur_prof', read one level up, i.e. on the lowest main level.
        explicit_flux_density: Output; only the surface row is written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The surface flux level, 'nlev' (Fortran 'k_sf = ke1').
        vertical_end: 'nlev + 1'.
    """
    _add_implicit_surface_flux_to_the_explicit_flux_density(
        explicit_flux_density=explicit_flux_density_at_the_surface,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        current_profile=current_profile,
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
