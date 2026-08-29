# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower boundary value of a profile whose surface condition is a flux density.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2493-2503 at icon commit 26d6b98cce), the block under 'IF (lsfgrduse)', reached from
'vertdiff' with 'lsfgrduse = lsfli(n)' (turb_vertdiff.f90:757). The scientific commentary in
those files is by Matthias Raschendorfer (DWD).

    cur_prof(i,k_sf) = cur_prof(i,k_sf-1) - diff_dep(i,k_sf)*eff_flux(i,k_sf)

The surface concentration is reconstructed from the lowest main level by walking down the
effective gradient over the transfer depth 'diff_dep(k_sf) = tkv(k_sf)/tsv'. It is the
counterpart of 'compute_surface_gradients_from_flux_densities', which produced that gradient
from the prescribed flux.

THIS OVERWRITES WHAT THE PROFILE PROGRAM PUT THERE. In the Fortran, 'cur_prof(ke1)' has already
been set to the surface value ('shfl_s' divided by 'eprs' for temperature, 'qvfl_s' for water
vapour) a few lines earlier, and this block replaces it unconditionally. The earlier assignment
is dead under 'lsfluse' and this port does not make it.

The Fortran's own note explains why this is not simply a flux boundary condition: with
'lsflucond = .FALSE.' the solve treats this reconstructed concentration implicitly, so the
surface flux it finally realises may deviate from the prescribed one. For the scalars
'vertdiff' runs with 'tdc%lsflcnd = .TRUE.' and no such deviation develops.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_profile_value_from_flux_gradient(
    current_profile: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    surface_gradient: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'cur_prof(ke) - diff_dep(ke1)*grad(ke1)'."""
    return current_profile(Koff[-1]) - diffusion_depth * surface_gradient


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_profile_value_from_flux_gradient(
    current_profile_above: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    surface_gradient: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Write the surface row of 'cur_prof' from the effective gradient.

    Args:
        current_profile_above: 'cur_prof' [variable units], read one level up. ICON has one
            array and a caller passes the same field here and below; the row read is never the
            row written.
        diffusion_depth: 'diff_dep' [m]; the surface row, 'tkv(ke1)/tsv', from
            'compute_surface_diffusion_momentum_and_depth'.
        surface_gradient: 'zvari(:,ke1,m)' [variable units per m], from
            'compute_surface_gradients_from_flux_densities'.
        current_profile: Output, 'cur_prof'; only the surface row is written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The surface half level, 'nlev' (Fortran 'k_sf = ke1').
        vertical_end: 'nlev + 1'.
    """
    _compute_surface_profile_value_from_flux_gradient(
        current_profile=current_profile_above,
        diffusion_depth=diffusion_depth,
        surface_gradient=surface_gradient,
        out=current_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
