# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion momentum and diffusion depth at the surface flux level.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2471-2478 at icon commit 26d6b98cce), reached from SUBROUTINE 'vertdiff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    expl_mom(i,k_sf) = rhon(i,k_sf)*tsv(i)
    diff_dep(i,k_sf) = tkv(i,k_sf)/tsv(i)

One Fortran loop, one program. The surface layer is not a finite difference between two model
levels: the transfer scheme has already reduced it to a transfer velocity 'tsv' -- 'tvm' for
momentum, 'tvh' for heat and moisture -- so the momentum is 'rho*tsv' directly and the depth
that goes with it is 'K/tsv', the depth at which that velocity reproduces the diffusion
coefficient. The Fortran's own note ("Einfuehrung von 'tsv': macht Unterschiede", "introducing
'tsv' makes a difference") records that this replaced the earlier 'rho*K/dzs' form.

BOTH ROWS ARE PER TYPE, and both are recomputed under 'lnewvtype'. The comment at :2479
("'tkmin' should be excluded for the surface level") is about the disabled floor in
'compute_diffusion_momentum' and has no effect here.

The Fortran's 'IF (tdc%lfreeslip) expl_mom(i,k_sf) = 0' (:2492-2499) is not ported: 'lfreeslip'
defaults to '.FALSE.' (mo_turbdiff_config.f90:284) and is documented "use for idealized runs
only!".
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_diffusion_momentum(
    air_density: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rhon(k_sf)*tsv' [kg/m2/s]."""
    return air_density * broadcast(surface_transfer_velocity, (dims.CellDim, dims.KDim))


@gtx.field_operator
def _compute_surface_diffusion_depth(
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'tkv(k_sf)/tsv' [m]."""
    return diffusion_coefficient / broadcast(surface_transfer_velocity, (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_diffusion_momentum_and_depth(
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
    diffusion_momentum: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Write the surface row of 'expl_mom' and of 'diff_dep'.

    Args:
        air_density: 'rhon' on half levels [kg/m3]; only the surface row is read.
        diffusion_coefficient: 'tkv' [m2/s]; only the surface row is read.
        surface_transfer_velocity: 'tsv' [m/s], 'tvm' for momentum and 'tvh' for scalars.
        diffusion_momentum: In-out, 'expl_mom' [kg/m2/s]; only the surface row is written, and
            it keeps the FULL momentum -- 'subtract_implicit_diffusion_momentum' stops one row
            short of it on purpose.
        diffusion_depth: In-out, 'diff_dep' [m]; only the surface row is written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The surface flux level, 'nlev' (Fortran 'k_sf = ke1').
        vertical_end: 'nlev + 1'.
    """
    _compute_surface_diffusion_momentum(
        air_density=air_density,
        surface_transfer_velocity=surface_transfer_velocity,
        out=diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_surface_diffusion_depth(
        diffusion_coefficient=diffusion_coefficient,
        surface_transfer_velocity=surface_transfer_velocity,
        out=diffusion_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
