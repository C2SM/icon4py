# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_horizontal_diffusion_and_update_scalar(
    scalar: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Apply the horizontal turbulent diffusion tendency and update a cell scalar.

    Port of the flux divergence and update loops of the conservative horizontal
    diffusion in 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        hori_tend = (sum_{e in C2E} nabla2_flux(e) * geofac_div(e)) / rho
        tend      = tend + hori_tend
        new       = scalar + tend * dtime

    ``tend`` holds the vertical diffusion tendency on entry (written by the
    vertical diffusion solver) and the total (vertical + horizontal) tendency
    on exit.

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        scalar: diffused cell scalar at full levels (old state)
        nabla2_flux: horizontal turbulent diffusion flux at full-level edges
        geofac_div: geometric factors of the cell-centered edge-flux divergence
        rho: air density at full levels [kg/m^3]
        tend: vertical diffusion tendency of the scalar at full levels
        dtime: time step [s]

    Returns:
        (updated scalar, total diffusion tendency) at full levels
    """
    hori_tend = neighbor_sum(nabla2_flux(C2E) * geofac_div, axis=C2EDim) / rho
    new_tend = tend + hori_tend
    new_scalar = scalar + new_tend * dtime
    return new_scalar, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_horizontal_diffusion_and_update_scalar(
    scalar: fa.CellKField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rho: fa.CellKField[wpfloat],
    new_scalar: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_horizontal_diffusion_and_update_scalar(
        scalar=scalar,
        nabla2_flux=nabla2_flux,
        geofac_div=geofac_div,
        rho=rho,
        tend=tend,
        dtime=dtime,
        out=(new_scalar, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
