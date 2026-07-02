# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_scalar_nabla2_flux(
    scalar: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    rturb_prandtl: wpfloat,
    prefac: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    Compute the horizontal turbulent diffusion flux of a cell scalar at edges.

    Port of the ``nabla2_e`` loops ("compute kh_ie * grad_horiz(state)") of the
    conservative horizontal diffusion in 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        nabla2_e(k) = 0.5 * prefac * rturb_prandtl * (km_ie(k) + km_ie(k+1))
                      * inv_dual_edge_length
                      * (scalar(E2C[1]) - scalar(E2C[0]))

    ``0.5 * rturb_prandtl * (km_ie(k) + km_ie(k+1))`` is the turbulent
    diffusivity ``kh`` averaged from the adjacent half levels to the full level
    ``k``, and ``inv_dual_edge_length * (scalar(E2C[1]) - scalar(E2C[0]))`` is
    the horizontal gradient normal to the edge. ``prefac = 1`` for the
    hydrometeors and ``prefac = zfactor`` (``scale_turb_energy_flux`` if
    enabled, else 1) for the energy.

    Domains (Fortran): jk = 1..nlev; edges from ``rl_start = grf_bdywidth_e``
    -> ``h_grid.Zone.NUDGING`` to ``rl_end = min_rledge_int - 1`` ->
    ``h_grid.Zone.HALO`` (halo edges are computed on purpose, the flux
    divergence is taken on halo-adjacent cells afterwards).

    Args:
        scalar: diffused cell scalar at full levels (halo values must be valid)
        km_ie: turbulent viscosity at half-level edges [m^2/s] (nlev + 1 rows)
        inv_dual_edge_length: inverse dual edge length [1/m]
        rturb_prandtl: reciprocal turbulent Prandtl number
        prefac: scaling factor of the turbulent flux

    Returns:
        horizontal turbulent diffusion flux at full-level edges
    """
    return (
        wpfloat("0.5")
        * prefac
        * rturb_prandtl
        * (km_ie + km_ie(KDim + 1))
        * inv_dual_edge_length
        * (scalar(E2C[1]) - scalar(E2C[0]))
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_scalar_nabla2_flux(
    scalar: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    nabla2_flux: fa.EdgeKField[wpfloat],
    rturb_prandtl: wpfloat,
    prefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_scalar_nabla2_flux(
        scalar=scalar,
        km_ie=km_ie,
        inv_dual_edge_length=inv_dual_edge_length,
        rturb_prandtl=rturb_prandtl,
        prefac=prefac,
        out=nabla2_flux,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
