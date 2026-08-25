# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_flux_rhs(
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    prefac: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Set the surface flux right-hand side of the vertical diffusion solve.

    Port of the right-hand-side rows of 'Compute_diffusion_hydrometeors' and
    'Compute_diffusion_temperature' (mo_vdf.f90):

        rhs(nlev) = - sfc_flx * prefac * inv_mair(nlev)

    with ``prefac = 1`` for the hydrometeors and ``prefac = zfactor``
    (``scale_turb_energy_flux`` if enabled, else 1) for the energy. All other
    rows of ``rhs`` are zero: the Fortran zero-initializes ``rhs`` and only
    writes the bottom row and the top row ``rhs(1) = + top_flx * inv_mair(1)``,
    where ``top_flx`` is always zero in tmx.

    This program is meant to be run on the single bottom K row (program domain
    ``KDim: (nlev - 1, nlev)``, so ``inv_air_mass`` is read at the bottom full
    level without any vertical offset); the caller must keep the other rows of
    ``rhs`` zero (zero-allocate and never write them elsewhere).

    Args:
        sfc_flx: grid-mean surface flux of the diffused quantity (2D cell field)
        inv_air_mass: inverse air mass per unit area at full levels [m^2/kg]
        prefac: scaling factor of the turbulent flux

    Returns:
        right-hand side of the vertical diffusion solve at the bottom full level
    """
    return wpfloat("0.0") - sfc_flx * prefac * inv_air_mass


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_flux_rhs(
    sfc_flx: fa.CellField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    prefac: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_surface_flux_rhs(
        sfc_flx=sfc_flx,
        inv_air_mass=inv_air_mass,
        prefac=prefac,
        out=rhs,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
