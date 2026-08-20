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
def _compute_static_energy(
    temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    spec_heat: wpfloat,
    grav: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the dry static energy at full-level cell centers.

    Port of ``compute_static_energy`` in ICON's ``mo_vdf_atmo.f90``:

        static_energy = spec_heat * temperature + grav * height_above_ground

    ``height_above_ground`` is the geometric height of the full levels above the
    surface (``ghf`` in the Fortran code, computed at init time by the granule's
    ``init_height_above_ground`` program), so
    ``grav * height_above_ground`` is the geopotential above ground. At the tmx
    call sites ``spec_heat`` is the specific heat of dry air at constant
    pressure, ``cpd``, giving ``ctgz = cpd * T + g * ghf``.

    The Fortran subroutine loops over the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the horizontal
    domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``, and over all full
    levels (Fortran jk = 1..nlev -> k = 0..nlev-1).

    Args:
        temperature: air temperature at full levels [K]
        height_above_ground: height of the full levels above the surface [m]
        spec_heat: specific heat [J/K/kg] (``cpd`` at the tmx call sites)
        grav: gravitational acceleration [m/s2]

    Returns:
        static energy at full levels [J/kg]
    """
    return spec_heat * temperature + grav * height_above_ground


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_static_energy(
    temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    static_energy: fa.CellKField[wpfloat],
    spec_heat: wpfloat,
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_static_energy(
        temperature=temperature,
        height_above_ground=height_above_ground,
        spec_heat=spec_heat,
        grav=grav,
        out=static_energy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
