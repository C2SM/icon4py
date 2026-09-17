# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_brunt_vaisala_frequency(
    theta_v: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    grav: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    """
    Compute the squared Brunt-Vaisala frequency at half-level cell centers.

    Port of ``brunt_vaisala_freq`` in ICON's ``mo_nh_vert_interp_les.f90``:

        bruvais(k) = grav * (theta_v(k-1) - theta_v(k)) * inv_ddqz_z_half(k)
                     / theta_v_ic(k)

    (Fortran jk = 2..nlev, 1-based -> k = 1..nlev-1, 0-based; ``k-1`` is the
    full level above ``k``, so ``bruvais`` is positive for stable
    stratification), with theta_v interpolated to the half levels as in
    ``vert_intp_full2half_cell_3d`` (interior rows only):

        theta_v_ic(k) = wgtfac_c(k) * theta_v(k)
                        + (1 - wgtfac_c(k)) * theta_v(k-1)

    The top and bottom half levels (k = 0 and k = nlev) are not computed; run
    the program with ``vertical_start = 1``, ``vertical_end = nlev`` so that
    the half-level shifts stay in bounds.

    Args:
        theta_v: virtual potential temperature at full levels [K] (nlev levels)
        wgtfac_c: interpolation weight on half levels
        inv_ddqz_z_half: inverse layer thickness centered at half levels [1/m]
        grav: gravitational acceleration [m/s2]

    Returns:
        squared Brunt-Vaisala frequency at half levels [1/s2]
    """
    theta_v_ic = _interpolate_cell_field_to_half_levels_wp(wgtfac_c=wgtfac_c, interpolant=theta_v)
    return (
        grav
        * (theta_v(dims.KHalfDim - 0.5) - theta_v(dims.KHalfDim + 0.5))
        * inv_ddqz_z_half
        / theta_v_ic
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_brunt_vaisala_frequency(  # noqa: PLR0917 [too-many-positional-arguments]
    theta_v: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    bruvais: fa.CellKHalfField[wpfloat],
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_brunt_vaisala_frequency(
        theta_v=theta_v,
        wgtfac_c=wgtfac_c,
        inv_ddqz_z_half=inv_ddqz_z_half,
        grav=grav,
        out=bruvais,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
