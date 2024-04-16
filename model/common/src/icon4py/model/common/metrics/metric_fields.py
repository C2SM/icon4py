# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next import (
    Field,
    GridType,
    broadcast,
    exp,
    field_operator,
    int32,
    maximum,
    program,
    sin,
    tanh,
    where,
)

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.math.helpers import (
    average_k_level_up,
    difference_k_level_down,
    difference_k_level_up,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


"""
Contains metric fields calculations for the vertical grid, ported from mo_vertical_grid.f90.
"""


@program(grid_type=GridType.UNSTRUCTURED)
def compute_z_mc(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute the geometric height of full levels from the geometric height of half levels (z_ifc).

    This assumes that the input field z_ifc is defined on half levels (KHalfDim) and the
    returned fields is defined on full levels (KDim)

    Args:
        z_ifc: Field[[CellDim, KDim], wpfloat] geometric height on half levels
        z_mc: Field[[CellDim, KDim], wpfloat] output, geometric height defined on full levels
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain

    """
    average_k_level_up(
        z_ifc,
        out=z_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_ddqz_z_half(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
) -> Field[[CellDim, KDim], wpfloat]:
    ddqz_z_half = where(
        (k > int32(0)) & (k < nlev),
        difference_k_level_down(z_mc),
        where(k == 0, 2.0 * (z_ifc - z_mc), 2.0 * (z_mc(Koff[-1]) - z_ifc)),
    )
    return ddqz_z_half


@program(grid_type=GridType.UNSTRUCTURED, backend=None)
def compute_ddqz_z_half(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    ddqz_z_half: Field[[CellDim, KDim], wpfloat],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute functional determinant of the metrics (is positive) on half levels.

    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        z_mc: geometric height on full levels
        k: vertical dimension index
        nlev: total number of levels
        ddqz_z_half: (output)
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_ddqz_z_half(
        z_ifc,
        z_mc,
        k,
        nlev,
        out=ddqz_z_half,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_ddqz_z_full(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    ddqz_z_full = difference_k_level_up(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ddqz_z_full(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute ddqz_z_full and its inverse inv_ddqz_z_full.

    Functional determinant of the metrics (is positive) on full levels and inverse inverse layer thickness(for runtime optimization).
    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        ddqz_z_full: (output)
        inv_ddqz_z_full: (output)
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index

    """
    _compute_ddqz_z_full(
        z_ifc,
        out=(ddqz_z_full, inv_ddqz_z_full),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_scalfac_dd3d(
    vct_a: Field[[KDim], wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: int32,
) -> Field[[KDim], wpfloat]:
    scalfac_dd3d = broadcast(1.0, (KDim,))
    if divdamp_type == 32:
        zf = 0.5 * (vct_a + vct_a(Koff[1]))
        scalfac_dd3d = where(zf >= divdamp_trans_end, 0.0, scalfac_dd3d)
        scalfac_dd3d = where(
            zf >= divdamp_trans_start,
            (divdamp_trans_end - zf) / (divdamp_trans_end - divdamp_trans_start),
            scalfac_dd3d,
        )
    return scalfac_dd3d


@program
def compute_scalfac_dd3d(
    vct_a: Field[[KDim], wpfloat],
    scalfac_dd3d: Field[[KDim], wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute scaling factor for 3D divergence damping terms.

    See mo_vertical_grid.f90

    Args:
        vct_a: Field[[KDim], float],
        scalfac_dd3d: (output)
        divdamp_trans_start: lower bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_trans_end: upper bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_type: type of divergence damping (2D or 3D divergence)
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_scalfac_dd3d(
        vct_a,
        divdamp_trans_start,
        divdamp_trans_end,
        divdamp_type,
        out=scalfac_dd3d,
        domain={KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_rayleigh_w(
    vct_a: Field[[KDim], wpfloat],
    vct_a_1: Field[[], wpfloat],
    damping_height: wpfloat,
    rayleigh_type: int32,
    rayleigh_classic: int32,
    rayleigh_klemp: int32,
    rayleigh_coeff: wpfloat,
    pi_const: wpfloat,
) -> Field[[KDim], wpfloat]:
    rayleigh_w = broadcast(0.0, (KDim,))
    z_sin_diff = maximum(0.0, vct_a - damping_height)
    z_tanh_diff = vct_a_1 - vct_a  # vct_a(1) - vct_a
    if rayleigh_type == rayleigh_classic:
        rayleigh_w = (
            rayleigh_coeff
            * sin(pi_const / 2.0 * z_sin_diff / maximum(0.001, vct_a_1 - damping_height)) ** 2
        )
    elif rayleigh_type == rayleigh_klemp:
        rayleigh_w = rayleigh_coeff * (
            1.0 - tanh(3.8 * z_tanh_diff / maximum(0.000001, vct_a_1 - damping_height))
        )
    return rayleigh_w


@program
def compute_rayleigh_w(
    rayleigh_w: Field[[KDim], wpfloat],
    vct_a: Field[[KDim], wpfloat],
    vct_a_1: Field[[], wpfloat],
    damping_height: wpfloat,
    rayleigh_type: int32,
    rayleigh_classic: int32,
    rayleigh_klemp: int32,
    rayleigh_coeff: wpfloat,
    pi_const: wpfloat,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_rayleigh_w(
        vct_a,
        vct_a_1,
        damping_height,
        rayleigh_type,
        rayleigh_classic,
        rayleigh_klemp,
        rayleigh_coeff,
        pi_const,
        out=rayleigh_w,
        domain={KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_coeff_dwdz(
    ddqz_z_full: Field[[CellDim, KDim], float], z_ifc: Field[[CellDim, KDim], wpfloat]
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    coeff1_dwdz = ddqz_z_full / ddqz_z_full(Koff[-1]) / (z_ifc(Koff[-1]) - z_ifc(Koff[1]))
    coeff2_dwdz = ddqz_z_full(Koff[-1]) / ddqz_z_full / (z_ifc(Koff[-1]) - z_ifc(Koff[1]))

    return coeff1_dwdz, coeff2_dwdz


@program(grid_type=GridType.UNSTRUCTURED)
def compute_coeff_dwdz(
    ddqz_z_full: Field[[CellDim, KDim], float],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_coeff_dwdz(
        ddqz_z_full,
        z_ifc,
        out=(coeff1_dwdz, coeff2_dwdz),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_d2dexdz2_fac_mc(
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], vpfloat],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    igradp_method: int32,
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    if igradp_method <= int32(3):
        d2dexdz2_fac1_mc = -grav / (cpd * theta_ref_mc**2) * inv_ddqz_z_full
        d2dexdz2_fac2_mc = (
            2.0
            * grav
            / (cpd * theta_ref_mc**3)
            * (grav / cpd - del_t_bg / h_scal_bg * exp(-z_mc / h_scal_bg))
            / exner_ref_mc
        )

    return d2dexdz2_fac1_mc, d2dexdz2_fac2_mc


@program(grid_type=GridType.UNSTRUCTURED)
def compute_d2dexdz2_fac_mc(
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], vpfloat],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    igradp_method: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_d2dexdz2_fac_mc(
        theta_ref_mc,
        inv_ddqz_z_full,
        exner_ref_mc,
        z_mc,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        cpd,
        grav,
        del_t_bg,
        h_scal_bg,
        igradp_method,
        out=(d2dexdz2_fac1_mc, d2dexdz2_fac2_mc),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
