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

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, Koff, VertexDim
from icon4py.model.common.math.helpers import (
    _grad_fd_tang,
    average_cell_kdim_level_up,
    average_edge_kdim_level_up,
    difference_k_level_down,
    difference_k_level_up,
    grad_fd_norm,
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
    average_cell_kdim_level_up(
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
        ddqz_z_half: (output) functional determinant of the metrics (is positive), half levels
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
        ddqz_z_full: (output) functional determinant of the metrics (is positive), full levels
        inv_ddqz_z_full: (output) inverse layer thickness (for runtime optimization)
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
        zf = 0.5 * (vct_a + vct_a(Koff[1]))  # depends on nshift_total, assumed to be always 0
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
        scalfac_dd3d: (output) scaling factor for 3D divergence damping terms, and start level from which they are > 0
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
    damping_height: wpfloat,
    rayleigh_type: int32,
    rayleigh_classic: int32,
    rayleigh_klemp: int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
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
    damping_height: wpfloat,
    rayleigh_type: int32,
    rayleigh_classic: int32,
    rayleigh_klemp: int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute rayleigh_w factor.

    See mo_vertical_grid.f90

    Args:
        rayleigh_w: (output) Rayleigh damping
        vct_a: Field[[KDim], float]
        vct_a_1: 1D of vct_a
        damping_height: height at which w-damping and sponge layer start
        rayleigh_type: type of Rayleigh damping (1: CLASSIC, 2: Klemp (2008))
        rayleigh_classic: classical Rayleigh damping, which makes use of a reference state.
        rayleigh_klemp: Klemp (2008) type Rayleigh damping
        rayleigh_coeff: Rayleigh damping coefficient in w-equation
        pi_const: pi constant
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_rayleigh_w(
        vct_a,
        damping_height,
        rayleigh_type,
        rayleigh_classic,
        rayleigh_klemp,
        rayleigh_coeff,
        vct_a_1,
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
    """
    Compute coeff1_dwdz and coeff2_dwdz factors.

    See mo_vertical_grid.f90

    Args:
        ddqz_z_full: functional determinant of the metrics (is positive), full levels
        z_ifc: geometric height of half levels
        coeff1_dwdz: coefficient for second-order acurate dw/dz term
        coeff2_dwdz: coefficient for second-order acurate dw/dz term
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_coeff_dwdz(
        ddqz_z_full,
        z_ifc,
        out=(coeff1_dwdz, coeff2_dwdz),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_d2dexdz2_fac1_mc(
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], vpfloat],
    cpd: float,
    grav: wpfloat,
    igradp_method: int32,
) -> Field[[CellDim, KDim], vpfloat]:
    if igradp_method <= int32(3):
        d2dexdz2_fac1_mc = -grav / (cpd * theta_ref_mc**2) * inv_ddqz_z_full

    return d2dexdz2_fac1_mc


@field_operator
def _compute_d2dexdz2_fac2_mc(
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    igradp_method: int32,
) -> Field[[CellDim, KDim], vpfloat]:
    if igradp_method <= int32(3):
        d2dexdz2_fac2_mc = (
            2.0
            * grav
            / (cpd * theta_ref_mc**3)
            * (grav / cpd - del_t_bg / h_scal_bg * exp(-z_mc / h_scal_bg))
            / exner_ref_mc
        )
    return d2dexdz2_fac2_mc


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
    """
    Compute d2dexdz2_fac1_mc and d2dexdz2_fac2_mc factors.

    See mo_vertical_grid.f90

    Args:
        theta_ref_mc: reference Potential temperature, full level mass points
        inv_ddqz_z_full: inverse layer thickness (for runtime optimization)
        exner_ref_mc: reference Exner pressure, full level mass points
        z_mc: geometric height defined on full levels
        d2dexdz2_fac1_mc: (output) first vertical derivative of reference Exner pressure, full level mass points, divided by theta_ref
        d2dexdz2_fac2_mc: (output) vertical derivative of d_exner_dz/theta_ref, full level mass points
        cpd: Specific heat at constant pressure [J/K/kg]
        grav: avergae gravitational acceleratio
        del_t_bg: difference between sea level temperature and asymptotic stratospheric temperature
        h_scal_bg: height scale for reference atmosphere [m]
        igradp_method: method for computing the horizontal presure gradient
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_d2dexdz2_fac1_mc(
        theta_ref_mc,
        inv_ddqz_z_full,
        d2dexdz2_fac1_mc,
        cpd,
        grav,
        igradp_method,
        out=d2dexdz2_fac1_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )

    _compute_d2dexdz2_fac2_mc(
        theta_ref_mc,
        exner_ref_mc,
        z_mc,
        d2dexdz2_fac2_mc,
        cpd,
        grav,
        del_t_bg,
        h_scal_bg,
        igradp_method,
        out=d2dexdz2_fac2_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@program
def compute_ddxn_z_half_e(
    z_ifc: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ddxn_z_half_e: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    grad_fd_norm(
        z_ifc,
        inv_dual_edge_length,
        out=ddxn_z_half_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_ddxt_z_half_e(
    z_ifv: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    ddxt_z_half_e: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _grad_fd_tang(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
        out=ddxt_z_half_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_ddxnt_z_full(
    z_ddxnt_z_half_e: Field[[EdgeDim, KDim], float], ddxn_z_full: Field[[EdgeDim, KDim], float]
):
    average_edge_kdim_level_up(z_ddxnt_z_half_e, out=ddxn_z_full)
