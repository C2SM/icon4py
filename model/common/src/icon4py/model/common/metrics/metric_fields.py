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
from dataclasses import dataclass
from typing import Final

from gt4py.next import (
    Field,
    GridType,
    abs,
    astype,
    broadcast,
    exp,
    field_operator,
    int32,
    maximum,
    minimum,
    program,
    sin,
    tanh,
    where,
)

from icon4py.model.common.dimension import (
    C2E,
    E2C,
    CellDim,
    E2CDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.math.helpers import (
    _grad_fd_tang,
    average_cell_kdim_level_up,
    average_edge_kdim_level_up,
    difference_k_level_up,
    grad_fd_norm,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


"""
Contains metric fields calculations for the vertical grid, ported from mo_vertical_grid.f90.
"""


@dataclass(frozen=True)
class MetricsConfig:
    #: Temporal extrapolation of Exner for computation of horizontal pressure gradient, defined in `mo_nonhydrostatic_nml.f90` used only in metrics fields calculation.
    exner_expol: Final[wpfloat] = 0.333


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
):
    # TODO: change this to concat_where once it's merged
    ddqz_z_half = where(k == 0, 2.0 * (z_ifc - z_mc), 0.0)
    ddqz_z_half = where((k > 0) & (k < nlev), z_mc(Koff[-1]) - z_mc, ddqz_z_half)
    ddqz_z_half = where(k == nlev, 2.0 * (z_mc(Koff[-1]) - z_ifc), ddqz_z_half)
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
def _compute_ddqz_z_full_and_inverse(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    ddqz_z_full = difference_k_level_up(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ddqz_z_full_and_inverse(
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
    _compute_ddqz_z_full_and_inverse(
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
def compute_ddxn_z_full(
    z_ddxnt_z_half_e: Field[[EdgeDim, KDim], float], ddxn_z_full: Field[[EdgeDim, KDim], float]
):
    average_edge_kdim_level_up(z_ddxnt_z_half_e, out=ddxn_z_full)


@field_operator
def _compute_vwind_expl_wgt(vwind_impl_wgt: Field[[CellDim], wpfloat]) -> Field[[CellDim], wpfloat]:
    return 1.0 - vwind_impl_wgt


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_expl_wgt(
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute vwind_expl_wgt.

    See mo_vertical_grid.f90

    Args:
        vwind_impl_wgt: offcentering in vertical mass flux
        vwind_expl_wgt: (output) 1 - of vwind_impl_wgt
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index

    """

    _compute_vwind_expl_wgt(
        vwind_impl_wgt=vwind_impl_wgt,
        out=vwind_expl_wgt,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_maxslp_maxhgtd(
    ddxn_z_full: Field[[EdgeDim, KDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    z_maxslp_0_1 = maximum(abs(ddxn_z_full(C2E[0])), abs(ddxn_z_full(C2E[1])))
    z_maxslp = maximum(z_maxslp_0_1, abs(ddxn_z_full(C2E[2])))

    z_maxhgtd_0_1 = maximum(
        abs(ddxn_z_full(C2E[0]) * dual_edge_length(C2E[0])),
        abs(ddxn_z_full(C2E[1]) * dual_edge_length(C2E[1])),
    )

    z_maxhgtd = maximum(z_maxhgtd_0_1, abs(ddxn_z_full(C2E[2]) * dual_edge_length(C2E[2])))
    return z_maxslp, z_maxhgtd


@field_operator
def _compute_exner_exfac(
    ddxn_z_full: Field[[EdgeDim, KDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    exner_expol: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    z_maxslp, z_maxhgtd = _compute_maxslp_maxhgtd(ddxn_z_full, dual_edge_length)

    exner_exfac = exner_expol * minimum(1.0 - (4.0 * z_maxslp) ** 2, 1.0 - (0.002 * z_maxhgtd) ** 2)
    exner_exfac = maximum(0.0, exner_exfac)
    exner_exfac = where(
        z_maxslp > 1.5, maximum(-1.0 / 6.0, 1.0 / 9.0 * (1.5 - z_maxslp)), exner_exfac
    )

    return exner_exfac


@program(grid_type=GridType.UNSTRUCTURED)
def compute_exner_exfac(
    ddxn_z_full: Field[[EdgeDim, KDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    exner_exfac: Field[[CellDim, KDim], wpfloat],
    exner_expol: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute exner_exfac.

    Exner extrapolation reaches zero for a slope of 1/4 or a height difference of 500 m between adjacent grid points (empirically determined values). See mo_vertical_grid.f90

    Args:
        ddxn_z_full: ddxn_z_full
        dual_edge_length: dual_edge_length
        exner_exfac: Exner factor
        exner_expol: Exner extrapolation factor
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index

    """
    _compute_exner_exfac(
        ddxn_z_full=ddxn_z_full,
        dual_edge_length=dual_edge_length,
        exner_expol=exner_expol,
        out=exner_exfac,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_vwind_impl_wgt_1(
    z_ddxn_z_half_e: Field[[EdgeDim], wpfloat],
    z_ddxt_z_half_e: Field[[EdgeDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    vwind_offctr: wpfloat,
) -> Field[[CellDim], wpfloat]:
    z_ddx_1 = maximum(abs(z_ddxn_z_half_e(C2E[0])), abs(z_ddxt_z_half_e(C2E[0])))
    z_ddx_2 = maximum(abs(z_ddxn_z_half_e(C2E[1])), abs(z_ddxt_z_half_e(C2E[1])))
    z_ddx_3 = maximum(abs(z_ddxn_z_half_e(C2E[2])), abs(z_ddxt_z_half_e(C2E[2])))
    z_ddx_1_2 = maximum(z_ddx_1, z_ddx_2)
    z_maxslope = maximum(z_ddx_1_2, z_ddx_3)

    z_diff_1_2 = maximum(
        abs(z_ddxn_z_half_e(C2E[0]) * dual_edge_length(C2E[0])),
        abs(z_ddxn_z_half_e(C2E[1]) * dual_edge_length(C2E[1])),
    )
    z_diff = maximum(z_diff_1_2, abs(z_ddxn_z_half_e(C2E[2]) * dual_edge_length(C2E[2])))
    z_offctr_1 = maximum(vwind_offctr, 0.425 * z_maxslope ** (0.75))
    z_offctr = maximum(z_offctr_1, minimum(0.25, 0.00025 * (z_diff - 250.0)))
    z_offctr = minimum(maximum(vwind_offctr, 0.75), z_offctr)
    vwind_impl_wgt = 0.5 + z_offctr
    return vwind_impl_wgt


@field_operator
def _compute_vwind_impl_wgt_2(
    vct_a: Field[[KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    z_diff_2 = (z_ifc - z_ifc(Koff[-1])) / (vct_a - vct_a(Koff[-1]))
    vwind_impl_wgt_k = where(
        z_diff_2 < 0.6, maximum(vwind_impl_wgt, 1.2 - z_diff_2), vwind_impl_wgt
    )
    return vwind_impl_wgt_k


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_impl_wgt(
    z_ddxn_z_half_e: Field[[EdgeDim], wpfloat],
    z_ddxt_z_half_e: Field[[EdgeDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    vct_a: Field[[KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    vwind_impl_wgt_k: Field[[CellDim, KDim], wpfloat],
    vwind_offctr: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute vwind_impl_wgt.

    See mo_vertical_grid.f90

    Args:
        z_ddxn_z_half_e: intermediate storage for field
        z_ddxt_z_half_e: intermediate storage for field
        dual_edge_length: dual_edge_length
        vct_a: Field[[KDim], float]
        z_ifc: geometric height on half levels
        vwind_impl_wgt: (output) offcentering in vertical mass flux
        vwind_offctr: off-centering in vertical wind solver
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_vwind_impl_wgt_1(
        z_ddxn_z_half_e=z_ddxn_z_half_e,
        z_ddxt_z_half_e=z_ddxt_z_half_e,
        dual_edge_length=dual_edge_length,
        vwind_offctr=vwind_offctr,
        out=vwind_impl_wgt,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )

    _compute_vwind_impl_wgt_2(
        vct_a=vct_a,
        z_ifc=z_ifc,
        vwind_impl_wgt=vwind_impl_wgt,
        out=vwind_impl_wgt_k,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@program
def compute_wgtfac_e(
    wgtfac_c: Field[[CellDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute wgtfac_e.

    See mo_vertical_grid.f90

    Args:
        wgtfac_c: weighting factor for quadratic interpolation to surface
        c_lin_e: interpolation field
        wgtfac_e: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _cell_2_edge_interpolation(
        in_field=wgtfac_c,
        coeff=c_lin_e,
        out=wgtfac_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_flat_idx(
    z_me: Field[[EdgeDim, KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k_lev: Field[[KDim], int],
) -> Field[[EdgeDim, KDim], int]:
    z_ifc_e_0 = z_ifc(E2C[0])
    z_ifc_e_k_0 = z_ifc_e_0(Koff[1])
    z_ifc_e_1 = z_ifc(E2C[1])
    z_ifc_e_k_1 = z_ifc_e_1(Koff[1])
    flat_idx = where(
        (z_me <= z_ifc_e_0) & (z_me >= z_ifc_e_k_0) & (z_me <= z_ifc_e_1) & (z_me >= z_ifc_e_k_1),
        k_lev,
        int(0),
    )
    return flat_idx


@field_operator
def _compute_z_aux2(
    z_ifc: Field[[CellDim], wpfloat],
) -> Field[[EdgeDim], wpfloat]:
    extrapol_dist = 5.0
    z_aux1 = maximum(z_ifc(E2C[0]), z_ifc(E2C[1]))
    z_aux2 = z_aux1 - extrapol_dist

    return z_aux2


@field_operator
def _compute_pg_edgeidx_vertidx(
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_aux2: Field[[EdgeDim], wpfloat],
    e_owner_mask: Field[[EdgeDim], bool],
    flat_idx_max: Field[[EdgeDim], int],
    e_lev: Field[[EdgeDim], int],
    k_lev: Field[[KDim], int],
    pg_edgeidx: Field[[EdgeDim, KDim], int],
    pg_vertidx: Field[[EdgeDim, KDim], int],
) -> tuple[Field[[EdgeDim, KDim], int], Field[[EdgeDim, KDim], int]]:
    e_lev = broadcast(e_lev, (EdgeDim, KDim))
    k_lev = broadcast(k_lev, (EdgeDim, KDim))
    z_mc = average_cell_kdim_level_up(z_ifc)
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
    pg_edgeidx = where(
        (k_lev >= (flat_idx_max + int(1))) & (z_me < z_aux2) & e_owner_mask, e_lev, pg_edgeidx
    )
    pg_vertidx = where(
        (k_lev >= (flat_idx_max + int(1))) & (z_me < z_aux2) & e_owner_mask, k_lev, pg_vertidx
    )
    return pg_edgeidx, pg_vertidx


@field_operator
def _compute_pg_exdist_dsl(
    z_me: Field[[EdgeDim, KDim], wpfloat],
    z_aux2: Field[[EdgeDim], wpfloat],
    e_owner_mask: Field[[EdgeDim], bool],
    flat_idx_max: Field[[EdgeDim], int],
    k_lev: Field[[KDim], int],
    pg_exdist_dsl: Field[[EdgeDim, KDim], wpfloat],
) -> Field[[EdgeDim, KDim], wpfloat]:
    k_lev = broadcast(k_lev, (EdgeDim, KDim))
    pg_exdist_dsl = where(
        (k_lev >= (flat_idx_max + int(1))) & (z_me < z_aux2) & e_owner_mask,
        z_me - z_aux2,
        pg_exdist_dsl,
    )
    return pg_exdist_dsl


@program
def compute_pg_exdist_dsl(
    z_aux2: Field[[EdgeDim], wpfloat],
    z_me: Field[[EdgeDim, KDim], float],
    e_owner_mask: Field[[EdgeDim], bool],
    flat_idx_max: Field[[EdgeDim], int],
    k_lev: Field[[KDim], int],
    pg_exdist_dsl: Field[[EdgeDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute pg_edgeidx_dsl.

    See mo_vertical_grid.f90

    Args:
        z_aux2: Local field
        z_me: Local field
        e_owner_mask: Field of booleans over edges
        flat_idx_max: Highest vertical index (counted from top to bottom) for which the edge point lies inside the cell box of the adjacent grid points
        k_lev: Field of K levels
        pg_exdist_dsl: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_pg_exdist_dsl(
        z_me=z_me,
        z_aux2=z_aux2,
        e_owner_mask=e_owner_mask,
        flat_idx_max=flat_idx_max,
        k_lev=k_lev,
        pg_exdist_dsl=pg_exdist_dsl,
        out=pg_exdist_dsl,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_pg_edgeidx_dsl(
    pg_edgeidx: Field[[EdgeDim, KDim], int],
    pg_vertidx: Field[[EdgeDim, KDim], int],
) -> Field[[EdgeDim, KDim], bool]:
    pg_edgeidx_dsl = where((pg_edgeidx > int(0)) & (pg_vertidx > int(0)), True, False)
    return pg_edgeidx_dsl


@program
def compute_pg_edgeidx_dsl(
    pg_edgeidx: Field[[EdgeDim, KDim], int],
    pg_vertidx: Field[[EdgeDim, KDim], int],
    pg_edgeidx_dsl: Field[[EdgeDim, KDim], bool],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute pg_edgeidx_dsl.

    See mo_vertical_grid.f90

    Args:
        pg_edgeidx: Index Edge values
        pg_vertidx: Index K values
        pg_edgeidx_dsl: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_pg_edgeidx_dsl(
        pg_edgeidx=pg_edgeidx,
        pg_vertidx=pg_vertidx,
        out=pg_edgeidx_dsl,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_mask_prog_halo_c(
    c_refin_ctrl: Field[[CellDim], int32], mask_prog_halo_c: Field[[CellDim], bool]
) -> Field[[CellDim], bool]:
    mask_prog_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), mask_prog_halo_c, True)
    return mask_prog_halo_c


@program
def compute_mask_prog_halo_c(
    c_refin_ctrl: Field[[CellDim], int32],
    mask_prog_halo_c: Field[[CellDim], bool],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute mask_prog_halo_c.

    See mo_vertical_grid.f90

    Args:
        c_refin_ctrl: Cell field of refin_ctrl
        mask_prog_halo_c: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_mask_prog_halo_c(
        c_refin_ctrl,
        mask_prog_halo_c,
        out=mask_prog_halo_c,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_bdy_halo_c(
    c_refin_ctrl: Field[[CellDim], int32],
    bdy_halo_c: Field[[CellDim], bool],
) -> Field[[CellDim], bool]:
    bdy_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), True, bdy_halo_c)
    return bdy_halo_c


@program
def compute_bdy_halo_c(
    c_refin_ctrl: Field[[CellDim], int32],
    bdy_halo_c: Field[[CellDim], bool],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute bdy_halo_c.

    See mo_vertical_grid.f90. mask_prog_halo_c_dsl_low_refin in ICON

    Args:
        c_refin_ctrl: Cell field of refin_ctrl
        bdy_halo_c: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_bdy_halo_c(
        c_refin_ctrl,
        bdy_halo_c,
        out=bdy_halo_c,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_hmask_dd3d(
    e_refin_ctrl: Field[[EdgeDim], int32], grf_nudge_start_e: int32, grf_nudgezone_width: int32
) -> Field[[EdgeDim], wpfloat]:
    hmask_dd3d = (
        1.0
        / (astype(grf_nudgezone_width, wpfloat) - 1.0)
        * (
            astype(e_refin_ctrl, wpfloat)
            - (astype(grf_nudge_start_e, wpfloat) + astype(grf_nudgezone_width, wpfloat) - 1.0)
        )
    )
    hmask_dd3d = where(
        (e_refin_ctrl <= 0) | (e_refin_ctrl >= (grf_nudge_start_e + 2 * (grf_nudgezone_width - 1))),
        1.0,
        hmask_dd3d,
    )
    hmask_dd3d = where(
        e_refin_ctrl <= (grf_nudge_start_e + grf_nudgezone_width - 1), 0.0, hmask_dd3d
    )
    return hmask_dd3d


@program
def compute_hmask_dd3d(
    e_refin_ctrl: Field[[EdgeDim], int32],
    hmask_dd3d: Field[[EdgeDim], wpfloat],
    grf_nudge_start_e: int32,
    grf_nudgezone_width: int32,
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute hmask_dd3d.

    See mo_vertical_grid.f90. Horizontal mask field for 3D divergence damping term.

    Args:
        e_refin_ctrl: Edge field of refin_ctrl
        hmask_dd3d: output
        grf_nudge_start_e: mo_impl_constants_grf constant
        grf_nudgezone_width: mo_impl_constants_grf constant
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_hmask_dd3d(
        e_refin_ctrl=e_refin_ctrl,
        grf_nudge_start_e=grf_nudge_start_e,
        grf_nudgezone_width=grf_nudgezone_width,
        out=hmask_dd3d,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )
