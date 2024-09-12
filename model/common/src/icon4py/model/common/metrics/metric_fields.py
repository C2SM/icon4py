# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    neighbor_sum,
    program,
    scan_operator,
    sin,
    tanh,
    where,
    log,
    exp
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa, settings
from icon4py.model.common.dimension import (
    C2E,
    C2E2C,
    C2E2CO,
    E2C,
    V2C,
    C2E2CODim,
    Koff,
    V2CDim,
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

# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@dataclass(frozen=True)
class MetricsConfig:
    #: Temporal extrapolation of Exner for computation of horizontal pressure gradient, defined in `mo_nonhydrostatic_nml.f90` used only in metrics fields calculation.
    exner_expol: Final[wpfloat] = 0.3333333333333


@program(grid_type=GridType.UNSTRUCTURED, backend=settings.backend)
def compute_z_mc(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute the geometric height of full levels from the geometric height of half levels (z_ifc).

    This assumes that the input field z_ifc is defined on half levels (KHalfDim) and the
    returned fields is defined on full levels (dims.KDim)

    Args:
        z_ifc: Field[Dims[dims.CellDim, dims.KDim], wpfloat] geometric height on half levels
        z_mc: Field[Dims[dims.CellDim, dims.KDim], wpfloat] output, geometric height defined on full levels
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain

    """
    average_cell_kdim_level_up(
        z_ifc,
        out=z_mc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_ddqz_z_half(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    nlev: int32,
):
    # TODO: change this to concat_where once it's merged
    ddqz_z_half = where(k == 0, 2.0 * (z_ifc - z_mc), 0.0)
    ddqz_z_half = where((k > 0) & (k < nlev), z_mc(Koff[-1]) - z_mc, ddqz_z_half)
    ddqz_z_half = where(k == nlev, 2.0 * (z_mc(Koff[-1]) - z_ifc), ddqz_z_half)
    return ddqz_z_half


@program(grid_type=GridType.UNSTRUCTURED, backend=settings.backend)
def compute_ddqz_z_half(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    ddqz_z_half: fa.CellKField[wpfloat],
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
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_ddqz_z_full_and_inverse(
    z_ifc: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    ddqz_z_full = difference_k_level_up(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ddqz_z_full_and_inverse(
    z_ifc: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
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
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_scalfac_dd3d(
    vct_a: fa.KField[wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: int32,
) -> fa.KField[wpfloat]:
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
    vct_a: fa.KField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
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
        vct_a: Field[Dims[dims.KDim], float],
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
    vct_a: fa.KField[wpfloat],
    damping_height: wpfloat,
    rayleigh_type: int32,
    rayleigh_classic: int32,
    rayleigh_klemp: int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
) -> fa.KField[wpfloat]:
    rayleigh_w = broadcast(0.0, (KDim,))
    z_sin_diff = maximum(0.0, vct_a - damping_height)
    z_tanh_diff = vct_a_1 - vct_a  # vct_a(1) - vct_a
    if rayleigh_type == rayleigh_classic:
        rayleigh_w = (
            rayleigh_coeff
            * (sin(pi_const / 2.0 * z_sin_diff / maximum(0.001, vct_a_1 - damping_height))) ** 2
        )

    elif rayleigh_type == rayleigh_klemp:
        rayleigh_w = rayleigh_coeff * (
            1.0 - tanh(3.8 * z_tanh_diff / maximum(0.000001, vct_a_1 - damping_height))
        )
    return rayleigh_w


@program
def compute_rayleigh_w(
    rayleigh_w: fa.KField[wpfloat],
    vct_a: fa.KField[wpfloat],
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
        vct_a: Field[Dims[dims.KDim], float]
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
    ddqz_z_full: fa.CellKField[wpfloat], z_ifc: fa.CellKField[wpfloat]
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    coeff1_dwdz = ddqz_z_full / ddqz_z_full(Koff[-1]) / (z_ifc(Koff[-1]) - z_ifc(Koff[1]))
    coeff2_dwdz = ddqz_z_full(Koff[-1]) / ddqz_z_full / (z_ifc(Koff[-1]) - z_ifc(Koff[1]))

    return coeff1_dwdz, coeff2_dwdz


@program(grid_type=GridType.UNSTRUCTURED)
def compute_coeff_dwdz(
    ddqz_z_full: fa.CellKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
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
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_d2dexdz2_fac1_mc(
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    cpd: float,
    grav: wpfloat,
    igradp_method: int32,
    igradp_constant: int32,
) -> fa.CellKField[vpfloat]:
    if igradp_method <= igradp_constant:
        d2dexdz2_fac1_mc = -grav / (cpd * theta_ref_mc**2) * inv_ddqz_z_full

    return d2dexdz2_fac1_mc


@field_operator
def _compute_d2dexdz2_fac2_mc(
    theta_ref_mc: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    igradp_method: int32,
    igradp_constant: int32,
) -> fa.CellKField[vpfloat]:
    if igradp_method <= igradp_constant:
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
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    igradp_method: int32,
    igradp_constant: int32,
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
        igradp_constant,
        out=d2dexdz2_fac1_mc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
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
        igradp_constant,
        out=d2dexdz2_fac2_mc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_ddxn_z_half_e(
    z_ifc: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddxn_z_half_e: fa.EdgeKField[wpfloat],
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
    z_ifv: Field[[dims.VertexDim, dims.KDim], float],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    ddxt_z_half_e: fa.EdgeKField[wpfloat],
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
    ddxnt_z_half_e: fa.EdgeKField[wpfloat],
    ddxn_z_full: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    average_edge_kdim_level_up(
        ddxnt_z_half_e,
        out=ddxn_z_full,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_vwind_expl_wgt(vwind_impl_wgt: fa.CellField[wpfloat]) -> fa.CellField[wpfloat]:
    return 1.0 - vwind_impl_wgt


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_expl_wgt(
    vwind_impl_wgt: fa.CellField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
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
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_maxslp_0_1 = maximum(abs(ddxn_z_full(C2E[0])), abs(ddxn_z_full(C2E[1])))
    z_maxslp = maximum(z_maxslp_0_1, abs(ddxn_z_full(C2E[2])))

    z_maxhgtd_0_1 = maximum(
        abs(ddxn_z_full(C2E[0]) * dual_edge_length(C2E[0])),
        abs(ddxn_z_full(C2E[1]) * dual_edge_length(C2E[1])),
    )

    z_maxhgtd = maximum(z_maxhgtd_0_1, abs(ddxn_z_full(C2E[2]) * dual_edge_length(C2E[2])))
    return z_maxslp, z_maxhgtd


@program
def compute_maxslp_maxhgtd(
    ddxn_z_full: Field[[dims.EdgeDim, dims.KDim], wpfloat],
    dual_edge_length: Field[[dims.EdgeDim], wpfloat],
    maxslp: Field[[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: Field[[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute z_maxslp and z_maxhgtd.

    See mo_vertical_grid.f90.

    Args:
        ddxn_z_full: dual_edge_length
        dual_edge_length: dual_edge_length
        maxslp: output
        maxhgtd: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_maxslp_maxhgtd(
        ddxn_z_full=ddxn_z_full,
        dual_edge_length=dual_edge_length,
        out=(maxslp, maxhgtd),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

@field_operator
def _exner_exfac_broadcast(exner_expol: wpfloat,) -> fa.CellKField[wpfloat]:
    return broadcast(exner_expol, (CellDim, KDim))

@field_operator
def _compute_exner_exfac(
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    exner_expol: wpfloat,
) -> fa.CellKField[wpfloat]:
    z_maxslp, z_maxhgtd = _compute_maxslp_maxhgtd(ddxn_z_full, dual_edge_length)

    exner_exfac = exner_expol * minimum(1.0 - (4.0 * z_maxslp) ** 2, 1.0 - (0.002 * z_maxhgtd) ** 2)
    exner_exfac = maximum(0.0, exner_exfac)
    exner_exfac = where(
        z_maxslp > 1.5, maximum(-1.0 / 6.0, 1.0 / 9.0 * (1.5 - z_maxslp)), exner_exfac
    )

    return exner_exfac


@program(grid_type=GridType.UNSTRUCTURED)
def compute_exner_exfac(
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    exner_exfac: fa.CellKField[wpfloat],
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
    _exner_exfac_broadcast(
        exner_expol,
        out=exner_exfac
    )
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
    z_ddxn_z_half_e: fa.EdgeField[wpfloat],
    z_ddxt_z_half_e: fa.EdgeField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    vwind_offctr: wpfloat,
) -> fa.CellField[wpfloat]:
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
    vct_a: fa.KField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_diff_2 = (z_ifc - z_ifc(Koff[-1])) / (vct_a - vct_a(Koff[-1]))
    vwind_impl_wgt_k = where(
        z_diff_2 < 0.6, maximum(vwind_impl_wgt, 1.2 - z_diff_2), vwind_impl_wgt
    )
    return vwind_impl_wgt_k


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_impl_wgt_partial(
    z_ddxn_z_half_e: fa.EdgeField[wpfloat],
    z_ddxt_z_half_e: fa.EdgeField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    vct_a: fa.KField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    vwind_impl_wgt_k: fa.CellKField[wpfloat],
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
        vct_a: Field[Dims[dims.KDim], float]
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
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_wgtfac_e(
    wgtfac_c: fa.CellKField[wpfloat],
    c_lin_e: Field[[dims.EdgeDim, dims.E2CDim], float],
    wgtfac_e: fa.EdgeKField[wpfloat],
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
    z_me: fa.EdgeKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    k_lev: fa.KField[int32],
) -> fa.EdgeKField[int32]:
    z_ifc_e_0 = z_ifc(E2C[0])
    z_ifc_e_k_0 = z_ifc_e_0(Koff[1])
    z_ifc_e_1 = z_ifc(E2C[1])
    z_ifc_e_k_1 = z_ifc_e_1(Koff[1])
    flat_idx = where(
        (z_me <= z_ifc_e_0) & (z_me >= z_ifc_e_k_0) & (z_me <= z_ifc_e_1) & (z_me >= z_ifc_e_k_1),
        k_lev,
        0,
    )
    return flat_idx


@program
def compute_flat_idx(
    z_me: fa.EdgeKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    k_lev: fa.KField[int32],
    flat_idx: fa.EdgeKField[int32],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_flat_idx(
        z_me=z_me,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_z_aux2(
    z_ifc: fa.CellField[wpfloat],
) -> fa.EdgeField[wpfloat]:
    extrapol_dist = 5.0
    z_aux1 = maximum(z_ifc(E2C[0]), z_ifc(E2C[1]))
    z_aux2 = z_aux1 - extrapol_dist

    return z_aux2


@program
def compute_z_aux2(
    z_ifc_sliced: fa.CellField[wpfloat],
    z_aux2: fa.EdgeField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
):
    _compute_z_aux2(
        z_ifc=z_ifc_sliced, out=z_aux2, domain={dims.EdgeDim: (horizontal_start, horizontal_end)}
    )


@field_operator
def _compute_pg_edgeidx_vertidx(
    c_lin_e: Field[[dims.EdgeDim, dims.E2CDim], float],
    z_ifc: fa.CellKField[wpfloat],
    z_aux2: fa.EdgeField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[int32],
    e_lev: fa.EdgeField[int32],
    k_lev: fa.KField[int32],
    pg_edgeidx: fa.EdgeKField[int32],
    pg_vertidx: fa.EdgeKField[int32],
) -> tuple[fa.EdgeKField[int32], fa.EdgeKField[int32]]:
    e_lev = broadcast(e_lev, (EdgeDim, KDim))
    k_lev = broadcast(k_lev, (EdgeDim, KDim))
    z_mc = average_cell_kdim_level_up(z_ifc)
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
    pg_edgeidx = where(
        (k_lev >= (flat_idx_max + 1)) & (z_me < z_aux2) & e_owner_mask, e_lev, pg_edgeidx
    )
    pg_vertidx = where(
        (k_lev >= (flat_idx_max + 1)) & (z_me < z_aux2) & e_owner_mask, k_lev, pg_vertidx
    )
    return pg_edgeidx, pg_vertidx


@program
def compute_pg_edgeidx_vertidx(
    c_lin_e: Field[[dims.EdgeDim, dims.E2CDim], float],
    z_ifc: fa.CellKField[wpfloat],
    z_aux2: fa.EdgeField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[int32],
    e_lev: fa.EdgeField[int32],
    k_lev: fa.KField[int32],
    pg_edgeidx: fa.EdgeKField[int32],
    pg_vertidx: fa.EdgeKField[int32],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_pg_edgeidx_vertidx(
        c_lin_e=c_lin_e,
        z_ifc=z_ifc,
        z_aux2=z_aux2,
        e_owner_mask=e_owner_mask,
        flat_idx_max=flat_idx_max,
        e_lev=e_lev,
        k_lev=k_lev,
        pg_edgeidx=pg_edgeidx,
        pg_vertidx=pg_vertidx,
        out=(pg_edgeidx, pg_vertidx),
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_pg_exdist_dsl(
    z_me: fa.EdgeKField[wpfloat],
    z_aux2: fa.EdgeField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[int32],
    k_lev: fa.KField[int32],
    pg_exdist_dsl: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    k_lev = broadcast(k_lev, (EdgeDim, KDim))
    pg_exdist_dsl = where(
        (k_lev >= (flat_idx_max + 1)) & (z_me < z_aux2) & e_owner_mask,
        z_me - z_aux2,
        pg_exdist_dsl,
    )
    return pg_exdist_dsl


@program
def compute_pg_exdist_dsl(
    z_aux2: fa.EdgeField[wpfloat],
    z_me: fa.EdgeKField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[int32],
    k_lev: fa.KField[int32],
    pg_exdist_dsl: fa.EdgeKField[wpfloat],
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
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_pg_edgeidx_dsl(
    pg_edgeidx: fa.EdgeKField[int32],
    pg_vertidx: fa.EdgeKField[int32],
) -> fa.EdgeKField[bool]:
    pg_edgeidx_dsl = where((pg_edgeidx > 0) & (pg_vertidx > 0), True, False)
    return pg_edgeidx_dsl


@program
def compute_pg_edgeidx_dsl(
    pg_edgeidx: fa.EdgeKField[int32],
    pg_vertidx: fa.EdgeKField[int32],
    pg_edgeidx_dsl: fa.EdgeKField[bool],
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
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[int32], mask_prog_halo_c: fa.CellField[bool]
) -> fa.CellField[bool]:
    mask_prog_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), mask_prog_halo_c, True)
    return mask_prog_halo_c


@program
def compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[int32],
    mask_prog_halo_c: fa.CellField[bool],
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
    c_refin_ctrl: fa.CellField[int32],
    bdy_halo_c: fa.CellField[bool],
) -> fa.CellField[bool]:
    bdy_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), True, bdy_halo_c)
    return bdy_halo_c


@program
def compute_bdy_halo_c(
    c_refin_ctrl: fa.CellField[int32],
    bdy_halo_c: fa.CellField[bool],
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
    e_refin_ctrl: fa.EdgeField[int32], grf_nudge_start_e: int32, grf_nudgezone_width: int32
) -> fa.EdgeField[wpfloat]:
    hmask_dd3d = (
        1
        / (grf_nudgezone_width - 1)
        * (e_refin_ctrl - (grf_nudge_start_e + grf_nudgezone_width - 1))
    )
    hmask_dd3d = where(e_refin_ctrl <= (grf_nudge_start_e + grf_nudgezone_width - 1), 0, hmask_dd3d)
    hmask_dd3d = where(
        (e_refin_ctrl <= 0) | (e_refin_ctrl >= (grf_nudge_start_e + 2 * (grf_nudgezone_width - 1))),
        1,
        hmask_dd3d,
    )
    return astype(hmask_dd3d, wpfloat)


@program
def compute_hmask_dd3d(
    e_refin_ctrl: fa.EdgeField[int32],
    hmask_dd3d: fa.EdgeField[wpfloat],
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


@field_operator
def _compute_weighted_cell_neighbor_sum(
    field: Field[[dims.CellDim, dims.KDim], wpfloat],
    c_bln_avg: Field[[dims.CellDim, C2E2CODim], wpfloat],
) -> Field[[dims.CellDim, dims.KDim], wpfloat]:
    field_avg = neighbor_sum(field(C2E2CO) * c_bln_avg, axis=C2E2CODim)
    return field_avg


@program
def compute_weighted_cell_neighbor_sum(
    maxslp: Field[[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: Field[[dims.CellDim, dims.KDim], wpfloat],
    c_bln_avg: Field[[dims.CellDim, C2E2CODim], wpfloat],
    z_maxslp_avg: Field[[dims.CellDim, dims.KDim], wpfloat],
    z_maxhgtd_avg: Field[[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute z_maxslp_avg and z_maxhgtd_avg.

    See mo_vertical_grid.f90.

    Args:
        maxslp: Max field over ddxn_z_full offset
        maxhgtd: Max field over ddxn_z_full offset*dual_edge_length offset
        c_bln_avg: Interpolation field
        z_maxslp_avg: output
        z_maxhgtd_avg: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_weighted_cell_neighbor_sum(
        field=maxslp,
        c_bln_avg=c_bln_avg,
        out=z_maxslp_avg,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

    _compute_weighted_cell_neighbor_sum(
        field=maxhgtd,
        c_bln_avg=c_bln_avg,
        out=z_maxhgtd_avg,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_max_nbhgt(
    z_mc_nlev: Field[[dims.CellDim], wpfloat],
) -> Field[[dims.CellDim], wpfloat]:
    max_nbhgt_0_1 = maximum(z_mc_nlev(C2E2C[0]), z_mc_nlev(C2E2C[1]))
    max_nbhgt = maximum(max_nbhgt_0_1, z_mc_nlev(C2E2C[2]))
    return max_nbhgt


@program
def compute_max_nbhgt(
    z_mc_nlev: Field[[dims.CellDim], wpfloat],
    max_nbhgt: Field[[dims.CellDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute max_nbhgt.

    See mo_vertical_grid.f90.

    Args:
        z_mc_nlev: Last K level of z_mc
        max_nbhgt: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_max_nbhgt(
        z_mc_nlev=z_mc_nlev,
        out=max_nbhgt,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


@scan_operator(axis=dims.KDim, forward=True, init=(0, False))
def _compute_param(
    param: tuple[int32, bool],
    z_me_jk: float,
    z_ifc_off: float,
    z_ifc_off_koff: float,
    lower: int32,
    nlev: int32,
) -> tuple[int32, bool]:
    param_0, param_1 = param
    if param_0 >= lower:
        if (param_0 == nlev) | (z_me_jk <= z_ifc_off) & (z_me_jk >= z_ifc_off_koff):
            param_1 = True
    return param_0 + 1, param_1


@field_operator(grid_type=GridType.UNSTRUCTURED)
def _compute_z_ifc_off_koff(
    z_ifc_off: Field[[dims.EdgeDim, dims.KDim], wpfloat],
) -> Field[[dims.EdgeDim, dims.KDim], wpfloat]:
    n = z_ifc_off(Koff[1])
    return n


# TODO: this field is already in `compute_cell_2_vertex_interpolation` file
# inquire if it is ok to move here
@field_operator
def _compute_cell_2_vertex_interpolation(
    cell_in: Field[[dims.CellDim, dims.KDim], wpfloat],
    c_int: Field[[dims.VertexDim, V2CDim], wpfloat],
) -> Field[[dims.VertexDim, dims.KDim], wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=V2CDim)
    return vert_out


program(grid_type=GridType.UNSTRUCTURED, backend=settings.backend)


@program
def compute_cell_2_vertex_interpolation(
    cell_in: Field[[dims.CellDim, dims.KDim], wpfloat],
    c_int: Field[[dims.VertexDim, V2CDim], wpfloat],
    vert_out: Field[[dims.VertexDim, dims.KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute the interpolation from cell to vertex field.

    Args:
        cell_in: input cell field
        c_int: interpolation coefficients
        vert_out: (output) vertex field
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_cell_2_vertex_interpolation(
        cell_in,
        c_int,
        out=vert_out,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

@field_operator
def _compute_theta_exner_ref_mc(
    z_mc: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    p0sl_bg: wpfloat,
    rd_o_cpd: wpfloat,
    p0ref: wpfloat,
):
    z_aux1 = p0sl_bg * exp(-grav / rd * h_scal_bg / (t0sl_bg - del_t_bg)
                    * log((exp(z_mc / h_scal_bg) *(t0sl_bg - del_t_bg) + del_t_bg) / t0sl_bg))
    exner_ref_mc = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = (t0sl_bg - del_t_bg) + del_t_bg * exp(-z_mc / h_scal_bg)
    theta_ref_mc = z_temp / exner_ref_mc
    return exner_ref_mc, theta_ref_mc


@program
def compute_theta_exner_ref_mc(
    z_mc: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    p0sl_bg: wpfloat,
    rd_o_cpd: wpfloat,
    p0ref: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_theta_exner_ref_mc(
        z_mc=z_mc,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
        h_scal_bg=h_scal_bg,
        grav=grav,
        rd=rd,
        p0sl_bg=p0sl_bg,
        rd_o_cpd=rd_o_cpd,
        p0ref=p0ref,
        out=(exner_ref_mc, theta_ref_mc),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
