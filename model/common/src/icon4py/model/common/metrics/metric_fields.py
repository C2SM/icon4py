# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType
from typing import Final

import gt4py.next as gtx
import numpy as np
from gt4py.next import (
    GridType,
    abs,  # noqa: A004
    astype,
    broadcast,
    exp,
    field_operator,
    int32,
    log,
    maximum,
    minimum,
    neighbor_sum,
    program,
    scan_operator,
    sin,
    tanh,
    where,
)
from gt4py.next.ffront.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_options
from icon4py.model.common.dimension import C2E, C2E2C, C2E2CO, E2C, C2E2CODim, Koff
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.math.helpers import (
    _grad_fd_tang,
    difference_level_plus1_on_cells,
    grad_fd_norm,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Contains metric fields calculations for the vertical grid, ported from mo_vertical_grid.f90.
"""


rayleigh_damping_options: Final = model_options.RayleighType()


# TODO(@nfarabullini): ddqz_z_half vertical dimension is khalf, use K2KHalf once merged for z_ifc and z_mc
# TODO(@nfarabullini): change dimension type hint for ddqz_z_half to cell, khalf
@field_operator
def _compute_ddqz_z_half(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    ddqz_z_half = concat_where((dims.KDim > 0) & (dims.KDim < nlev), 0.0, 2.0 * (z_ifc - z_mc))
    ddqz_z_half = concat_where(
        (0 < dims.KDim) & (dims.KDim < nlev), z_mc(Koff[-1]) - z_mc, ddqz_z_half
    )
    ddqz_z_half = concat_where(dims.KDim == nlev, 2.0 * (z_mc(Koff[-1]) - z_ifc), ddqz_z_half)
    return ddqz_z_half


@program(grid_type=GridType.UNSTRUCTURED, backend=None)
def compute_ddqz_z_half(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        nlev,
        out=ddqz_z_half,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_ddqz_z_full_and_inverse(
    z_ifc: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    ddqz_z_full = difference_level_plus1_on_cells(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ddqz_z_full_and_inverse(
    z_ifc: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_scaling_factor_for_3d_divdamp(
    vct_a: fa.KField[wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: gtx.int32,
) -> fa.KField[wpfloat]:
    scaling_factor_for_3d_divdamp = broadcast(1.0, (dims.KDim,))
    if divdamp_type == 32:
        zf = 0.5 * (vct_a + vct_a(Koff[1]))  # depends on nshift_total, assumed to be always 0
        scaling_factor_for_3d_divdamp = where(
            zf >= divdamp_trans_end, 0.0, scaling_factor_for_3d_divdamp
        )
        scaling_factor_for_3d_divdamp = where(
            zf >= divdamp_trans_start,
            (divdamp_trans_end - zf) / (divdamp_trans_end - divdamp_trans_start),
            scaling_factor_for_3d_divdamp,
        )
    return scaling_factor_for_3d_divdamp


@program
def compute_scaling_factor_for_3d_divdamp(
    vct_a: fa.KField[wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute scaling factor for 3D divergence damping terms (declared as scalfac_dd3d in ICON).

    See mo_vertical_grid.f90

    Args:
        vct_a: Field[Dims[dims.KDim], float],
        scaling_factor_for_3d_divdamp: (output) scaling factor for 3D divergence damping terms, and start level from which they are > 0
        divdamp_trans_start: lower bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_trans_end: upper bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_type: type of divergence damping (2D or 3D divergence)
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_scaling_factor_for_3d_divdamp(
        vct_a,
        divdamp_trans_start,
        divdamp_trans_end,
        divdamp_type,
        out=scaling_factor_for_3d_divdamp,
        domain={dims.KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_rayleigh_w(
    vct_a: fa.KField[wpfloat],
    damping_height: wpfloat,
    rayleigh_type: gtx.int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
) -> fa.KField[wpfloat]:
    rayleigh_w = broadcast(0.0, (dims.KDim,))
    z_sin_diff = maximum(0.0, vct_a - damping_height)
    z_tanh_diff = vct_a_1 - vct_a  # vct_a(1) - vct_a
    if rayleigh_type == rayleigh_damping_options.CLASSIC:
        rayleigh_w = (
            rayleigh_coeff
            * (sin(pi_const / 2.0 * z_sin_diff / maximum(0.001, vct_a_1 - damping_height))) ** 2
        )

    elif rayleigh_type == rayleigh_damping_options.KLEMP:
        rayleigh_w = rayleigh_coeff * (
            1.0 - tanh(3.8 * z_tanh_diff / maximum(0.000001, vct_a_1 - damping_height))
        )
    return rayleigh_w


@program
def compute_rayleigh_w(
    rayleigh_w: fa.KField[wpfloat],
    vct_a: fa.KField[wpfloat],
    damping_height: wpfloat,
    rayleigh_type: gtx.int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        rayleigh_coeff,
        vct_a_1,
        pi_const,
        out=rayleigh_w,
        domain={dims.KDim: (vertical_start, vertical_end)},
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
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_ddxn_z_half_e(
    z_ifc: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddxn_z_half_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    grad_fd_norm(
        z_ifc,
        inv_dual_edge_length,
        out=ddxn_z_half_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_ddxt_z_half_e(
    cell_in: fa.CellKField[wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
):
    z_ifv = _compute_cell_2_vertex_interpolation(cell_in, c_int)
    ddxt_z_half_e = _grad_fd_tang(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
    )
    return ddxt_z_half_e


@program
def compute_ddxt_z_half_e(
    cell_in: fa.CellKField[wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    ddxt_z_half_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ddxt_z_half_e(
        cell_in,
        c_int,
        inv_primal_edge_length,
        tangent_orientation,
        out=ddxt_z_half_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_exner_w_explicit_weight_parameter(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    return 1.0 - exner_w_implicit_weight_parameter


@program(grid_type=GridType.UNSTRUCTURED)
def compute_exner_w_explicit_weight_parameter(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute exner_w_explicit_weight_parameter.

    See mo_vertical_grid.f90

    Args:
        exner_w_implicit_weight_parameter: offcentering in vertical mass flux
        exner_w_explicit_weight_parameter: (output) 1 - exner_w_implicit_weight_parameter
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index

    """

    _compute_exner_w_explicit_weight_parameter(
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        out=exner_w_explicit_weight_parameter,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_maxslp_maxhgtd(
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_maxslp_0_1 = maximum(abs(ddxn_z_full(C2E[0])), abs(ddxn_z_full(C2E[1])))
    maxslp = maximum(z_maxslp_0_1, abs(ddxn_z_full(C2E[2])))

    z_maxhgtd_0_1 = maximum(
        abs(ddxn_z_full(C2E[0]) * dual_edge_length(C2E[0])),
        abs(ddxn_z_full(C2E[1]) * dual_edge_length(C2E[1])),
    )

    maxhgtd = maximum(z_maxhgtd_0_1, abs(ddxn_z_full(C2E[2]) * dual_edge_length(C2E[2])))
    return maxslp, maxhgtd


@program
def compute_maxslp_maxhgtd(
    ddxn_z_full: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], wpfloat],
    dual_edge_length: gtx.Field[gtx.Dims[dims.EdgeDim], wpfloat],
    maxslp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_exner_exfac(
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
    exner_expol: wpfloat,
    lateral_boundary_level_2: gtx.int32,
) -> fa.CellKField[wpfloat]:
    z_maxslp, z_maxhgtd = _compute_maxslp_maxhgtd(ddxn_z_full, dual_edge_length)

    exner_exfac = concat_where(
        dims.CellDim >= lateral_boundary_level_2,
        exner_expol * minimum(1.0 - (4.0 * z_maxslp) ** 2, 1.0 - (0.002 * z_maxhgtd) ** 2),
        exner_expol,
    )
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
    lateral_boundary_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        lateral_boundary_level_2=lateral_boundary_level_2,
        out=exner_exfac,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@program
def compute_wgtfac_e(
    wgtfac_c: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float],
    wgtfac_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_flat_idx(
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    k_lev: fa.KField[gtx.int32],
) -> fa.EdgeKField[gtx.int32]:
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
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


@program(grid_type=GridType.UNSTRUCTURED)
def compute_flat_idx(
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    k_lev: fa.KField[int32],
    flat_idx: fa.EdgeKField[int32],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_flat_idx(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        z_ifc=z_ifc,
        k_lev=k_lev,
        out=flat_idx,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def compute_max_index(
    flat_idx: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    """
    Reduces a 2d array to a 1d array by taking the maximum value along axis 1.

    Usage example in ICON: to compute the max index of flat levels along a horizontal dimension.
    """
    max_idx = array_ns.amax(flat_idx, axis=1)
    return max_idx


@field_operator
def _compute_downward_extrapolation_distance(
    z_ifc: fa.CellField[wpfloat],
) -> fa.EdgeField[wpfloat]:
    extrapol_dist = 5.0
    z_aux1 = maximum(z_ifc(E2C[0]), z_ifc(E2C[1]))
    z_aux2 = z_aux1 - extrapol_dist
    return z_aux2


@field_operator
def _compute_pressure_gradient_downward_extrapolation_mask_distance(
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    topography: fa.CellField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[gtx.int32],
    e_lev: fa.EdgeField[gtx.int32],
    k_lev: fa.KField[gtx.int32],
    horizontal_start_distance: int32,
    horizontal_end_distance: int32,
) -> tuple[fa.EdgeKField[bool], fa.EdgeKField[wpfloat]]:
    """
    Compute an edge mask and extrapolation distance for grid points requiring downward extrapolation of the pressure gradient.

    See pg_edgeidx and pg_exdist in mo_vertical_grid.f90

    Args:
        z_mc: height of cells [m]
        c_lin_e:  interpolation coefficient from cells to edges
        topography: ground level height of cells [m]
        e_owner_mask: mask edges owned by PE.
        flat_idx_max: level from where edge levels start to become flat
        e_lev: edge indices
        k_lev: k-level indices
        horizontal_start_distance: start index in edge fields from where extrapolation distance is computed
        horizontal_end_distance: end index in edge fields until where extrapolation distance is computed

    Returns:
        pg_edge_mask: edge index mask for points requiring downward extrapolation
        pg_exdist_dsl: extrapolation distance

    """

    e_lev = broadcast(e_lev, (dims.EdgeDim, dims.KDim))
    k_lev = broadcast(k_lev, (dims.EdgeDim, dims.KDim))
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
    downward_distance = _compute_downward_extrapolation_distance(topography)
    extrapolation_distance = concat_where(
        (horizontal_start_distance <= dims.EdgeDim) & (dims.EdgeDim < horizontal_end_distance),
        downward_distance,
        0.0,
    )
    flatness_condition = (k_lev >= (flat_idx_max + 1)) & (z_me < downward_distance) & e_owner_mask
    pg_edgeidx, pg_vertidx = where(flatness_condition, (e_lev, k_lev), (0, 0))
    pg_edge_mask = (pg_edgeidx > 0) & (pg_vertidx > 0)

    pg_exdist_dsl = where(
        (k_lev >= (flat_idx_max + 1)) & (z_me < extrapolation_distance) & e_owner_mask,
        z_me - extrapolation_distance,
        0.0,
    )

    return pg_edge_mask, pg_exdist_dsl


@program(grid_type=GridType.UNSTRUCTURED)
def compute_pressure_gradient_downward_extrapolation_mask_distance(
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float],
    topography: fa.CellField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[gtx.int32],
    e_lev: fa.EdgeField[gtx.int32],
    k_lev: fa.KField[gtx.int32],
    pg_edgeidx_dsl: fa.EdgeKField[bool],
    pg_exdist_dsl: fa.EdgeKField[wpfloat],
    horizontal_start_distance: int32,
    horizontal_end_distance: int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_pressure_gradient_downward_extrapolation_mask_distance(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        topography=topography,
        flat_idx_max=flat_idx_max,
        e_owner_mask=e_owner_mask,
        e_lev=e_lev,
        k_lev=k_lev,
        horizontal_start_distance=horizontal_start_distance,
        horizontal_end_distance=horizontal_end_distance,
        out=(pg_edgeidx_dsl, pg_exdist_dsl),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[gtx.int32], mask_prog_halo_c: fa.CellField[bool]
) -> fa.CellField[bool]:
    mask_prog_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), mask_prog_halo_c, True)
    return mask_prog_halo_c


# TODO (@halungge) not registered in factory
@program(grid_type=GridType.UNSTRUCTURED)
def compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[gtx.int32],
    mask_prog_halo_c: fa.CellField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
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
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_bdy_halo_c(
    c_refin_ctrl: fa.CellField[int32],
) -> fa.CellField[bool]:
    bdy_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), True, False)
    return bdy_halo_c


# TODO (@halungge) not registered in factory
@program(grid_type=GridType.UNSTRUCTURED)
def compute_bdy_halo_c(
    c_refin_ctrl: fa.CellField[gtx.int32],
    bdy_halo_c: fa.CellField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute bdy_halo_c.

    See mo_vertical_grid.f90. bdy_halo_c_dsl_low_refin in ICON

    Args:
        c_refin_ctrl: Cell field of refin_ctrl
        bdy_halo_c: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_bdy_halo_c(
        c_refin_ctrl,
        out=bdy_halo_c,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@program(grid_type=GridType.UNSTRUCTURED)
def compute_mask_bdy_halo_c(
    c_refin_ctrl: fa.CellField[int32],
    mask_prog_halo_c: fa.CellField[bool],
    bdy_halo_c: fa.CellField[bool],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute bdy_halo_c.
    Compute mask_prog_halo_c.


    See mo_vertical_grid.f90. bdy_halo_c_dsl_low_refin in ICON

    Args:
        c_refin_ctrl: Cell field of refin_ctrl
        bdy_halo_c: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_mask_prog_halo_c(
        c_refin_ctrl,
        mask_prog_halo_c,
        out=mask_prog_halo_c,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )

    _compute_bdy_halo_c(
        c_refin_ctrl,
        out=bdy_halo_c,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_horizontal_mask_for_3d_divdamp(
    e_refin_ctrl: fa.EdgeField[gtx.int32],
    grf_nudge_start_e: gtx.int32,
    grf_nudgezone_width: gtx.int32,
) -> fa.EdgeField[wpfloat]:
    e_refin_ctrl_wp = astype(e_refin_ctrl, wpfloat)
    grf_nudge_start_e_wp = astype(grf_nudge_start_e, wpfloat)
    grf_nudgezone_width_wp = astype(grf_nudgezone_width, wpfloat)
    horizontal_mask_for_3d_divdamp = where(
        (e_refin_ctrl > (grf_nudge_start_e + grf_nudgezone_width - 1)),
        1.0
        / (grf_nudgezone_width_wp - 1.0)
        * (e_refin_ctrl_wp - (grf_nudge_start_e_wp + grf_nudgezone_width_wp - 1.0)),
        0.0,
    )
    horizontal_mask_for_3d_divdamp = where(
        (e_refin_ctrl <= 0)
        | (e_refin_ctrl_wp >= (grf_nudge_start_e_wp + 2.0 * (grf_nudgezone_width_wp - 1.0))),
        1.0,
        horizontal_mask_for_3d_divdamp,
    )
    return horizontal_mask_for_3d_divdamp


@program(grid_type=GridType.UNSTRUCTURED)
def compute_horizontal_mask_for_3d_divdamp(
    e_refin_ctrl: fa.EdgeField[gtx.int32],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[wpfloat],
    grf_nudge_start_e: gtx.int32,
    grf_nudgezone_width: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute horizontal_mask_for_3d_divdamp (declared as hmask_dd3d in ICON).

    See mo_vertical_grid.f90. Horizontal mask field for 3D divergence damping term.

    Args:
        e_refin_ctrl: Edge field of refin_ctrl
        horizontal_mask_for_3d_divdamp: output
        grf_nudge_start_e: mo_impl_constants_grf constant
        grf_nudgezone_width: mo_impl_constants_grf constant
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_horizontal_mask_for_3d_divdamp(
        e_refin_ctrl=e_refin_ctrl,
        grf_nudge_start_e=grf_nudge_start_e,
        grf_nudgezone_width=grf_nudgezone_width,
        out=horizontal_mask_for_3d_divdamp,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_weighted_cell_neighbor_sum(
    field: fa.CellKField[wpfloat],
    c_bln_avg: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> fa.CellKField[wpfloat]:
    field_avg = neighbor_sum(field(C2E2CO) * c_bln_avg, axis=C2E2CODim)
    return field_avg


@program(grid_type=GridType.UNSTRUCTURED)
def compute_weighted_cell_neighbor_sum(
    maxslp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    c_bln_avg: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    maxslp_avg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd_avg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute maxslp_avg and maxhgtd_avg.

    See mo_vertical_grid.f90.

    Args:
        maxslp: Max field over ddxn_z_full offset
        maxhgtd: Max field over ddxn_z_full offset*dual_edge_length offset
        c_bln_avg: Interpolation field
        maxslp_avg: output
        maxhgtd_avg: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_weighted_cell_neighbor_sum(
        field=maxslp,
        c_bln_avg=c_bln_avg,
        out=maxslp_avg,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

    _compute_weighted_cell_neighbor_sum(
        field=maxhgtd,
        c_bln_avg=c_bln_avg,
        out=maxhgtd_avg,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_max_nbhgt(
    z_mc_nlev: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    max_nbhgt_0_1 = maximum(z_mc_nlev(C2E2C[0]), z_mc_nlev(C2E2C[1]))
    max_nbhgt = maximum(max_nbhgt_0_1, z_mc_nlev(C2E2C[2]))
    return max_nbhgt


@program(grid_type=GridType.UNSTRUCTURED)
def compute_max_nbhgt(
    z_mc_nlev: fa.CellField[wpfloat],
    max_nbhgt: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
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
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@scan_operator(axis=dims.KDim, forward=True, init=(0, False))
def _compute_param(
    param: tuple[gtx.int32, bool],
    z_me_jk: float,
    z_ifc_off: float,
    z_ifc_off_koff: float,
    lower: gtx.int32,
    nlev: gtx.int32,
) -> tuple[gtx.int32, bool]:
    param_0, param_1 = param
    if param_0 >= lower:
        if (param_0 == nlev) | (z_me_jk <= z_ifc_off) & (z_me_jk >= z_ifc_off_koff):
            param_1 = True
    return param_0 + 1, param_1


@field_operator(grid_type=GridType.UNSTRUCTURED)
def _compute_z_ifc_off_koff(
    z_ifc_off: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    n = z_ifc_off(Koff[1])
    return n


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
    z_aux1 = p0sl_bg * exp(
        -grav
        / rd
        * h_scal_bg
        / (t0sl_bg - del_t_bg)
        * log((exp(z_mc / h_scal_bg) * (t0sl_bg - del_t_bg) + del_t_bg) / t0sl_bg)
    )
    exner_ref_mc = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = (t0sl_bg - del_t_bg) + del_t_bg * exp(-z_mc / h_scal_bg)
    theta_ref_mc = z_temp / exner_ref_mc
    return exner_ref_mc, theta_ref_mc


# TODO @halungge: duplicate program - see reference_atmosphere.py
@program(grid_type=GridType.UNSTRUCTURED)
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def compute_exner_w_implicit_weight_parameter(
    c2e: data_alloc.NDArray,
    vct_a: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    z_ddxn_z_half_e: data_alloc.NDArray,
    z_ddxt_z_half_e: data_alloc.NDArray,
    dual_edge_length: data_alloc.NDArray,
    vwind_offctr: float,
    nlev: int,
    horizontal_start_cell: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    factor = max(vwind_offctr, 0.75)

    zn_off = array_ns.abs(z_ddxn_z_half_e[:, nlev][c2e])
    zt_off = array_ns.abs(z_ddxt_z_half_e[:, nlev][c2e])
    stacked = array_ns.concatenate((zn_off, zt_off), axis=1)
    maxslope = 0.425 * array_ns.amax(stacked, axis=1) ** (0.75)
    diff = array_ns.minimum(
        0.25, 0.00025 * (np.amax(np.abs(zn_off * dual_edge_length[c2e]), axis=1) - 250.0)
    )
    offctr = array_ns.minimum(
        factor, array_ns.maximum(vwind_offctr, array_ns.maximum(maxslope, diff))
    )
    exner_w_implicit_weight_parameter = 0.5 + offctr

    k_start = max(0, nlev - 9)

    zdiff2 = (z_ifc[:, 0:nlev] - z_ifc[:, 1 : nlev + 1]) / (vct_a[0:nlev] - vct_a[1 : nlev + 1])

    for jk in range(k_start, nlev):
        zdiff2_sliced = zdiff2[horizontal_start_cell:, jk]
        index_for_k = np.where(zdiff2_sliced < 0.6)[0]
        max_value_k = np.maximum(
            1.2 - zdiff2_sliced, exner_w_implicit_weight_parameter[horizontal_start_cell:]
        )
        exner_w_implicit_weight_parameter[index_for_k + horizontal_start_cell] = max_value_k[
            index_for_k
        ]

    return exner_w_implicit_weight_parameter
