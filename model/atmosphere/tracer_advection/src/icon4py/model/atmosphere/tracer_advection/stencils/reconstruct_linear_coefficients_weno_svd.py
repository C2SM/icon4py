# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import gt4py.next as gtx
from gt4py.next import minimum, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C, C2E2CDim
from icon4py.model.common.type_alias import wpfloat


#: p of alpha_j = d_j / (beta_j + eps)^p
DEFAULT_WENO_SMOOTHNESS_EXPONENT: Final[wpfloat] = wpfloat(2.0)


# Linear WENO reconstruction (ihadv_tracer=102) over the 3-point C2E2C stencil
# (mo_advection_hflux.f90 upwind_hflux_miura_weno, 1488-1524). Per cell it blends 3
# linear least-squares candidates -- candidate i uses the pseudoinverse that drops
# direct neighbour i -- by the inverse-square smoothness weight of each candidate's
# gradient magnitude. z_b is the increment of the neighbour cell averages relative
# to the center cell; each candidate pseudoinverse is a slice of the Task-1
# (n_cells, 3, 2, 3) array, split into zonal (component 0) and meridional
# (component 1) rows over C2E2C. The conservative branch (llsq_lin_consv, f90
# 1507-1514) is intentionally not ported (Fortran default off), so the constant
# coefficient z_lsq_coeff(1) equals p_cc for every candidate and its smoothness
# blend collapses to p_cc exactly.


@gtx.field_operator
def _weno_candidate_gradients(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # f90 1488-1490: z_b = p_cc(neighbour) - p_cc(center) on the C2E2C rows
    z_b = p_cc(C2E2C) - p_cc

    # f90 1494-1519: per-candidate zonal/meridional gradient
    cx_1 = neighbor_sum(lsq_pseudoinv_zonal_c1 * z_b, axis=C2E2CDim)
    cy_1 = neighbor_sum(lsq_pseudoinv_meridional_c1 * z_b, axis=C2E2CDim)
    beta_1 = cx_1 * cx_1 + cy_1 * cy_1
    denom_1 = beta_1 + wpfloat(1.0e-20)

    cx_2 = neighbor_sum(lsq_pseudoinv_zonal_c2 * z_b, axis=C2E2CDim)
    cy_2 = neighbor_sum(lsq_pseudoinv_meridional_c2 * z_b, axis=C2E2CDim)
    beta_2 = cx_2 * cx_2 + cy_2 * cy_2
    denom_2 = beta_2 + wpfloat(1.0e-20)

    cx_3 = neighbor_sum(lsq_pseudoinv_zonal_c3 * z_b, axis=C2E2CDim)
    cy_3 = neighbor_sum(lsq_pseudoinv_meridional_c3 * z_b, axis=C2E2CDim)
    beta_3 = cx_3 * cx_3 + cy_3 * cy_3
    denom_3 = beta_3 + wpfloat(1.0e-20)

    return cx_1, cy_1, denom_1, cx_2, cy_2, denom_2, cx_3, cy_3, denom_3


@gtx.field_operator
def _weno_blend(
    p_cc: fa.CellKField[ta.wpfloat],
    cx_1: fa.CellKField[ta.wpfloat],
    cy_1: fa.CellKField[ta.wpfloat],
    s_1: fa.CellKField[ta.wpfloat],
    cx_2: fa.CellKField[ta.wpfloat],
    cy_2: fa.CellKField[ta.wpfloat],
    s_2: fa.CellKField[ta.wpfloat],
    cx_3: fa.CellKField[ta.wpfloat],
    cy_3: fa.CellKField[ta.wpfloat],
    s_3: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """omega_j = alpha_j / sum_k alpha_k applied to the candidate gradients. This gives us the desired coefficients of the linear scheme"""
    smooth_sum = s_1 + s_2 + s_3
    p_coeff_1_dsl = p_cc
    p_coeff_2_dsl = (cx_1 * s_1 + cx_2 * s_2 + cx_3 * s_3) / smooth_sum
    p_coeff_3_dsl = (cy_1 * s_1 + cy_2 * s_2 + cy_3 * s_3) / smooth_sum
    return p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl


@gtx.field_operator
def _reconstruct_linear_coefficients_weno_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    cx_1, cy_1, denom_1, cx_2, cy_2, denom_2, cx_3, cy_3, denom_3 = _weno_candidate_gradients(
        p_cc,
        lsq_pseudoinv_zonal_c1,
        lsq_pseudoinv_zonal_c2,
        lsq_pseudoinv_zonal_c3,
        lsq_pseudoinv_meridional_c1,
        lsq_pseudoinv_meridional_c2,
        lsq_pseudoinv_meridional_c3,
    )

    s_1 = wpfloat(1.0) / (denom_1 * denom_1)
    s_2 = wpfloat(1.0) / (denom_2 * denom_2)
    s_3 = wpfloat(1.0) / (denom_3 * denom_3)

    return _weno_blend(p_cc, cx_1, cy_1, s_1, cx_2, cy_2, s_2, cx_3, cy_3, s_3)


@gtx.field_operator
def _reconstruct_linear_coefficients_weno_svd_exponent(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    smoothness_exponent: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    cx_1, cy_1, denom_1, cx_2, cy_2, denom_2, cx_3, cy_3, denom_3 = _weno_candidate_gradients(
        p_cc,
        lsq_pseudoinv_zonal_c1,
        lsq_pseudoinv_zonal_c2,
        lsq_pseudoinv_zonal_c3,
        lsq_pseudoinv_meridional_c1,
        lsq_pseudoinv_meridional_c2,
        lsq_pseudoinv_meridional_c3,
    )

    denom_min = minimum(denom_1, minimum(denom_2, denom_3))
    s_1 = (denom_min / denom_1) ** smoothness_exponent
    s_2 = (denom_min / denom_2) ** smoothness_exponent
    s_3 = (denom_min / denom_3) ** smoothness_exponent

    return _weno_blend(p_cc, cx_1, cy_1, s_1, cx_2, cy_2, s_2, cx_3, cy_3, s_3)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_linear_coefficients_weno_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_linear_coefficients_weno_svd(
        p_cc=p_cc,
        lsq_pseudoinv_zonal_c1=lsq_pseudoinv_zonal_c1,
        lsq_pseudoinv_zonal_c2=lsq_pseudoinv_zonal_c2,
        lsq_pseudoinv_zonal_c3=lsq_pseudoinv_zonal_c3,
        lsq_pseudoinv_meridional_c1=lsq_pseudoinv_meridional_c1,
        lsq_pseudoinv_meridional_c2=lsq_pseudoinv_meridional_c2,
        lsq_pseudoinv_meridional_c3=lsq_pseudoinv_meridional_c3,
        out=(p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_linear_coefficients_weno_svd_exponent(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    smoothness_exponent: ta.wpfloat,
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_linear_coefficients_weno_svd_exponent(
        p_cc=p_cc,
        lsq_pseudoinv_zonal_c1=lsq_pseudoinv_zonal_c1,
        lsq_pseudoinv_zonal_c2=lsq_pseudoinv_zonal_c2,
        lsq_pseudoinv_zonal_c3=lsq_pseudoinv_zonal_c3,
        lsq_pseudoinv_meridional_c1=lsq_pseudoinv_meridional_c1,
        lsq_pseudoinv_meridional_c2=lsq_pseudoinv_meridional_c2,
        lsq_pseudoinv_meridional_c3=lsq_pseudoinv_meridional_c3,
        smoothness_exponent=smoothness_exponent,
        out=(p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
