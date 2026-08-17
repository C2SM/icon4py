# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C, C2E2CDim


# Linear WENO reconstruction (ihadv_tracer=102) over the 3-point C2E2C stencil
# (mo_advection_hflux.f90 upwind_hflux_miura_weno, 983-1027). Per cell it blends 3
# linear least-squares candidates -- candidate i uses the pseudoinverse that drops
# direct neighbour i -- by the inverse-square smoothness weight of each candidate's
# gradient magnitude. z_b is the increment of the neighbour cell averages relative
# to the center cell; each candidate pseudoinverse is a slice of the Task-1
# (n_cells, 3, 2, 3) array, split into zonal (component 0) and meridional
# (component 1) rows over C2E2C. The conservative branch (llsq_lin_consv, f90
# 1005-1013) is intentionally not ported (Fortran default off), so the constant
# coefficient z_lsq_coeff(1) equals p_cc for every candidate and its smoothness
# blend collapses to p_cc exactly.


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
    # f90 991-993: z_b = p_cc(neighbour) - p_cc(center) on the C2E2C rows
    z_b = p_cc(C2E2C) - p_cc

    # f90 995-1017: per-candidate zonal/meridional gradient and inverse-square
    # smoothness weight s = 1 / ((cx**2 + cy**2) + 1e-20)**2
    cx_1 = neighbor_sum(lsq_pseudoinv_zonal_c1 * z_b, axis=C2E2CDim)
    cy_1 = neighbor_sum(lsq_pseudoinv_meridional_c1 * z_b, axis=C2E2CDim)
    denom_1 = (cx_1 * cx_1 + cy_1 * cy_1) + 1.0e-20
    s_1 = 1.0 / (denom_1 * denom_1)

    cx_2 = neighbor_sum(lsq_pseudoinv_zonal_c2 * z_b, axis=C2E2CDim)
    cy_2 = neighbor_sum(lsq_pseudoinv_meridional_c2 * z_b, axis=C2E2CDim)
    denom_2 = (cx_2 * cx_2 + cy_2 * cy_2) + 1.0e-20
    s_2 = 1.0 / (denom_2 * denom_2)

    cx_3 = neighbor_sum(lsq_pseudoinv_zonal_c3 * z_b, axis=C2E2CDim)
    cy_3 = neighbor_sum(lsq_pseudoinv_meridional_c3 * z_b, axis=C2E2CDim)
    denom_3 = (cx_3 * cx_3 + cy_3 * cy_3) + 1.0e-20
    s_3 = 1.0 / (denom_3 * denom_3)

    # f90 1018-1019: smoothness-weighted average over the 3 candidates
    smooth_sum = s_1 + s_2 + s_3
    p_coeff_1_dsl = p_cc
    p_coeff_2_dsl = (cx_1 * s_1 + cx_2 * s_2 + cx_3 * s_3) / smooth_sum
    p_coeff_3_dsl = (cy_1 * s_1 + cy_2 * s_2 + cy_3 * s_3) / smooth_sum

    return p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl


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
