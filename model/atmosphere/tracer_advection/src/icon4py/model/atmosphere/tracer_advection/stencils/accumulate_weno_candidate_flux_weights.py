# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import fortran_sp_float, wpfloat


# f90 3008: literal `1d-20` regularization added to the smoothness before squaring. The gtfn
# backend does not fold a module-level constant referenced inside a field operator into the IR
# ("Symbols not found"), so the field operator inlines this literal; the tests import _WENO_EPS
# so both use one value.
_WENO_EPS = 1e-20


# WENO smoothness weighting for one of the 27 candidate stencils (mo_advection_hflux.f90
# 2996-3011, upwind_hflux_miura3_weno; the hybrid's WENO branch, 3672-3687, is the same with
# l_weights_s = 1). The Fortran loops over cells and scatters to the three edges owned by the upwind
# cell; here each edge gathers the candidate coefficients (and the area) of its upwind cell,
# selected by p_cell_rel_idx_dsl (0 or 1) into E2C. The weighted sums z_lsq_weighted and
# smooth_sum are accumulated over the 27 candidates, so the accumulators are read and written.


@gtx.field_operator
def _accumulate_weno_candidate_flux_weights(
    p_coeff_1: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    cell_area: fa.CellField[ta.wpfloat],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    z_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    z_lsq_weighted_1: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_2: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_3: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_4: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_5: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_6: fa.EdgeKField[ta.wpfloat],
    smooth_sum: fa.EdgeKField[ta.wpfloat],
    l_weight_s: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
]:
    # gather the upwind cell's coefficients and area onto the edge (f90 backward trajectory:
    # ptr_ilc/ptr_ibc select the upwind cell, mirrored by p_cell_rel_idx_dsl into E2C)
    c1 = where(p_cell_rel_idx_dsl == 1, p_coeff_1(E2C[1]), p_coeff_1(E2C[0]))
    c2 = where(p_cell_rel_idx_dsl == 1, p_coeff_2(E2C[1]), p_coeff_2(E2C[0]))
    c3 = where(p_cell_rel_idx_dsl == 1, p_coeff_3(E2C[1]), p_coeff_3(E2C[0]))
    c4 = where(p_cell_rel_idx_dsl == 1, p_coeff_4(E2C[1]), p_coeff_4(E2C[0]))
    c5 = where(p_cell_rel_idx_dsl == 1, p_coeff_5(E2C[1]), p_coeff_5(E2C[0]))
    c6 = where(p_cell_rel_idx_dsl == 1, p_coeff_6(E2C[1]), p_coeff_6(E2C[0]))
    area = where(p_cell_rel_idx_dsl == 1, cell_area(E2C[1]), cell_area(E2C[0]))

    # smoothness vector (f90 2996-3005) in the Fortran's REAL(sp): zlc(1:6) = z_lsq_coeff(1:6)
    # and area are single precision there (f90 2643, 2956), and the literal 2e0 is a default-
    # real (single) literal; fortran_sp_float stands in for that kind (type_alias). Unknowns
    # [c0, x, y, x^2, y^2, xy]: smooth_2/3/6 use the raw c4/c5/c6, the rest their squares (f90
    # squares zlc(4:6) in place at 3000-3002, i.e. after smooth_2/3/6 and before smooth_4/5/1).
    zlc_2 = astype(c2, fortran_sp_float)
    zlc_3 = astype(c3, fortran_sp_float)
    zlc_4 = astype(c4, fortran_sp_float)
    zlc_5 = astype(c5, fortran_sp_float)
    zlc_6 = astype(c6, fortran_sp_float)
    area_sp = astype(area, fortran_sp_float)
    smooth_2 = fortran_sp_float(2.0) * (zlc_2 * zlc_4 + zlc_3 * zlc_6)
    smooth_3 = fortran_sp_float(2.0) * (zlc_2 * zlc_6 + zlc_3 * zlc_5)
    smooth_6 = fortran_sp_float(2.0) * zlc_6 * (zlc_4 + zlc_5)
    zlc_4_sq = zlc_4 * zlc_4
    zlc_5_sq = zlc_5 * zlc_5
    zlc_6_sq = zlc_6 * zlc_6
    smooth_4 = fortran_sp_float(2.0) * (zlc_4_sq + zlc_6_sq)
    smooth_5 = fortran_sp_float(2.0) * (zlc_5_sq + zlc_6_sq)
    smooth_1 = zlc_2 * zlc_2 + zlc_3 * zlc_3 + area_sp * (zlc_4_sq + zlc_5_sq + zlc_6_sq)

    # f90 3007: DOT_PRODUCT(z_lsq_smooth, real(z_quad_vector_sum)) is a single-precision dot
    # product (z_quad_vector_sum rounded to default real), assigned to the REAL(wp) smoothness
    beta_sp = (
        smooth_1 * astype(z_quad_vector_sum_1, fortran_sp_float)
        + smooth_2 * astype(z_quad_vector_sum_2, fortran_sp_float)
        + smooth_3 * astype(z_quad_vector_sum_3, fortran_sp_float)
        + smooth_4 * astype(z_quad_vector_sum_4, fortran_sp_float)
        + smooth_5 * astype(z_quad_vector_sum_5, fortran_sp_float)
        + smooth_6 * astype(z_quad_vector_sum_6, fortran_sp_float)
    )
    beta = astype(beta_sp, wpfloat)
    # f90 3008: smoothness = l_weights_s / (smoothness + 1d-20)**2 in REAL(wp): the 1d-20 is a
    # double literal and the square is taken of the double sum (written as a product: x**2 is
    # a pow call on GPU). 1e-20 == _WENO_EPS (see note above); wpfloat(...) folds in gtfn, a
    # module symbol does not
    w = l_weight_s / ((beta + wpfloat(1e-20)) * (beta + wpfloat(1e-20)))

    # f90 3009-3010: accumulate weighted coefficients and weights over the candidates
    return (
        z_lsq_weighted_1 + c1 * w,
        z_lsq_weighted_2 + c2 * w,
        z_lsq_weighted_3 + c3 * w,
        z_lsq_weighted_4 + c4 * w,
        z_lsq_weighted_5 + c5 * w,
        z_lsq_weighted_6 + c6 * w,
        smooth_sum + w,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def accumulate_weno_candidate_flux_weights(
    p_coeff_1: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    cell_area: fa.CellField[ta.wpfloat],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    z_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    z_lsq_weighted_1: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_2: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_3: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_4: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_5: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_6: fa.EdgeKField[ta.wpfloat],
    smooth_sum: fa.EdgeKField[ta.wpfloat],
    l_weight_s: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _accumulate_weno_candidate_flux_weights(
        p_coeff_1=p_coeff_1,
        p_coeff_2=p_coeff_2,
        p_coeff_3=p_coeff_3,
        p_coeff_4=p_coeff_4,
        p_coeff_5=p_coeff_5,
        p_coeff_6=p_coeff_6,
        cell_area=cell_area,
        p_cell_rel_idx_dsl=p_cell_rel_idx_dsl,
        z_quad_vector_sum_1=z_quad_vector_sum_1,
        z_quad_vector_sum_2=z_quad_vector_sum_2,
        z_quad_vector_sum_3=z_quad_vector_sum_3,
        z_quad_vector_sum_4=z_quad_vector_sum_4,
        z_quad_vector_sum_5=z_quad_vector_sum_5,
        z_quad_vector_sum_6=z_quad_vector_sum_6,
        z_lsq_weighted_1=z_lsq_weighted_1,
        z_lsq_weighted_2=z_lsq_weighted_2,
        z_lsq_weighted_3=z_lsq_weighted_3,
        z_lsq_weighted_4=z_lsq_weighted_4,
        z_lsq_weighted_5=z_lsq_weighted_5,
        z_lsq_weighted_6=z_lsq_weighted_6,
        smooth_sum=smooth_sum,
        l_weight_s=l_weight_s,
        out=(
            z_lsq_weighted_1,
            z_lsq_weighted_2,
            z_lsq_weighted_3,
            z_lsq_weighted_4,
            z_lsq_weighted_5,
            z_lsq_weighted_6,
            smooth_sum,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
