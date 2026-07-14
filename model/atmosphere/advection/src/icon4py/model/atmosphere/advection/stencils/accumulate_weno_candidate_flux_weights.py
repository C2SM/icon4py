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
from icon4py.model.common.type_alias import wpfloat


# f90 2509: literal `1d-20` regularization added to the smoothness before squaring. The gtfn
# backend does not fold a module-level constant referenced inside a field operator into the IR
# ("Symbols not found"), so the field operator inlines this literal; the tests import _WENO_EPS
# so both use one value.
_WENO_EPS = 1e-20


# WENO smoothness weighting for one of the 27 candidate stencils (mo_advection_hflux.f90
# 2497-2512). The Fortran loops over cells and scatters to the three edges owned by the upwind
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

    # smoothness vector (f90 2497-2506); zlc == z_lsq_coeff, unknowns [c0, x, y, x^2, y^2, xy].
    # smooth_2/3/6 use the raw c4/c5/c6, the rest use their squares (f90 squares zlc(4:6) in
    # place at 2501-2503, i.e. after smooth_2/3/6 and before smooth_4/5/1).
    smooth_2 = 2.0 * (c2 * c4 + c3 * c6)
    smooth_3 = 2.0 * (c2 * c6 + c3 * c5)
    smooth_6 = 2.0 * c6 * (c4 + c5)
    c4_sq = c4 * c4
    c5_sq = c5 * c5
    c6_sq = c6 * c6
    smooth_4 = 2.0 * (c4_sq + c6_sq)
    smooth_5 = 2.0 * (c5_sq + c6_sq)
    smooth_1 = c2 * c2 + c3 * c3 + area * (c4_sq + c5_sq + c6_sq)

    # f90 2508-2509: smoothness = l_weights_s / (z_lsq_smooth . z_quad_vector_sum + eps)^2
    beta = (
        smooth_1 * astype(z_quad_vector_sum_1, wpfloat)
        + smooth_2 * astype(z_quad_vector_sum_2, wpfloat)
        + smooth_3 * astype(z_quad_vector_sum_3, wpfloat)
        + smooth_4 * astype(z_quad_vector_sum_4, wpfloat)
        + smooth_5 * astype(z_quad_vector_sum_5, wpfloat)
        + smooth_6 * astype(z_quad_vector_sum_6, wpfloat)
    )
    w = l_weight_s / ((beta + 1e-20) * (beta + 1e-20))  # 1e-20 == _WENO_EPS (see note above)

    # f90 2510-2511: accumulate weighted coefficients and weights over the candidates
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
