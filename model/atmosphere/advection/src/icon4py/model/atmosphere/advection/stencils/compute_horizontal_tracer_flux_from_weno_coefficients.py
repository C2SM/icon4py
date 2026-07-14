# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


# Final flux of the miura3 WENO scheme (mo_advection_hflux.f90 2514-2521): the candidate
# coefficients accumulated by accumulate_weno_candidate_flux_weights are normalized by the
# accumulated smoothness weights and dotted with the quadrature vector, then multiplied by the
# mass flux. The quadrature vector holds the departure-region AREA AVERAGES of the monomials
# (prepare_gauss_quadrature_quadratic_miura3), so there is no division by the region area.


@gtx.field_operator
def _compute_horizontal_tracer_flux_from_weno_coefficients(
    z_lsq_weighted_1: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_2: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_3: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_4: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_5: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_6: fa.EdgeKField[ta.wpfloat],
    smooth_sum: fa.EdgeKField[ta.wpfloat],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    # f90 2514: z_lsq_weighted(1:6,ie) = z_lsq_weighted(1:6,ie) / smooth_sum(ie); the division
    # happens on the coefficients, before the dot product
    c_1 = z_lsq_weighted_1 / smooth_sum
    c_2 = z_lsq_weighted_2 / smooth_sum
    c_3 = z_lsq_weighted_3 / smooth_sum
    c_4 = z_lsq_weighted_4 / smooth_sum
    c_5 = z_lsq_weighted_5 / smooth_sum
    c_6 = z_lsq_weighted_6 / smooth_sum

    # f90 2516-2518: DOT_PRODUCT(z_lsq_weighted(1:6), z_quad_vector_sum(1:6)) * p_mass_flx_e
    p_out_e = (
        c_1 * astype(p_quad_vector_sum_1, wpfloat)
        + c_2 * astype(p_quad_vector_sum_2, wpfloat)
        + c_3 * astype(p_quad_vector_sum_3, wpfloat)
        + c_4 * astype(p_quad_vector_sum_4, wpfloat)
        + c_5 * astype(p_quad_vector_sum_5, wpfloat)
        + c_6 * astype(p_quad_vector_sum_6, wpfloat)
    ) * p_mass_flx_e

    return p_out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_tracer_flux_from_weno_coefficients(
    z_lsq_weighted_1: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_2: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_3: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_4: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_5: fa.EdgeKField[ta.wpfloat],
    z_lsq_weighted_6: fa.EdgeKField[ta.wpfloat],
    smooth_sum: fa.EdgeKField[ta.wpfloat],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_tracer_flux_from_weno_coefficients(
        z_lsq_weighted_1=z_lsq_weighted_1,
        z_lsq_weighted_2=z_lsq_weighted_2,
        z_lsq_weighted_3=z_lsq_weighted_3,
        z_lsq_weighted_4=z_lsq_weighted_4,
        z_lsq_weighted_5=z_lsq_weighted_5,
        z_lsq_weighted_6=z_lsq_weighted_6,
        smooth_sum=smooth_sum,
        p_quad_vector_sum_1=p_quad_vector_sum_1,
        p_quad_vector_sum_2=p_quad_vector_sum_2,
        p_quad_vector_sum_3=p_quad_vector_sum_3,
        p_quad_vector_sum_4=p_quad_vector_sum_4,
        p_quad_vector_sum_5=p_quad_vector_sum_5,
        p_quad_vector_sum_6=p_quad_vector_sum_6,
        p_mass_flx_e=p_mass_flx_e,
        out=p_out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
