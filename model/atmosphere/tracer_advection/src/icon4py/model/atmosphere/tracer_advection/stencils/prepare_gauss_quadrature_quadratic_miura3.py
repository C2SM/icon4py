# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


# Quadrature vector for the quadratic reconstruction of the miura3 WENO scheme: port of
# prep_gauss_quadrature_q_miura3 (mo_advection_quadrature.f90 561-703). The departure region is
# a parallelogram, so the Jacobian of the bilinear mapping is constant and equals a quarter of
# its area; with the weights z_wgt = 0.25 * wgt the quadrature directly yields the AREA AVERAGE
# of the monomials {1, x, y, x^2, y^2, xy} - hence, unlike the plain and cubic variants, no
# Jacobian determinant and no p_dreg_area output. The shape functions and Gauss weights are the
# gaussq_2d_o2 setup (init_2D_gauss_quad, mo_advection_config.f90 1084-1136), passed as scalars
# like in prepare_numerical_quadrature_for_cubic_reconstruction.


@gtx.field_operator
def _prepare_gauss_quadrature_quadratic_miura3(
    p_coords_dreg_v_1_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[ta.vpfloat],
    shape_func_1_1: ta.wpfloat,
    shape_func_2_1: ta.wpfloat,
    shape_func_3_1: ta.wpfloat,
    shape_func_4_1: ta.wpfloat,
    shape_func_1_2: ta.wpfloat,
    shape_func_2_2: ta.wpfloat,
    shape_func_3_2: ta.wpfloat,
    shape_func_4_2: ta.wpfloat,
    shape_func_1_3: ta.wpfloat,
    shape_func_2_3: ta.wpfloat,
    shape_func_3_3: ta.wpfloat,
    shape_func_4_3: ta.wpfloat,
    shape_func_1_4: ta.wpfloat,
    shape_func_2_4: ta.wpfloat,
    shape_func_3_4: ta.wpfloat,
    shape_func_4_4: ta.wpfloat,
    wgt_zeta_1: ta.wpfloat,
    wgt_zeta_2: ta.wpfloat,
    wgt_eta_1: ta.wpfloat,
    wgt_eta_2: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    # f90 641: z_wgt(1:4) = 0.25 * gq%wgt(1:4), with the wgt_zeta/wgt_eta flattening of
    # mo_advection_config.f90 1127-1136
    z_wgt_1 = 0.25 * wgt_zeta_1 * wgt_eta_1
    z_wgt_2 = 0.25 * wgt_zeta_1 * wgt_eta_2
    z_wgt_3 = 0.25 * wgt_zeta_2 * wgt_eta_1
    z_wgt_4 = 0.25 * wgt_zeta_2 * wgt_eta_2

    p_coords_dreg_v_1_x_wp = astype(p_coords_dreg_v_1_x, wpfloat)
    p_coords_dreg_v_2_x_wp = astype(p_coords_dreg_v_2_x, wpfloat)
    p_coords_dreg_v_3_x_wp = astype(p_coords_dreg_v_3_x, wpfloat)
    p_coords_dreg_v_4_x_wp = astype(p_coords_dreg_v_4_x, wpfloat)
    p_coords_dreg_v_1_y_wp = astype(p_coords_dreg_v_1_y, wpfloat)
    p_coords_dreg_v_2_y_wp = astype(p_coords_dreg_v_2_y, wpfloat)
    p_coords_dreg_v_3_y_wp = astype(p_coords_dreg_v_3_y, wpfloat)
    p_coords_dreg_v_4_y_wp = astype(p_coords_dreg_v_4_y, wpfloat)

    # f90 667-668: coordinates of the Gauss points in physical space (bilinear mapping)
    z_gauss_pts_1_x = (
        shape_func_1_1 * p_coords_dreg_v_1_x_wp
        + shape_func_2_1 * p_coords_dreg_v_2_x_wp
        + shape_func_3_1 * p_coords_dreg_v_3_x_wp
        + shape_func_4_1 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_1_y = (
        shape_func_1_1 * p_coords_dreg_v_1_y_wp
        + shape_func_2_1 * p_coords_dreg_v_2_y_wp
        + shape_func_3_1 * p_coords_dreg_v_3_y_wp
        + shape_func_4_1 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_2_x = (
        shape_func_1_2 * p_coords_dreg_v_1_x_wp
        + shape_func_2_2 * p_coords_dreg_v_2_x_wp
        + shape_func_3_2 * p_coords_dreg_v_3_x_wp
        + shape_func_4_2 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_2_y = (
        shape_func_1_2 * p_coords_dreg_v_1_y_wp
        + shape_func_2_2 * p_coords_dreg_v_2_y_wp
        + shape_func_3_2 * p_coords_dreg_v_3_y_wp
        + shape_func_4_2 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_3_x = (
        shape_func_1_3 * p_coords_dreg_v_1_x_wp
        + shape_func_2_3 * p_coords_dreg_v_2_x_wp
        + shape_func_3_3 * p_coords_dreg_v_3_x_wp
        + shape_func_4_3 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_3_y = (
        shape_func_1_3 * p_coords_dreg_v_1_y_wp
        + shape_func_2_3 * p_coords_dreg_v_2_y_wp
        + shape_func_3_3 * p_coords_dreg_v_3_y_wp
        + shape_func_4_3 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_4_x = (
        shape_func_1_4 * p_coords_dreg_v_1_x_wp
        + shape_func_2_4 * p_coords_dreg_v_2_x_wp
        + shape_func_3_4 * p_coords_dreg_v_3_x_wp
        + shape_func_4_4 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_4_y = (
        shape_func_1_4 * p_coords_dreg_v_1_y_wp
        + shape_func_2_4 * p_coords_dreg_v_2_y_wp
        + shape_func_3_4 * p_coords_dreg_v_3_y_wp
        + shape_func_4_4 * p_coords_dreg_v_4_y_wp
    )

    # f90 670-689: quadrature vector {1, x, y, x^2, y^2, xy} per Gauss point, weighted and
    # summed over the 4 points
    p_quad_vector_sum_1 = z_wgt_1 + z_wgt_2 + z_wgt_3 + z_wgt_4
    p_quad_vector_sum_2 = (
        z_wgt_1 * z_gauss_pts_1_x
        + z_wgt_2 * z_gauss_pts_2_x
        + z_wgt_3 * z_gauss_pts_3_x
        + z_wgt_4 * z_gauss_pts_4_x
    )
    p_quad_vector_sum_3 = (
        z_wgt_1 * z_gauss_pts_1_y
        + z_wgt_2 * z_gauss_pts_2_y
        + z_wgt_3 * z_gauss_pts_3_y
        + z_wgt_4 * z_gauss_pts_4_y
    )
    p_quad_vector_sum_4 = (
        z_wgt_1 * z_gauss_pts_1_x * z_gauss_pts_1_x
        + z_wgt_2 * z_gauss_pts_2_x * z_gauss_pts_2_x
        + z_wgt_3 * z_gauss_pts_3_x * z_gauss_pts_3_x
        + z_wgt_4 * z_gauss_pts_4_x * z_gauss_pts_4_x
    )
    p_quad_vector_sum_5 = (
        z_wgt_1 * z_gauss_pts_1_y * z_gauss_pts_1_y
        + z_wgt_2 * z_gauss_pts_2_y * z_gauss_pts_2_y
        + z_wgt_3 * z_gauss_pts_3_y * z_gauss_pts_3_y
        + z_wgt_4 * z_gauss_pts_4_y * z_gauss_pts_4_y
    )
    p_quad_vector_sum_6 = (
        z_wgt_1 * z_gauss_pts_1_x * z_gauss_pts_1_y
        + z_wgt_2 * z_gauss_pts_2_x * z_gauss_pts_2_y
        + z_wgt_3 * z_gauss_pts_3_x * z_gauss_pts_3_y
        + z_wgt_4 * z_gauss_pts_4_x * z_gauss_pts_4_y
    )

    return (
        astype(broadcast(p_quad_vector_sum_1, (dims.EdgeDim, dims.KDim)), vpfloat),
        astype(p_quad_vector_sum_2, vpfloat),
        astype(p_quad_vector_sum_3, vpfloat),
        astype(p_quad_vector_sum_4, vpfloat),
        astype(p_quad_vector_sum_5, vpfloat),
        astype(p_quad_vector_sum_6, vpfloat),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_gauss_quadrature_quadratic_miura3(
    p_coords_dreg_v_1_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    shape_func_1_1: ta.wpfloat,
    shape_func_2_1: ta.wpfloat,
    shape_func_3_1: ta.wpfloat,
    shape_func_4_1: ta.wpfloat,
    shape_func_1_2: ta.wpfloat,
    shape_func_2_2: ta.wpfloat,
    shape_func_3_2: ta.wpfloat,
    shape_func_4_2: ta.wpfloat,
    shape_func_1_3: ta.wpfloat,
    shape_func_2_3: ta.wpfloat,
    shape_func_3_3: ta.wpfloat,
    shape_func_4_3: ta.wpfloat,
    shape_func_1_4: ta.wpfloat,
    shape_func_2_4: ta.wpfloat,
    shape_func_3_4: ta.wpfloat,
    shape_func_4_4: ta.wpfloat,
    wgt_zeta_1: ta.wpfloat,
    wgt_zeta_2: ta.wpfloat,
    wgt_eta_1: ta.wpfloat,
    wgt_eta_2: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_gauss_quadrature_quadratic_miura3(
        p_coords_dreg_v_1_x=p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x=p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x=p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x=p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y=p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y=p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y=p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y=p_coords_dreg_v_4_y,
        shape_func_1_1=shape_func_1_1,
        shape_func_2_1=shape_func_2_1,
        shape_func_3_1=shape_func_3_1,
        shape_func_4_1=shape_func_4_1,
        shape_func_1_2=shape_func_1_2,
        shape_func_2_2=shape_func_2_2,
        shape_func_3_2=shape_func_3_2,
        shape_func_4_2=shape_func_4_2,
        shape_func_1_3=shape_func_1_3,
        shape_func_2_3=shape_func_2_3,
        shape_func_3_3=shape_func_3_3,
        shape_func_4_3=shape_func_4_3,
        shape_func_1_4=shape_func_1_4,
        shape_func_2_4=shape_func_2_4,
        shape_func_3_4=shape_func_3_4,
        shape_func_4_4=shape_func_4_4,
        wgt_zeta_1=wgt_zeta_1,
        wgt_zeta_2=wgt_zeta_2,
        wgt_eta_1=wgt_eta_1,
        wgt_eta_2=wgt_eta_2,
        out=(
            p_quad_vector_sum_1,
            p_quad_vector_sum_2,
            p_quad_vector_sum_3,
            p_quad_vector_sum_4,
            p_quad_vector_sum_5,
            p_quad_vector_sum_6,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
