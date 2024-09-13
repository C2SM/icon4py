# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    abs,
    maximum,
    where,
)

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _prepare_numerical_quadrature_for_cubic_reconstruction(
    p_coords_dreg_v_1_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[vpfloat],
    shape_func_1_1: wpfloat,
    shape_func_2_1: wpfloat,
    shape_func_3_1: wpfloat,
    shape_func_4_1: wpfloat,
    shape_func_1_2: wpfloat,
    shape_func_2_2: wpfloat,
    shape_func_3_2: wpfloat,
    shape_func_4_2: wpfloat,
    shape_func_1_3: wpfloat,
    shape_func_2_3: wpfloat,
    shape_func_3_3: wpfloat,
    shape_func_4_3: wpfloat,
    shape_func_1_4: wpfloat,
    shape_func_2_4: wpfloat,
    shape_func_3_4: wpfloat,
    shape_func_4_4: wpfloat,
    zeta_1: wpfloat,
    zeta_2: wpfloat,
    zeta_3: wpfloat,
    zeta_4: wpfloat,
    eta_1: wpfloat,
    eta_2: wpfloat,
    eta_3: wpfloat,
    eta_4: wpfloat,
    wgt_zeta_1: wpfloat,
    wgt_zeta_2: wpfloat,
    wgt_eta_1: wpfloat,
    wgt_eta_2: wpfloat,
    dbl_eps: wpfloat,
    eps: wpfloat,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    z_wgt_1 = wpfloat(0.0625) * wgt_zeta_1 * wgt_eta_1
    z_wgt_2 = wpfloat(0.0625) * wgt_zeta_1 * wgt_eta_2
    z_wgt_3 = wpfloat(0.0625) * wgt_zeta_2 * wgt_eta_1
    z_wgt_4 = wpfloat(0.0625) * wgt_zeta_2 * wgt_eta_2

    z_eta_1_1 = wpfloat(1.0) - eta_1
    z_eta_2_1 = wpfloat(1.0) - eta_2
    z_eta_3_1 = wpfloat(1.0) - eta_3
    z_eta_4_1 = wpfloat(1.0) - eta_4
    z_eta_1_2 = wpfloat(1.0) + eta_1
    z_eta_2_2 = wpfloat(1.0) + eta_2
    z_eta_3_2 = wpfloat(1.0) + eta_3
    z_eta_4_2 = wpfloat(1.0) + eta_4
    z_eta_1_3 = wpfloat(1.0) - zeta_1
    z_eta_2_3 = wpfloat(1.0) - zeta_2
    z_eta_3_3 = wpfloat(1.0) - zeta_3
    z_eta_4_3 = wpfloat(1.0) - zeta_4
    z_eta_1_4 = wpfloat(1.0) + zeta_1
    z_eta_2_4 = wpfloat(1.0) + zeta_2
    z_eta_3_4 = wpfloat(1.0) + zeta_3
    z_eta_4_4 = wpfloat(1.0) + zeta_4

    wgt_t_detjac_1 = dbl_eps + z_wgt_1 * (
        (
            z_eta_1_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
            + z_eta_1_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
        )
        * (
            z_eta_1_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
            - z_eta_1_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
        )
        - (
            z_eta_1_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
            + z_eta_1_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
        )
        * (
            z_eta_1_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
            - z_eta_1_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
        )
    )
    wgt_t_detjac_2 = dbl_eps + z_wgt_2 * (
        (
            z_eta_2_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
            + z_eta_2_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
        )
        * (
            z_eta_2_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
            - z_eta_2_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
        )
        - (
            z_eta_2_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
            + z_eta_2_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
        )
        * (
            z_eta_2_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
            - z_eta_2_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
        )
    )
    wgt_t_detjac_3 = dbl_eps + z_wgt_3 * (
        (
            z_eta_3_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
            + z_eta_3_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
        )
        * (
            z_eta_3_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
            - z_eta_3_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
        )
        - (
            z_eta_3_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
            + z_eta_3_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
        )
        * (
            z_eta_3_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
            - z_eta_3_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
        )
    )
    wgt_t_detjac_4 = dbl_eps + z_wgt_4 * (
        (
            z_eta_4_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
            + z_eta_4_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
        )
        * (
            z_eta_4_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
            - z_eta_4_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
        )
        - (
            z_eta_4_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
            + z_eta_4_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
        )
        * (
            z_eta_4_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
            - z_eta_4_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
        )
    )

    z_gauss_pts_1_x = (
        shape_func_1_1 * p_coords_dreg_v_1_x
        + shape_func_2_1 * p_coords_dreg_v_2_x
        + shape_func_3_1 * p_coords_dreg_v_3_x
        + shape_func_4_1 * p_coords_dreg_v_4_x
    )
    z_gauss_pts_1_y = (
        shape_func_1_1 * p_coords_dreg_v_1_y
        + shape_func_2_1 * p_coords_dreg_v_2_y
        + shape_func_3_1 * p_coords_dreg_v_3_y
        + shape_func_4_1 * p_coords_dreg_v_4_y
    )
    z_gauss_pts_2_x = (
        shape_func_1_2 * p_coords_dreg_v_1_x
        + shape_func_2_2 * p_coords_dreg_v_2_x
        + shape_func_3_2 * p_coords_dreg_v_3_x
        + shape_func_4_2 * p_coords_dreg_v_4_x
    )
    z_gauss_pts_2_y = (
        shape_func_1_2 * p_coords_dreg_v_1_y
        + shape_func_2_2 * p_coords_dreg_v_2_y
        + shape_func_3_2 * p_coords_dreg_v_3_y
        + shape_func_4_2 * p_coords_dreg_v_4_y
    )
    z_gauss_pts_3_x = (
        shape_func_1_3 * p_coords_dreg_v_1_x
        + shape_func_2_3 * p_coords_dreg_v_2_x
        + shape_func_3_3 * p_coords_dreg_v_3_x
        + shape_func_4_3 * p_coords_dreg_v_4_x
    )
    z_gauss_pts_3_y = (
        shape_func_1_3 * p_coords_dreg_v_1_y
        + shape_func_2_3 * p_coords_dreg_v_2_y
        + shape_func_3_3 * p_coords_dreg_v_3_y
        + shape_func_4_3 * p_coords_dreg_v_4_y
    )
    z_gauss_pts_4_x = (
        shape_func_1_4 * p_coords_dreg_v_1_x
        + shape_func_2_4 * p_coords_dreg_v_2_x
        + shape_func_3_4 * p_coords_dreg_v_3_x
        + shape_func_4_4 * p_coords_dreg_v_4_x
    )
    z_gauss_pts_4_y = (
        shape_func_1_4 * p_coords_dreg_v_1_y
        + shape_func_2_4 * p_coords_dreg_v_2_y
        + shape_func_3_4 * p_coords_dreg_v_3_y
        + shape_func_4_4 * p_coords_dreg_v_4_y
    )

    p_quad_vector_sum_1 = wgt_t_detjac_1 + wgt_t_detjac_2 + wgt_t_detjac_3 + wgt_t_detjac_4
    p_quad_vector_sum_2 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x
        + wgt_t_detjac_2 * z_gauss_pts_2_x
        + wgt_t_detjac_3 * z_gauss_pts_3_x
        + wgt_t_detjac_4 * z_gauss_pts_4_x
    )
    p_quad_vector_sum_3 = (
        wgt_t_detjac_1 * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_y
    )
    p_quad_vector_sum_4 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x
        + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x
        + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x
        + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x
    )
    p_quad_vector_sum_5 = (
        wgt_t_detjac_1 * z_gauss_pts_1_y * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_y * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_y * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_y * z_gauss_pts_4_y
    )
    p_quad_vector_sum_6 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_y
    )
    p_quad_vector_sum_7 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x * z_gauss_pts_1_x
        + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x * z_gauss_pts_2_x
        + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x * z_gauss_pts_3_x
        + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x * z_gauss_pts_4_x
    )
    p_quad_vector_sum_8 = (
        wgt_t_detjac_1 * z_gauss_pts_1_y * z_gauss_pts_1_y * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_y * z_gauss_pts_2_y * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_y * z_gauss_pts_3_y * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_y * z_gauss_pts_4_y * z_gauss_pts_4_y
    )
    p_quad_vector_sum_9 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x * z_gauss_pts_4_y
    )
    p_quad_vector_sum_10 = (
        wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_y * z_gauss_pts_1_y
        + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_y * z_gauss_pts_2_y
        + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_y * z_gauss_pts_3_y
        + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_y * z_gauss_pts_4_y
    )

    z_area = p_quad_vector_sum_1
    p_dreg_area_out = where(
        z_area >= vpfloat(0.0), maximum(eps, abs(z_area)), -maximum(eps, abs(z_area))
    )

    return (
        p_quad_vector_sum_1,
        p_quad_vector_sum_2,
        p_quad_vector_sum_3,
        p_quad_vector_sum_4,
        p_quad_vector_sum_5,
        p_quad_vector_sum_6,
        p_quad_vector_sum_7,
        p_quad_vector_sum_8,
        p_quad_vector_sum_9,
        p_quad_vector_sum_10,
        p_dreg_area_out,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def prepare_numerical_quadrature_for_cubic_reconstruction(
    p_coords_dreg_v_1_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[vpfloat],
    shape_func_1_1: wpfloat,
    shape_func_2_1: wpfloat,
    shape_func_3_1: wpfloat,
    shape_func_4_1: wpfloat,
    shape_func_1_2: wpfloat,
    shape_func_2_2: wpfloat,
    shape_func_3_2: wpfloat,
    shape_func_4_2: wpfloat,
    shape_func_1_3: wpfloat,
    shape_func_2_3: wpfloat,
    shape_func_3_3: wpfloat,
    shape_func_4_3: wpfloat,
    shape_func_1_4: wpfloat,
    shape_func_2_4: wpfloat,
    shape_func_3_4: wpfloat,
    shape_func_4_4: wpfloat,
    zeta_1: wpfloat,
    zeta_2: wpfloat,
    zeta_3: wpfloat,
    zeta_4: wpfloat,
    eta_1: wpfloat,
    eta_2: wpfloat,
    eta_3: wpfloat,
    eta_4: wpfloat,
    wgt_zeta_1: wpfloat,
    wgt_zeta_2: wpfloat,
    wgt_eta_1: wpfloat,
    wgt_eta_2: wpfloat,
    dbl_eps: wpfloat,
    eps: wpfloat,
    p_quad_vector_sum_1: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_7: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_8: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_9: fa.EdgeKField[vpfloat],
    p_quad_vector_sum_10: fa.EdgeKField[vpfloat],
    p_dreg_area_out: fa.EdgeKField[vpfloat],
):
    _prepare_numerical_quadrature_for_cubic_reconstruction(
        p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y,
        shape_func_1_1,
        shape_func_2_1,
        shape_func_3_1,
        shape_func_4_1,
        shape_func_1_2,
        shape_func_2_2,
        shape_func_3_2,
        shape_func_4_2,
        shape_func_1_3,
        shape_func_2_3,
        shape_func_3_3,
        shape_func_4_3,
        shape_func_1_4,
        shape_func_2_4,
        shape_func_3_4,
        shape_func_4_4,
        zeta_1,
        zeta_2,
        zeta_3,
        zeta_4,
        eta_1,
        eta_2,
        eta_3,
        eta_4,
        wgt_zeta_1,
        wgt_zeta_2,
        wgt_eta_1,
        wgt_eta_2,
        dbl_eps,
        eps,
        out=(
            p_quad_vector_sum_1,
            p_quad_vector_sum_2,
            p_quad_vector_sum_3,
            p_quad_vector_sum_4,
            p_quad_vector_sum_5,
            p_quad_vector_sum_6,
            p_quad_vector_sum_7,
            p_quad_vector_sum_8,
            p_quad_vector_sum_9,
            p_quad_vector_sum_10,
            p_dreg_area_out,
        ),
    )
