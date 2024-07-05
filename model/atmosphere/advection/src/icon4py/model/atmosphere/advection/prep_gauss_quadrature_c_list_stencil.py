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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _prep_gauss_quadrature_c_list_stencil(
    famask_int: Field[[EdgeDim, KDim], int32],
    p_coords_dreg_v_1_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_1_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_y: Field[[EdgeDim, KDim], float],
    shape_func_1_1: float,
    shape_func_2_1: float,
    shape_func_3_1: float,
    shape_func_4_1: float,
    shape_func_1_2: float,
    shape_func_2_2: float,
    shape_func_3_2: float,
    shape_func_4_2: float,
    shape_func_1_3: float,
    shape_func_2_3: float,
    shape_func_3_3: float,
    shape_func_4_3: float,
    shape_func_1_4: float,
    shape_func_2_4: float,
    shape_func_3_4: float,
    shape_func_4_4: float,
    zeta_1: float,
    zeta_2: float,
    zeta_3: float,
    zeta_4: float,
    eta_1: float,
    eta_2: float,
    eta_3: float,
    eta_4: float,
    wgt_zeta_1: float,
    wgt_zeta_2: float,
    wgt_eta_1: float,
    wgt_eta_2: float,
    dbl_eps: float,
    eps: float,
    p_dreg_area_in: Field[[EdgeDim, KDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    z_wgt_1 = 0.0625 * wgt_zeta_1 * wgt_eta_1
    z_wgt_2 = 0.0625 * wgt_zeta_1 * wgt_eta_2
    z_wgt_3 = 0.0625 * wgt_zeta_2 * wgt_eta_1
    z_wgt_4 = 0.0625 * wgt_zeta_2 * wgt_eta_2

    z_eta_1_1 = 1.0 - eta_1
    z_eta_2_1 = 1.0 - eta_2
    z_eta_3_1 = 1.0 - eta_3
    z_eta_4_1 = 1.0 - eta_4
    z_eta_1_2 = 1.0 + eta_1
    z_eta_2_2 = 1.0 + eta_2
    z_eta_3_2 = 1.0 + eta_3
    z_eta_4_2 = 1.0 + eta_4
    z_eta_1_3 = 1.0 - zeta_1
    z_eta_2_3 = 1.0 - zeta_2
    z_eta_3_3 = 1.0 - zeta_3
    z_eta_4_3 = 1.0 - zeta_4
    z_eta_1_4 = 1.0 + zeta_1
    z_eta_2_4 = 1.0 + zeta_2
    z_eta_3_4 = 1.0 + zeta_3
    z_eta_4_4 = 1.0 + zeta_4

    famask_bool = where(famask_int == 1, True, False)
    p_coords_dreg_v_1_x = where(famask_bool, p_coords_dreg_v_1_x, 0.0)
    p_coords_dreg_v_2_x = where(famask_bool, p_coords_dreg_v_2_x, 0.0)
    p_coords_dreg_v_3_x = where(famask_bool, p_coords_dreg_v_3_x, 0.0)
    p_coords_dreg_v_4_x = where(famask_bool, p_coords_dreg_v_4_x, 0.0)
    p_coords_dreg_v_1_y = where(famask_bool, p_coords_dreg_v_1_y, 0.0)
    p_coords_dreg_v_2_y = where(famask_bool, p_coords_dreg_v_2_y, 0.0)
    p_coords_dreg_v_3_y = where(famask_bool, p_coords_dreg_v_3_y, 0.0)
    p_coords_dreg_v_4_y = where(famask_bool, p_coords_dreg_v_4_y, 0.0)

    wgt_t_detjac_1 = where(
        famask_bool,
        dbl_eps
        + z_wgt_1
        * (
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
        ),
        0.0,
    )
    wgt_t_detjac_2 = where(
        famask_bool,
        dbl_eps
        + z_wgt_2
        * (
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
        ),
        0.0,
    )
    wgt_t_detjac_3 = where(
        famask_bool,
        dbl_eps
        + z_wgt_3
        * (
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
        ),
        0.0,
    )
    wgt_t_detjac_4 = where(
        famask_bool,
        dbl_eps
        + z_wgt_4
        * (
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
        ),
        0.0,
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

    p_dreg_area = p_dreg_area_in + p_quad_vector_sum_1

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
        p_dreg_area,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def prep_gauss_quadrature_c_list_stencil(
    famask_int: Field[[EdgeDim, KDim], int32],
    p_coords_dreg_v_1_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_x: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_1_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_2_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_3_y: Field[[EdgeDim, KDim], float],
    p_coords_dreg_v_4_y: Field[[EdgeDim, KDim], float],
    shape_func_1_1: float,
    shape_func_2_1: float,
    shape_func_3_1: float,
    shape_func_4_1: float,
    shape_func_1_2: float,
    shape_func_2_2: float,
    shape_func_3_2: float,
    shape_func_4_2: float,
    shape_func_1_3: float,
    shape_func_2_3: float,
    shape_func_3_3: float,
    shape_func_4_3: float,
    shape_func_1_4: float,
    shape_func_2_4: float,
    shape_func_3_4: float,
    shape_func_4_4: float,
    zeta_1: float,
    zeta_2: float,
    zeta_3: float,
    zeta_4: float,
    eta_1: float,
    eta_2: float,
    eta_3: float,
    eta_4: float,
    wgt_zeta_1: float,
    wgt_zeta_2: float,
    wgt_eta_1: float,
    wgt_eta_2: float,
    dbl_eps: float,
    eps: float,
    p_dreg_area_in: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_1: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_2: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_3: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_4: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_5: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_6: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_7: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_8: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_9: Field[[EdgeDim, KDim], float],
    p_quad_vector_sum_10: Field[[EdgeDim, KDim], float],
    p_dreg_area: Field[[EdgeDim, KDim], float],
):
    _prep_gauss_quadrature_c_list_stencil(
        famask_int,
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
        p_dreg_area_in,
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
            p_dreg_area,
        ),
    )
