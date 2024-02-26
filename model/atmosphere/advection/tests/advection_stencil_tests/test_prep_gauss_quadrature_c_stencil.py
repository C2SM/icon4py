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

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.prep_gauss_quadrature_c_stencil import (
    prep_gauss_quadrature_c_stencil,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


@pytest.mark.slow_tests
class TestPrepGaussQuadratureCStencil(StencilTest):
    PROGRAM = prep_gauss_quadrature_c_stencil
    OUTPUTS = (
        "p_quad_vector_sum_1",
        "p_quad_vector_sum_2",
        "p_quad_vector_sum_3",
        "p_quad_vector_sum_4",
        "p_quad_vector_sum_5",
        "p_quad_vector_sum_6",
        "p_quad_vector_sum_7",
        "p_quad_vector_sum_8",
        "p_quad_vector_sum_9",
        "p_quad_vector_sum_10",
        "p_dreg_area_out",
    )

    @staticmethod
    def _compute_wgt_t_detjac(
        wgt_zeta_1,
        wgt_zeta_2,
        wgt_eta_1,
        wgt_eta_2,
        dbl_eps,
        p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y,
        zeta_1,
        zeta_2,
        zeta_3,
        zeta_4,
        eta_1,
        eta_2,
        eta_3,
        eta_4,
    ):
        # Compute z_wgt
        z_wgt_1 = 0.0625 * wgt_zeta_1 * wgt_eta_1
        z_wgt_2 = 0.0625 * wgt_zeta_1 * wgt_eta_2
        z_wgt_3 = 0.0625 * wgt_zeta_2 * wgt_eta_1
        z_wgt_4 = 0.0625 * wgt_zeta_2 * wgt_eta_2

        # Compute z_eta
        z_eta_1_1, z_eta_2_1, z_eta_3_1, z_eta_4_1 = (
            1.0 - eta_1,
            1.0 - eta_2,
            1.0 - eta_3,
            1.0 - eta_4,
        )
        z_eta_1_2, z_eta_2_2, z_eta_3_2, z_eta_4_2 = (
            1.0 + eta_1,
            1.0 + eta_2,
            1.0 + eta_3,
            1.0 + eta_4,
        )
        z_eta_1_3, z_eta_2_3, z_eta_3_3, z_eta_4_3 = (
            1.0 - zeta_1,
            1.0 - zeta_2,
            1.0 - zeta_3,
            1.0 - zeta_4,
        )
        z_eta_1_4, z_eta_2_4, z_eta_3_4, z_eta_4_4 = (
            1.0 + zeta_1,
            1.0 + zeta_2,
            1.0 + zeta_3,
            1.0 + zeta_4,
        )

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
        return wgt_t_detjac_1, wgt_t_detjac_2, wgt_t_detjac_3, wgt_t_detjac_4

    @staticmethod
    def _compute_z_gauss_points(
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
    ):
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
        return (
            z_gauss_pts_1_x,
            z_gauss_pts_1_y,
            z_gauss_pts_2_x,
            z_gauss_pts_2_y,
            z_gauss_pts_3_x,
            z_gauss_pts_3_y,
            z_gauss_pts_4_x,
            z_gauss_pts_4_y,
        )

    @staticmethod
    def _compute_vector_sums(
        wgt_t_detjac_1,
        wgt_t_detjac_2,
        wgt_t_detjac_3,
        wgt_t_detjac_4,
        z_gauss_pts_1_x,
        z_gauss_pts_1_y,
        z_gauss_pts_2_x,
        z_gauss_pts_2_y,
        z_gauss_pts_3_x,
        z_gauss_pts_3_y,
        z_gauss_pts_4_x,
        z_gauss_pts_4_y,
    ):
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
        )

    @classmethod
    def reference(
        cls,
        grid,
        p_coords_dreg_v_1_x: np.array,
        p_coords_dreg_v_2_x: np.array,
        p_coords_dreg_v_3_x: np.array,
        p_coords_dreg_v_4_x: np.array,
        p_coords_dreg_v_1_y: np.array,
        p_coords_dreg_v_2_y: np.array,
        p_coords_dreg_v_3_y: np.array,
        p_coords_dreg_v_4_y: np.array,
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
        **kwargs,
    ):
        wgt_t_detjac_1, wgt_t_detjac_2, wgt_t_detjac_3, wgt_t_detjac_4 = cls._compute_wgt_t_detjac(
            wgt_zeta_1,
            wgt_zeta_2,
            wgt_eta_1,
            wgt_eta_2,
            dbl_eps,
            p_coords_dreg_v_1_x,
            p_coords_dreg_v_2_x,
            p_coords_dreg_v_3_x,
            p_coords_dreg_v_4_x,
            p_coords_dreg_v_1_y,
            p_coords_dreg_v_2_y,
            p_coords_dreg_v_3_y,
            p_coords_dreg_v_4_y,
            zeta_1,
            zeta_2,
            zeta_3,
            zeta_4,
            eta_1,
            eta_2,
            eta_3,
            eta_4,
        )

        (
            z_gauss_pts_1_x,
            z_gauss_pts_1_y,
            z_gauss_pts_2_x,
            z_gauss_pts_2_y,
            z_gauss_pts_3_x,
            z_gauss_pts_3_y,
            z_gauss_pts_4_x,
            z_gauss_pts_4_y,
        ) = cls._compute_z_gauss_points(
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
        )

        (
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
        ) = cls._compute_vector_sums(
            wgt_t_detjac_1,
            wgt_t_detjac_2,
            wgt_t_detjac_3,
            wgt_t_detjac_4,
            z_gauss_pts_1_x,
            z_gauss_pts_1_y,
            z_gauss_pts_2_x,
            z_gauss_pts_2_y,
            z_gauss_pts_3_x,
            z_gauss_pts_3_y,
            z_gauss_pts_4_x,
            z_gauss_pts_4_y,
        )

        z_area = p_quad_vector_sum_1
        p_dreg_area_out = np.where(
            z_area >= 0.0,
            np.maximum(eps, np.absolute(z_area)),
            -np.maximum(eps, np.absolute(z_area)),
        )
        return dict(
            p_quad_vector_sum_1=p_quad_vector_sum_1,
            p_quad_vector_sum_2=p_quad_vector_sum_2,
            p_quad_vector_sum_3=p_quad_vector_sum_3,
            p_quad_vector_sum_4=p_quad_vector_sum_4,
            p_quad_vector_sum_5=p_quad_vector_sum_5,
            p_quad_vector_sum_6=p_quad_vector_sum_6,
            p_quad_vector_sum_7=p_quad_vector_sum_7,
            p_quad_vector_sum_8=p_quad_vector_sum_8,
            p_quad_vector_sum_9=p_quad_vector_sum_9,
            p_quad_vector_sum_10=p_quad_vector_sum_10,
            p_dreg_area_out=p_dreg_area_out,
        )

    @pytest.fixture
    def input_data(self, grid):
        p_coords_dreg_v_1_x = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_2_x = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_3_x = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_4_x = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_1_y = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_2_y = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_3_y = random_field(grid, EdgeDim, KDim)
        p_coords_dreg_v_4_y = random_field(grid, EdgeDim, KDim)
        shape_func_1_1 = 0.001
        shape_func_2_1 = 0.001
        shape_func_3_1 = 0.001
        shape_func_4_1 = 0.001
        shape_func_1_2 = 0.001
        shape_func_2_2 = 0.001
        shape_func_3_2 = 0.001
        shape_func_4_2 = 0.001
        shape_func_1_3 = 0.001
        shape_func_2_3 = 0.001
        shape_func_3_3 = 0.001
        shape_func_4_3 = 0.001
        shape_func_1_4 = 0.001
        shape_func_2_4 = 0.001
        shape_func_3_4 = 0.001
        shape_func_4_4 = 0.001
        zeta_1 = 0.002
        zeta_2 = 0.002
        zeta_3 = 0.002
        zeta_4 = 0.002
        eta_1 = 0.5
        eta_2 = 0.5
        eta_3 = 0.5
        eta_4 = 0.5
        wgt_zeta_1 = 0.003
        wgt_zeta_2 = 0.003
        wgt_eta_1 = 0.002
        wgt_eta_2 = 0.007
        dbl_eps = np.float64(0.1)
        eps = 0.1
        p_quad_vector_sum_1 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_2 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_3 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_4 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_5 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_6 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_7 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_8 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_9 = zero_field(grid, EdgeDim, KDim)
        p_quad_vector_sum_10 = zero_field(grid, EdgeDim, KDim)
        p_dreg_area_out = zero_field(grid, EdgeDim, KDim)
        return dict(
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
            zeta_1=zeta_1,
            zeta_2=zeta_2,
            zeta_3=zeta_3,
            zeta_4=zeta_4,
            eta_1=eta_1,
            eta_2=eta_2,
            eta_3=eta_3,
            eta_4=eta_4,
            wgt_zeta_1=wgt_zeta_1,
            wgt_zeta_2=wgt_zeta_2,
            wgt_eta_1=wgt_eta_1,
            wgt_eta_2=wgt_eta_2,
            dbl_eps=dbl_eps,
            eps=eps,
            p_quad_vector_sum_1=p_quad_vector_sum_1,
            p_quad_vector_sum_2=p_quad_vector_sum_2,
            p_quad_vector_sum_3=p_quad_vector_sum_3,
            p_quad_vector_sum_4=p_quad_vector_sum_4,
            p_quad_vector_sum_5=p_quad_vector_sum_5,
            p_quad_vector_sum_6=p_quad_vector_sum_6,
            p_quad_vector_sum_7=p_quad_vector_sum_7,
            p_quad_vector_sum_8=p_quad_vector_sum_8,
            p_quad_vector_sum_9=p_quad_vector_sum_9,
            p_quad_vector_sum_10=p_quad_vector_sum_10,
            p_dreg_area_out=p_dreg_area_out,
        )
