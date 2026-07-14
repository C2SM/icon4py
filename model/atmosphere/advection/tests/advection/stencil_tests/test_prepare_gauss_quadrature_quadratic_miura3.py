# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.prepare_gauss_quadrature_quadratic_miura3 import (
    prepare_gauss_quadrature_quadratic_miura3,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


# Gauss-Legendre points and weights of the 2nd-order 2D quadrature and the bilinear shape
# functions evaluated there (init_2D_gauss_quad, mo_advection_config.f90 1084-1136)
_GAUSS = 1.0 / np.sqrt(3.0)
_ZETA = (-_GAUSS, _GAUSS, _GAUSS, -_GAUSS)
_ETA = (-_GAUSS, -_GAUSS, _GAUSS, _GAUSS)


class TestPrepareGaussQuadratureQuadraticMiura3(stencil_tests.StencilTest):
    PROGRAM = prepare_gauss_quadrature_quadratic_miura3
    OUTPUTS = (
        "p_quad_vector_sum_1",
        "p_quad_vector_sum_2",
        "p_quad_vector_sum_3",
        "p_quad_vector_sum_4",
        "p_quad_vector_sum_5",
        "p_quad_vector_sum_6",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_coords_dreg_v_1_x: np.ndarray,
        p_coords_dreg_v_2_x: np.ndarray,
        p_coords_dreg_v_3_x: np.ndarray,
        p_coords_dreg_v_4_x: np.ndarray,
        p_coords_dreg_v_1_y: np.ndarray,
        p_coords_dreg_v_2_y: np.ndarray,
        p_coords_dreg_v_3_y: np.ndarray,
        p_coords_dreg_v_4_y: np.ndarray,
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
        wgt_zeta_1: float,
        wgt_zeta_2: float,
        wgt_eta_1: float,
        wgt_eta_2: float,
        **kwargs: Any,
    ) -> dict:
        # literal port of prep_gauss_quadrature_q_miura3 (mo_advection_quadrature.f90 561-703)
        coords_x = (
            p_coords_dreg_v_1_x,
            p_coords_dreg_v_2_x,
            p_coords_dreg_v_3_x,
            p_coords_dreg_v_4_x,
        )
        coords_y = (
            p_coords_dreg_v_1_y,
            p_coords_dreg_v_2_y,
            p_coords_dreg_v_3_y,
            p_coords_dreg_v_4_y,
        )
        shape_func = np.array(
            [
                [shape_func_1_1, shape_func_1_2, shape_func_1_3, shape_func_1_4],
                [shape_func_2_1, shape_func_2_2, shape_func_2_3, shape_func_2_4],
                [shape_func_3_1, shape_func_3_2, shape_func_3_3, shape_func_3_4],
                [shape_func_4_1, shape_func_4_2, shape_func_4_3, shape_func_4_4],
            ]
        )
        # f90 641: z_wgt(1:4) = 0.25 * gq%wgt(1:4), with the wgt_zeta/wgt_eta flattening of
        # mo_advection_config.f90 1127-1136
        z_wgt = 0.25 * np.array(
            [
                wgt_zeta_1 * wgt_eta_1,
                wgt_zeta_1 * wgt_eta_2,
                wgt_zeta_2 * wgt_eta_1,
                wgt_zeta_2 * wgt_eta_2,
            ]
        )

        p_quad_vector_sum = [np.zeros_like(p_coords_dreg_v_1_x) for _ in range(6)]
        for jg in range(4):
            # f90 667-668: gauss point coordinates via the bilinear mapping
            z_gauss_pt_x = sum(shape_func[i, jg] * coords_x[i] for i in range(4))
            z_gauss_pt_y = sum(shape_func[i, jg] * coords_y[i] for i in range(4))
            # f90 670-681: quadrature vector {1, x, y, x^2, y^2, xy}, summed at 684-689
            p_quad_vector_sum[0] = p_quad_vector_sum[0] + z_wgt[jg]
            p_quad_vector_sum[1] = p_quad_vector_sum[1] + z_wgt[jg] * z_gauss_pt_x
            p_quad_vector_sum[2] = p_quad_vector_sum[2] + z_wgt[jg] * z_gauss_pt_y
            p_quad_vector_sum[3] = p_quad_vector_sum[3] + z_wgt[jg] * z_gauss_pt_x * z_gauss_pt_x
            p_quad_vector_sum[4] = p_quad_vector_sum[4] + z_wgt[jg] * z_gauss_pt_y * z_gauss_pt_y
            p_quad_vector_sum[5] = p_quad_vector_sum[5] + z_wgt[jg] * z_gauss_pt_x * z_gauss_pt_y

        return dict(
            p_quad_vector_sum_1=p_quad_vector_sum[0],
            p_quad_vector_sum_2=p_quad_vector_sum[1],
            p_quad_vector_sum_3=p_quad_vector_sum[2],
            p_quad_vector_sum_4=p_quad_vector_sum[3],
            p_quad_vector_sum_5=p_quad_vector_sum[4],
            p_quad_vector_sum_6=p_quad_vector_sum[5],
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        input_data = dict(
            p_coords_dreg_v_1_x=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_2_x=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_3_x=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_4_x=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_1_y=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_2_y=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_3_y=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_coords_dreg_v_4_y=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_1=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_2=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_3=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_4=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_5=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_6=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            # the live Gauss-Legendre O2 values (mo_advection_config.f90 1095-1136)
            wgt_zeta_1=1.0,
            wgt_zeta_2=1.0,
            wgt_eta_1=1.0,
            wgt_eta_2=1.0,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
        for jg in range(4):
            input_data[f"shape_func_1_{jg + 1}"] = 0.25 * (1.0 - _ZETA[jg]) * (1.0 - _ETA[jg])
            input_data[f"shape_func_2_{jg + 1}"] = 0.25 * (1.0 + _ZETA[jg]) * (1.0 - _ETA[jg])
            input_data[f"shape_func_3_{jg + 1}"] = 0.25 * (1.0 + _ZETA[jg]) * (1.0 + _ETA[jg])
            input_data[f"shape_func_4_{jg + 1}"] = 0.25 * (1.0 - _ZETA[jg]) * (1.0 + _ETA[jg])
        return input_data
