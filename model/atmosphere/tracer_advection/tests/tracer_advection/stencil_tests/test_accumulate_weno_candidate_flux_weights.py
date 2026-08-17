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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.tracer_advection.stencils.accumulate_weno_candidate_flux_weights import (
    _WENO_EPS,
    accumulate_weno_candidate_flux_weights,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


class TestAccumulateWenoCandidateFluxWeights(stencil_tests.StencilTest):
    PROGRAM = accumulate_weno_candidate_flux_weights
    OUTPUTS = (
        "z_lsq_weighted_1",
        "z_lsq_weighted_2",
        "z_lsq_weighted_3",
        "z_lsq_weighted_4",
        "z_lsq_weighted_5",
        "z_lsq_weighted_6",
        "smooth_sum",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_coeff_1: np.ndarray,
        p_coeff_2: np.ndarray,
        p_coeff_3: np.ndarray,
        p_coeff_4: np.ndarray,
        p_coeff_5: np.ndarray,
        p_coeff_6: np.ndarray,
        cell_area: np.ndarray,
        p_cell_rel_idx_dsl: np.ndarray,
        z_quad_vector_sum_1: np.ndarray,
        z_quad_vector_sum_2: np.ndarray,
        z_quad_vector_sum_3: np.ndarray,
        z_quad_vector_sum_4: np.ndarray,
        z_quad_vector_sum_5: np.ndarray,
        z_quad_vector_sum_6: np.ndarray,
        z_lsq_weighted_1: np.ndarray,
        z_lsq_weighted_2: np.ndarray,
        z_lsq_weighted_3: np.ndarray,
        z_lsq_weighted_4: np.ndarray,
        z_lsq_weighted_5: np.ndarray,
        z_lsq_weighted_6: np.ndarray,
        smooth_sum: np.ndarray,
        l_weight_s: float,
        **kwargs: Any,
    ) -> dict:
        z_lsq_weighted_1_cp = z_lsq_weighted_1.copy()
        z_lsq_weighted_2_cp = z_lsq_weighted_2.copy()
        z_lsq_weighted_3_cp = z_lsq_weighted_3.copy()
        z_lsq_weighted_4_cp = z_lsq_weighted_4.copy()
        z_lsq_weighted_5_cp = z_lsq_weighted_5.copy()
        z_lsq_weighted_6_cp = z_lsq_weighted_6.copy()
        smooth_sum_cp = smooth_sum.copy()

        e2c = connectivities[dims.E2CDim]

        def upwind(coeff: np.ndarray) -> np.ndarray:
            coeff_e2c = coeff[e2c]
            return np.where(p_cell_rel_idx_dsl == 1, coeff_e2c[:, 1], coeff_e2c[:, 0])

        # gather the upwind cell's coefficients and area onto the edge
        c1 = upwind(p_coeff_1)
        c2 = upwind(p_coeff_2)
        c3 = upwind(p_coeff_3)
        c4 = upwind(p_coeff_4)
        c5 = upwind(p_coeff_5)
        c6 = upwind(p_coeff_6)
        area_e2c = cell_area[e2c]
        area = np.where(p_cell_rel_idx_dsl == 1, area_e2c[:, 1:2], area_e2c[:, 0:1])

        # smoothness vector (f90 2497-2506): smooth_2/3/6 use raw c4/c5/c6, the rest their squares
        smooth_2 = 2.0 * (c2 * c4 + c3 * c6)
        smooth_3 = 2.0 * (c2 * c6 + c3 * c5)
        smooth_6 = 2.0 * c6 * (c4 + c5)
        c4_sq = c4 * c4
        c5_sq = c5 * c5
        c6_sq = c6 * c6
        smooth_4 = 2.0 * (c4_sq + c6_sq)
        smooth_5 = 2.0 * (c5_sq + c6_sq)
        smooth_1 = c2 * c2 + c3 * c3 + area * (c4_sq + c5_sq + c6_sq)

        # f90 2508-2509: w = l_weights_s / (z_lsq_smooth . z_quad_vector_sum + eps)^2
        beta = (
            smooth_1 * z_quad_vector_sum_1
            + smooth_2 * z_quad_vector_sum_2
            + smooth_3 * z_quad_vector_sum_3
            + smooth_4 * z_quad_vector_sum_4
            + smooth_5 * z_quad_vector_sum_5
            + smooth_6 * z_quad_vector_sum_6
        )
        w = l_weight_s / (beta + _WENO_EPS) ** 2

        # f90 2510-2511: accumulate onto the incoming weighted sums and weight sum
        z_lsq_weighted_1 = z_lsq_weighted_1 + c1 * w
        z_lsq_weighted_2 = z_lsq_weighted_2 + c2 * w
        z_lsq_weighted_3 = z_lsq_weighted_3 + c3 * w
        z_lsq_weighted_4 = z_lsq_weighted_4 + c4 * w
        z_lsq_weighted_5 = z_lsq_weighted_5 + c5 * w
        z_lsq_weighted_6 = z_lsq_weighted_6 + c6 * w
        smooth_sum = smooth_sum + w

        # restriction of execution domain
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        for computed, original in (
            (z_lsq_weighted_1, z_lsq_weighted_1_cp),
            (z_lsq_weighted_2, z_lsq_weighted_2_cp),
            (z_lsq_weighted_3, z_lsq_weighted_3_cp),
            (z_lsq_weighted_4, z_lsq_weighted_4_cp),
            (z_lsq_weighted_5, z_lsq_weighted_5_cp),
            (z_lsq_weighted_6, z_lsq_weighted_6_cp),
            (smooth_sum, smooth_sum_cp),
        ):
            computed[0:horizontal_start, :] = original[0:horizontal_start, :]
            computed[horizontal_end:, :] = original[horizontal_end:, :]

        return dict(
            z_lsq_weighted_1=z_lsq_weighted_1,
            z_lsq_weighted_2=z_lsq_weighted_2,
            z_lsq_weighted_3=z_lsq_weighted_3,
            z_lsq_weighted_4=z_lsq_weighted_4,
            z_lsq_weighted_5=z_lsq_weighted_5,
            z_lsq_weighted_6=z_lsq_weighted_6,
            smooth_sum=smooth_sum,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_coeff_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_coeff_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_coeff_3 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_coeff_4 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_coeff_5 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_coeff_6 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        cell_area = data_alloc.random_field(grid, dims.CellDim, low=1.0, high=2.0)

        # checkerboard rel_idx guarantees both upwind selections (0 -> E2C[0], 1 -> E2C[1])
        rel_idx_np = (
            np.add.outer(np.arange(grid.num_edges), np.arange(grid.num_levels)) % 2
        ).astype(gtx.int32)
        p_cell_rel_idx_dsl = gtx.as_field((dims.EdgeDim, dims.KDim), rel_idx_np)

        z_quad_vector_sum_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_quad_vector_sum_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_quad_vector_sum_3 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_quad_vector_sum_4 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_quad_vector_sum_5 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_quad_vector_sum_6 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        # nonzero initial accumulators verify that the stencil accumulates rather than assigns
        z_lsq_weighted_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_lsq_weighted_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_lsq_weighted_3 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_lsq_weighted_4 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_lsq_weighted_5 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_lsq_weighted_6 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        smooth_sum = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))

        return dict(
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
            l_weight_s=2.991549980478795,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
