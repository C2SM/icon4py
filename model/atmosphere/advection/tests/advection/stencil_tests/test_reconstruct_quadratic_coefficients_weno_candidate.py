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
from icon4py.model.atmosphere.advection.stencils.reconstruct_quadratic_coefficients_weno_candidate import (
    reconstruct_quadratic_coefficients_weno_candidate,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


class TestReconstructQuadraticCoefficientsWenoCandidate(stencil_tests.StencilTest):
    PROGRAM = reconstruct_quadratic_coefficients_weno_candidate
    OUTPUTS = (
        "p_coeff_1_dsl",
        "p_coeff_2_dsl",
        "p_coeff_3_dsl",
        "p_coeff_4_dsl",
        "p_coeff_5_dsl",
        "p_coeff_6_dsl",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        lsq_pseudoinv_direct_1: np.ndarray,
        lsq_pseudoinv_direct_2: np.ndarray,
        lsq_pseudoinv_direct_3: np.ndarray,
        lsq_pseudoinv_direct_4: np.ndarray,
        lsq_pseudoinv_direct_5: np.ndarray,
        lsq_pseudoinv_butterfly_1: np.ndarray,
        lsq_pseudoinv_butterfly_2: np.ndarray,
        lsq_pseudoinv_butterfly_3: np.ndarray,
        lsq_pseudoinv_butterfly_4: np.ndarray,
        lsq_pseudoinv_butterfly_5: np.ndarray,
        lsq_moments_1: np.ndarray,
        lsq_moments_2: np.ndarray,
        lsq_moments_3: np.ndarray,
        lsq_moments_4: np.ndarray,
        lsq_moments_5: np.ndarray,
        p_coeff_1_dsl: np.ndarray,
        p_coeff_2_dsl: np.ndarray,
        p_coeff_3_dsl: np.ndarray,
        p_coeff_4_dsl: np.ndarray,
        p_coeff_5_dsl: np.ndarray,
        p_coeff_6_dsl: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_coeff_1_dsl_cp = p_coeff_1_dsl.copy()
        p_coeff_2_dsl_cp = p_coeff_2_dsl.copy()
        p_coeff_3_dsl_cp = p_coeff_3_dsl.copy()
        p_coeff_4_dsl_cp = p_coeff_4_dsl.copy()
        p_coeff_5_dsl_cp = p_coeff_5_dsl.copy()
        p_coeff_6_dsl_cp = p_coeff_6_dsl.copy()

        c2e2c = connectivities[dims.C2E2CDim]
        c2e2c2e2c = connectivities[dims.C2E2C2E2CDim]

        # f90 2448: increments of the neighbour cell averages relative to the center cell
        zb_direct = p_cc[c2e2c] - p_cc[:, np.newaxis, :]
        zb_butterfly = p_cc[c2e2c2e2c] - p_cc[:, np.newaxis, :]

        def reconstruct(direct: np.ndarray, butterfly: np.ndarray) -> np.ndarray:
            # f90 2463-2472: pseudoinv . z_b, split over the direct and butterfly rows
            return np.sum(direct[:, :, np.newaxis] * zb_direct, axis=1) + np.sum(
                butterfly[:, :, np.newaxis] * zb_butterfly, axis=1
            )

        p_coeff_2_dsl = reconstruct(lsq_pseudoinv_direct_1, lsq_pseudoinv_butterfly_1)
        p_coeff_3_dsl = reconstruct(lsq_pseudoinv_direct_2, lsq_pseudoinv_butterfly_2)
        p_coeff_4_dsl = reconstruct(lsq_pseudoinv_direct_3, lsq_pseudoinv_butterfly_3)
        p_coeff_5_dsl = reconstruct(lsq_pseudoinv_direct_4, lsq_pseudoinv_butterfly_4)
        p_coeff_6_dsl = reconstruct(lsq_pseudoinv_direct_5, lsq_pseudoinv_butterfly_5)

        # f90 2494-2495: c0 from the linear constraint c0 = p_cc - coeff(2:6) . moments(1:5)
        p_coeff_1_dsl = p_cc - (
            p_coeff_2_dsl * lsq_moments_1[:, np.newaxis]
            + p_coeff_3_dsl * lsq_moments_2[:, np.newaxis]
            + p_coeff_4_dsl * lsq_moments_3[:, np.newaxis]
            + p_coeff_5_dsl * lsq_moments_4[:, np.newaxis]
            + p_coeff_6_dsl * lsq_moments_5[:, np.newaxis]
        )

        # restriction of execution domain
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        for computed, original in (
            (p_coeff_1_dsl, p_coeff_1_dsl_cp),
            (p_coeff_2_dsl, p_coeff_2_dsl_cp),
            (p_coeff_3_dsl, p_coeff_3_dsl_cp),
            (p_coeff_4_dsl, p_coeff_4_dsl_cp),
            (p_coeff_5_dsl, p_coeff_5_dsl_cp),
            (p_coeff_6_dsl, p_coeff_6_dsl_cp),
        ):
            computed[0:horizontal_start, :] = original[0:horizontal_start, :]
            computed[horizontal_end:, :] = original[horizontal_end:, :]

        return dict(
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
            p_coeff_4_dsl=p_coeff_4_dsl,
            p_coeff_5_dsl=p_coeff_5_dsl,
            p_coeff_6_dsl=p_coeff_6_dsl,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        lsq_pseudoinv_direct_1 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_direct_2 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_direct_3 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_direct_4 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_direct_5 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_butterfly_1 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_butterfly_2 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_butterfly_3 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_butterfly_4 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_butterfly_5 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_moments_1 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_2 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_3 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_4 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_5 = data_alloc.random_field(grid, dims.CellDim)
        p_coeff_1_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_2_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_3_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_4_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_5_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_6_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        return dict(
            p_cc=p_cc,
            lsq_pseudoinv_direct_1=lsq_pseudoinv_direct_1,
            lsq_pseudoinv_direct_2=lsq_pseudoinv_direct_2,
            lsq_pseudoinv_direct_3=lsq_pseudoinv_direct_3,
            lsq_pseudoinv_direct_4=lsq_pseudoinv_direct_4,
            lsq_pseudoinv_direct_5=lsq_pseudoinv_direct_5,
            lsq_pseudoinv_butterfly_1=lsq_pseudoinv_butterfly_1,
            lsq_pseudoinv_butterfly_2=lsq_pseudoinv_butterfly_2,
            lsq_pseudoinv_butterfly_3=lsq_pseudoinv_butterfly_3,
            lsq_pseudoinv_butterfly_4=lsq_pseudoinv_butterfly_4,
            lsq_pseudoinv_butterfly_5=lsq_pseudoinv_butterfly_5,
            lsq_moments_1=lsq_moments_1,
            lsq_moments_2=lsq_moments_2,
            lsq_moments_3=lsq_moments_3,
            lsq_moments_4=lsq_moments_4,
            lsq_moments_5=lsq_moments_5,
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
            p_coeff_4_dsl=p_coeff_4_dsl,
            p_coeff_5_dsl=p_coeff_5_dsl,
            p_coeff_6_dsl=p_coeff_6_dsl,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
