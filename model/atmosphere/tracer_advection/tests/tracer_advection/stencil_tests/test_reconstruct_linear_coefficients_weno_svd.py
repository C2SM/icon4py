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
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_weno_svd import (
    reconstruct_linear_coefficients_weno_svd,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


class TestReconstructLinearCoefficientsWenoSvd(stencil_tests.StencilTest):
    PROGRAM = reconstruct_linear_coefficients_weno_svd
    OUTPUTS = (
        "p_coeff_1_dsl",
        "p_coeff_2_dsl",
        "p_coeff_3_dsl",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        lsq_pseudoinv_zonal_c1: np.ndarray,
        lsq_pseudoinv_zonal_c2: np.ndarray,
        lsq_pseudoinv_zonal_c3: np.ndarray,
        lsq_pseudoinv_meridional_c1: np.ndarray,
        lsq_pseudoinv_meridional_c2: np.ndarray,
        lsq_pseudoinv_meridional_c3: np.ndarray,
        p_coeff_1_dsl: np.ndarray,
        p_coeff_2_dsl: np.ndarray,
        p_coeff_3_dsl: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_coeff_1_dsl_cp = p_coeff_1_dsl.copy()
        p_coeff_2_dsl_cp = p_coeff_2_dsl.copy()
        p_coeff_3_dsl_cp = p_coeff_3_dsl.copy()

        c2e2c = connectivities[dims.C2E2CDim]

        # f90 991-993: increments of the neighbour cell averages relative to the center cell
        z_b = p_cc[c2e2c] - p_cc[:, np.newaxis, :]

        def gradient(pseudoinv: np.ndarray) -> np.ndarray:
            # f90 997-1004: candidate gradient = pseudoinv . z_b over the 3 C2E2C rows
            return np.sum(pseudoinv[:, :, np.newaxis] * z_b, axis=1)

        # f90 995-1017: per-candidate zonal/meridional gradient and smoothness weight
        cx_1 = gradient(lsq_pseudoinv_zonal_c1)
        cy_1 = gradient(lsq_pseudoinv_meridional_c1)
        s_1 = 1.0 / ((cx_1**2 + cy_1**2) + 1.0e-20) ** 2

        cx_2 = gradient(lsq_pseudoinv_zonal_c2)
        cy_2 = gradient(lsq_pseudoinv_meridional_c2)
        s_2 = 1.0 / ((cx_2**2 + cy_2**2) + 1.0e-20) ** 2

        cx_3 = gradient(lsq_pseudoinv_zonal_c3)
        cy_3 = gradient(lsq_pseudoinv_meridional_c3)
        s_3 = 1.0 / ((cx_3**2 + cy_3**2) + 1.0e-20) ** 2

        # f90 1018-1019: smoothness-weighted average over the 3 candidates. The
        # constant coefficient is p_cc for every candidate (llsq_lin_consv off),
        # so its blend collapses to p_cc.
        smooth_sum = s_1 + s_2 + s_3
        p_coeff_1_dsl = p_cc
        p_coeff_2_dsl = (cx_1 * s_1 + cx_2 * s_2 + cx_3 * s_3) / smooth_sum
        p_coeff_3_dsl = (cy_1 * s_1 + cy_2 * s_2 + cy_3 * s_3) / smooth_sum

        # restriction of execution domain
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        for computed, original in (
            (p_coeff_1_dsl, p_coeff_1_dsl_cp),
            (p_coeff_2_dsl, p_coeff_2_dsl_cp),
            (p_coeff_3_dsl, p_coeff_3_dsl_cp),
        ):
            computed[0:horizontal_start, :] = original[0:horizontal_start, :]
            computed[horizontal_end:, :] = original[horizontal_end:, :]

        return dict(
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        lsq_pseudoinv_zonal_c1 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_zonal_c2 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_zonal_c3 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_meridional_c1 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_meridional_c2 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        lsq_pseudoinv_meridional_c3 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CDim)
        p_coeff_1_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_2_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_3_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        return dict(
            p_cc=p_cc,
            lsq_pseudoinv_zonal_c1=lsq_pseudoinv_zonal_c1,
            lsq_pseudoinv_zonal_c2=lsq_pseudoinv_zonal_c2,
            lsq_pseudoinv_zonal_c3=lsq_pseudoinv_zonal_c3,
            lsq_pseudoinv_meridional_c1=lsq_pseudoinv_meridional_c1,
            lsq_pseudoinv_meridional_c2=lsq_pseudoinv_meridional_c2,
            lsq_pseudoinv_meridional_c3=lsq_pseudoinv_meridional_c3,
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
