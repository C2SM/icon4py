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

from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_weno_coefficients import (
    compute_horizontal_tracer_flux_from_weno_coefficients,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeHorizontalTracerFluxFromWenoCoefficients(stencil_tests.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_weno_coefficients
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_lsq_weighted_1: np.ndarray,
        z_lsq_weighted_2: np.ndarray,
        z_lsq_weighted_3: np.ndarray,
        z_lsq_weighted_4: np.ndarray,
        z_lsq_weighted_5: np.ndarray,
        z_lsq_weighted_6: np.ndarray,
        smooth_sum: np.ndarray,
        p_quad_vector_sum_1: np.ndarray,
        p_quad_vector_sum_2: np.ndarray,
        p_quad_vector_sum_3: np.ndarray,
        p_quad_vector_sum_4: np.ndarray,
        p_quad_vector_sum_5: np.ndarray,
        p_quad_vector_sum_6: np.ndarray,
        p_mass_flx_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        # literal port of the miura3 WENO flux write (mo_advection_hflux.f90 2514-2521): the
        # normalization by smooth_sum is applied to the coefficients BEFORE the dot product
        weighted = [
            coeff / smooth_sum
            for coeff in (
                z_lsq_weighted_1,
                z_lsq_weighted_2,
                z_lsq_weighted_3,
                z_lsq_weighted_4,
                z_lsq_weighted_5,
                z_lsq_weighted_6,
            )
        ]
        quad = (
            p_quad_vector_sum_1,
            p_quad_vector_sum_2,
            p_quad_vector_sum_3,
            p_quad_vector_sum_4,
            p_quad_vector_sum_5,
            p_quad_vector_sum_6,
        )
        p_out_e = sum(w * q for w, q in zip(weighted, quad, strict=True)) * p_mass_flx_e
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        return dict(
            z_lsq_weighted_1=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            z_lsq_weighted_2=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            z_lsq_weighted_3=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            z_lsq_weighted_4=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            z_lsq_weighted_5=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            z_lsq_weighted_6=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.1),
            # the accumulated smoothness weights are sums of positive terms; keep them away
            # from zero
            smooth_sum=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, low=0.5, high=2.0),
            p_quad_vector_sum_1=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_2=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_3=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_4=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_5=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_quad_vector_sum_6=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_mass_flx_e=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_out_e=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
