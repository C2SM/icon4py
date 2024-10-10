# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients import (
    compute_horizontal_tracer_flux_from_linear_coefficients,
)
from icon4py.model.common import dimension as dims


class TestComputeHorizontalTracerFluxFromLinearCoefficients(helpers.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_linear_coefficients
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        z_lsq_coeff_1: np.array,
        z_lsq_coeff_2: np.array,
        z_lsq_coeff_3: np.array,
        distv_bary_1: np.array,
        distv_bary_2: np.array,
        p_mass_flx_e: np.array,
        cell_rel_idx_dsl: np.array,
        **kwargs,
    ):
        e2c = grid.connectivities[dims.E2CDim]
        z_lsq_coeff_1_e2c = z_lsq_coeff_1[e2c]
        z_lsq_coeff_2_e2c = z_lsq_coeff_2[e2c]
        z_lsq_coeff_3_e2c = z_lsq_coeff_3[e2c]

        p_out_e = (
            np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_1_e2c[:, 1],
                z_lsq_coeff_1_e2c[:, 0],
            )
            + distv_bary_1
            * np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_2_e2c[:, 1],
                z_lsq_coeff_2_e2c[:, 0],
            )
            + distv_bary_2
            * np.where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_3_e2c[:, 1],
                z_lsq_coeff_3_e2c[:, 0],
            )
        ) * p_mass_flx_e

        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid):
        z_lsq_coeff_1 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_2 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_3 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        distv_bary_1 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        distv_bary_2 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_mass_flx_e = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        cell_rel_idx_dsl = helpers.constant_field(grid, 0, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_out_e = helpers.zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_lsq_coeff_1=z_lsq_coeff_1,
            z_lsq_coeff_2=z_lsq_coeff_2,
            z_lsq_coeff_3=z_lsq_coeff_3,
            distv_bary_1=distv_bary_1,
            distv_bary_2=distv_bary_2,
            p_mass_flx_e=p_mass_flx_e,
            cell_rel_idx_dsl=cell_rel_idx_dsl,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )