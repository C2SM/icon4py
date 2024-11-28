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
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid


class TestComputeHorizontalTracerFluxFromLinearCoefficientsAlt(helpers.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_linear_coefficients_alt
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
        p_vn: np.array,
        p_out_e: np.array,
        **kwargs,
    ) -> dict:
        p_out_e_cp = p_out_e.copy()
        e2c = grid.connectivities[dims.E2CDim]
        z_lsq_coeff_1_e2c = z_lsq_coeff_1[e2c]
        z_lsq_coeff_2_e2c = z_lsq_coeff_2[e2c]
        z_lsq_coeff_3_e2c = z_lsq_coeff_3[e2c]

        lvn_pos_inv = p_vn < 0.0

        p_out_e = (
            np.where(lvn_pos_inv, z_lsq_coeff_1_e2c[:, 1], z_lsq_coeff_1_e2c[:, 0])
            + distv_bary_1 * np.where(lvn_pos_inv, z_lsq_coeff_2_e2c[:, 1], z_lsq_coeff_2_e2c[:, 0])
            + distv_bary_2 * np.where(lvn_pos_inv, z_lsq_coeff_3_e2c[:, 1], z_lsq_coeff_3_e2c[:, 0])
        ) * p_mass_flx_e

        # restriction of execution domain
        p_out_e[0 : kwargs["horizontal_start"], :] = p_out_e_cp[0 : kwargs["horizontal_start"], :]
        p_out_e[kwargs["horizontal_end"] :, :] = p_out_e_cp[kwargs["horizontal_end"] :, :]

        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        z_lsq_coeff_1 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_2 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_lsq_coeff_3 = helpers.random_field(grid, dims.CellDim, dims.KDim)
        distv_bary_1 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        distv_bary_2 = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_mass_flx_e = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vn = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = helpers.zero_field(grid, dims.EdgeDim, dims.KDim)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = (
            grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
            if hasattr(grid, "start_index")
            else 0
        )

        return dict(
            z_lsq_coeff_1=z_lsq_coeff_1,
            z_lsq_coeff_2=z_lsq_coeff_2,
            z_lsq_coeff_3=z_lsq_coeff_3,
            distv_bary_1=distv_bary_1,
            distv_bary_2=distv_bary_2,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
            p_out_e=p_out_e,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
