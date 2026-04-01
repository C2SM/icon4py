# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


class TestComputeHorizontalTracerFluxFromLinearCoefficientsAlt(stencil_tests.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_from_linear_coefficients_alt
    OUTPUTS = ("p_out_e",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        z_lsq_coeff_1: np.ndarray,
        z_lsq_coeff_2: np.ndarray,
        z_lsq_coeff_3: np.ndarray,
        distv_bary_1: np.ndarray,
        distv_bary_2: np.ndarray,
        p_mass_flx_e: np.ndarray,
        p_vn: np.ndarray,
        p_out_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        p_out_e_cp = p_out_e.copy()
        e2c = connectivities[dims.E2CDim]
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

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        z_lsq_coeff_1 = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        z_lsq_coeff_2 = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        z_lsq_coeff_3 = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        distv_bary_1 = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        distv_bary_2 = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_mass_flx_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_out_e = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))

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
