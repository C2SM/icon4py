# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


def reconstruct_linear_coefficients_svd_numpy(
    p_cc: np.ndarray,
    lsq_pseudoinv_1: np.ndarray,
    lsq_pseudoinv_2: np.ndarray,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = p_cc[connectivities[dims.C2E2C]] - p_cc[:, np.newaxis, :]
    p_coeff_2 = np.sum(np.expand_dims(lsq_pseudoinv_1, axis=-1) * diff, axis=1)
    p_coeff_3 = np.sum(np.expand_dims(lsq_pseudoinv_2, axis=-1) * diff, axis=1)
    return p_cc, p_coeff_2, p_coeff_3


class TestReconstructLinearCoefficientsSvd(stencil_tests.StencilTest):
    PROGRAM = reconstruct_linear_coefficients_svd
    OUTPUTS = ("p_coeff_1_dsl", "p_coeff_2_dsl", "p_coeff_3_dsl")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        p_cc: np.ndarray,
        lsq_pseudoinv_1: np.ndarray,
        lsq_pseudoinv_2: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        coeffs = reconstruct_linear_coefficients_svd_numpy(
            p_cc, lsq_pseudoinv_1, lsq_pseudoinv_2, connectivities
        )
        out = []
        for coeff in coeffs:
            restricted = np.zeros_like(coeff)
            restricted[horizontal_start:horizontal_end] = coeff[horizontal_start:horizontal_end]
            out.append(restricted)
        return dict(p_coeff_1_dsl=out[0], p_coeff_2_dsl=out[1], p_coeff_3_dsl=out[2])

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        cell_domain = h_grid.domain(dims.CellDim)
        return dict(
            p_cc=data_alloc.random_field(dims.CellDim, dims.KDim),
            lsq_pseudoinv_1=data_alloc.random_field(dims.CellDim, dims.C2E2CDim),
            lsq_pseudoinv_2=data_alloc.random_field(dims.CellDim, dims.C2E2CDim),
            p_coeff_1_dsl=data_alloc.zero_field(dims.CellDim, dims.KDim),
            p_coeff_2_dsl=data_alloc.zero_field(dims.CellDim, dims.KDim),
            p_coeff_3_dsl=data_alloc.zero_field(dims.CellDim, dims.KDim),
            # boundary cells have fewer than three neighbouring cells
            horizontal_start=grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)),
            horizontal_end=grid.end_index(cell_domain(h_grid.Zone.END)),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
