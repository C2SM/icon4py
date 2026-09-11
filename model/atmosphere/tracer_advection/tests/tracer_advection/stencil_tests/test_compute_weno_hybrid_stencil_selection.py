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

import icon4py.model.common.type_alias as ta
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.tracer_advection.stencils.compute_weno_hybrid_stencil_selection import (
    compute_weno_hybrid_stencil_selection,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestComputeWenoHybridStencilSelection(stencil_tests.StencilTest):
    """The mask on random data; the single-precision arithmetic is mirrored step by step.

    The threshold is put in the middle of the residual distribution so both outcomes
    occur; cells within round-off of the threshold cannot be compared and the inputs
    are chosen so there are none (the residual is O(1), the threshold O(1)).
    """

    PROGRAM = compute_weno_hybrid_stencil_selection
    OUTPUTS = ("use_weno",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        p_coeff_2: np.ndarray,
        p_coeff_3: np.ndarray,
        p_coeff_4: np.ndarray,
        p_coeff_5: np.ndarray,
        p_coeff_6: np.ndarray,
        lsq_error_direct_1: np.ndarray,
        lsq_error_direct_2: np.ndarray,
        lsq_error_direct_3: np.ndarray,
        lsq_error_direct_4: np.ndarray,
        lsq_error_direct_5: np.ndarray,
        lsq_error_butterfly_1: np.ndarray,
        lsq_error_butterfly_2: np.ndarray,
        lsq_error_butterfly_3: np.ndarray,
        lsq_error_butterfly_4: np.ndarray,
        lsq_error_butterfly_5: np.ndarray,
        lsq_butterfly_active: np.ndarray,
        selection_threshold: float,
        selection_eps: float,
        **kwargs,
    ) -> dict:
        c2e2c = connectivities[dims.C2E2CDim]
        c2e2c2e2c = connectivities[dims.C2E2C2E2CDim]
        f32 = ta.fortran_sp_float  # the Fortran's REAL(sp), as the port resolves it
        zlc = [c.astype(f32) for c in (p_coeff_2, p_coeff_3, p_coeff_4, p_coeff_5, p_coeff_6)]
        zb_direct = (p_cc[c2e2c] - p_cc[:, np.newaxis]).astype(f32)  # (cells, 3, k)
        zb_butterfly = (p_cc[c2e2c2e2c] - p_cc[:, np.newaxis]).astype(f32) * lsq_butterfly_active[
            :, :, np.newaxis
        ].astype(f32)
        direct = (
            lsq_error_direct_1,
            lsq_error_direct_2,
            lsq_error_direct_3,
            lsq_error_direct_4,
            lsq_error_direct_5,
        )
        butterfly = (
            lsq_error_butterfly_1,
            lsq_error_butterfly_2,
            lsq_error_butterfly_3,
            lsq_error_butterfly_4,
            lsq_error_butterfly_5,
        )
        residual_direct = (
            sum(
                (
                    a[:, :, np.newaxis].astype(f32) * z[:, np.newaxis, :]
                    for a, z in zip(direct, zlc, strict=True)
                ),
                start=f32(0.0),
            )
            - zb_direct
        )
        residual_butterfly = (
            sum(
                (
                    a[:, :, np.newaxis].astype(f32) * z[:, np.newaxis, :]
                    for a, z in zip(butterfly, zlc, strict=True)
                ),
                start=f32(0.0),
            )
            - zb_butterfly
        )
        lsqe = np.sum(residual_direct * residual_direct, axis=1) + np.sum(
            residual_butterfly * residual_butterfly, axis=1
        )
        threshold = selection_threshold * ((p_cc + selection_eps) * (p_cc + selection_eps))
        return dict(use_weno=lsqe.astype(ta.wpfloat) > threshold)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        def direct(low=-1.0, high=1.0):
            return data_alloc.random_field(
                grid, dims.CellDim, dims.C2E2CDim, low=low, high=high, dtype=ta.fortran_sp_float
            )

        def butterfly(low=-1.0, high=1.0):
            return data_alloc.random_field(
                grid,
                dims.CellDim,
                dims.C2E2C2E2CDim,
                low=low,
                high=high,
                dtype=ta.fortran_sp_float,
            )

        active = data_alloc.random_mask(grid, dims.CellDim, dims.C2E2C2E2CDim, dtype=np.int32)
        return dict(
            p_cc=data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=1.5),
            **{
                f"p_coeff_{c}": data_alloc.random_field(grid, dims.CellDim, dims.KDim)
                for c in (2, 3, 4, 5, 6)
            },
            **{f"lsq_error_direct_{u}": direct() for u in (1, 2, 3, 4, 5)},
            **{f"lsq_error_butterfly_{u}": butterfly() for u in (1, 2, 3, 4, 5)},
            lsq_butterfly_active=active,
            use_weno=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=bool),
            selection_threshold=ta.wpfloat(2.0),
            selection_eps=ta.wpfloat(ta.fortran_sp_literal(1e-10)),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
