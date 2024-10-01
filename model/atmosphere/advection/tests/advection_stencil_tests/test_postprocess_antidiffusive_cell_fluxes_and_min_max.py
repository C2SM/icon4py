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
from icon4py.model.atmosphere.advection.stencils.postprocess_antidiffusive_cell_fluxes_and_min_max import (
    postprocess_antidiffusive_cell_fluxes_and_min_max,
)
from icon4py.model.common import dimension as dims


class TestPostprocessAntidiffusiveCellFluxesAndMinMax(helpers.StencilTest):
    PROGRAM = postprocess_antidiffusive_cell_fluxes_and_min_max
    OUTPUTS = ("z_tracer_new_low", "z_tracer_max", "z_tracer_min")

    @staticmethod
    def reference(
        grid,
        refin_ctrl: np.ndarray,
        p_cc: np.ndarray,
        z_tracer_new_low: np.ndarray,
        z_tracer_max: np.ndarray,
        z_tracer_min: np.ndarray,
        lo_bound: float,
        hi_bound: float,
        **kwargs,
    ):
        refin_ctrl = np.expand_dims(refin_ctrl, axis=1)
        condition = np.logical_or(
            np.equal(refin_ctrl, lo_bound * np.ones(refin_ctrl.shape, dtype=gtx.int32)),
            np.equal(refin_ctrl, hi_bound * np.ones(refin_ctrl.shape, dtype=gtx.int32)),
        )
        z_tracer_new_out = np.where(
            condition,
            np.minimum(1.1 * p_cc, np.maximum(0.9 * p_cc, z_tracer_new_low)),
            z_tracer_new_low,
        )
        z_tracer_max_out = np.where(condition, np.maximum(p_cc, z_tracer_new_out), z_tracer_max)
        z_tracer_min_out = np.where(condition, np.minimum(p_cc, z_tracer_new_out), z_tracer_min)
        return dict(
            z_tracer_new_low=z_tracer_new_out,
            z_tracer_max=z_tracer_max_out,
            z_tracer_min=z_tracer_min_out,
        )

    @pytest.fixture()
    def input_data(self, grid):
        hi_bound, lo_bound = 3, 1
        refin_ctrl = helpers.constant_field(grid, 2, dims.CellDim, dtype=gtx.int32)
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_new_low_in = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_max_in = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_min_in = helpers.random_field(grid, dims.CellDim, dims.KDim)

        z_tracer_new_low_out = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_max_out = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_min_out = helpers.zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            refin_ctrl=refin_ctrl,
            p_cc=p_cc,
            z_tracer_new_low=z_tracer_new_low_in,
            z_tracer_max=z_tracer_max_in,
            z_tracer_min=z_tracer_min_in,
            lo_bound=lo_bound,
            hi_bound=hi_bound,
            z_tracer_new_low_out=z_tracer_new_low_out,
            z_tracer_max_out=z_tracer_max_out,
            z_tracer_min_out=z_tracer_min_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
