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

from icon4py.model.atmosphere.dycore.stencils.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field, random_mask
from icon4py.model.testing.helpers import StencilTest


class TestApplyHydrostaticCorrectionToHorizontalGradientOfExnerPressure(StencilTest):
    PROGRAM = apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        ipeidx_dsl: np.ndarray,
        pg_exdist: np.ndarray,
        z_hydro_corr: np.ndarray,
        z_gradh_exner: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_gradh_exner = np.where(
            ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner
        )

        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        ipeidx_dsl = random_mask(grid, dims.EdgeDim, dims.KDim)
        pg_exdist = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_hydro_corr = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_gradh_exner = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            z_hydro_corr=z_hydro_corr,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
