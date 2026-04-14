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

from icon4py.model.atmosphere.dycore.stencils.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def update_dynamical_exner_time_increment_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    exner: np.ndarray,
    exner_tendency_due_to_slow_physics: np.ndarray,
    exner_dynamical_increment: np.ndarray,
    ndyn_substeps_var: float,
    dtime: float,
) -> np.ndarray:
    exner_dynamical_increment = exner - (exner_dynamical_increment + ndyn_substeps_var * dtime * exner_tendency_due_to_slow_physics)
    return exner_dynamical_increment


class TestUpdateDynamicalExnerTimeIncrement(StencilTest):
    PROGRAM = update_dynamical_exner_time_increment
    OUTPUTS = ("exner_dynamical_increment",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        exner: np.ndarray,
        exner_tendency_due_to_slow_physics: np.ndarray,
        exner_dynamical_increment: np.ndarray,
        ndyn_substeps_var: float,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        exner_dynamical_increment = update_dynamical_exner_time_increment_numpy(
            connectivities,
            exner,
            exner_tendency_due_to_slow_physics,
            exner_dynamical_increment,
            ndyn_substeps_var,
            dtime,
        )
        return dict(exner_dynamical_increment=exner_dynamical_increment)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        ndyn_substeps_var, dtime = wpfloat("10.0"), wpfloat("12.0")
        exner = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_tendency_due_to_slow_physics = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        exner_dynamical_increment = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            exner=exner,
            exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
            exner_dynamical_increment=exner_dynamical_increment,
            ndyn_substeps_var=ndyn_substeps_var,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
