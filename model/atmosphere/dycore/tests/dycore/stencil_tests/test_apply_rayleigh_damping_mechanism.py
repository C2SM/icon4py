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

from icon4py.model.atmosphere.dycore.stencils.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_rayleigh_damping_mechanism_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    rayleigh_damping_factor: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    rayleigh_damping_factor = np.expand_dims(rayleigh_damping_factor, axis=0)
    w = rayleigh_damping_factor * w
    return w


class TestApplyRayleighDampingMechanism(StencilTest):
    PROGRAM = apply_rayleigh_damping_mechanism
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        rayleigh_damping_factor: np.ndarray,
        w: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        w = apply_rayleigh_damping_mechanism_numpy(connectivities, rayleigh_damping_factor, w)
        return dict(w=w)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rayleigh_damping_factor = random_field(grid, dims.KDim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            rayleigh_damping_factor=rayleigh_damping_factor,
            w=w,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
