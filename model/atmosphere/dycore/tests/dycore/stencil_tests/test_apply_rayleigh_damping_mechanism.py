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

from icon4py.model.atmosphere.dycore.stencils.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def apply_rayleigh_damping_mechanism_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    z_raylfac: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    z_raylfac = np.expand_dims(z_raylfac, axis=0)
    w = z_raylfac * w
    return w


class TestApplyRayleighDampingMechanism(StencilTest):
    PROGRAM = apply_rayleigh_damping_mechanism
    OUTPUTS = ("w",)

    @static_reference
    def reference(
        grid: base.Grid,
        z_raylfac: np.ndarray,
        w: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        w = apply_rayleigh_damping_mechanism_numpy(connectivities, z_raylfac, w)
        return dict(w=w)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_raylfac = self.data_alloc.random_field(dims.KDim, dtype=wpfloat)
        w = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            z_raylfac=z_raylfac,
            w=w,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
