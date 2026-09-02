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

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.stencils.compute_dry_static_energy import (
    compute_dry_static_energy,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_dry_static_energy_numpy(
    temperature: np.ndarray,
    height_above_ground: np.ndarray,
    *,
    grav: float,
) -> np.ndarray:
    return constants.CPD * temperature + grav * height_above_ground


class TestComputeStaticEnergy(stencil_tests.StencilTest):
    PROGRAM = compute_dry_static_energy
    OUTPUTS = ("dry_static_energy",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        temperature: np.ndarray,
        height_above_ground: np.ndarray,
        grav: float,
        **kwargs,
    ) -> dict:
        dry_static_energy = compute_dry_static_energy_numpy(
            temperature,
            height_above_ground,
            grav=grav,
        )
        return dict(dry_static_energy=dry_static_energy)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        temperature = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        )
        height_above_ground = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        )
        dry_static_energy = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            temperature=temperature,
            height_above_ground=height_above_ground,
            dry_static_energy=dry_static_energy,
            grav=constants.GRAV,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
