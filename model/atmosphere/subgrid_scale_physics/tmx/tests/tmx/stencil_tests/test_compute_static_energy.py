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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_static_energy import (
    compute_static_energy,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_static_energy_numpy(
    temperature: np.ndarray,
    height_above_ground: np.ndarray,
    *,
    spec_heat: float,
    grav: float,
) -> np.ndarray:
    return spec_heat * temperature + grav * height_above_ground


class TestComputeStaticEnergy(StencilTest):
    PROGRAM = compute_static_energy
    OUTPUTS = ("static_energy",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        temperature: np.ndarray,
        height_above_ground: np.ndarray,
        spec_heat: float,
        grav: float,
        **kwargs,
    ) -> dict:
        static_energy = compute_static_energy_numpy(
            temperature,
            height_above_ground,
            spec_heat=spec_heat,
            grav=grav,
        )
        return dict(static_energy=static_energy)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        temperature = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        )
        height_above_ground = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        )
        static_energy = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            temperature=temperature,
            height_above_ground=height_above_ground,
            static_energy=static_energy,
            spec_heat=constants.CPD,
            grav=constants.GRAV,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
