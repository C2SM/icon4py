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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.init_smagorinsky_mixing_length import (
    init_smagorinsky_mixing_length,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def init_smagorinsky_mixing_length_numpy(
    dz_ic: np.ndarray,
    geopot_agl_ic: np.ndarray,
    cell_area: np.ndarray,
    *,
    smag_constant: float,
    max_turb_scale: float,
    grav: float,
) -> np.ndarray:
    kappa = 0.4
    z_agl = geopot_agl_ic * (1.0 / grav)
    les_filter = smag_constant * np.minimum(
        max_turb_scale, (dz_ic * cell_area[:, np.newaxis]) ** 0.33333
    )
    return (
        (les_filter * z_agl)
        * (les_filter * z_agl)
        / ((les_filter / kappa) * (les_filter / kappa) + z_agl * z_agl)
    )


class TestInitSmagorinskyMixingLength(StencilTest):
    PROGRAM = init_smagorinsky_mixing_length
    OUTPUTS = ("mixing_length_sq",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        dz_ic: np.ndarray,
        geopot_agl_ic: np.ndarray,
        cell_area: np.ndarray,
        smag_constant: float,
        max_turb_scale: float,
        grav: float,
        **kwargs,
    ) -> dict:
        mixing_length_sq = init_smagorinsky_mixing_length_numpy(
            dz_ic,
            geopot_agl_ic,
            cell_area,
            smag_constant=smag_constant,
            max_turb_scale=max_turb_scale,
            grav=grav,
        )
        return dict(mixing_length_sq=mixing_length_sq)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dz_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=10.0,
            high=500.0,
            dtype=wpfloat,
            extend={dims.KDim: 1},
        )
        geopot_agl_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.0,
            high=1.0e5,
            dtype=wpfloat,
            extend={dims.KDim: 1},
        )
        cell_area = data_alloc.random_field(
            grid, dims.CellDim, low=1.0e6, high=1.0e8, dtype=wpfloat
        )
        mixing_length_sq = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            dz_ic=dz_ic,
            geopot_agl_ic=geopot_agl_ic,
            cell_area=cell_area,
            mixing_length_sq=mixing_length_sq,
            smag_constant=wpfloat(0.23),
            max_turb_scale=wpfloat(300.0),
            grav=constants.GRAV,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
