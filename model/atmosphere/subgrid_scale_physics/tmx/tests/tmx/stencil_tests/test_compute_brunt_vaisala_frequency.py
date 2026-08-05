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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_brunt_vaisala_frequency import (
    compute_brunt_vaisala_frequency,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_brunt_vaisala_frequency_numpy(
    theta_v: np.ndarray,
    wgtfac_c: np.ndarray,
    inv_ddqz_z_half: np.ndarray,
    *,
    grav: float,
) -> np.ndarray:
    nlev = theta_v.shape[1]
    bruvais = np.zeros((theta_v.shape[0], nlev + 1), dtype=theta_v.dtype)
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based); the top and
    # bottom half levels (k = 0 and k = nlev) stay untouched (zero-initialized).
    theta_v_ic = (
        wgtfac_c[:, 1:nlev] * theta_v[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * theta_v[:, 0 : nlev - 1]
    )
    bruvais[:, 1:nlev] = (
        grav
        * (theta_v[:, 0 : nlev - 1] - theta_v[:, 1:nlev])
        * inv_ddqz_z_half[:, 1:nlev]
        / theta_v_ic
    )
    return bruvais


class TestComputeBruntVaisalaFrequency(StencilTest):
    PROGRAM = compute_brunt_vaisala_frequency
    OUTPUTS = ("bruvais",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        theta_v: np.ndarray,
        wgtfac_c: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        grav: float,
        **kwargs,
    ) -> dict:
        bruvais = compute_brunt_vaisala_frequency_numpy(
            theta_v, wgtfac_c, inv_ddqz_z_half, grav=grav
        )
        return dict(bruvais=bruvais)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        theta_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=270.0, high=350.0, dtype=wpfloat
        )
        wgtfac_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=wpfloat, extend={dims.KDim: 1}
        )
        inv_ddqz_z_half = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=1.0e-3,
            high=1.0e-1,
            dtype=wpfloat,
            extend={dims.KDim: 1},
        )
        bruvais = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            theta_v=theta_v,
            wgtfac_c=wgtfac_c,
            inv_ddqz_z_half=inv_ddqz_z_half,
            bruvais=bruvais,
            grav=constants.GRAV,
            # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based half levels)
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
