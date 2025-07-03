# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Final

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.update_theta_v import update_theta_v
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field, random_mask
from icon4py.model.testing.helpers import StencilTest


dycore_consts: Final = constants.PhysicsConstants()


class TestUpdateThetaV(StencilTest):
    PROGRAM = update_theta_v
    OUTPUTS = ("theta_v_new",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        mask_prog_halo_c: np.ndarray,
        rho_now: np.ndarray,
        theta_v_now: np.ndarray,
        exner_new: np.ndarray,
        exner_now: np.ndarray,
        rho_new: np.ndarray,
        theta_v_new: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        mask_prog_halo_c = np.expand_dims(mask_prog_halo_c, axis=-1)

        theta_v_new = np.where(
            mask_prog_halo_c,
            rho_now
            * theta_v_now
            * ((exner_new / exner_now - 1) * dycore_consts.cvd_o_rd + 1.0)
            / rho_new,
            theta_v_new,
        )
        return dict(theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        mask_prog_halo_c = random_mask(grid, dims.CellDim)
        rho_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_new = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_new = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_new = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            mask_prog_halo_c=mask_prog_halo_c,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_new=exner_new,
            exner_now=exner_now,
            rho_new=rho_new,
            theta_v_new=theta_v_new,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
