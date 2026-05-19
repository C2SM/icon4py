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

from icon4py.model.atmosphere.dycore.stencils.update_theta_and_exner_in_halo import (
    update_theta_and_exner_in_halo,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestUpdateThetaV(StencilTest):
    PROGRAM = update_theta_and_exner_in_halo
    OUTPUTS = (
        "theta_v_new",
        "exner_new",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        mask_prog_halo_c: np.ndarray,
        current_rho: np.ndarray,
        rho_new: np.ndarray,
        current_theta_v: np.ndarray,
        theta_v_new: np.ndarray,
        current_exner: np.ndarray,
        exner_new: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        mask_prog_halo_c = np.expand_dims(mask_prog_halo_c, axis=-1)

        theta_v_new = np.where(mask_prog_halo_c != 1, exner_new, theta_v_new)
        exner_new = np.where(
            mask_prog_halo_c != 1,
            np.exp(constants.RD_O_CVD * np.log(constants.RD_O_P0REF * rho_new * exner_new)),
            exner_new,
        )

        theta_v_new = np.where(
            mask_prog_halo_c,
            current_rho
            * current_theta_v
            * ((exner_new / current_exner - 1) * constants.CVD_O_RD + 1.0)
            / rho_new,
            theta_v_new,
        )
        return dict(theta_v_new=theta_v_new, exner_new=exner_new)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        mask_prog_halo_c = data_alloc.random_mask(grid, dims.CellDim)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            mask_prog_halo_c=mask_prog_halo_c,
            current_rho=current_rho,
            rho_new=rho_new,
            current_theta_v=current_theta_v,
            theta_v_new=theta_v_new,
            current_exner=current_exner,
            exner_new=exner_new,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
