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

from icon4py.model.atmosphere.dycore.stencils.update_mass_flux_weighted import (
    update_mass_flux_weighted,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


@pytest.mark.continuous_benchmarking
class TestUpdateMassFluxWeighted(StencilTest):
    PROGRAM = update_mass_flux_weighted
    OUTPUTS = ("mass_flx_ic",)
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        rho_ic: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        exner_w_implicit_weight_parameter: np.ndarray,
        w_now: np.ndarray,
        w_new: np.ndarray,
        w_concorr_c: np.ndarray,
        mass_flx_ic: np.ndarray,
        r_nsubsteps: float,
        **kwargs: Any,
    ) -> dict:
        exner_w_explicit_weight_parameter = np.expand_dims(exner_w_explicit_weight_parameter, axis=-1)
        exner_w_implicit_weight_parameter = np.expand_dims(exner_w_implicit_weight_parameter, axis=-1)
        mass_flx_ic = mass_flx_ic + (
            r_nsubsteps * rho_ic * (exner_w_explicit_weight_parameter * w_now + exner_w_implicit_weight_parameter * w_new - w_concorr_c)
        )
        return dict(mass_flx_ic=mass_flx_ic)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        r_nsubsteps = wpfloat("10.0")
        rho_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_w_explicit_weight_parameter = random_field(grid, dims.CellDim, dtype=wpfloat)
        exner_w_implicit_weight_parameter = random_field(grid, dims.CellDim, dtype=wpfloat)
        w_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_new = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_concorr_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        mass_flx_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            rho_ic=rho_ic,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            w_now=w_now,
            w_new=w_new,
            w_concorr_c=w_concorr_c,
            mass_flx_ic=mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
