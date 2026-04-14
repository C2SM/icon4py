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

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def add_analysis_increments_from_data_assimilation_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    rho_explicit_term: np.ndarray,
    rho_iau_increment: np.ndarray,
    exner_explicit_term: np.ndarray,
    exner_iau_increment: np.ndarray,
    iau_wgt_dyn: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho_explicit_term = rho_explicit_term + iau_wgt_dyn * rho_iau_increment
    exner_explicit_term = exner_explicit_term + iau_wgt_dyn * exner_iau_increment
    return (rho_explicit_term, exner_explicit_term)


class TestAddAnalysisIncrementsFromDataAssimilation(StencilTest):
    PROGRAM = add_analysis_increments_from_data_assimilation
    OUTPUTS = ("rho_explicit_term", "exner_explicit_term")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        rho_explicit_term: np.ndarray,
        rho_iau_increment: np.ndarray,
        exner_explicit_term: np.ndarray,
        exner_iau_increment: np.ndarray,
        iau_wgt_dyn: float,
        **kwargs: Any,
    ) -> dict:
        rho_explicit_term, exner_explicit_term = add_analysis_increments_from_data_assimilation_numpy(
            connectivities,
            rho_explicit_term=rho_explicit_term,
            rho_iau_increment=rho_iau_increment,
            exner_explicit_term=exner_explicit_term,
            exner_iau_increment=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        return dict(rho_explicit_term=rho_explicit_term, exner_explicit_term=exner_explicit_term)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        exner_explicit_term = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        rho_explicit_term = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_iau_increment = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        iau_wgt_dyn = ta.wpfloat("8.0")

        return dict(
            rho_explicit_term=rho_explicit_term,
            exner_explicit_term=exner_explicit_term,
            rho_iau_increment=rho_iau_increment,
            exner_iau_increment=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
