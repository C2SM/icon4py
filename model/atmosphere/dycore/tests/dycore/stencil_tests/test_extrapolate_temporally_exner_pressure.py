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

from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    extrapolate_temporally_exner_pressure,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def extrapolate_temporally_exner_pressure_numpy(
    connectivities: Mapping[gtx.Dimension, np.ndarray],
    exner: np.ndarray,
    exner_ref_mc: np.ndarray,
    exner_pr: np.ndarray,
    exner_exfac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_exner_ex_pr = (1 + exner_exfac) * (exner - exner_ref_mc) - exner_exfac * exner_pr
    exner_pr = exner - exner_ref_mc
    return (z_exner_ex_pr, exner_pr)


class TestExtrapolateTemporallyExnerPressure(StencilTest):
    PROGRAM = extrapolate_temporally_exner_pressure
    OUTPUTS = ("z_exner_ex_pr", "exner_pr")

    @static_reference
    def reference(
        grid: base.Grid,
        exner: np.ndarray,
        exner_ref_mc: np.ndarray,
        exner_pr: np.ndarray,
        exner_exfac: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        (z_exner_ex_pr, exner_pr) = extrapolate_temporally_exner_pressure_numpy(
            connectivities,
            exner=exner,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
            exner_exfac=exner_exfac,
        )

        return dict(z_exner_ex_pr=z_exner_ex_pr, exner_pr=exner_pr)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        exner = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_ref_mc = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        exner_pr = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_exfac = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_exner_ex_pr = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            exner_exfac=exner_exfac,
            exner=exner,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
            z_exner_ex_pr=z_exner_ex_pr,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
