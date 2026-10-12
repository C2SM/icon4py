# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import numpy as np

from icon4py.model.atmosphere.diffusion.stencils.limit_smag_coefficients import (
    _limit_smag_coefficients,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


class TestLimitSmagCoefficients(stencil_tests.StencilTest):
    PROGRAM = _limit_smag_coefficients
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        kh_smag: np.ndarray,
        smag_limit: np.ndarray,
        smag_offset: float,
        **kwargs: Any,
    ) -> dict:
        kh_smag_limited = np.maximum(0.0, kh_smag - smag_offset)
        kh_smag_limited = np.minimum(kh_smag_limited, smag_limit)
        return dict(out=kh_smag_limited)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        kh_smag = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        smag_limit = data_alloc.random_field(dims.KDim, dtype=ta.vpfloat)
        smag_offset = ta.vpfloat("9.0")

        return dict(
            kh_smag=kh_smag,
            smag_limit=smag_limit,
            smag_offset=smag_offset,
            out=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.vpfloat),
        )
