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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils.compute_curl import _compute_curl
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import reference_funcs, stencil_tests


class TestComputeCurl(stencil_tests.StencilTest):
    PROGRAM = _compute_curl
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vec_e: np.ndarray,
        geofac_rot: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(out=reference_funcs.compute_curl_numpy(connectivities, vec_e, geofac_rot))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        vec_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        geofac_rot = data_alloc.random_field(dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        return dict(
            vec_e=vec_e,
            geofac_rot=geofac_rot,
            out=data_alloc.zero_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.VertexDim: (0, gtx.int32(grid.num_vertices)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )
