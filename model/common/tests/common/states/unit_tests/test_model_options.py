# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from icon4py.model.common.model_backends import BACKENDS
from icon4py.model.common import dimension as dims
from icon4py.model.common.model_options import customize_backend
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common.grid.simple import simple_grid
from icon4py.model.testing import test_utils
import numpy as np


def test_custom_backend():
    backend_options = {
        "device": "cpu",  # or whatever it will be after the discussion
        "backend_kind": "gtfn",  # or the corresponding maker if we go that way
    }
    backend = customize_backend(**backend_options)
    field = data_alloc.zero_field(
        simple_grid(), dims.EdgeDim, dims.KDim, backend=backend, dtype=vpfloat
    )
    assert test_utils.dallclose(
        field.asnumpy(), np.zeros((simple_grid().num_edges, simple_grid().num_levels))
    )
