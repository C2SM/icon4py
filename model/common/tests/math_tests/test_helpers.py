# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import fields as field_utils
from icon4py.model.common.grid import simple
from icon4py.model.common.math import helpers
from icon4py.model.common.settings import xp
from icon4py.model.testing import helpers as test_helpers


def test_cross_product(backend):
    mesh = simple.SimpleGrid()
    x1 = field_utils.random_field(mesh, dims.EdgeDim)
    y1 = field_utils.random_field(mesh, dims.EdgeDim)
    z1 = field_utils.random_field(mesh, dims.EdgeDim)
    x2 = field_utils.random_field(mesh, dims.EdgeDim)
    y2 = field_utils.random_field(mesh, dims.EdgeDim)
    z2 = field_utils.random_field(mesh, dims.EdgeDim)
    x = field_utils.zero_field(mesh, dims.EdgeDim)
    y = field_utils.zero_field(mesh, dims.EdgeDim)
    z = field_utils.zero_field(mesh, dims.EdgeDim)

    helpers.cross_product_on_edges.with_backend(backend)(
        x1, x2, y1, y2, z1, z2, out=(x, y, z), offset_provider={}
    )
    a = xp.column_stack((x1.ndarray, y1.ndarray, z1.ndarray))
    b = xp.column_stack((x2.ndarray, y2.ndarray, z2.ndarray))
    c = xp.cross(a, b)

    assert test_helpers.dallclose(c[:, 0], x.ndarray)
    assert test_helpers.dallclose(c[:, 1], y.ndarray)
    assert test_helpers.dallclose(c[:, 2], z.ndarray)
