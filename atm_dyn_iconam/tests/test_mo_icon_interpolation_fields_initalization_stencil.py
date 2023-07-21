# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
#from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.field_management.interpolation_fields import (
    InterpolationFields,
)
from icon4py.common.dimension import EdgeDim, KDim

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def test_mo_icon_interpolation_fields_initalization():
    mesh = SimpleMesh()

    edge_cell_length = random_field(mesh, EdgeDim)
    dual_edge_length = random_field(mesh, EdgeDim)
    c_lin_e = (random_field(mesh, EdgeDim), random_field(mesh, EdgeDim))

    c_lin_e = InterpolationFields.initialization_1st_numpy(
        np.asarray(edge_cell_length),
        np.asarray(dual_edge_length),
    )

    c_lin_e_ref = interpolation_state.c_lin_e

    assert np.allclose(c_lin_e, c_lin_e_ref)
