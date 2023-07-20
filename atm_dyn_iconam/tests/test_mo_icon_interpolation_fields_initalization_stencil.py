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

from icon4py.field_management.mo_icon_interpolation_fields_initalization_stencil import (
    mo_icon_interpolation_fields_initalization_stencil,
)
from icon4py.common.dimension import EdgeDim, KDim

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_icon_interpolation_fields_initalization_stencil_numpy(
    edge_cell_length: np.array,
    dual_edge_length: np.array,
) -> np.array:
    c_lin_e = (edge_cell_length[:] / dual_edge_length[:], 1.0 - edge_cell_length[:] / dual_edge_length[:])
#    c_lin_e[1, :] = 1.0 - c_lin_e[0, :]
    return c_lin_e


def test_mo_icon_interpolation_fields_initalization_stencil():
    mesh = SimpleMesh()

    edge_cell_length = random_field(mesh, EdgeDim)
    dual_edge_length = random_field(mesh, EdgeDim)
    c_lin_e = (random_field(mesh, EdgeDim), random_field(mesh, EdgeDim))

    c_lin_e_ref = mo_icon_interpolation_fields_initalization_stencil_numpy(
        np.asarray(edge_cell_length),
        np.asarray(dual_edge_length),
    )

    mo_icon_interpolation_fields_initalization_stencil(
        edge_cell_length,
        dual_edge_length,
        c_lin_e,
        offset_provider={
                "E2V": mesh.get_e2v_offset_provider(),
                "E2C": mesh.get_e2c_offset_provider(),
#                "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
                "E2EC": EdgeDim,
                "Koff": KDim,
        },
    )

    assert np.allclose(c_lin_e, c_lin_e_ref)
