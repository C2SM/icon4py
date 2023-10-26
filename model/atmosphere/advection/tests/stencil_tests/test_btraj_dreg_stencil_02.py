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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atmosphere.advection.btraj_dreg_stencil_02 import btraj_dreg_stencil_02
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, zero_field


def btraj_dreg_stencil_02_numpy(
    p_vn: np.array,
    p_vt: np.array,
    edge_cell_length: np.array,
    p_dt: float,
):
    lvn_pos = np.where(p_vn >= 0.0, True, False)

    traj_length = np.sqrt(p_vn**2 + p_vt**2) * p_dt

    edge_cell_length = np.expand_dims(edge_cell_length, axis=-1)
    e2c_length = np.where(lvn_pos, edge_cell_length[:, 0], edge_cell_length[:, 1])

    opt_famask_dsl = np.where(
        traj_length > (1.25 * np.broadcast_to(e2c_length, p_vn.shape)),
        int32(1),
        int32(0),
    )

    return opt_famask_dsl


def test_btraj_dreg_stencil_02():
    grid = SimpleGrid()
    p_vn = random_field(grid, EdgeDim, KDim)
    p_vt = random_field(grid, EdgeDim, KDim)
    edge_cell_length = np.asarray(grid.connectivities[E2CDim], dtype=float)
    edge_cell_length_new = as_1D_sparse_field(edge_cell_length, ECDim)
    p_dt = 1.0
    opt_famask_dsl = zero_field(grid, EdgeDim, KDim, dtype=int32)

    ref = btraj_dreg_stencil_02_numpy(
        np.asarray(p_vn), np.asarray(p_vt), np.asarray(edge_cell_length), p_dt
    )

    btraj_dreg_stencil_02(
        p_vn,
        p_vt,
        edge_cell_length_new,
        p_dt,
        opt_famask_dsl,
        offset_provider={
            "E2C": grid.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, grid.size[E2CDim]),
        },
    )

    assert np.allclose(ref, opt_famask_dsl)
