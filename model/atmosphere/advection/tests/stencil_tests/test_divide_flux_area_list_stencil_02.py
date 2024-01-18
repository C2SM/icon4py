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
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.divide_flux_area_list_stencil_02 import (
    divide_flux_area_list_stencil_02,
)
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    numpy_to_1D_sparse_field,
    random_field,
    random_mask,
    reshape,
)


class TestDivideFluxAreaListStencil02(StencilTest):
    PROGRAM = divide_flux_area_list_stencil_02
    OUTPUTS = (
        "dreg_patch1_1_lon_vmask",
        "dreg_patch1_1_lat_vmask",
        "dreg_patch1_2_lon_vmask",
        "dreg_patch1_2_lat_vmask",
        "dreg_patch1_3_lon_vmask",
        "dreg_patch1_3_lat_vmask",
        "dreg_patch1_4_lon_vmask",
        "dreg_patch1_4_lat_vmask",
        "dreg_patch2_1_lon_vmask",
        "dreg_patch2_1_lat_vmask",
        "dreg_patch2_2_lon_vmask",
        "dreg_patch2_2_lat_vmask",
        "dreg_patch2_3_lon_vmask",
        "dreg_patch2_3_lat_vmask",
        "dreg_patch2_4_lon_vmask",
        "dreg_patch2_4_lat_vmask",
        "patch1_cell_idx_vmask",
        "patch1_cell_blk_vmask",
        "patch2_cell_idx_vmask",
        "patch2_cell_blk_vmask",
    )

    @staticmethod
    def reference(
        grid,
        famask_int: np.array,
        p_vn: np.array,
        bf_cc_patch1_lon: np.array,
        bf_cc_patch1_lat: np.array,
        bf_cc_patch2_lon: np.array,
        bf_cc_patch2_lat: np.array,
        butterfly_idx_patch1_vnpos: np.array,
        butterfly_idx_patch1_vnneg: np.array,
        butterfly_blk_patch1_vnpos: np.array,
        butterfly_blk_patch1_vnneg: np.array,
        butterfly_idx_patch2_vnpos: np.array,
        butterfly_idx_patch2_vnneg: np.array,
        butterfly_blk_patch2_vnpos: np.array,
        butterfly_blk_patch2_vnneg: np.array,
        dreg_patch1_1_lon_vmask: np.array,
        dreg_patch1_1_lat_vmask: np.array,
        dreg_patch1_2_lon_vmask: np.array,
        dreg_patch1_2_lat_vmask: np.array,
        dreg_patch1_3_lon_vmask: np.array,
        dreg_patch1_3_lat_vmask: np.array,
        dreg_patch1_4_lon_vmask: np.array,
        dreg_patch1_4_lat_vmask: np.array,
        dreg_patch2_1_lon_vmask: np.array,
        dreg_patch2_1_lat_vmask: np.array,
        dreg_patch2_2_lon_vmask: np.array,
        dreg_patch2_2_lat_vmask: np.array,
        dreg_patch2_3_lon_vmask: np.array,
        dreg_patch2_3_lat_vmask: np.array,
        dreg_patch2_4_lon_vmask: np.array,
        dreg_patch2_4_lat_vmask: np.array,
        **kwargs,
    ):
        e2c = grid.connectivities[E2CDim]
        famask_bool = np.where(famask_int == int32(1), True, False)
        lvn_pos = np.where(p_vn >= np.broadcast_to(0.0, p_vn.shape), True, False)
        # Translation of patch 1 and patch 2 in system relative to respective cell
        bf_cc_patch1_lon = reshape(bf_cc_patch1_lon, e2c.shape)
        bf_cc_patch1_lon_e = np.expand_dims(bf_cc_patch1_lon, axis=-1)
        bf_cc_patch1_lat = reshape(bf_cc_patch1_lat, e2c.shape)
        bf_cc_patch1_lat_e = np.expand_dims(bf_cc_patch1_lat, axis=-1)
        bf_cc_patch2_lon = reshape(bf_cc_patch2_lon, e2c.shape)
        bf_cc_patch2_lon_e = np.expand_dims(bf_cc_patch2_lon, axis=-1)
        bf_cc_patch2_lat = reshape(bf_cc_patch2_lat, e2c.shape)
        bf_cc_patch2_lat_e = np.expand_dims(bf_cc_patch2_lat, axis=-1)

        bf_cc_patch1_lon = np.where(
            famask_bool,
            np.where(lvn_pos, bf_cc_patch1_lon_e[:, 0], bf_cc_patch1_lon_e[:, 1]),
            0.0,
        )
        bf_cc_patch1_lat = np.where(
            famask_bool,
            np.where(lvn_pos, bf_cc_patch1_lat_e[:, 0], bf_cc_patch1_lat_e[:, 1]),
            0.0,
        )
        bf_cc_patch2_lon = np.where(
            famask_bool,
            np.where(lvn_pos, bf_cc_patch2_lon_e[:, 0], bf_cc_patch2_lon_e[:, 1]),
            0.0,
        )
        bf_cc_patch2_lat = np.where(
            famask_bool,
            np.where(lvn_pos, bf_cc_patch2_lat_e[:, 0], bf_cc_patch2_lat_e[:, 1]),
            0.0,
        )

        # patch1 in translated system
        dreg_patch1_1_lon_vmask = dreg_patch1_1_lon_vmask - bf_cc_patch1_lon
        dreg_patch1_1_lat_vmask = dreg_patch1_1_lat_vmask - bf_cc_patch1_lat
        dreg_patch1_2_lon_vmask = dreg_patch1_2_lon_vmask - bf_cc_patch1_lon
        dreg_patch1_2_lat_vmask = dreg_patch1_2_lat_vmask - bf_cc_patch1_lat
        dreg_patch1_3_lon_vmask = dreg_patch1_3_lon_vmask - bf_cc_patch1_lon
        dreg_patch1_3_lat_vmask = dreg_patch1_3_lat_vmask - bf_cc_patch1_lat
        dreg_patch1_4_lon_vmask = dreg_patch1_4_lon_vmask - bf_cc_patch1_lon
        dreg_patch1_4_lat_vmask = dreg_patch1_4_lat_vmask - bf_cc_patch1_lat
        # patch2 in translated system
        dreg_patch2_1_lon_vmask = dreg_patch2_1_lon_vmask - bf_cc_patch2_lon
        dreg_patch2_1_lat_vmask = dreg_patch2_1_lat_vmask - bf_cc_patch2_lat
        dreg_patch2_2_lon_vmask = dreg_patch2_2_lon_vmask - bf_cc_patch2_lon
        dreg_patch2_2_lat_vmask = dreg_patch2_2_lat_vmask - bf_cc_patch2_lat
        dreg_patch2_3_lon_vmask = dreg_patch2_3_lon_vmask - bf_cc_patch2_lon
        dreg_patch2_3_lat_vmask = dreg_patch2_3_lat_vmask - bf_cc_patch2_lat
        dreg_patch2_4_lon_vmask = dreg_patch2_4_lon_vmask - bf_cc_patch2_lon
        dreg_patch2_4_lat_vmask = dreg_patch2_4_lat_vmask - bf_cc_patch2_lat

        # Store global index of the underlying grid cell
        # Adapt dimensions to fit ofr multiple levels
        butterfly_idx_patch1_vnpos_3d = np.broadcast_to(
            np.expand_dims(butterfly_idx_patch1_vnpos, axis=-1), p_vn.shape
        )
        butterfly_idx_patch1_vnneg_3d = np.broadcast_to(
            np.expand_dims(butterfly_idx_patch1_vnneg, axis=-1), p_vn.shape
        )
        butterfly_idx_patch2_vnpos_3d = np.broadcast_to(
            np.expand_dims(butterfly_idx_patch2_vnpos, axis=-1), p_vn.shape
        )
        butterfly_idx_patch2_vnneg_3d = np.broadcast_to(
            np.expand_dims(butterfly_idx_patch2_vnneg, axis=-1), p_vn.shape
        )
        butterfly_blk_patch1_vnpos_3d = np.broadcast_to(
            np.expand_dims(butterfly_blk_patch1_vnpos, axis=-1), p_vn.shape
        )
        butterfly_blk_patch1_vnneg_3d = np.broadcast_to(
            np.expand_dims(butterfly_blk_patch1_vnneg, axis=-1), p_vn.shape
        )
        butterfly_blk_patch2_vnpos_3d = np.broadcast_to(
            np.expand_dims(butterfly_blk_patch2_vnpos, axis=-1), p_vn.shape
        )
        butterfly_blk_patch2_vnneg_3d = np.broadcast_to(
            np.expand_dims(butterfly_blk_patch2_vnneg, axis=-1), p_vn.shape
        )
        patch1_cell_idx_vmask = np.where(
            famask_bool,
            np.where(lvn_pos, butterfly_idx_patch1_vnpos_3d, butterfly_idx_patch1_vnneg_3d),
            int32(0),
        )
        patch2_cell_idx_vmask = np.where(
            famask_bool,
            np.where(lvn_pos, butterfly_idx_patch2_vnpos_3d, butterfly_idx_patch2_vnneg_3d),
            int32(0),
        )
        patch1_cell_blk_vmask = np.where(
            famask_bool,
            np.where(lvn_pos, butterfly_blk_patch1_vnpos_3d, butterfly_blk_patch1_vnneg_3d),
            int32(0),
        )
        patch2_cell_blk_vmask = np.where(
            famask_bool,
            np.where(lvn_pos, butterfly_blk_patch2_vnpos_3d, butterfly_blk_patch2_vnneg_3d),
            int32(0),
        )

        return dict(
            dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
            dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
            patch1_cell_idx_vmask=patch1_cell_idx_vmask,
            patch1_cell_blk_vmask=patch1_cell_blk_vmask,
            patch2_cell_idx_vmask=patch2_cell_idx_vmask,
            patch2_cell_blk_vmask=patch2_cell_blk_vmask,
        )

    @pytest.fixture
    def input_data(self, grid):
        famask_int = random_mask(grid, EdgeDim, KDim, dtype=int32)
        p_vn = random_field(grid, EdgeDim, KDim)
        bf_cc_patch1_lon = random_field(grid, EdgeDim, E2CDim)
        bf_cc_patch1_lon_field = as_1D_sparse_field(bf_cc_patch1_lon, ECDim)
        bf_cc_patch1_lat = random_field(grid, EdgeDim, E2CDim)
        bf_cc_patch1_lat_field = as_1D_sparse_field(bf_cc_patch1_lat, ECDim)
        bf_cc_patch2_lon = random_field(grid, EdgeDim, E2CDim)
        bf_cc_patch2_lon_field = as_1D_sparse_field(bf_cc_patch2_lon, ECDim)
        bf_cc_patch2_lat = random_field(grid, EdgeDim, E2CDim)
        bf_cc_patch2_lat_field = as_1D_sparse_field(bf_cc_patch2_lat, ECDim)
        butterfly_idx_patch1_vnpos = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_idx_patch1_vnneg = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_blk_patch1_vnpos = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_blk_patch1_vnneg = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_idx_patch2_vnpos = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_idx_patch2_vnneg = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_blk_patch2_vnpos = random_mask(grid, EdgeDim, dtype=int32)
        butterfly_blk_patch2_vnneg = random_mask(grid, EdgeDim, dtype=int32)
        dreg_patch1_1_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_1_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_2_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_2_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_3_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_3_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_4_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch1_4_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_1_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_1_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_2_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_2_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_3_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_3_lat_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_4_lon_vmask = random_field(grid, EdgeDim, KDim)
        dreg_patch2_4_lat_vmask = random_field(grid, EdgeDim, KDim)
        patch1_cell_idx_vmask = random_mask(grid, EdgeDim, KDim, dtype=int32)
        patch1_cell_blk_vmask = random_mask(grid, EdgeDim, KDim, dtype=int32)
        patch2_cell_idx_vmask = random_mask(grid, EdgeDim, KDim, dtype=int32)
        patch2_cell_blk_vmask = random_mask(grid, EdgeDim, KDim, dtype=int32)

        return dict(
            famask_int=famask_int,
            p_vn=p_vn,
            bf_cc_patch1_lon=bf_cc_patch1_lon_field,
            bf_cc_patch1_lat=bf_cc_patch1_lat_field,
            bf_cc_patch2_lon=bf_cc_patch2_lon_field,
            bf_cc_patch2_lat=bf_cc_patch2_lat_field,
            butterfly_idx_patch1_vnpos=butterfly_idx_patch1_vnpos,
            butterfly_idx_patch1_vnneg=butterfly_idx_patch1_vnneg,
            butterfly_blk_patch1_vnpos=butterfly_blk_patch1_vnpos,
            butterfly_blk_patch1_vnneg=butterfly_blk_patch1_vnneg,
            butterfly_idx_patch2_vnpos=butterfly_idx_patch2_vnpos,
            butterfly_idx_patch2_vnneg=butterfly_idx_patch2_vnneg,
            butterfly_blk_patch2_vnpos=butterfly_blk_patch2_vnpos,
            butterfly_blk_patch2_vnneg=butterfly_blk_patch2_vnneg,
            dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
            dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
            patch1_cell_idx_vmask=patch1_cell_idx_vmask,
            patch1_cell_blk_vmask=patch1_cell_blk_vmask,
            patch2_cell_idx_vmask=patch2_cell_idx_vmask,
            patch2_cell_blk_vmask=patch2_cell_blk_vmask,
        )
