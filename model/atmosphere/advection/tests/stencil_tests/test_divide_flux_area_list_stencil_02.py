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

from icon4py.model.atmosphere.advection.divide_flux_area_list_stencil_02 import (
    divide_flux_area_list_stencil_02,
)
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, random_mask


def divide_flux_area_list_stencil_02_numpy(
    e2c: np.array,
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
):
    famask_bool = np.where(famask_int == int32(1), True, False)
    lvn_pos = np.where(p_vn >= np.broadcast_to(0.0, p_vn.shape), True, False)
    # Translation of patch 1 and patch 2 in system relative to respective cell
    bf_cc_patch1_lon_e = np.expand_dims(bf_cc_patch1_lon, axis=-1)
    bf_cc_patch1_lat_e = np.expand_dims(bf_cc_patch1_lat, axis=-1)
    bf_cc_patch2_lon_e = np.expand_dims(bf_cc_patch2_lon, axis=-1)
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
    return (
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
        patch1_cell_idx_vmask,
        patch1_cell_blk_vmask,
        patch2_cell_idx_vmask,
        patch2_cell_blk_vmask,
    )


def test_divide_flux_area_list_stencil_02(backend):
    grid = SimpleGrid()

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

    (
        ref_1,
        ref_2,
        ref_3,
        ref_4,
        ref_5,
        ref_6,
        ref_7,
        ref_8,
        ref_9,
        ref_10,
        ref_11,
        ref_12,
        ref_13,
        ref_14,
        ref_15,
        ref_16,
        ref_17,
        ref_18,
        ref_19,
        ref_20,
    ) = divide_flux_area_list_stencil_02_numpy(
        grid.connectivities[E2CDim],
        famask_int.asnumpy(),
        p_vn.asnumpy(),
        bf_cc_patch1_lon.asnumpy(),
        bf_cc_patch1_lat.asnumpy(),
        bf_cc_patch2_lon.asnumpy(),
        bf_cc_patch2_lat.asnumpy(),
        butterfly_idx_patch1_vnpos.asnumpy(),
        butterfly_idx_patch1_vnneg.asnumpy(),
        butterfly_blk_patch1_vnpos.asnumpy(),
        butterfly_blk_patch1_vnneg.asnumpy(),
        butterfly_idx_patch2_vnpos.asnumpy(),
        butterfly_idx_patch2_vnneg.asnumpy(),
        butterfly_blk_patch2_vnpos.asnumpy(),
        butterfly_blk_patch2_vnneg.asnumpy(),
        dreg_patch1_1_lon_vmask.asnumpy(),
        dreg_patch1_1_lat_vmask.asnumpy(),
        dreg_patch1_2_lon_vmask.asnumpy(),
        dreg_patch1_2_lat_vmask.asnumpy(),
        dreg_patch1_3_lon_vmask.asnumpy(),
        dreg_patch1_3_lat_vmask.asnumpy(),
        dreg_patch1_4_lon_vmask.asnumpy(),
        dreg_patch1_4_lat_vmask.asnumpy(),
        dreg_patch2_1_lon_vmask.asnumpy(),
        dreg_patch2_1_lat_vmask.asnumpy(),
        dreg_patch2_2_lon_vmask.asnumpy(),
        dreg_patch2_2_lat_vmask.asnumpy(),
        dreg_patch2_3_lon_vmask.asnumpy(),
        dreg_patch2_3_lat_vmask.asnumpy(),
        dreg_patch2_4_lon_vmask.asnumpy(),
        dreg_patch2_4_lat_vmask.asnumpy(),
    )

    divide_flux_area_list_stencil_02.with_backend(backend)(
        famask_int,
        p_vn,
        bf_cc_patch1_lon_field,
        bf_cc_patch1_lat_field,
        bf_cc_patch2_lon_field,
        bf_cc_patch2_lat_field,
        butterfly_idx_patch1_vnpos,
        butterfly_idx_patch1_vnneg,
        butterfly_blk_patch1_vnpos,
        butterfly_blk_patch1_vnneg,
        butterfly_idx_patch2_vnpos,
        butterfly_idx_patch2_vnneg,
        butterfly_blk_patch2_vnpos,
        butterfly_blk_patch2_vnneg,
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
        patch1_cell_idx_vmask,
        patch1_cell_blk_vmask,
        patch2_cell_idx_vmask,
        patch2_cell_blk_vmask,
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, grid.size[E2CDim]),
        },
    )
    assert np.allclose(dreg_patch1_1_lon_vmask.asnumpy(), ref_1)
    assert np.allclose(dreg_patch1_1_lat_vmask.asnumpy(), ref_2)
    assert np.allclose(dreg_patch1_2_lon_vmask.asnumpy(), ref_3)
    assert np.allclose(dreg_patch1_2_lat_vmask.asnumpy(), ref_4)
    assert np.allclose(dreg_patch1_3_lon_vmask.asnumpy(), ref_5)
    assert np.allclose(dreg_patch1_3_lat_vmask.asnumpy(), ref_6)
    assert np.allclose(dreg_patch1_4_lon_vmask.asnumpy(), ref_7)
    assert np.allclose(dreg_patch1_4_lat_vmask.asnumpy(), ref_8)
    assert np.allclose(dreg_patch2_1_lon_vmask.asnumpy(), ref_9)
    assert np.allclose(dreg_patch2_1_lat_vmask.asnumpy(), ref_10)
    assert np.allclose(dreg_patch2_2_lon_vmask.asnumpy(), ref_11)
    assert np.allclose(dreg_patch2_2_lat_vmask.asnumpy(), ref_12)
    assert np.allclose(dreg_patch2_3_lon_vmask.asnumpy(), ref_13)
    assert np.allclose(dreg_patch2_3_lat_vmask.asnumpy(), ref_14)
    assert np.allclose(dreg_patch2_4_lon_vmask.asnumpy(), ref_15)
    assert np.allclose(dreg_patch2_4_lat_vmask.asnumpy(), ref_16)
    assert np.allclose(patch1_cell_idx_vmask.asnumpy(), ref_17)
    assert np.allclose(patch1_cell_blk_vmask.asnumpy(), ref_18)
    assert np.allclose(patch2_cell_idx_vmask.asnumpy(), ref_19)
    assert np.allclose(patch2_cell_blk_vmask.asnumpy(), ref_20)
