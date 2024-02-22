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

import sys

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.model.common.dimension import E2EC, ECDim, EdgeDim, KDim


sys.setrecursionlimit(5500)


@field_operator
def _divide_flux_area_list_stencil_02(
    famask_int: Field[[EdgeDim, KDim], int32],
    p_vn: Field[[EdgeDim, KDim], float],
    bf_cc_patch1_lon: Field[[ECDim], float],
    bf_cc_patch1_lat: Field[[ECDim], float],
    bf_cc_patch2_lon: Field[[ECDim], float],
    bf_cc_patch2_lat: Field[[ECDim], float],
    butterfly_idx_patch1_vnpos: Field[[EdgeDim], int32],
    butterfly_idx_patch1_vnneg: Field[[EdgeDim], int32],
    butterfly_blk_patch1_vnpos: Field[[EdgeDim], int32],
    butterfly_blk_patch1_vnneg: Field[[EdgeDim], int32],
    butterfly_idx_patch2_vnpos: Field[[EdgeDim], int32],
    butterfly_idx_patch2_vnneg: Field[[EdgeDim], int32],
    butterfly_blk_patch2_vnpos: Field[[EdgeDim], int32],
    butterfly_blk_patch2_vnneg: Field[[EdgeDim], int32],
    dreg_patch1_1_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_1_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_2_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_2_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_3_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_3_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_4_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_4_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_1_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_1_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_2_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_2_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_3_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_3_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_4_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_4_lat_vmask: Field[[EdgeDim, KDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
    Field[[EdgeDim, KDim], int32],
]:
    famask_bool = where(famask_int == int32(1), True, False)
    lvn_pos = where(p_vn >= 0.0, True, False)
    # Translation of patch 1 and patch 2 in system relative to respective cell
    bf_cc_patch1_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lon(E2EC[0]), bf_cc_patch1_lon(E2EC[1])),
        0.0,
    )
    bf_cc_patch1_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lat(E2EC[0]), bf_cc_patch1_lat(E2EC[1])),
        0.0,
    )
    bf_cc_patch2_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lon(E2EC[0]), bf_cc_patch2_lon(E2EC[1])),
        0.0,
    )
    bf_cc_patch2_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lat(E2EC[0]), bf_cc_patch2_lat(E2EC[1])),
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
    butterfly_idx_patch1_vnpos_3d = broadcast(butterfly_idx_patch1_vnpos, (EdgeDim, KDim))
    butterfly_idx_patch1_vnneg_3d = broadcast(butterfly_idx_patch1_vnneg, (EdgeDim, KDim))
    butterfly_idx_patch2_vnpos_3d = broadcast(butterfly_idx_patch2_vnpos, (EdgeDim, KDim))
    butterfly_idx_patch2_vnneg_3d = broadcast(butterfly_idx_patch2_vnneg, (EdgeDim, KDim))
    butterfly_blk_patch1_vnpos_3d = broadcast(butterfly_blk_patch1_vnpos, (EdgeDim, KDim))
    butterfly_blk_patch1_vnneg_3d = broadcast(butterfly_blk_patch1_vnneg, (EdgeDim, KDim))
    butterfly_blk_patch2_vnpos_3d = broadcast(butterfly_blk_patch2_vnpos, (EdgeDim, KDim))
    butterfly_blk_patch2_vnneg_3d = broadcast(butterfly_blk_patch2_vnneg, (EdgeDim, KDim))
    patch1_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch1_vnpos_3d, butterfly_idx_patch1_vnneg_3d),
        int32(0),
    )
    patch2_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch2_vnpos_3d, butterfly_idx_patch2_vnneg_3d),
        int32(0),
    )
    patch1_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch1_vnpos_3d, butterfly_blk_patch1_vnneg_3d),
        int32(0),
    )
    patch2_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch2_vnpos_3d, butterfly_blk_patch2_vnneg_3d),
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


@program(grid_type=GridType.UNSTRUCTURED)
def divide_flux_area_list_stencil_02(
    famask_int: Field[[EdgeDim, KDim], int32],
    p_vn: Field[[EdgeDim, KDim], float],
    bf_cc_patch1_lon: Field[[ECDim], float],
    bf_cc_patch1_lat: Field[[ECDim], float],
    bf_cc_patch2_lon: Field[[ECDim], float],
    bf_cc_patch2_lat: Field[[ECDim], float],
    butterfly_idx_patch1_vnpos: Field[[EdgeDim], int32],
    butterfly_idx_patch1_vnneg: Field[[EdgeDim], int32],
    butterfly_blk_patch1_vnpos: Field[[EdgeDim], int32],
    butterfly_blk_patch1_vnneg: Field[[EdgeDim], int32],
    butterfly_idx_patch2_vnpos: Field[[EdgeDim], int32],
    butterfly_idx_patch2_vnneg: Field[[EdgeDim], int32],
    butterfly_blk_patch2_vnpos: Field[[EdgeDim], int32],
    butterfly_blk_patch2_vnneg: Field[[EdgeDim], int32],
    dreg_patch1_1_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_1_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_2_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_2_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_3_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_3_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_4_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch1_4_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_1_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_1_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_2_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_2_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_3_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_3_lat_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_4_lon_vmask: Field[[EdgeDim, KDim], float],
    dreg_patch2_4_lat_vmask: Field[[EdgeDim, KDim], float],
    patch1_cell_idx_vmask: Field[[EdgeDim, KDim], int32],
    patch1_cell_blk_vmask: Field[[EdgeDim, KDim], int32],
    patch2_cell_idx_vmask: Field[[EdgeDim, KDim], int32],
    patch2_cell_blk_vmask: Field[[EdgeDim, KDim], int32],
):
    _divide_flux_area_list_stencil_02(
        famask_int,
        p_vn,
        bf_cc_patch1_lon,
        bf_cc_patch1_lat,
        bf_cc_patch2_lon,
        bf_cc_patch2_lat,
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
        out=(
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
        ),
    )
