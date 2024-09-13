# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2EC
from icon4py.model.common.type_alias import vpfloat, wpfloat


sys.setrecursionlimit(5500)


@field_operator
def _compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[int32],
    p_vn: fa.EdgeKField[wpfloat],
    bf_cc_patch1_lon: Field[[dims.ECDim], wpfloat],
    bf_cc_patch1_lat: Field[[dims.ECDim], wpfloat],
    bf_cc_patch2_lon: Field[[dims.ECDim], wpfloat],
    bf_cc_patch2_lat: Field[[dims.ECDim], wpfloat],
    butterfly_idx_patch1_vnpos: fa.EdgeField[int32],
    butterfly_idx_patch1_vnneg: fa.EdgeField[int32],
    butterfly_blk_patch1_vnpos: fa.EdgeField[int32],
    butterfly_blk_patch1_vnneg: fa.EdgeField[int32],
    butterfly_idx_patch2_vnpos: fa.EdgeField[int32],
    butterfly_idx_patch2_vnneg: fa.EdgeField[int32],
    butterfly_blk_patch2_vnpos: fa.EdgeField[int32],
    butterfly_blk_patch2_vnneg: fa.EdgeField[int32],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[vpfloat],
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[int32],
    fa.EdgeKField[int32],
    fa.EdgeKField[int32],
    fa.EdgeKField[int32],
]:
    famask_bool = where(famask_int == 1, True, False)
    lvn_pos = where(p_vn >= wpfloat(0.0), True, False)
    # Translation of patch 1 and patch 2 in system relative to respective cell
    bf_cc_patch1_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lon(E2EC[0]), bf_cc_patch1_lon(E2EC[1])),
        wpfloat(0.0),
    )
    bf_cc_patch1_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lat(E2EC[0]), bf_cc_patch1_lat(E2EC[1])),
        wpfloat(0.0),
    )
    bf_cc_patch2_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lon(E2EC[0]), bf_cc_patch2_lon(E2EC[1])),
        wpfloat(0.0),
    )
    bf_cc_patch2_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lat(E2EC[0]), bf_cc_patch2_lat(E2EC[1])),
        wpfloat(0.0),
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
    butterfly_idx_patch1_vnpos_3d = broadcast(butterfly_idx_patch1_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch1_vnneg_3d = broadcast(butterfly_idx_patch1_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch2_vnpos_3d = broadcast(butterfly_idx_patch2_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch2_vnneg_3d = broadcast(butterfly_idx_patch2_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch1_vnpos_3d = broadcast(butterfly_blk_patch1_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch1_vnneg_3d = broadcast(butterfly_blk_patch1_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch2_vnpos_3d = broadcast(butterfly_blk_patch2_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch2_vnneg_3d = broadcast(butterfly_blk_patch2_vnneg, (dims.EdgeDim, dims.KDim))
    patch1_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch1_vnpos_3d, butterfly_idx_patch1_vnneg_3d),
        0,
    )
    patch2_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch2_vnpos_3d, butterfly_idx_patch2_vnneg_3d),
        0,
    )
    patch1_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch1_vnpos_3d, butterfly_blk_patch1_vnneg_3d),
        0,
    )
    patch2_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch2_vnpos_3d, butterfly_blk_patch2_vnneg_3d),
        0,
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
def compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[int32],
    p_vn: fa.EdgeKField[wpfloat],
    bf_cc_patch1_lon: Field[[dims.ECDim], wpfloat],
    bf_cc_patch1_lat: Field[[dims.ECDim], wpfloat],
    bf_cc_patch2_lon: Field[[dims.ECDim], wpfloat],
    bf_cc_patch2_lat: Field[[dims.ECDim], wpfloat],
    butterfly_idx_patch1_vnpos: fa.EdgeField[int32],
    butterfly_idx_patch1_vnneg: fa.EdgeField[int32],
    butterfly_blk_patch1_vnpos: fa.EdgeField[int32],
    butterfly_blk_patch1_vnneg: fa.EdgeField[int32],
    butterfly_idx_patch2_vnpos: fa.EdgeField[int32],
    butterfly_idx_patch2_vnneg: fa.EdgeField[int32],
    butterfly_blk_patch2_vnpos: fa.EdgeField[int32],
    butterfly_blk_patch2_vnneg: fa.EdgeField[int32],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[vpfloat],
    patch1_cell_idx_vmask: fa.EdgeKField[int32],
    patch1_cell_blk_vmask: fa.EdgeKField[int32],
    patch2_cell_idx_vmask: fa.EdgeKField[int32],
    patch2_cell_blk_vmask: fa.EdgeKField[int32],
):
    _compute_ffsl_flux_area_list(
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
