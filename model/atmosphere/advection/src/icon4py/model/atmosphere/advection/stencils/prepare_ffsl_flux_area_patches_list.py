# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import astype, broadcast, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2EC
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


sys.setrecursionlimit(5500)


# FUNCTIONS
# Checking turn when travelling along three points, used to check whether lines inters.
@gtx.field_operator
def ccw(
    p0_lon: fa.EdgeKField[ta.wpfloat],
    p0_lat: fa.EdgeKField[ta.wpfloat],
    p1_lon: fa.EdgeKField[ta.wpfloat],
    p1_lat: fa.EdgeKField[ta.wpfloat],
    p2_lon: fa.EdgeKField[ta.wpfloat],
    p2_lat: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[gtx.int32]:
    dx1 = p1_lon - p0_lon
    dy1 = p1_lat - p0_lat

    dx2 = p2_lon - p0_lon
    dy2 = p2_lat - p0_lat

    dx1dy2 = dx1 * dy2
    dy1dx2 = dy1 * dx2

    lccw = where(dx1dy2 > dy1dx2, True, False)
    ccw_out = where(lccw, 1, -1)  # 1: clockwise, -1: counterclockwise
    return ccw_out


# Checks whether two lines intersect
@gtx.field_operator
def lintersect(
    line1_p1_lon: fa.EdgeKField[ta.wpfloat],
    line1_p1_lat: fa.EdgeKField[ta.wpfloat],
    line1_p2_lon: fa.EdgeKField[ta.wpfloat],
    line1_p2_lat: fa.EdgeKField[ta.wpfloat],
    line2_p1_lon: fa.EdgeKField[ta.wpfloat],
    line2_p1_lat: fa.EdgeKField[ta.wpfloat],
    line2_p2_lon: fa.EdgeKField[ta.wpfloat],
    line2_p2_lat: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[bool]:
    intersect1 = ccw(
        line1_p1_lon,
        line1_p1_lat,
        line1_p2_lon,
        line1_p2_lat,
        line2_p1_lon,
        line2_p1_lat,
    ) * ccw(
        line1_p1_lon,
        line1_p1_lat,
        line1_p2_lon,
        line1_p2_lat,
        line2_p2_lon,
        line2_p2_lat,
    )
    intersect2 = ccw(
        line2_p1_lon,
        line2_p1_lat,
        line2_p2_lon,
        line2_p2_lat,
        line1_p1_lon,
        line1_p1_lat,
    ) * ccw(
        line2_p1_lon,
        line2_p1_lat,
        line2_p2_lon,
        line2_p2_lat,
        line1_p2_lon,
        line1_p2_lat,
    )
    lintersect_out = where((intersect1 + intersect2) == -2, True, False)

    return lintersect_out


# Compute intersection point of two lines in 2D
@gtx.field_operator
def line_intersect(
    line1_p1_lon: fa.EdgeKField[ta.wpfloat],
    line1_p1_lat: fa.EdgeKField[ta.wpfloat],
    line1_p2_lon: fa.EdgeKField[ta.wpfloat],
    line1_p2_lat: fa.EdgeKField[ta.wpfloat],
    line2_p1_lon: fa.EdgeKField[ta.wpfloat],
    line2_p1_lat: fa.EdgeKField[ta.wpfloat],
    line2_p2_lon: fa.EdgeKField[ta.wpfloat],
    line2_p2_lat: fa.EdgeKField[ta.wpfloat],
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat]]:
    # avoid division with zero
    d1 = line1_p2_lon - line1_p1_lon
    d1 = where(d1 != 0.0, d1, line1_p2_lon)

    d2 = line2_p2_lon - line2_p1_lon
    d2 = where(d2 != 0.0, d2, line2_p2_lon)

    m1 = (line1_p2_lat - line1_p1_lat) / d1
    m2 = (line2_p2_lat - line2_p1_lat) / d2

    intersect_1 = (line2_p1_lat - line1_p1_lat + m1 * line1_p1_lon - m2 * line2_p1_lon) / (m1 - m2)
    intersect_2 = line1_p1_lat + m1 * (intersect_1 - line1_p1_lon)

    return intersect_1, intersect_2


@gtx.field_operator
def _prepare_ffsl_flux_area_patches_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    ptr_v3_lon: gtx.Field[gtx.Dims[dims.ECDim], ta.wpfloat],
    ptr_v3_lat: gtx.Field[gtx.Dims[dims.ECDim], ta.wpfloat],
    tangent_orientation_dsl: fa.EdgeField[ta.wpfloat],
    dreg_patch0_1_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_1_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lat_dsl: fa.EdgeKField[ta.vpfloat],
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    arrival_pts_1_lon_dsl = dreg_patch0_1_lon_dsl
    arrival_pts_1_lat_dsl = dreg_patch0_1_lat_dsl
    arrival_pts_2_lon_dsl = dreg_patch0_2_lon_dsl
    arrival_pts_2_lat_dsl = dreg_patch0_2_lat_dsl
    depart_pts_1_lon_dsl = dreg_patch0_4_lon_dsl  # indices have to be switched so that dep 1 belongs to arr 1 (and d2->a2)
    depart_pts_1_lat_dsl = dreg_patch0_4_lat_dsl
    depart_pts_2_lon_dsl = dreg_patch0_3_lon_dsl
    depart_pts_2_lat_dsl = dreg_patch0_3_lat_dsl

    lvn_pos = where(p_vn >= 0.0, True, False)

    # get flux area departure-line segment
    fl_line_p1_lon = depart_pts_1_lon_dsl
    fl_line_p1_lat = depart_pts_1_lat_dsl
    fl_line_p2_lon = depart_pts_2_lon_dsl
    fl_line_p2_lat = depart_pts_2_lat_dsl

    # get triangle edge 1 (A1V3)
    tri_line1_p1_lon = arrival_pts_1_lon_dsl
    tri_line1_p1_lat = arrival_pts_1_lat_dsl
    tri_line1_p2_lon = astype(
        where(
            lvn_pos,
            broadcast(ptr_v3_lon(E2EC[0]), (dims.EdgeDim, dims.KDim)),
            broadcast(ptr_v3_lon(E2EC[1]), (dims.EdgeDim, dims.KDim)),
        ),
        vpfloat,
    )
    tri_line1_p2_lat = astype(
        where(
            lvn_pos,
            broadcast(ptr_v3_lat(E2EC[0]), (dims.EdgeDim, dims.KDim)),
            broadcast(ptr_v3_lat(E2EC[1]), (dims.EdgeDim, dims.KDim)),
        ),
        vpfloat,
    )

    # get triangle edge 2 (A2V3)
    tri_line2_p1_lon = arrival_pts_2_lon_dsl
    tri_line2_p1_lat = arrival_pts_2_lat_dsl
    tri_line2_p2_lon = astype(
        where(
            lvn_pos,
            broadcast(ptr_v3_lon(E2EC[0]), (dims.EdgeDim, dims.KDim)),
            broadcast(ptr_v3_lon(E2EC[1]), (dims.EdgeDim, dims.KDim)),
        ),
        vpfloat,
    )
    tri_line2_p2_lat = astype(
        where(
            lvn_pos,
            broadcast(ptr_v3_lat(E2EC[0]), (dims.EdgeDim, dims.KDim)),
            broadcast(ptr_v3_lat(E2EC[1]), (dims.EdgeDim, dims.KDim)),
        ),
        vpfloat,
    )

    # Create first mask does departure-line segment intersects with A1V3
    lintersect_line1 = lintersect(
        astype(fl_line_p1_lon, wpfloat),
        astype(fl_line_p1_lat, wpfloat),
        astype(fl_line_p2_lon, wpfloat),
        astype(fl_line_p2_lat, wpfloat),
        astype(tri_line1_p1_lon, wpfloat),
        astype(tri_line1_p1_lat, wpfloat),
        astype(tri_line1_p2_lon, wpfloat),
        astype(tri_line1_p2_lat, wpfloat),
    )
    # Create first mask does departure-line segment intersects with A2V3
    lintersect_line2 = lintersect(
        astype(fl_line_p1_lon, wpfloat),
        astype(fl_line_p1_lat, wpfloat),
        astype(fl_line_p2_lon, wpfloat),
        astype(fl_line_p2_lat, wpfloat),
        astype(tri_line2_p1_lon, wpfloat),
        astype(tri_line2_p1_lat, wpfloat),
        astype(tri_line2_p2_lon, wpfloat),
        astype(tri_line2_p2_lat, wpfloat),
    )

    lvn_sys_pos = where(
        (p_vn * broadcast(tangent_orientation_dsl, (dims.EdgeDim, dims.KDim))) >= 0.0,
        True,
        False,
    )
    famask_bool = where(famask_int == 1, True, False)
    # ------------------------------------------------- Case 1
    mask_case1 = lintersect_line1 & lintersect_line2 & famask_bool
    ps1_x, ps1_y = line_intersect(
        astype(fl_line_p1_lon, wpfloat),
        astype(fl_line_p1_lat, wpfloat),
        astype(fl_line_p2_lon, wpfloat),
        astype(fl_line_p2_lat, wpfloat),
        astype(tri_line1_p1_lon, wpfloat),
        astype(tri_line1_p1_lat, wpfloat),
        astype(tri_line1_p2_lon, wpfloat),
        astype(tri_line1_p2_lat, wpfloat),
    )
    ps2_x, ps2_y = line_intersect(
        astype(fl_line_p1_lon, wpfloat),
        astype(fl_line_p1_lat, wpfloat),
        astype(fl_line_p2_lon, wpfloat),
        astype(fl_line_p2_lat, wpfloat),
        astype(tri_line2_p1_lon, wpfloat),
        astype(tri_line2_p1_lat, wpfloat),
        astype(tri_line2_p2_lon, wpfloat),
        astype(tri_line2_p2_lat, wpfloat),
    )

    # Case 1 - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case1, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case1, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case1,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, astype(ps1_x, vpfloat)),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case1,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, astype(ps1_y, vpfloat)),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case1, astype(ps2_x, vpfloat), dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case1, astype(ps2_y, vpfloat), dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case1,
        where(lvn_sys_pos, astype(ps1_x, vpfloat), arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case1,
        where(lvn_sys_pos, astype(ps1_y, vpfloat), arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 1 - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, vpfloat(0.0))
    dreg_patch1_1_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, vpfloat(0.0))
    dreg_patch1_4_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, vpfloat(0.0))
    dreg_patch1_4_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, vpfloat(0.0))
    dreg_patch1_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, astype(ps1_x, vpfloat), depart_pts_1_lon_dsl), vpfloat(0.0)
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, astype(ps1_y, vpfloat), depart_pts_1_lat_dsl), vpfloat(0.0)
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lon_dsl, astype(ps1_x, vpfloat)), vpfloat(0.0)
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lat_dsl, astype(ps1_y, vpfloat)), vpfloat(0.0)
    )
    # Case 1 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, vpfloat(0.0))
    dreg_patch2_1_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, vpfloat(0.0))
    dreg_patch2_4_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, vpfloat(0.0))
    dreg_patch2_4_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, vpfloat(0.0))
    dreg_patch2_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lon_dsl, astype(ps2_x, vpfloat)), vpfloat(0.0)
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lat_dsl, astype(ps2_y, vpfloat)), vpfloat(0.0)
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, astype(ps2_x, vpfloat), depart_pts_2_lon_dsl), vpfloat(0.0)
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, astype(ps2_y, vpfloat), depart_pts_2_lat_dsl), vpfloat(0.0)
    )

    # ------------------------------------------------- Case 2a
    mask_case2a = lintersect_line1 & (not lintersect_line2) & famask_bool
    # Case 2a - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, astype(ps1_x, vpfloat)),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, astype(ps1_y, vpfloat)),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case2a, depart_pts_2_lon_dsl, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2a, depart_pts_2_lat_dsl, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, astype(ps1_x, vpfloat), arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, astype(ps1_y, vpfloat), arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2a - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_4_lat_vmask)
    dreg_patch1_2_lon_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, astype(ps1_x, vpfloat), depart_pts_1_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, astype(ps1_y, vpfloat), depart_pts_1_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, astype(ps1_x, vpfloat)),
        dreg_patch1_3_lon_vmask,
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, astype(ps1_y, vpfloat)),
        dreg_patch1_3_lat_vmask,
    )
    # Case 2a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2a, vpfloat(0.0), dreg_patch2_4_lat_vmask)

    # -------------------------------------------------- Case 2b
    mask_case2b = lintersect_line2 & (not lintersect_line1) & famask_bool
    # Case 2b - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case2b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case2b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case2b, astype(ps2_x, vpfloat), dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2b, astype(ps2_y, vpfloat), dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2b - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2b, vpfloat(0.0), dreg_patch1_4_lat_vmask)
    # Case 2b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_4_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_2_lon_dsl, astype(ps2_x, vpfloat)),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_2_lat_dsl, astype(ps2_y, vpfloat)),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, astype(ps2_x, vpfloat), depart_pts_2_lon_dsl),
        dreg_patch2_3_lon_vmask,
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, astype(ps2_y, vpfloat), depart_pts_2_lat_dsl),
        dreg_patch2_3_lat_vmask,
    )

    # flux area edge 1 and 2
    fl_e1_p1_lon = arrival_pts_1_lon_dsl
    fl_e1_p1_lat = arrival_pts_1_lat_dsl
    fl_e1_p2_lon = depart_pts_1_lon_dsl
    fl_e1_p2_lat = depart_pts_1_lat_dsl
    fl_e2_p1_lon = arrival_pts_2_lon_dsl
    fl_e2_p1_lat = arrival_pts_2_lat_dsl
    fl_e2_p2_lon = depart_pts_2_lon_dsl
    fl_e2_p2_lat = depart_pts_2_lat_dsl

    # ----------------------------------------------- Case 3a
    # Check whether flux area edge 2 intersects with triangle edge 1
    lintersect_e2_line1 = lintersect(
        astype(fl_e2_p1_lon, wpfloat),
        astype(fl_e2_p1_lat, wpfloat),
        astype(fl_e2_p2_lon, wpfloat),
        astype(fl_e2_p2_lat, wpfloat),
        astype(tri_line1_p1_lon, wpfloat),
        astype(tri_line1_p1_lat, wpfloat),
        astype(tri_line1_p2_lon, wpfloat),
        astype(tri_line1_p2_lat, wpfloat),
    )
    mask_case3a = lintersect_e2_line1 & famask_bool
    pi1_x, pi1_y = line_intersect(
        astype(fl_e2_p1_lon, wpfloat),
        astype(fl_e2_p1_lat, wpfloat),
        astype(fl_e2_p2_lon, wpfloat),
        astype(fl_e2_p2_lat, wpfloat),
        astype(tri_line1_p1_lon, wpfloat),
        astype(tri_line1_p1_lat, wpfloat),
        astype(tri_line1_p2_lon, wpfloat),
        astype(tri_line1_p2_lat, wpfloat),
    )
    # Case 3a - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case3a, astype(ps2_x, vpfloat), dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case3a, astype(ps2_y, vpfloat), dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 3a - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, astype(pi1_x, vpfloat), depart_pts_2_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, astype(pi1_y, vpfloat), depart_pts_2_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = where(mask_case3a, depart_pts_1_lon_dsl, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3a, depart_pts_1_lat_dsl, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, astype(pi1_x, vpfloat)),
        dreg_patch1_4_lon_vmask,
    )
    dreg_patch1_4_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, astype(pi1_y, vpfloat)),
        dreg_patch1_4_lat_vmask,
    )
    # Case 3a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case3a, vpfloat(0.0), dreg_patch2_4_lat_vmask)

    # ------------------------------------------------ Case 3b
    # Check whether flux area edge 1 intersects with triangle edge 2
    lintersect_e1_line2 = lintersect(
        astype(fl_e1_p1_lon, wpfloat),
        astype(fl_e1_p1_lat, wpfloat),
        astype(fl_e1_p2_lon, wpfloat),
        astype(fl_e1_p2_lat, wpfloat),
        astype(tri_line2_p1_lon, wpfloat),
        astype(tri_line2_p1_lat, wpfloat),
        astype(tri_line2_p2_lon, wpfloat),
        astype(tri_line2_p2_lat, wpfloat),
    )
    mask_case3b = lintersect_e1_line2 & famask_bool
    pi2_x, pi2_y = line_intersect(
        astype(fl_e1_p1_lon, wpfloat),
        astype(fl_e1_p1_lat, wpfloat),
        astype(fl_e1_p2_lon, wpfloat),
        astype(fl_e1_p2_lat, wpfloat),
        astype(tri_line2_p1_lon, wpfloat),
        astype(tri_line2_p1_lat, wpfloat),
        astype(tri_line2_p2_lon, wpfloat),
        astype(tri_line2_p2_lat, wpfloat),
    )
    # Case 3b - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_4_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_4_lon_dsl)
    dreg_patch0_4_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_4_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, astype(pi2_x, vpfloat)),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, astype(pi2_y, vpfloat)),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, astype(pi2_x, vpfloat), arrival_pts_2_lon_dsl),
        dreg_patch0_3_lon_dsl,
    )
    dreg_patch0_3_lat_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, astype(pi2_y, vpfloat), arrival_pts_2_lat_dsl),
        dreg_patch0_3_lat_dsl,
    )
    # Case 3b - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case3b, vpfloat(0.0), dreg_patch1_4_lat_vmask)
    # Case 3b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, depart_pts_2_lon_dsl, astype(pi2_x, vpfloat)),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, depart_pts_2_lat_dsl, astype(pi2_y, vpfloat)),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = where(mask_case3b, depart_pts_1_lon_dsl, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3b, depart_pts_1_lat_dsl, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, astype(pi2_x, vpfloat), depart_pts_2_lon_dsl),
        dreg_patch2_4_lon_vmask,
    )
    dreg_patch2_4_lat_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, astype(pi2_y, vpfloat), depart_pts_2_lat_dsl),
        dreg_patch2_4_lat_vmask,
    )

    # --------------------------------------------- Case 4
    # NB: Next line acts as the "ELSE IF", indices that already previously matched one of the above conditions
    # can't be overwritten by this new condition.
    indices_previously_matched = mask_case3b | mask_case3a | mask_case2b | mask_case2a | mask_case1
    #    mask_case4 = (abs(p_vn) < 0.1) & famask_bool & (not indices_previously_matched) we insert also the error indices
    mask_case4 = famask_bool & (not indices_previously_matched)
    # Case 4 - patch 0 - no change
    # Case 4 - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch1_4_lat_vmask)
    # Case 4 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case4, vpfloat(0.0), dreg_patch2_4_lat_vmask)

    return (
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
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
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def prepare_ffsl_flux_area_patches_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    ptr_v3_lon: gtx.Field[gtx.Dims[dims.ECDim], ta.wpfloat],
    ptr_v3_lat: gtx.Field[gtx.Dims[dims.ECDim], ta.wpfloat],
    tangent_orientation_dsl: fa.EdgeField[ta.wpfloat],
    dreg_patch0_1_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_1_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _prepare_ffsl_flux_area_patches_list(
        famask_int,
        p_vn,
        ptr_v3_lon,
        ptr_v3_lat,
        tangent_orientation_dsl,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
        out=(
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
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
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
