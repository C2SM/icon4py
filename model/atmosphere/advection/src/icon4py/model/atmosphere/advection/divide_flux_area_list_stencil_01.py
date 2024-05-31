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
from gt4py.next.ffront.fbuiltins import Field, broadcast, where
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import E2EC, ECDim, EdgeDim, KDim


sys.setrecursionlimit(5500)


# FUNCTIONS
# Checking turn when travelling along three points, used to check whether lines inters.
@field_operator
def ccw(
    p0_lon: Field[[EdgeDim, KDim], float],
    p0_lat: Field[[EdgeDim, KDim], float],
    p1_lon: Field[[EdgeDim, KDim], float],
    p1_lat: Field[[EdgeDim, KDim], float],
    p2_lon: Field[[EdgeDim, KDim], float],
    p2_lat: Field[[EdgeDim, KDim], float],
) -> fa.EKintField:
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
@field_operator
def lintersect(
    line1_p1_lon: Field[[EdgeDim, KDim], float],
    line1_p1_lat: Field[[EdgeDim, KDim], float],
    line1_p2_lon: Field[[EdgeDim, KDim], float],
    line1_p2_lat: Field[[EdgeDim, KDim], float],
    line2_p1_lon: Field[[EdgeDim, KDim], float],
    line2_p1_lat: Field[[EdgeDim, KDim], float],
    line2_p2_lon: Field[[EdgeDim, KDim], float],
    line2_p2_lat: Field[[EdgeDim, KDim], float],
) -> fa.EKboolField:
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
@field_operator
def line_intersect(
    line1_p1_lon: Field[[EdgeDim, KDim], float],
    line1_p1_lat: Field[[EdgeDim, KDim], float],
    line1_p2_lon: Field[[EdgeDim, KDim], float],
    line1_p2_lat: Field[[EdgeDim, KDim], float],
    line2_p1_lon: Field[[EdgeDim, KDim], float],
    line2_p1_lat: Field[[EdgeDim, KDim], float],
    line2_p2_lon: Field[[EdgeDim, KDim], float],
    line2_p2_lat: Field[[EdgeDim, KDim], float],
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
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


@field_operator
def _divide_flux_area_list_stencil_01(
    famask_int: fa.EKintField,
    p_vn: Field[[EdgeDim, KDim], float],
    ptr_v3_lon: Field[[ECDim], float],
    ptr_v3_lat: Field[[ECDim], float],
    tangent_orientation_dsl: Field[[EdgeDim], float],
    dreg_patch0_1_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_1_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_2_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_2_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_3_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_3_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_4_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_4_lat_dsl: Field[[EdgeDim, KDim], float],
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
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
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
    tri_line1_p2_lon = where(
        lvn_pos,
        broadcast(ptr_v3_lon(E2EC[0]), (EdgeDim, KDim)),
        broadcast(ptr_v3_lon(E2EC[1]), (EdgeDim, KDim)),
    )
    tri_line1_p2_lat = where(
        lvn_pos,
        broadcast(ptr_v3_lat(E2EC[0]), (EdgeDim, KDim)),
        broadcast(ptr_v3_lat(E2EC[1]), (EdgeDim, KDim)),
    )

    # get triangle edge 2 (A2V3)
    tri_line2_p1_lon = arrival_pts_2_lon_dsl
    tri_line2_p1_lat = arrival_pts_2_lat_dsl
    tri_line2_p2_lon = where(
        lvn_pos,
        broadcast(ptr_v3_lon(E2EC[0]), (EdgeDim, KDim)),
        broadcast(ptr_v3_lon(E2EC[1]), (EdgeDim, KDim)),
    )
    tri_line2_p2_lat = where(
        lvn_pos,
        broadcast(ptr_v3_lat(E2EC[0]), (EdgeDim, KDim)),
        broadcast(ptr_v3_lat(E2EC[1]), (EdgeDim, KDim)),
    )

    # Create first mask does departure-line segment intersects with A1V3
    lintersect_line1 = lintersect(
        fl_line_p1_lon,
        fl_line_p1_lat,
        fl_line_p2_lon,
        fl_line_p2_lat,
        tri_line1_p1_lon,
        tri_line1_p1_lat,
        tri_line1_p2_lon,
        tri_line1_p2_lat,
    )
    # Create first mask does departure-line segment intersects with A2V3
    lintersect_line2 = lintersect(
        fl_line_p1_lon,
        fl_line_p1_lat,
        fl_line_p2_lon,
        fl_line_p2_lat,
        tri_line2_p1_lon,
        tri_line2_p1_lat,
        tri_line2_p2_lon,
        tri_line2_p2_lat,
    )

    lvn_sys_pos = where(
        (p_vn * broadcast(tangent_orientation_dsl, (EdgeDim, KDim))) >= 0.0, True, False
    )
    famask_bool = where(famask_int == 1, True, False)
    # ------------------------------------------------- Case 1
    mask_case1 = lintersect_line1 & lintersect_line2 & famask_bool
    ps1_x, ps1_y = line_intersect(
        fl_line_p1_lon,
        fl_line_p1_lat,
        fl_line_p2_lon,
        fl_line_p2_lat,
        tri_line1_p1_lon,
        tri_line1_p1_lat,
        tri_line1_p2_lon,
        tri_line1_p2_lat,
    )
    ps2_x, ps2_y = line_intersect(
        fl_line_p1_lon,
        fl_line_p1_lat,
        fl_line_p2_lon,
        fl_line_p2_lat,
        tri_line2_p1_lon,
        tri_line2_p1_lat,
        tri_line2_p2_lon,
        tri_line2_p2_lat,
    )

    # Case 1 - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case1, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case1, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case1,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case1,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case1, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case1, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case1,
        where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case1,
        where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 1 - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_1_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_4_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_4_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl), 0.0
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl), 0.0
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x), 0.0
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y), 0.0
    )
    # Case 1 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_1_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_4_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_4_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x), 0.0
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y), 0.0
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl), 0.0
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl), 0.0
    )

    # ------------------------------------------------- Case 2a
    mask_case2a = lintersect_line1 & (not lintersect_line2) & famask_bool
    # Case 2a - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case2a, depart_pts_2_lon_dsl, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2a, depart_pts_2_lat_dsl, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case2a,
        where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2a - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_4_lat_vmask)
    dreg_patch1_2_lon_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x),
        dreg_patch1_3_lon_vmask,
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case2a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y),
        dreg_patch1_3_lat_vmask,
    )
    # Case 2a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_4_lat_vmask)

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
    dreg_patch0_3_lon_dsl = where(mask_case2b, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2b, ps2_y, dreg_patch0_3_lat_dsl)
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
    dreg_patch1_1_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 2b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_4_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl),
        dreg_patch2_3_lon_vmask,
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case2b,
        where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl),
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
        fl_e2_p1_lon,
        fl_e2_p1_lat,
        fl_e2_p2_lon,
        fl_e2_p2_lat,
        tri_line1_p1_lon,
        tri_line1_p1_lat,
        tri_line1_p2_lon,
        tri_line1_p2_lat,
    )
    mask_case3a = lintersect_e2_line1 & famask_bool
    pi1_x, pi1_y = line_intersect(
        fl_e2_p1_lon,
        fl_e2_p1_lat,
        fl_e2_p2_lon,
        fl_e2_p2_lat,
        tri_line1_p1_lon,
        tri_line1_p1_lat,
        tri_line1_p2_lon,
        tri_line1_p2_lat,
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
    dreg_patch0_3_lon_dsl = where(mask_case3a, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case3a, ps2_y, dreg_patch0_3_lat_dsl)
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
        where(lvn_sys_pos, pi1_x, depart_pts_2_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, pi1_y, depart_pts_2_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = where(mask_case3a, depart_pts_1_lon_dsl, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3a, depart_pts_1_lat_dsl, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, pi1_x),
        dreg_patch1_4_lon_vmask,
    )
    dreg_patch1_4_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, pi1_y),
        dreg_patch1_4_lat_vmask,
    )
    # Case 3a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_4_lat_vmask)

    # ------------------------------------------------ Case 3b
    # Check whether flux area edge 1 intersects with triangle edge 2
    lintersect_e1_line2 = lintersect(
        fl_e1_p1_lon,
        fl_e1_p1_lat,
        fl_e1_p2_lon,
        fl_e1_p2_lat,
        tri_line2_p1_lon,
        tri_line2_p1_lat,
        tri_line2_p2_lon,
        tri_line2_p2_lat,
    )
    mask_case3b = lintersect_e1_line2 & famask_bool
    pi2_x, pi2_y = line_intersect(
        fl_e1_p1_lon,
        fl_e1_p1_lat,
        fl_e1_p2_lon,
        fl_e1_p2_lat,
        tri_line2_p1_lon,
        tri_line2_p1_lat,
        tri_line2_p2_lon,
        tri_line2_p2_lat,
    )
    # Case 3b - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_4_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_4_lon_dsl)
    dreg_patch0_4_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_4_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, pi2_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, pi2_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, pi2_x, arrival_pts_2_lon_dsl),
        dreg_patch0_3_lon_dsl,
    )
    dreg_patch0_3_lat_dsl = where(
        mask_case3b,
        where(lvn_sys_pos, pi2_y, arrival_pts_2_lat_dsl),
        dreg_patch0_3_lat_dsl,
    )
    # Case 3b - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 3b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, depart_pts_2_lon_dsl, pi2_x),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, depart_pts_2_lat_dsl, pi2_y),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = where(mask_case3b, depart_pts_1_lon_dsl, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3b, depart_pts_1_lat_dsl, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, pi2_x, depart_pts_2_lon_dsl),
        dreg_patch2_4_lon_vmask,
    )
    dreg_patch2_4_lat_vmask = where(
        mask_case3b,
        where(lvn_sys_pos, pi2_y, depart_pts_2_lat_dsl),
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
    dreg_patch1_1_lon_vmask = where(mask_case4, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case4, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case4, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case4, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case4, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case4, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case4, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case4, 0.0, dreg_patch1_4_lat_vmask)
    # Case 4 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case4, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case4, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case4, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case4, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case4, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case4, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case4, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case4, 0.0, dreg_patch2_4_lat_vmask)

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


@program(grid_type=GridType.UNSTRUCTURED)
def divide_flux_area_list_stencil_01(
    famask_int: fa.EKintField,
    p_vn: Field[[EdgeDim, KDim], float],
    ptr_v3_lon: Field[[ECDim], float],
    ptr_v3_lat: Field[[ECDim], float],
    tangent_orientation_dsl: Field[[EdgeDim], float],
    dreg_patch0_1_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_1_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_2_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_2_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_3_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_3_lat_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_4_lon_dsl: Field[[EdgeDim, KDim], float],
    dreg_patch0_4_lat_dsl: Field[[EdgeDim, KDim], float],
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
):
    _divide_flux_area_list_stencil_01(
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
    )
