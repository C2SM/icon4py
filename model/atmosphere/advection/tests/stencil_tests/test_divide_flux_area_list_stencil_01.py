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
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atmosphere.advection.divide_flux_area_list_stencil_01 import (
    divide_flux_area_list_stencil_01,
)
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    random_field,
    random_mask,
    zero_field,
)


# FUNCTIONS
# Checking turn when travelling along three points, used to check whether lines inters.
def ccw(
    p0_lon: np.array,
    p0_lat: np.array,
    p1_lon: np.array,
    p1_lat: np.array,
    p2_lon: np.array,
    p2_lat: np.array,
) -> np.array:

    dx1 = p1_lon - p0_lon
    dy1 = p1_lat - p0_lat

    dx2 = p2_lon - p0_lon
    dy2 = p2_lat - p0_lat

    dx1dy2 = dx1 * dy2
    dy1dx2 = dy1 * dx2

    lccw = np.where(dx1dy2 > dy1dx2, True, False)
    ccw_out = np.where(lccw, int32(1), int32(-1))  # 1: clockwise, -1: counterclockwise
    return ccw_out


# Checks whether two lines intersect
def lintersect(
    line1_p1_lon: np.array,
    line1_p1_lat: np.array,
    line1_p2_lon: np.array,
    line1_p2_lat: np.array,
    line2_p1_lon: np.array,
    line2_p1_lat: np.array,
    line2_p2_lon: np.array,
    line2_p2_lat: np.array,
) -> np.array:

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
    lintersect_out = np.where((intersect1 + intersect2) == -2, True, False)

    return lintersect_out


# Compute intersection point of two lines in 2D
def line_intersect(
    line1_p1_lon: np.array,
    line1_p1_lat: np.array,
    line1_p2_lon: np.array,
    line1_p2_lat: np.array,
    line2_p1_lon: np.array,
    line2_p1_lat: np.array,
    line2_p2_lon: np.array,
    line2_p2_lat: np.array,
) -> tuple[np.array]:

    m1 = (line1_p2_lat - line1_p1_lat) / (line1_p2_lon - line1_p1_lon)
    m2 = (line2_p2_lat - line2_p1_lat) / (line2_p2_lon - line2_p1_lon)

    intersect_1 = (line2_p1_lat - line1_p1_lat + m1 * line1_p1_lon - m2 * line2_p1_lon) / (m1 - m2)
    intersect_2 = line1_p1_lat + m1 * (intersect_1 - line1_p1_lon)

    return intersect_1, intersect_2


def divide_flux_area_list_stencil_01_numpy(
    e2c: np.array,
    famask_int: np.array,
    p_vn: np.array,
    ptr_v3_lon: np.array,
    ptr_v3_lat: np.array,
    tangent_orientation_dsl: np.array,
    dreg_patch0_1_lon_dsl: np.array,
    dreg_patch0_1_lat_dsl: np.array,
    dreg_patch0_2_lon_dsl: np.array,
    dreg_patch0_2_lat_dsl: np.array,
    dreg_patch0_3_lon_dsl: np.array,
    dreg_patch0_3_lat_dsl: np.array,
    dreg_patch0_4_lon_dsl: np.array,
    dreg_patch0_4_lat_dsl: np.array,
):
    ptr_v3_lon_e = np.expand_dims(ptr_v3_lon, axis=-1)
    ptr_v3_lat_e = np.expand_dims(ptr_v3_lat, axis=-1)
    ptr_v3_lon_e = np.expand_dims(ptr_v3_lon, axis=-1)
    ptr_v3_lat_e = np.expand_dims(ptr_v3_lat, axis=-1)
    tangent_orientation_dsl = np.expand_dims(tangent_orientation_dsl, axis=-1)

    arrival_pts_1_lon_dsl = dreg_patch0_1_lon_dsl
    arrival_pts_1_lat_dsl = dreg_patch0_1_lat_dsl
    arrival_pts_2_lon_dsl = dreg_patch0_2_lon_dsl
    arrival_pts_2_lat_dsl = dreg_patch0_2_lat_dsl
    depart_pts_1_lon_dsl = dreg_patch0_4_lon_dsl  # indices have to be switched so that dep 1 belongs to arr 1 (and d2->a2)
    depart_pts_1_lat_dsl = dreg_patch0_4_lat_dsl
    depart_pts_2_lon_dsl = dreg_patch0_3_lon_dsl
    depart_pts_2_lat_dsl = dreg_patch0_3_lat_dsl

    lvn_pos = np.where(p_vn >= 0.0, True, False)

    # get flux area departure-line segment
    fl_line_p1_lon = depart_pts_1_lon_dsl
    fl_line_p1_lat = depart_pts_1_lat_dsl
    fl_line_p2_lon = depart_pts_2_lon_dsl
    fl_line_p2_lat = depart_pts_2_lat_dsl

    # get triangle edge 1 (A1V3)
    tri_line1_p1_lon = arrival_pts_1_lon_dsl
    tri_line1_p1_lat = arrival_pts_1_lat_dsl
    tri_line1_p2_lon = np.where(
        lvn_pos,
        np.broadcast_to(ptr_v3_lon_e[:, 0], p_vn.shape),
        np.broadcast_to(ptr_v3_lon_e[:, 1], p_vn.shape),
    )
    tri_line1_p2_lat = np.where(
        lvn_pos,
        np.broadcast_to(ptr_v3_lat_e[:, 0], p_vn.shape),
        np.broadcast_to(ptr_v3_lat_e[:, 1], p_vn.shape),
    )

    # get triangle edge 2 (A2V3)
    tri_line2_p1_lon = arrival_pts_2_lon_dsl
    tri_line2_p1_lat = arrival_pts_2_lat_dsl
    tri_line2_p2_lon = np.where(
        lvn_pos,
        np.broadcast_to(ptr_v3_lon_e[:, 0], p_vn.shape),
        np.broadcast_to(ptr_v3_lon_e[:, 1], p_vn.shape),
    )
    tri_line2_p2_lat = np.where(
        lvn_pos,
        np.broadcast_to(ptr_v3_lat_e[:, 0], p_vn.shape),
        np.broadcast_to(ptr_v3_lat_e[:, 1], p_vn.shape),
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

    lvn_sys_pos = np.where(
        (p_vn * np.broadcast_to(tangent_orientation_dsl, p_vn.shape)) >= 0.0,
        True,
        False,
    )
    famask_bool = np.where(famask_int == int32(1), True, False)
    # ------------------------------------------------- Case 1
    mask_case1 = np.logical_and.reduce([lintersect_line1, lintersect_line2, famask_bool])
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
    dreg_patch0_1_lon_dsl = np.where(mask_case1, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = np.where(mask_case1, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = np.where(
        mask_case1,
        np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = np.where(
        mask_case1,
        np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = np.where(mask_case1, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = np.where(mask_case1, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = np.where(
        mask_case1,
        np.where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = np.where(
        mask_case1,
        np.where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 1 - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_1_lat_vmask = np.where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_4_lon_vmask = np.where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_4_lat_vmask = np.where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_2_lon_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl), 0.0
    )
    dreg_patch1_2_lat_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl), 0.0
    )
    dreg_patch1_3_lon_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x), 0.0
    )
    dreg_patch1_3_lat_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y), 0.0
    )
    # Case 1 - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_1_lat_vmask = np.where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_4_lon_vmask = np.where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_4_lat_vmask = np.where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_2_lon_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x), 0.0
    )
    dreg_patch2_2_lat_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y), 0.0
    )
    dreg_patch2_3_lon_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl), 0.0
    )
    dreg_patch2_3_lat_vmask = np.where(
        mask_case1, np.where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl), 0.0
    )

    # ------------------------------------------------- Case 2a
    mask_case2a = np.logical_and.reduce(
        [lintersect_line1, np.logical_not(lintersect_line2), famask_bool]
    )
    # Case 2a - patch 0
    dreg_patch0_1_lon_dsl = np.where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = np.where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = np.where(mask_case2a, depart_pts_2_lon_dsl, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = np.where(mask_case2a, depart_pts_2_lat_dsl, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2a - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = np.where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_4_lon_vmask = np.where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = np.where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_4_lat_vmask)
    dreg_patch1_2_lon_vmask = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x),
        dreg_patch1_3_lon_vmask,
    )
    dreg_patch1_3_lat_vmask = np.where(
        mask_case2a,
        np.where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y),
        dreg_patch1_3_lat_vmask,
    )
    # Case 2a - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_4_lat_vmask)

    # -------------------------------------------------- Case 2b
    mask_case2b = np.logical_and.reduce(
        [lintersect_line2, np.logical_not(lintersect_line1), famask_bool]
    )
    # Case 2b - patch 0
    dreg_patch0_1_lon_dsl = np.where(mask_case2b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = np.where(mask_case2b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = np.where(mask_case2b, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = np.where(mask_case2b, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2b - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case2b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = np.where(mask_case2b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = np.where(mask_case2b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = np.where(mask_case2b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = np.where(mask_case2b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = np.where(mask_case2b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = np.where(mask_case2b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = np.where(mask_case2b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 2b - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = np.where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_4_lon_vmask = np.where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = np.where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_4_lat_vmask)
    dreg_patch2_2_lon_vmask = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl),
        dreg_patch2_3_lon_vmask,
    )
    dreg_patch2_3_lat_vmask = np.where(
        mask_case2b,
        np.where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl),
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
    mask_case3a = np.logical_and(lintersect_e2_line1, famask_bool)
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
    dreg_patch0_1_lon_dsl = np.where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = np.where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = np.where(mask_case3a, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = np.where(mask_case3a, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 3a - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = np.where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, pi1_x, depart_pts_2_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, pi1_y, depart_pts_2_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = np.where(mask_case3a, depart_pts_1_lon_dsl, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = np.where(mask_case3a, depart_pts_1_lat_dsl, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, depart_pts_1_lon_dsl, pi1_x),
        dreg_patch1_4_lon_vmask,
    )
    dreg_patch1_4_lat_vmask = np.where(
        mask_case3a,
        np.where(lvn_sys_pos, depart_pts_1_lat_dsl, pi1_y),
        dreg_patch1_4_lat_vmask,
    )
    # Case 3a - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_4_lat_vmask)

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
    dreg_patch0_1_lon_dsl = np.where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = np.where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_4_lon_dsl = np.where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_4_lon_dsl)
    dreg_patch0_4_lat_dsl = np.where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_4_lat_dsl)
    dreg_patch0_2_lon_dsl = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, pi2_x),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, pi2_y),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, pi2_x, arrival_pts_2_lon_dsl),
        dreg_patch0_3_lon_dsl,
    )
    dreg_patch0_3_lat_dsl = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, pi2_y, arrival_pts_2_lat_dsl),
        dreg_patch0_3_lat_dsl,
    )
    # Case 3b - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 3b - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case3b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = np.where(mask_case3b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, depart_pts_2_lon_dsl, pi2_x),
        dreg_patch2_2_lon_vmask,
    )
    dreg_patch2_2_lat_vmask = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, depart_pts_2_lat_dsl, pi2_y),
        dreg_patch2_2_lat_vmask,
    )
    dreg_patch2_3_lon_vmask = np.where(mask_case3b, depart_pts_1_lon_dsl, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = np.where(mask_case3b, depart_pts_1_lat_dsl, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, pi2_x, depart_pts_2_lon_dsl),
        dreg_patch2_4_lon_vmask,
    )
    dreg_patch2_4_lat_vmask = np.where(
        mask_case3b,
        np.where(lvn_sys_pos, pi2_y, depart_pts_2_lat_dsl),
        dreg_patch2_4_lat_vmask,
    )

    # --------------------------------------------- Case 4
    # NB: Next line acts as the "ELSE IF", indices that already previously matched one of the above conditions
    # can't be overwritten by this new condition.
    indices_previously_matched = np.logical_or.reduce(
        [mask_case3b, mask_case3a, mask_case2b, mask_case2a, mask_case1]
    )
    #    mask_case4 = (abs(p_vn) < 0.1) & famask_bool & (not indices_previously_matched) we insert also the error indices
    mask_case4 = np.logical_and.reduce([famask_bool, np.logical_not(indices_previously_matched)])
    # Case 4 - patch 0 - no change
    # Case 4 - patch 1
    dreg_patch1_1_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_4_lat_vmask)
    # Case 4 - patch 2
    dreg_patch2_1_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_4_lon_vmask)

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


@pytest.mark.slow_tests
def test_divide_flux_area_list_stencil_01():
    grid = SimpleGrid()

    famask_int = random_mask(grid, EdgeDim, KDim, dtype=int32)
    p_vn = random_field(grid, EdgeDim, KDim)
    ptr_v3_lon = random_field(grid, EdgeDim, E2CDim)
    ptr_v3_lon_field = as_1D_sparse_field(ptr_v3_lon, ECDim)
    ptr_v3_lat = random_field(grid, EdgeDim, E2CDim)
    ptr_v3_lat_field = as_1D_sparse_field(ptr_v3_lat, ECDim)
    tangent_orientation_dsl = random_field(grid, EdgeDim)
    dreg_patch0_1_lon_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_1_lat_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_2_lon_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_2_lat_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_3_lon_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_3_lat_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_4_lon_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch0_4_lat_dsl = random_field(grid, EdgeDim, KDim)
    dreg_patch1_1_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_1_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_2_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_2_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_3_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_3_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_4_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch1_4_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_1_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_1_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_2_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_2_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_3_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_3_lat_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_4_lon_vmask = zero_field(grid, EdgeDim, KDim)
    dreg_patch2_4_lat_vmask = zero_field(grid, EdgeDim, KDim)

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
        ref_21,
        ref_22,
        ref_23,
        ref_24,
    ) = divide_flux_area_list_stencil_01_numpy(
        grid.connectivities[E2CDim],
        np.asarray(famask_int),
        np.asarray(p_vn),
        np.asarray(ptr_v3_lon),
        np.asarray(ptr_v3_lat),
        np.asarray(tangent_orientation_dsl),
        np.asarray(dreg_patch0_1_lon_dsl),
        np.asarray(dreg_patch0_1_lat_dsl),
        np.asarray(dreg_patch0_2_lon_dsl),
        np.asarray(dreg_patch0_2_lat_dsl),
        np.asarray(dreg_patch0_3_lon_dsl),
        np.asarray(dreg_patch0_3_lat_dsl),
        np.asarray(dreg_patch0_4_lon_dsl),
        np.asarray(dreg_patch0_4_lat_dsl),
    )

    divide_flux_area_list_stencil_01(
        famask_int,
        p_vn,
        ptr_v3_lon_field,
        ptr_v3_lat_field,
        tangent_orientation_dsl,
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
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, grid.size[E2CDim]),
        },
    )
    assert np.allclose(dreg_patch0_1_lon_dsl, ref_1)
    assert np.allclose(dreg_patch0_1_lat_dsl, ref_2)
    assert np.allclose(dreg_patch0_2_lon_dsl, ref_3)
    assert np.allclose(dreg_patch0_2_lat_dsl, ref_4)
    assert np.allclose(dreg_patch0_3_lon_dsl, ref_5)
    assert np.allclose(dreg_patch0_3_lat_dsl, ref_6)
    assert np.allclose(dreg_patch0_4_lon_dsl, ref_7)
    assert np.allclose(dreg_patch0_4_lat_dsl, ref_8)
    assert np.allclose(dreg_patch1_1_lon_vmask, ref_9)
    assert np.allclose(dreg_patch1_1_lat_vmask, ref_10)
    assert np.allclose(dreg_patch1_2_lon_vmask, ref_11)
    assert np.allclose(dreg_patch1_2_lat_vmask, ref_12)
    assert np.allclose(dreg_patch1_3_lon_vmask, ref_13)
    assert np.allclose(dreg_patch1_3_lat_vmask, ref_14)
    assert np.allclose(dreg_patch1_4_lon_vmask, ref_15)
    assert np.allclose(dreg_patch1_4_lat_vmask, ref_16)
    assert np.allclose(dreg_patch2_1_lon_vmask, ref_17)
    assert np.allclose(dreg_patch2_1_lat_vmask, ref_18)
    assert np.allclose(dreg_patch2_2_lon_vmask, ref_19)
    assert np.allclose(dreg_patch2_2_lat_vmask, ref_20)
    assert np.allclose(dreg_patch2_3_lon_vmask, ref_21)
    assert np.allclose(dreg_patch2_3_lat_vmask, ref_22)
    assert np.allclose(dreg_patch2_4_lon_vmask, ref_23)
    assert np.allclose(dreg_patch2_4_lat_vmask, ref_24)
