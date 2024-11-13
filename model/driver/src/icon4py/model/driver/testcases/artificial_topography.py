# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, exp, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _gaussian_hill(
    cell_center_lon: fa.CellField[ta.wpfloat],
    cell_center_lat: fa.CellField[ta.wpfloat],
    mount_lon: ta.wpfloat,
    mount_lat: ta.wpfloat,
    mount_height: ta.wpfloat,
    mount_width: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:
    """
    Generates a Gaussian hill elevation profile on the given grid.

    Parameters:
    cartesian_x: The x-coordinates of the grid cells.
    cartesian_y: The y-coordinates of the grid cells.
    mount_lon: The longitude of the hill.
    mount_lat: The latitude of the hill.
    mount_height: The height of the hill in meters.
    mount_width: The "width" of the hill.

    Returns:
    The generated elevation field.
    """

    z_distance = sqrt((cell_center_lon - broadcast(mount_lon, (dims.CellDim,))) ** 2 + (cell_center_lat - broadcast(mount_lat, (dims.CellDim,))) ** 2)
    topography = mount_height * exp(-((z_distance / mount_width) ** 2))

    return topography


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def gaussian_hill(
    cell_center_lon: fa.CellField[ta.wpfloat],
    cell_center_lat: fa.CellField[ta.wpfloat],
    topography: fa.CellField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    mount_lon: ta.wpfloat = 0.0,
    mount_lat: ta.wpfloat = 0.0,
    mount_height: ta.wpfloat = 100.0,
    mount_width: ta.wpfloat = 500.0,
):
    _gaussian_hill(
        cell_center_lon=cell_center_lon,
        cell_center_lat=cell_center_lat,
        mount_lon=mount_lon,
        mount_lat=mount_lat,
        mount_height=mount_height,
        mount_width=mount_width,
        out=topography,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
