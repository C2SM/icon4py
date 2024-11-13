# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import exp, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import geometry, icon as icon_grid
from icon4py.model.common.settings import xp


@gtx.field_operator
def _gaussian_hill(
    cell_center_lon: fa.CellField[ta.wpfloat],
    cell_center_lat: fa.CellField[ta.wpfloat],
    mount_x: ta.wpfloat,
    mount_y: ta.wpfloat,
    mount_height: ta.wpfloat,
    mount_width: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the elevation of a parameterized Gaussian hill.
    """

    torus_domain_length = 5000 # TODO get this from the grid when actually using it
    if mount_lon < 0:
        mount_lon = torus_domain_length / 2
    if mount_lat < 0:
        mount_lat = 0

    cartesian_x = cell_center_lon * torus_domain_length / xp.pi / 2,
    cartesian_y = cell_center_lat * torus_domain_length / xp.pi / 2,

    z_distance = sqrt((cartesian_x - mount_x) ** 2 + (cartesian_y - mount_y) ** 2)
    topography = mount_height * exp(-((z_distance / mount_width) ** 2))

    return topography


def gaussian_hill(
    grid: icon_grid.IconGrid,
    cell_geometry: geometry.CellParams,
    mount_lon: ta.wpfloat = -1,
    mount_lat: ta.wpfloat = -1,
    mount_height: ta.wpfloat = 100,
    mount_width: ta.wpfloat = 2000,
) -> fa.CellField[ta.wpfloat]:
    """
    Generates a Gaussian hill topography on the given grid.

    Parameters:
    grid: The grid on which the topography is to be generated.
    cell_geometry: The parameters defining the cell geometry.
    mount_lon: The longitude of the mountain peak. Defaults to -1.
    mount_lat: The latitude of the mountain peak. Defaults to -1.
    mount_height: The height of the mountain. Defaults to 100.
    mount_width: The width of the mountain. Defaults to 2000.

    Returns:
    The generated elevation field.
    """
    topography = gtx.as_field((dims.CellDim,), xp.zeros((grid.num_cells,), dtype=ta.wpfloat))

    _gaussian_hill(
        cell_center_lon=cell_geometry.cell_center_lon,
        cell_center_lat=cell_geometry.cell_center_lat,
        mount_x=mount_lon,
        mount_y=mount_lat,
        mount_height=mount_height,
        mount_width=mount_width,
        out=topography,
        offset_provider={},
    )
    return topography
